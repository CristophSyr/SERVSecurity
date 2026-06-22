from datetime import datetime
from typing import Optional


# ── Tipos de evento ────────────────────────────────────────────────────────────
EVENT_INTRUSION        = "Intrusión en zona restringida"
EVENT_PERMANENCIA      = "Permanencia prolongada"
EVENT_HORARIO          = "Acceso fuera de horario"
EVENT_PRESENCIA        = "Presencia detectada"
EVENT_COMPORTAMIENTO_R = "Movimiento errático (Carrera/Forcejeo)"
EVENT_COMPORTAMIENTO_P = "Postura inusual (Caída/Agachamiento)"


def point_in_rect(px: int, py: int, rect: tuple) -> bool:
    """
    Determina si el punto (px, py) se encuentra dentro del rectángulo.

    Args:
        px, py: Coordenadas del punto (generalmente el centro de la persona).
        rect: Tupla (x1, y1, x2, y2) del rectángulo.

    Returns:
        True si el punto está dentro o en el borde del rectángulo.
    """
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


def bbox_overlaps_rect(bbox: tuple, rect: tuple) -> bool:
    """
    Verifica si el bounding box de una persona se superpone con la zona restringida.
    Más conservador que solo verificar el centro.

    Args:
        bbox: (x1, y1, x2, y2) del bounding box de la persona.
        rect: (x1, y1, x2, y2) de la zona restringida.

    Returns:
        True si hay superposición.
    """
    bx1, by1, bx2, by2 = bbox
    rx1, ry1, rx2, ry2 = rect
    return not (bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2)


def is_within_schedule(allowed_start: str, allowed_end: str) -> bool:
    """
    Verifica si la hora actual está dentro del rango horario permitido.
    Acepta formatos flexibles: '8:00', '08:00', '8:0', '20:30'.
    """
    now = datetime.now().time()
    try:
        start = datetime.strptime(allowed_start.strip(), "%H:%M").time()
        end = datetime.strptime(allowed_end.strip(), "%H:%M").time()

        if start <= end:
            return start <= now <= end
        else:
            return now >= start or now <= end
    except (ValueError, AttributeError):
        return True


class TrackingState:
    """
    Almacena el estado de seguimiento de UNA persona detectada.
    Se identifica de manera aproximada por la posición del centro.
    """

    def __init__(self, detection_id: int, center: tuple, timestamp: datetime):
        self.detection_id    = detection_id
        self.center          = center
        self.first_seen      = timestamp
        self.last_seen       = timestamp
        self.in_zone         = False
        self.zone_entry_time: Optional[datetime] = None
        self.time_in_zone    = 0.0       # segundos acumulados en zona
        self.alert_fired     = False     # para no disparar la misma alerta repetida
        self.behavior_alert  = False     # para no repetir alerta de comportamiento
        self.velocity        = 0.0       # píxeles por segundo
        self.is_authorized: Optional[bool] = None # None=Pendiente, True=OK, False=Desconocido
        self.auth_pending    = False     # True si ya se envio a analizar
        self.auth_attempts   = 0         # Intentos de reconocimiento
        self.last_auth_time  = 0.0       # Timestamp del ultimo intento


class RulesEngine:
    """
    Motor de reglas que analiza las detecciones y genera eventos de anomalía.
    """

    def __init__(
        self,
        zone: tuple = (100, 100, 400, 400),
        max_permanence_sec: float = 5.0,
        allowed_start: str = "08:00",
        allowed_end: str   = "20:00",
        use_overlap: bool  = True,
        max_velocity_px_sec: float = 800.0,
        max_aspect_ratio: float = 1.3,
        authenticator = None,
    ):
        """
        Args:
            zone: Zona restringida (x1, y1, x2, y2) en píxeles.
            max_permanence_sec: Segundos máximos permitidos dentro de la zona.
            allowed_start: Hora de inicio del horario permitido (HH:MM).
            allowed_end:   Hora de fin   del horario permitido (HH:MM).
            use_overlap:   Si True, usa superposición de bbox; si False, usa solo el centro.
        """
        self.zone               = zone
        self.max_permanence_sec = max_permanence_sec
        self.allowed_start      = allowed_start
        self.allowed_end        = allowed_end
        self.use_overlap        = use_overlap
        self.max_velocity_px_sec = max_velocity_px_sec
        self.max_aspect_ratio   = max_aspect_ratio
        self.authenticator      = authenticator

        # Rastreo persistente entre frames (clave: ID aproximado)
        self._tracks: dict[int, TrackingState] = {}
        self._next_id = 0

    def update_zone(self, zone: tuple):
        """Actualiza la zona restringida."""
        self.zone = zone

    def _find_or_create_track(self, center: tuple, now: datetime) -> TrackingState:
        """
        Asocia una detección actual con una pista existente (por proximidad)
        o crea una nueva.
        """
        DIST_THRESHOLD = 80   # píxeles
        best_id   = None
        best_dist = float("inf")

        for tid, track in self._tracks.items():
            dx = center[0] - track.center[0]
            dy = center[1] - track.center[1]
            dist = (dx**2 + dy**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id   = tid

        if best_id is not None and best_dist < DIST_THRESHOLD:
            track = self._tracks[best_id]
            # Calcular velocidad (píxeles por segundo)
            dt = (now - track.last_seen).total_seconds()
            if dt > 0:
                track.velocity = best_dist / dt
            else:
                track.velocity = 0.0
                
            track.center    = center
            track.last_seen = now
            return track

        # Nueva pista
        new_id = self._next_id
        self._next_id += 1
        track = TrackingState(new_id, center, now)
        self._tracks[new_id] = track
        return track

    def _cleanup_old_tracks(self, now: datetime, timeout_sec: float = 3.0):
        """Elimina pistas que no han sido actualizadas recientemente."""
        to_delete = []
        for tid, track in self._tracks.items():
            elapsed = (now - track.last_seen).total_seconds()
            if elapsed > timeout_sec:
                to_delete.append(tid)
        for tid in to_delete:
            del self._tracks[tid]

    def evaluate(
        self,
        detections: list[dict],
        frame_timestamp: Optional[datetime] = None,
    ) -> tuple[list[bool], list[dict]]:
        """
        Evalúa las detecciones del frame actual y determina alertas.

        Args:
            detections: Lista de detecciones del PersonDetector.
            frame_timestamp: Momento del frame (por defecto: ahora).

        Returns:
            alerts: Lista booleana paralela a detections.
            events: Lista de eventos generados (para registrar en BD).
        """
        now = frame_timestamp or datetime.now()
        alerts = [False] * len(detections)
        events = []

        # Horario permitido
        within_schedule = is_within_schedule(self.allowed_start, self.allowed_end)

        for i, det in enumerate(detections):
            center = det["center"]
            bbox   = det["bbox"]
            kpts   = det.get("keypoints")
            crop   = det.get("crop")

            track = self._find_or_create_track(center, now)
            
            # Pasar el estado de autenticacion a la deteccion para que se dibuje
            det["is_authorized"] = track.is_authorized
            
            # Lanzar validación facial asíncrona si está pendiente y tenemos crop
            if self.authenticator and crop is not None and crop.size > 0:
                # Reintentar hasta 5 veces si es desconocido (por si estaba muy lejos al inicio)
                needs_auth = (track.is_authorized is None) or (track.is_authorized is False and track.auth_attempts < 5)
                
                if needs_auth and not track.auth_pending:
                    if now.timestamp() - track.last_auth_time > 1.5: # 1.5 seg entre intentos
                        track.auth_pending = True
                        track.last_auth_time = now.timestamp()
                        track.auth_attempts += 1
                        self.authenticator.authenticate_async(crop, track)

            # ── Análisis de Comportamiento (Behavioral Anomaly) ────────────────
            behavior_anomalous = False
            behavior_reason = ""
            
            # 1. Postura Inusual (Caída / Gateo)
            # Evaluar proporción del Bounding Box (ancho vs alto)
            bx1, by1, bx2, by2 = bbox
            w = bx2 - bx1
            h = by2 - by1
            aspect_ratio = w / h if h > 0 else 0
            
            # Evaluar Keypoints si están disponibles (Nariz debajo de la cadera)
            head_below_waist = False
            if kpts is not None and len(kpts) >= 13:
                nose = kpts[0]
                hip_l = kpts[11]
                hip_r = kpts[12]
                if nose[0] != 0 and hip_l[0] != 0:
                    # Coordenada Y crece hacia abajo en imágenes
                    if nose[1] > hip_l[1]:
                        head_below_waist = True

            if aspect_ratio > self.max_aspect_ratio or head_below_waist:
                behavior_anomalous = True
                behavior_reason = EVENT_COMPORTAMIENTO_P
                alerts[i] = True
                
            # 2. Movimiento Errático / Rápido (Carrera / Forcejeo)
            if track.velocity > self.max_velocity_px_sec:
                behavior_anomalous = True
                behavior_reason = EVENT_COMPORTAMIENTO_R
                alerts[i] = True

            if behavior_anomalous and not track.behavior_alert:
                track.behavior_alert = True
                events.append({
                    "tipo_evento": behavior_reason,
                    "estado":      "sospechoso",
                    "duracion":    0.0,
                    "detalle":     f"Velocidad: {track.velocity:.0f} px/s | Aspecto: {aspect_ratio:.2f}",
                })
            elif not behavior_anomalous:
                # Resetear alerta si vuelve a comportamiento normal
                track.behavior_alert = False

            # ── Regla 1: persona en zona restringida ───────────────────────────
            if self.use_overlap:
                in_zone = bbox_overlaps_rect(bbox, self.zone)
            else:
                in_zone = point_in_rect(center[0], center[1], self.zone)

            if in_zone:
                if not track.in_zone:
                    # Acaba de entrar a la zona
                    track.in_zone         = True
                    track.zone_entry_time = now
                    track.alert_fired     = False
                    
                    # Logica de Autenticación Facial:
                    # Si es Desconocido (is_authorized == False), es Intrusión Inmediata
                    if track.is_authorized is False:
                        alerts[i] = True
                        events.append({
                            "tipo_evento": EVENT_INTRUSION,
                            "estado":      "sospechoso",
                            "duracion":    0.0,
                            "detalle":     f"ROSTRO DESCONOCIDO. Centro: {center}",
                        })
                    # Si no esta autorizado, y esta fuera de horario, tambien es sospechoso
                    elif not within_schedule:
                        alerts[i] = True
                        events.append({
                            "tipo_evento": EVENT_HORARIO,
                            "estado":      "sospechoso",
                            "duracion":    0.0,
                            "detalle":     f"Acceso a las {now.strftime('%H:%M')} (horario: {self.allowed_start}–{self.allowed_end})",
                        })
                    else:
                        # Ingreso normal (en horario permitido y validando/validado)
                        auth_status = "Autorizado" if track.is_authorized else "Analizando rostro..."
                        events.append({
                            "tipo_evento": "Ingreso a zona",
                            "estado":      "normal",
                            "duracion":    0.0,
                            "detalle":     f"{auth_status}. Centro: {center}",
                        })

                # Mantener la alerta activa si es un desconocido o fuera de horario
                if track.is_authorized is False or not within_schedule:
                    alerts[i] = True

                # ── Regla 2: permanencia prolongada ───────────────────────────
                if track.zone_entry_time:
                    time_in_zone = (now - track.zone_entry_time).total_seconds()
                    track.time_in_zone = time_in_zone

                    if time_in_zone >= self.max_permanence_sec and not track.alert_fired:
                        track.alert_fired = True
                        alerts[i] = True
                        events.append({
                            "tipo_evento": EVENT_PERMANENCIA,
                            "estado":      "sospechoso",
                            "duracion":    round(time_in_zone, 1),
                            "detalle":     f"Permanencia: {time_in_zone:.1f}s (máx {self.max_permanence_sec}s)",
                        })
                    
                    # Mantener alerta visual si ya se excedio el tiempo
                    if track.alert_fired:
                        alerts[i] = True

            else:
                # Salió de la zona
                if track.in_zone:
                    track.in_zone     = False
                    track.alert_fired = False

            # ── Evento de presencia normal (fuera de zona, en horario) ─────────
            if not in_zone and within_schedule:
                events.append({
                    "tipo_evento": EVENT_PRESENCIA,
                    "estado":      "normal",
                    "duracion":    0.0,
                    "detalle":     f"Persona detectada. Conf: {det['confidence']:.0%}",
                })

        self._cleanup_old_tracks(now)
        return alerts, events
