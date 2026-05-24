"""
rules.py – Motor de reglas para detección de anomalías de comportamiento.

Las anomalías se definen como:
  1. Persona dentro de la zona restringida.
  2. Permanencia mayor a X segundos dentro de la zona.
  3. Ingreso durante horario no permitido.
"""

from datetime import datetime, time as dtime
from typing import Optional


# ── Tipos de evento ────────────────────────────────────────────────────────────
EVENT_INTRUSION        = "Intrusión en zona restringida"
EVENT_PERMANENCIA      = "Permanencia prolongada"
EVENT_HORARIO          = "Acceso fuera de horario"
EVENT_PRESENCIA        = "Presencia detectada"


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

    Args:
        allowed_start: Hora de inicio permitida en formato "HH:MM".
        allowed_end:   Hora de fin   permitida en formato "HH:MM".

    Returns:
        True si la hora actual está dentro del rango.
    """
    now = datetime.now().time()
    try:
        h_start, m_start = map(int, allowed_start.split(":"))
        h_end, m_end     = map(int, allowed_end.split(":"))
        start = dtime(h_start, m_start)
        end   = dtime(h_end,   m_end)

        if start <= end:
            return start <= now <= end
        else:
            # Rango cruza medianoche (ej. 22:00 – 06:00)
            return now >= start or now <= end
    except (ValueError, AttributeError):
        return True   # Si hay error de parseo, no bloquear


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
            self._tracks[best_id].center    = center
            self._tracks[best_id].last_seen = now
            return self._tracks[best_id]

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

            track = self._find_or_create_track(center, now)

            # ── Regla 1: persona en zona restringida ───────────────────────────
            if self.use_overlap:
                in_zone = bbox_overlaps_rect(bbox, self.zone)
            else:
                in_zone = point_in_rect(center[0], center[1], self.zone)

            if in_zone:
                alerts[i] = True

                if not track.in_zone:
                    # Acaba de entrar a la zona
                    track.in_zone         = True
                    track.zone_entry_time = now
                    track.alert_fired     = False
                    events.append({
                        "tipo_evento": EVENT_INTRUSION,
                        "estado":      "sospechoso",
                        "duracion":    0.0,
                        "detalle":     f"Centro: {center}",
                    })

                # ── Regla 2: permanencia prolongada ───────────────────────────
                if track.zone_entry_time:
                    time_in_zone = (now - track.zone_entry_time).total_seconds()
                    track.time_in_zone = time_in_zone

                    if time_in_zone >= self.max_permanence_sec and not track.alert_fired:
                        track.alert_fired = True
                        events.append({
                            "tipo_evento": EVENT_PERMANENCIA,
                            "estado":      "sospechoso",
                            "duracion":    round(time_in_zone, 1),
                            "detalle":     f"Permanencia: {time_in_zone:.1f}s (máx {self.max_permanence_sec}s)",
                        })

                # ── Regla 3: fuera de horario ──────────────────────────────────
                if not within_schedule:
                    alerts[i] = True
                    events.append({
                        "tipo_evento": EVENT_HORARIO,
                        "estado":      "sospechoso",
                        "duracion":    0.0,
                        "detalle":     f"Acceso a las {now.strftime('%H:%M')} (horario: {self.allowed_start}–{self.allowed_end})",
                    })

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
