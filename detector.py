"""
detector.py – Módulo de detección de personas con YOLOv8 preentrenado.
Utiliza el modelo yolov8n-pose.pt (nano) para extraer puntos articulares (esqueleto) y analizar comportamiento.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime

# Importación lazy para no fallar si ultralytics no está instalado en tiempo de import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Clase única para detectar: persona (class_id = 0 en COCO)
PERSON_CLASS_ID = 0

# Colores BGR
COLOR_BOX_NORMAL     = (0, 255, 0)       # verde
COLOR_BOX_ALERT      = (0, 0, 255)       # rojo
COLOR_ZONE_NORMAL    = (0, 165, 255)     # naranja
COLOR_ZONE_ALERT     = (0, 0, 255)       # rojo
COLOR_TEXT           = (255, 255, 255)   # blanco
COLOR_TEXT_SHADOW    = (0, 0, 0)         # negro


class PersonDetector:
    """
    Encapsula el modelo YOLOv8 y la lógica de detección de personas.
    """

    def __init__(self, model_name: str = "yolov8n-pose.pt", confidence: float = 0.75):
        """
        Args:
            model_name: Nombre del modelo YOLOv8 a cargar.
            confidence: Umbral mínimo de confianza para aceptar una detección.
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics no está instalado. Ejecuta: pip install ultralytics"
            )

        self.confidence = confidence
        self.model = YOLO(model_name)
        # Precalentamiento del modelo
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detecta personas en un fotograma.

        Args:
            frame: Imagen BGR (numpy array).

        Returns:
            Lista de detecciones, cada una es:
            {
                'bbox': (x1, y1, x2, y2),
                'confidence': float,
                'center': (cx, cy)
            }
        """
        results = self.model(frame, verbose=False, conf=self.confidence, classes=[PERSON_CLASS_ID])
        detections = []

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            
            if boxes is None:
                continue
                
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Extraer keypoints si el modelo lo soporta (yolov8-pose)
                kpts = None
                if keypoints is not None and len(keypoints) > i:
                    # kpts: (17, 3) donde [2] es la confianza del keypoint
                    # Usamos .data[0] para obtener el tensor de la persona actual
                    kpt_data = keypoints.data[i].cpu().numpy()
                    kpts = kpt_data.tolist()
                    
                    # Filtro anti-alucinaciones (motos, macetas, objetos complejos):
                    # Una moto puede engañar a la IA y generar keypoints con ~0.5 de confianza.
                    # Exigimos al menos 5 puntos anatómicos con una certeza MUY alta (> 0.70).
                    valid_kpts = sum(1 for kp in kpts if len(kp) >= 3 and kp[2] > 0.70)
                    if valid_kpts < 5:
                        continue # Descartar esta detección, es muy probable que sea un objeto

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "center": (cx, cy),
                    "keypoints": kpts,
                    "crop": frame[y1:y2, x1:x2].copy()
                })

        return detections


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    alerts: list[bool],
    zone: Optional[tuple] = None,
    zone_active: bool = True,
) -> np.ndarray:
    """
    Dibuja las bounding boxes, estado de alerta y zona restringida sobre el frame.

    Args:
        frame: Imagen BGR original.
        detections: Lista de detecciones del detector.
        alerts: Lista booleana paralela a detections (True = alerta activa).
        zone: Tupla (x1, y1, x2, y2) de la zona restringida.
        zone_active: Si True, muestra la zona restringida.

    Returns:
        Frame anotado.
    """
    output = frame.copy()
    h, w = output.shape[:2]

    # ── Zona restringida ──────────────────────────────────────────────────────
    if zone_active and zone is not None:
        zx1, zy1, zx2, zy2 = zone
        any_alert = any(alerts) if alerts else False
        zone_color = COLOR_ZONE_ALERT if any_alert else COLOR_ZONE_NORMAL

        # Overlay semitransparente
        overlay = output.copy()
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), zone_color, -1)
        cv2.addWeighted(overlay, 0.15, output, 0.85, 0, output)
        cv2.rectangle(output, (zx1, zy1), (zx2, zy2), zone_color, 2)

        # Etiqueta de zona
        label = "ZONA RESTRINGIDA"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output, (zx1, zy1 - lh - 8), (zx1 + lw + 8, zy1), zone_color, -1)
        cv2.putText(output, label, (zx1 + 4, zy1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)

    # ── Bounding boxes de personas ────────────────────────────────────────────
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        is_alert = alerts[i] if i < len(alerts) else False

        box_color = COLOR_BOX_ALERT if is_alert else COLOR_BOX_NORMAL
        auth_status = det.get("is_authorized")
        if auth_status is True:
            auth_text = " [AUTORIZADO]"
            box_color = (255, 165, 0) # Celeste/Verde claro si esta autorizado
        elif auth_status is False:
            auth_text = " [DESCONOCIDO]"
            box_color = COLOR_BOX_ALERT # Rojo si es desconocido
        else:
            auth_text = " [Analizando...]"

        status_text = "[!] ALERTA" if is_alert else "Persona"
        label = f"{status_text}{auth_text} {conf:.0%}"

        # Caja
        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)

        # Etiqueta con fondo
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(output, (x1, y1 - lh - 8), (x1 + lw + 6, y1), box_color, -1)
        cv2.putText(output, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2)

        # ── Dibujar Esqueleto (Keypoints) ─────────────────────────────────────
        kpts = det.get("keypoints")
        if kpts is not None:
            # Lista de conexiones de articulaciones (COCO format)
            skeleton = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
                (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
                (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
                (3, 5), (4, 6)
            ]
            
            # Dibujar uniones (huesos)
            for j1, j2 in skeleton:
                if j1 < len(kpts) and j2 < len(kpts):
                    p1 = kpts[j1]
                    p2 = kpts[j2]
                    # Validar si tienen confianza > 0.5 (si la confianza existe en kpts[i][2])
                    if (len(p1) > 2 and p1[2] < 0.5) or (len(p2) > 2 and p2[2] < 0.5):
                        continue
                    if p1[0] == 0 or p2[0] == 0:  # Keypoints no detectados
                        continue
                        
                    pt1 = (int(p1[0]), int(p1[1]))
                    pt2 = (int(p2[0]), int(p2[1]))
                    # Dibujar linea semitransparente
                    cv2.line(output, pt1, pt2, box_color, 2)
            
            # Dibujar puntos (articulaciones)
            for p in kpts:
                if p[0] != 0 and p[1] != 0:
                    if len(p) > 2 and p[2] < 0.5:
                        continue
                    cv2.circle(output, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)

        # Punto central
        cx, cy = det["center"]
        cv2.circle(output, (cx, cy), 5, box_color, -1)

    # ── Timestamp en esquina ──────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(output, ts, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_SHADOW, 3)
    cv2.putText(output, ts, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    # ── Marca de agua ─────────────────────────────────────────────────────────
    cv2.putText(output, "SERVSecurity v1.0", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_SHADOW, 3)
    cv2.putText(output, "SERVSecurity v1.0", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)

    return output
