"""
detector.py – Módulo de detección de personas con YOLOv8 preentrenado.
Utiliza el modelo yolov8n.pt (nano) para mayor velocidad en demo.
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

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.45):
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
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "center": (cx, cy),
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
        status_text = "[!] ALERTA" if is_alert else "Persona"
        label = f"{status_text} {conf:.0%}"

        # Caja
        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)

        # Etiqueta con fondo
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(output, (x1, y1 - lh - 8), (x1 + lw + 6, y1), box_color, -1)
        cv2.putText(output, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2)

        # Punto central
        cx, cy = det["center"]
        cv2.circle(output, (cx, cy), 4, box_color, -1)

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
