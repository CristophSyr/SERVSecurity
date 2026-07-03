"""
utils.py – Utilidades comunes para SERVSecurity.

Incluye:
  - Guardado de capturas de pantalla.
  - Conversión de frames para Streamlit.
  - Helpers de tiempo y formato.
"""

import cv2
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import base64
import io
import os


# ── Detección de entorno ──────────────────────────────────────────────────────
IS_CLOUD = sys.platform.startswith("linux")


CAPTURES_DIR = Path("captures")

# Contador interno para ejecutar cleanup solo cada N capturas.
# Evita glob()+sort() en disco en cada imagen guardada.
_capture_save_count = 0
_CLEANUP_EVERY_N_SAVES = 50


def ensure_captures_dir():
    """Crea el directorio de capturas si no existe."""
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)


def save_capture(frame: np.ndarray, prefix: str = "event") -> str:
    """
    Guarda un fotograma como imagen JPEG en /captures.

    Args:
        frame:  Imagen BGR (numpy array).
        prefix: Prefijo para el nombre del archivo.

    Returns:
        Ruta relativa del archivo guardado (str).
    """
    global _capture_save_count
    ensure_captures_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{ts}.jpg"
    filepath = CAPTURES_DIR / filename

    # Guardar con compresión moderada para ahorrar espacio
    cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Limpieza perezosa: solo cada N capturas para no hacer glob+sort en cada frame
    _capture_save_count += 1
    if _capture_save_count % _CLEANUP_EVERY_N_SAVES == 0:
        cleanup_captures()

    return str(filepath)

def cleanup_captures(max_files: int = 500):
    """Mantiene como máximo `max_files` imágenes en el directorio de capturas."""
    try:
        # Ordenar archivos por fecha de modificación (más antiguo primero)
        captures = sorted(CAPTURES_DIR.glob("*.jpg"), key=os.path.getmtime)
        if len(captures) > max_files:
            # Eliminar los más antiguos
            for f in captures[:-max_files]:
                try:
                    f.unlink()
                except OSError:
                    pass
    except Exception:
        pass


def frame_to_bytes(frame: np.ndarray) -> bytes:
    """
    Convierte un frame BGR a bytes JPEG para mostrar en Streamlit.

    Args:
        frame: Imagen BGR.

    Returns:
        Bytes JPEG.
    """
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes()


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convierte de BGR a RGB para mostrar con st.image."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = None) -> bytes:
    """
    Convierte un frame BGR directamente a bytes JPEG.
    Mucho más eficiente que enviar arrays numpy a st.image():
    - Array numpy ~900KB por frame → JPEG ~30-50KB.
    - Streamlit acepta bytes directamente, evitando re-encoding interno.

    Args:
        frame:   Imagen BGR.
        quality: Calidad JPEG (0-100). Si None, usa 75 en nube y 85 en local.

    Returns:
        Bytes JPEG listos para st.image().
    """
    if quality is None:
        quality = 70 if IS_CLOUD else 85
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()


def resize_frame(frame: np.ndarray, max_width: int = 800) -> np.ndarray:
    """
    Redimensiona el frame manteniendo la relación de aspecto.

    Args:
        frame:     Imagen original.
        max_width: Ancho máximo en píxeles.

    Returns:
        Frame redimensionado.
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def format_duration(seconds: float) -> str:
    """
    Formatea una duración en segundos a una cadena legible.

    Args:
        seconds: Duración en segundos.

    Returns:
        Cadena con formato '1m 23s' o '45s'.
    """
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    secs    = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs:.1f}s" if seconds < 10 else f"{secs}s"


def get_status_emoji(estado: str) -> str:
    """Retorna un emoji representativo del estado del evento."""
    return "🔴" if estado == "sospechoso" else "🟢"


def get_event_color(tipo_evento: str) -> str:
    """Retorna un color hex para cada tipo de evento (uso en dashboard)."""
    colors = {
        "Intrusión en zona restringida": "#ff4b4b",
        "Permanencia prolongada":        "#ffa500",
        "Acceso fuera de horario":       "#ff00ff",
        "Presencia detectada":           "#00cc88",
    }
    return colors.get(tipo_evento, "#888888")


def clamp_rect(rect: tuple, frame_w: int, frame_h: int) -> tuple:
    """
    Asegura que los valores de la zona restringida estén dentro de los límites del frame.

    Args:
        rect: (x1, y1, x2, y2).
        frame_w, frame_h: Dimensiones del frame.

    Returns:
        Rectángulo ajustado.
    """
    x1, y1, x2, y2 = rect
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))
    return (x1, y1, x2, y2)


def normalize_zone(
    zone_pct: tuple,
    frame_w: int,
    frame_h: int,
) -> tuple:
    """
    Convierte porcentajes (0–100) a coordenadas de píxeles.

    Args:
        zone_pct: (x1%, y1%, x2%, y2%) en valores 0–100.
        frame_w, frame_h: Dimensiones del frame.

    Returns:
        (x1, y1, x2, y2) en píxeles.
    """
    x1p, y1p, x2p, y2p = zone_pct
    x1 = int(x1p / 100 * frame_w)
    y1 = int(y1p / 100 * frame_h)
    x2 = int(x2p / 100 * frame_w)
    y2 = int(y2p / 100 * frame_h)
    return clamp_rect((x1, y1, x2, y2), frame_w, frame_h)
