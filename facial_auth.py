import os
import cv2
import numpy as np
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import io
import time

# Configurar TensorFlow para usar estrictamente la CPU antes de importar DeepFace.
# Esto evita conflictos críticos de VRAM y deadlocks entre PyTorch (YOLOv8) y TensorFlow (DeepFace).
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    print("[FacialAuth] TensorFlow configurado en modo CPU para estabilidad.")
except Exception as e:
    print(f"[FacialAuth] No se pudo configurar TensorFlow en CPU: {e}")

# DeepFace se importa al top para que _prebuild_index y _verify lo compartan.
# El import tarda ~2s la primera vez (carga Keras/TF), pero solo ocurre una vez.
from deepface import DeepFace

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

FACES_DIR = "authorized_faces"
os.makedirs(FACES_DIR, exist_ok=True)

# Detectar entorno
_IS_CLOUD = sys.platform.startswith("linux")

# Usar siempre 1 worker para evitar ejecuciones concurrentes de DeepFace y prevenir deadlocks/crashes
_MAX_WORKERS = 1

# Timeout máximo por verificación (segundos)
_VERIFY_TIMEOUT = 15 if _IS_CLOUD else 10


class FacialAuthenticator:
    """
    Gestor de autenticación biométrica en segundo plano.

    Mejoras respecto a la versión anterior:
    - detector_backend="skip": DeepFace NO intenta detectar cara en el crop
      (ya lo hizo YOLO). Esto evita el fallo con crops pequeños (<60px) que
      OpenCV no puede procesar.
    - Pre-indexado en __init__: el índice .pkl se construye UNA SOLA VEZ al
      arrancar, no en cada llamada a find(). Esto elimina el retraso de 2-5s
      en la primera verificación de cada persona.
    - max_workers adaptativo: 1 en nube para evitar OOM, 2 en local.
    - Timeout de verificación: el hilo se abandona si tarda más de N segundos.
    """

    def __init__(self):
        self.db_path = FACES_DIR
        self.model_name = "Facenet"
        self.executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS)
        self._lock = threading.Lock()
        self._index_ready = False  # True cuando el índice .pkl ya existe y es válido

        # FIX: Eliminar caché de DeepFace (.pkl) al iniciar.
        # DeepFace guarda rutas absolutas. Si mueves el disco a otra PC la ruta
        # cambia y falla. Al borrarlo forzamos regeneración con la ruta correcta.
        for pkl_file in Path(self.db_path).glob("*.pkl"):
            try:
                pkl_file.unlink()
            except OSError:
                pass

        # Pre-construir el índice en un hilo para no bloquear el arranque de la app.
        # Cuando termine, _index_ready=True y las verificaciones serán instantáneas.
        valid_files = [
            f for f in os.listdir(self.db_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if valid_files:
            self.executor.submit(self._prebuild_index)

    def _prebuild_index(self):
        """
        Construye el índice de DeepFace (archivo .pkl) con una imagen dummy.
        Así la primera verificación real no paga el coste de indexación.
        """
        # Seleccionamos la primera foto válida de la base para crear un índice real
        first_image_path = None
        for f in os.listdir(self.db_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                first_image_path = os.path.join(self.db_path, f)
                break
        if first_image_path is None:
            # No hay fotos, usamos dummy como fallback
            dummy_img = np.zeros((112, 112, 3), dtype=np.uint8)
            img_to_use = dummy_img
        else:
            img = cv2.imread(first_image_path)
            if img is None:
                # Si la lectura falla, caer a dummy
                dummy_img = np.zeros((112, 112, 3), dtype=np.uint8)
                img_to_use = dummy_img
            else:
                img_to_use = img
        # Usa opencv para extraer caras reales del/las foto(s) y crear .pkl
        DeepFace.find(
            img_path=img_to_use,
            db_path=self.db_path,
            model_name=self.model_name,
            detector_backend="opencv",
            enforce_detection=False,
            silent=True,
        )

        # Verificar que el .pkl generado sea válido (> 1 KB)
        pkl_files = list(Path(self.db_path).glob("*.pkl"))
        if pkl_files:
            pkl_size = pkl_files[0].stat().st_size
            print(f"[FacialAuth] Índice pre-construido. Tamaño .pkl: {pkl_size} bytes")
            if pkl_size < 1024:
                print("[FacialAuth] ADVERTENCIA: .pkl parece vacío o corrupto. Se reintentará en la próxima verificación.")
                return  # No marcar como ready si el índice es inválido
        else:
            print("[FacialAuth] ADVERTENCIA: No se generó ningún .pkl. Verifica que las fotos tengan caras detectables.")
            return

        with self._lock:
            self._index_ready = True
        print("[FacialAuth] _index_ready = True. Reconocimiento facial activado.")

    def authenticate_async(self, frame_crop: np.ndarray, track, callback=None):
        """
        Lanza la verificación facial en un hilo separado.
        Usa un Lock para sincronizar el acceso a track.auth_pending
        y evitar que rules.py lance verificaciones duplicadas.
        """
        valid_files = [
            f for f in os.listdir(self.db_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not valid_files:
            track.is_authorized = False
            return

        with self._lock:
            if not self._index_ready:
                # Todavía se está pre-construyendo el índice con opencv.
                # Si dejamos pasar esto, DeepFace intentará construir el índice
                # aquí mismo usando "skip", lo cual corromperá las representaciones.
                track.auth_pending = False
                return

            if track.auth_pending:
                return
            track.auth_pending = True

        def _verify():
            try:
                # Asegurar tamaño mínimo del crop (Facenet necesita al menos 80x80)
                crop = frame_crop
                h, w = crop.shape[:2]
                if h < 80 or w < 80:
                    crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_CUBIC)

                dfs = DeepFace.find(
                    img_path=crop,
                    db_path=self.db_path,
                    model_name=self.model_name,
                    # "skip": asume que el crop YA ES una cara (lo recortó YOLO).
                    # Evita el fallo con crops pequeños donde OpenCV no detecta nada.
                    detector_backend="skip",
                    enforce_detection=False,
                    silent=True,
                )
                if len(dfs) > 0 and not dfs[0].empty:
                    track.is_authorized = True
                else:
                    track.is_authorized = False
            except Exception as e:
                print(f"[DeepFace Error] {e}")
                track.is_authorized = False
            finally:
                with self._lock:
                    track.auth_pending = False
                if callback:
                    callback(track)

        # submit con timeout manejado externamente mediante Future
        future = self.executor.submit(_verify)

        # Lanzar watchdog para no dejar el flag auth_pending colgado si el hilo muere
        def _watchdog():
            time.sleep(_VERIFY_TIMEOUT)
            if not future.done():
                # El hilo tardó demasiado: liberar el flag para que se reintente
                with self._lock:
                    track.auth_pending = False
                print("[FacialAuth] Timeout de verificación. Liberando track.")

        threading.Thread(target=_watchdog, daemon=True).start()
