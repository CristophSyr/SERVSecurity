import os
import cv2
import numpy as np
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import io

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

FACES_DIR = "authorized_faces"
os.makedirs(FACES_DIR, exist_ok=True)


class FacialAuthenticator:
    """
    Gestor de autenticación biométrica en segundo plano.
    Usa un Lock para evitar race conditions entre el hilo de verificación
    y el hilo principal que consulta track.auth_pending.
    """
    def __init__(self):
        self.db_path = FACES_DIR
        self.model_name = "Facenet"
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.Lock()
        
        # FIX: Eliminar caché de DeepFace (.pkl) al iniciar.
        # DeepFace guarda rutas absolutas en este archivo. Si mueves el disco duro
        # a otra PC, la ruta cambia y falla. Al borrarlo, forzamos a que lo regenere.
        for pkl_file in Path(self.db_path).glob("*.pkl"):
            try:
                pkl_file.unlink()
            except OSError:
                pass

    def authenticate_async(self, frame_crop: np.ndarray, track, callback=None):
        """
        Lanza la verificación facial en un hilo separado.
        Usa un Lock para sincronizar el acceso a track.auth_pending
        y evitar que rules.py lance verificaciones duplicadas.
        """
        valid_files = [f for f in os.listdir(self.db_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not valid_files:
            track.is_authorized = False
            return

        with self._lock:
            if track.auth_pending:
                return
            track.auth_pending = True

        def _verify():
            try:
                from deepface import DeepFace
                dfs = DeepFace.find(
                    img_path=frame_crop,
                    db_path=self.db_path,
                    model_name=self.model_name,
                    detector_backend="opencv",
                    enforce_detection=False,
                    silent=True
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

        self.executor.submit(_verify)
