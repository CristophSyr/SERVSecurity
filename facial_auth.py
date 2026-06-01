import os
import cv2
import numpy as np
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import io

# Parche para evitar que DeepFace crashee la terminal de Windows al imprimir emojis (ej. 🔗) durante la descarga
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

FACES_DIR = "authorized_faces"
os.makedirs(FACES_DIR, exist_ok=True)

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False


class FacialAuthenticator:
    """
    Gestor de autenticación biométrica en segundo plano.
    Evita bloquear el hilo principal de video de Streamlit.
    """
    def __init__(self):
        self.db_path = FACES_DIR
        self.model_name = "Facenet"
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Opcional: pre-crear embeddings si ya hay imágenes
        if DEEPFACE_AVAILABLE and os.listdir(self.db_path):
            try:
                # Pre-cargar representaciones (pkl)
                _ = DeepFace.find(
                    img_path=np.zeros((224, 224, 3), dtype=np.uint8), 
                    db_path=self.db_path, 
                    model_name=self.model_name,
                    detector_backend="retinaface",
                    enforce_detection=False,
                    silent=True
                )
            except:
                pass

    def authenticate_async(self, frame_crop: np.ndarray, track, callback=None):
        """
        Lanza la verificación facial en un hilo separado.
        Actualizará el atributo track.is_authorized cuando termine.
        """
        if not DEEPFACE_AVAILABLE:
            track.is_authorized = False
            return
            
        # Verificar si hay rostros autorizados configurados
        valid_files = [f for f in os.listdir(self.db_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not valid_files:
            track.is_authorized = False
            return

        def _verify():
            try:
                dfs = DeepFace.find(
                    img_path=frame_crop,
                    db_path=self.db_path,
                    model_name=self.model_name,
                    detector_backend="retinaface",
                    enforce_detection=False,
                    silent=True
                )
                
                # dfs[0] contiene los matches si los hay
                if len(dfs) > 0 and not dfs[0].empty:
                    track.is_authorized = True
                else:
                    track.is_authorized = False
                    
            except Exception as e:
                # Falló la extracción de cara
                track.is_authorized = False
                
            if callback:
                callback(track)

        self.executor.submit(_verify)
