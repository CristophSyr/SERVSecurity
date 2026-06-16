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

# Carga diferida de DeepFace para prevenir Segmentation Faults con TensorFlow en el arranque de Streamlit
DEEPFACE_AVAILABLE = True


class FacialAuthenticator:
    """
    Gestor de autenticación biométrica en segundo plano.
    Evita bloquear el hilo principal de video de Streamlit.
    """
    def __init__(self):
        self.db_path = FACES_DIR
        # Usar OpenFace (15MB) en lugar de Facenet (92MB) para prevenir OOM (Out of Memory) en la Nube
        self.model_name = "OpenFace"
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Eliminado el pre-calentamiento de DeepFace para evitar pico de memoria OOM en el arranque de Streamlit Cloud

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
                from deepface import DeepFace
                dfs = DeepFace.find(
                    img_path=frame_crop,
                    db_path=self.db_path,
                    model_name=self.model_name,
                    detector_backend="opencv",
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
                import traceback
                print(f"[DeepFace Error] {e}")
                traceback.print_exc()
                track.is_authorized = False
            
            # Liberar el flag de pendiente para que rules_engine pueda reintentar
            track.auth_pending = False
                
            if callback:
                callback(track)

        self.executor.submit(_verify)
