"""
Script para descargar una muestra representativa del dataset CCTV Anomaly.
Solo descarga 2 videos pequeños por categoria - no los 100 GB completos.
"""
import os
from kaggle import KaggleApi

api = KaggleApi()
api.authenticate()
print("Autenticado OK")

DATASET = "webadvisor/real-time-anomaly-detection-in-cctv-surveillance"
OUTPUT  = r"E:\SERVSecurity_datasets\cctv_anomaly"

# 2 videos pequeños por categoria (elegidos por menor tamaño)
SAMPLES = [
    "data/abuse/Abuse002_x264.mp4",
    "data/abuse/Abuse005_x264.mp4",
    "data/arrest/Arrest001_x264.mp4",
    "data/arrest/Arrest002_x264.mp4",
    "data/arson/Arson001_x264.mp4",
    "data/arson/Arson002_x264.mp4",
    "data/assault/Assault001_x264.mp4",
    "data/assault/Assault002_x264.mp4",
    "data/burglary/Burglary001_x264.mp4",
    "data/burglary/Burglary002_x264.mp4",
    "data/explosion/Explosion001_x264.mp4",
    "data/explosion/Explosion002_x264.mp4",
    "data/fighting/Fighting001_x264.mp4",
    "data/fighting/Fighting002_x264.mp4",
    "data/robbery/Robbery001_x264.mp4",
    "data/robbery/Robbery002_x264.mp4",
    "data/shooting/Shooting001_x264.mp4",
    "data/shooting/Shooting002_x264.mp4",
    "data/shoplifting/Shoplifting001_x264.mp4",
    "data/shoplifting/Shoplifting002_x264.mp4",
    "data/stealing/Stealing001_x264.mp4",
    "data/stealing/Stealing002_x264.mp4",
    "data/vandalism/Vandalism001_x264.mp4",
    "data/vandalism/Vandalism002_x264.mp4",
    "data/normal/Normal_Videos_001_x264.mp4",
    "data/normal/Normal_Videos_002_x264.mp4",
]

os.makedirs(OUTPUT, exist_ok=True)

total = len(SAMPLES)
for i, file_path in enumerate(SAMPLES, 1):
    category = file_path.split("/")[1]
    filename  = file_path.split("/")[-1]
    dest_dir  = os.path.join(OUTPUT, category)
    dest_file = os.path.join(dest_dir, filename)

    if os.path.exists(dest_file):
        print(f"[{i}/{total}] Ya existe: {filename} - saltando")
        continue

    os.makedirs(dest_dir, exist_ok=True)
    print(f"[{i}/{total}] Descargando {file_path} ...", flush=True)
    try:
        api.dataset_download_file(
            DATASET,
            file_path,
            path=dest_dir,
            force=False,
            quiet=False
        )
        # Descomprimir si llego como .zip
        zip_path = dest_file + ".zip"
        if os.path.exists(zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(dest_dir)
            os.remove(zip_path)
            print(f"  -> Descomprimido OK")
        else:
            print(f"  -> Guardado OK")
    except Exception as e:
        print(f"  -> ERROR: {e}")

print("\nDescarga de muestras CCTV completada.")
print(f"Archivos en: {OUTPUT}")
