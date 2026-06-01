"""
Descarga el dataset completo (100GB), extrae los videos de muestra, y elimina el ZIP.
"""
import os
import zipfile
import shutil

OUTPUT     = r"E:\SERVSecurity_datasets\cctv_anomaly"
ZIP_PATH   = r"E:\SERVSecurity_datasets\real-time-anomaly-detection-in-cctv-surveillance.zip"

TARGETS = [
    "data/arson/Arson001_x264.mp4",
    "data/assault/Assault001_x264.mp4",
    "data/burglary/Burglary001_x264.mp4",
    "data/explosion/Explosion001_x264.mp4",
    "data/fighting/Fighting001_x264.mp4",
    "data/robbery/Robbery001_x264.mp4",
    "data/shooting/Shooting001_x264.mp4",
    "data/shoplifting/Shoplifting001_x264.mp4",
    "data/stealing/Stealing001_x264.mp4",
    "data/vandalism/Vandalism001_x264.mp4",
    "data/normal/Normal_Videos_001_x264.mp4",
]

print("1. Iniciando descarga completa del dataset (100 GB) con Kaggle CLI...", flush=True)
os.system(f'kaggle datasets download webadvisor/real-time-anomaly-detection-in-cctv-surveillance -p "E:\SERVSecurity_datasets"')

if not os.path.exists(ZIP_PATH):
    print("Error: No se encontro el archivo ZIP descargado.")
    exit(1)

print("\n2. Descarga finalizada. Extrayendo archivos de muestra...", flush=True)
os.makedirs(OUTPUT, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    for target in TARGETS:
        try:
            filename = target.split("/")[-1]
            category = target.split("/")[-2] if "/" in target else "misc"
            dest_dir = os.path.join(OUTPUT, category)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, filename)
            
            if os.path.exists(dest_path):
                print(f"Ya existe: {filename}")
                continue
                
            print(f"Extrayendo {filename}...", flush=True)
            with z.open(target) as source, open(dest_path, "wb") as dest:
                shutil.copyfileobj(source, dest)
        except Exception as e:
            print(f"Error extrayendo {target}: {e}")

print("\n3. Eliminando el archivo ZIP gigante para liberar espacio...", flush=True)
os.remove(ZIP_PATH)
print("=== PROCESO COMPLETADO ===")
