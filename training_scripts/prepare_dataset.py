import os
import cv2
import random
import shutil

# CONFIGURACIÓN
# Carpeta donde se descargaron los 100GB
SOURCE_DIR = r"E:\SERVSecurity_datasets\cctv_anomaly"
# Carpeta donde se guardará el dataset ligero de imágenes
TARGET_DIR = r"E:\SERVSecurity_datasets\anomaly_images"

# Número de imágenes que extraeremos por cada video (para no saturar el disco)
FRAMES_PER_VIDEO = 10
# Porcentaje de imágenes que se irán a validación (examen final de la IA)
VAL_SPLIT = 0.2

# Categorías a procesar
CATEGORIES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Fighting", "Robbery", "Shooting", "Normal"]

def prepare_dataset():
    print("--- INICIANDO EXTRACCIÓN INTELIGENTE DE FOTOGRAMAS ---")
    
    # Crear carpetas train y val
    for split in ['train', 'val']:
        for cat in CATEGORIES:
            os.makedirs(os.path.join(TARGET_DIR, split, cat), exist_ok=True)
            
    # Buscar todos los videos en el disco E:
    for cat in CATEGORIES:
        cat_path = None
        for folder in os.listdir(SOURCE_DIR):
            if folder.lower() == cat.lower():
                cat_path = os.path.join(SOURCE_DIR, folder)
                break
                
        if not cat_path or not os.path.isdir(cat_path):
            print(f"⚠️ Categoría {cat} no encontrada en los videos. Saltando...")
            continue
            
        videos = [f for f in os.listdir(cat_path) if f.endswith('.mp4')]
        print(f"\n📂 Procesando categoría {cat} ({len(videos)} videos encontrados)...")
        
        for i, video_file in enumerate(videos, 1):
            video_path = os.path.join(cat_path, video_file)
            
            # Decidir si este video va a Train (80%) o Val (20%)
            split = 'val' if random.random() < VAL_SPLIT else 'train'
            save_dir = os.path.join(TARGET_DIR, split, cat)
            
            # Extraer fotogramas
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error abriendo video {video_file}")
                continue
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                continue
                
            # Calcular en qué frames vamos a tomar la foto
            step = max(1, total_frames // FRAMES_PER_VIDEO)
            frame_indices = [j * step for j in range(FRAMES_PER_VIDEO)]
            
            extracted = 0
            for frame_idx in frame_indices:
                if frame_idx >= total_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Guardar imagen
                    img_name = f"{cat}_{video_file.replace('.mp4', '')}_frame{frame_idx}.jpg"
                    img_path = os.path.join(save_dir, img_name)
                    cv2.imwrite(img_path, frame)
                    extracted += 1
            
            cap.release()
            if i % 10 == 0 or i == len(videos):
                print(f"[{i}/{len(videos)}] Extraídos {extracted} fotogramas de {video_file} -> {split}")

    print("\n✅ ¡EXTRACCIÓN FINALIZADA!")
    print(f"Revisa tu carpeta {TARGET_DIR} para ver las imágenes puras.")

if __name__ == "__main__":
    prepare_dataset()
