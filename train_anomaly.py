from ultralytics import YOLO
import torch

# Verificamos que tu RTX 3060 esté lista para la acción
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Tarjeta Gráfica: {torch.cuda.get_device_name(0)}")
else:
    print("ADVERTENCIA: La tarjeta gráfica no fue detectada. El entrenamiento será muy lento.")

def main():
    print("\n--- INICIANDO ENTRENAMIENTO DE IA (CLASIFICACIÓN) ---")
    
    # 1. Cargamos a nuestro estudiante (YOLOv8 Nano Classifier pre-entrenado)
    model = YOLO('yolov8n-cls.pt')  # Descargará el modelo base automáticamente

    # 2. Le pasamos el directorio con las imágenes que extrajimos
    # YOLO asume automáticamente que adentro hay una carpeta 'train' y 'val'
    dataset_dir = r"E:\SERVSecurity_datasets\anomaly_images"
    
    print(f"\nIniciando estudio intensivo del dataset en: {dataset_dir}")
    print("Escucha cómo se aceleran los ventiladores de tu computadora... 🏎️\n")

    # 3. ¡Entrenamos!
    # - data: ubicación de nuestras fotos
    # - epochs: 20 "pasadas completas" al libro de estudio (puedes subir a 50 si quieres que aprenda más, o bajar a 10 para terminar rápido)
    # - imgsz: Tamaño al que redimensionará las imágenes (224x224 es estándar para clasificación y entrena rapidísimo)
    # - device=0: Fuerza a usar tu NVIDIA RTX 3060
    results = model.train(
        data=dataset_dir,
        epochs=20,
        imgsz=224,
        device=0,
        name='servsecurity_anomaly_model' # Nombre de la carpeta donde guardará el resultado
    )
    
    print("\n✅ ¡ENTRENAMIENTO FINALIZADO CON ÉXITO!")
    print("Tu nuevo cerebro matemático ha nacido.")
    print("Busca tu modelo 'best.pt' dentro de la carpeta 'runs/classify/servsecurity_anomaly_model/weights/'.")

if __name__ == '__main__':
    main()
