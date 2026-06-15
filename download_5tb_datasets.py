import os
import sys

# ==============================================================================
# CONFIGURACIÓN DEL DISCO DE 5TB
# ==============================================================================
# Cambia esta ruta a la letra o carpeta de tu nuevo disco de 5TB.
# Por ejemplo: "D:\Datasets", "E:\CCTV", o "G:\Mi unidad\Tesis_Datasets"
DISCO_5TB_PATH = r"E:\SERVSecurity_datasets"
# ==============================================================================

DATASETS = [
    {
        "nombre": "Imágenes de Vigilancia para Detección de Personas",
        "kaggle_id": "luiscrmartins/surveillance-images-for-person-detection",
        "carpeta_destino": "Person_Detection_Images"
    },
    {
        "nombre": "Detección de Anomalías en CCTV en Tiempo Real (100GB+)",
        "kaggle_id": "webadvisor/real-time-anomaly-detection-in-cctv-surveillance",
        "carpeta_destino": "CCTV_Anomalies_Full"
    }
]

def descargar_masivo():
    print(f"--- INICIANDO DESCARGA MASIVA HACIA {DISCO_5TB_PATH} ---")
    
    # Crear el directorio principal en el disco de 5TB si no existe
    try:
        os.makedirs(DISCO_5TB_PATH, exist_ok=True)
    except Exception as e:
        print(f"ERROR: No se pudo crear la carpeta en {DISCO_5TB_PATH}.")
        print(f"Detalle: {e}")
        print("¿Asegúrate de haber cambiado la variable DISCO_5TB_PATH a la letra correcta de tu disco!")
        sys.exit(1)

    for ds in DATASETS:
        print(f"\n[{ds['nombre']}]")
        print("Iniciando descarga y extracción automática (esto tomará varias horas)...")
        
        destino_final = os.path.join(DISCO_5TB_PATH, ds["carpeta_destino"])
        os.makedirs(destino_final, exist_ok=True)
        
        # El comando 'kaggle' descargará y usará '--unzip' para descomprimir al vuelo.
        comando = f'kaggle datasets download {ds["kaggle_id"]} -p "{destino_final}" --unzip'
        
        # Ejecutar en consola
        codigo_salida = os.system(comando)
        
        if codigo_salida == 0:
            print(f"✅ ¡Descarga y extracción exitosa en: {destino_final}!")
        else:
            print(f"❌ Ocurrió un error descargando {ds['nombre']}.")
            print("Asegúrate de que Kaggle esté bien configurado en tu PC.")

    print("\n=== TODAS LAS DESCARGAS COMPLETADAS ===")

if __name__ == "__main__":
    descargar_masivo()
