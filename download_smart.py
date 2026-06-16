import os
import subprocess
import time
import shutil

# CONFIGURACIÓN
DATASET = "webadvisor/real-time-anomaly-detection-in-cctv-surveillance"
TARGET_DIR = r"E:\SERVSecurity_datasets\cctv_anomaly"

def get_all_files():
    print("Escaneando todos los archivos del dataset en Kaggle...")
    all_files = []
    page_token = None
    
    while True:
        cmd = ["kaggle", "datasets", "files", DATASET]
        if page_token:
            # Algunas versiones antiguas de kaggle CLI no soportan page-token,
            # pero asumiremos que funciona o solo nos traerá los primeros.
            # Realmente Kaggle CLI requiere usar la API de Python para paginación confiable,
            # pero existe un truco aún mejor: Kaggle CLI permite descargar carpetas completas en algunas versiones?
            # No, usaremos la API nativa de Python para paginar de forma segura.
            pass
        break # Placeholder

# Para evitar el dolor de cabeza de la paginación, usaremos la librería oficial de Kaggle en Python
# que maneja todo internamente.
def download_piecemeal():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    print("Conectando a Kaggle para obtener el índice del gigantesco dataset...")
    
    # La API de Python puede iterar sobre todas las páginas si usamos el cliente raw
    # o si simplemente llamamos a api.datasets_list_files(dataset, page_token=...)
    all_files = []
    token = None
    
    while True:
        response = api.dataset_list_files(DATASET, page_token=token)
        all_files.extend(response.files)
        
        # El objeto de respuesta crudo contiene el nextPageToken
        # Hack para extraerlo:
        raw_dict = getattr(response, '_raw_dict', None) or response.__dict__
        if hasattr(response, 'nextPageToken') and response.nextPageToken:
            token = response.nextPageToken
        elif 'nextPageToken' in str(raw_dict):
            # Hack extremo si la API es rebelde
            token = str(raw_dict).split("'nextPageToken': '")[1].split("'")[0] if "'nextPageToken': '" in str(raw_dict) else None
        else:
            token = None
            
        print(f"Obtenidos {len(all_files)} archivos hasta ahora...")
        if not token:
            break
            
        # Evitar el Error 429 (Too Many Requests) de Kaggle
        time.sleep(1)

    print(f"\n¡Se encontraron {len(all_files)} videos en total!")
    
    # ¡Ordenamos la lista para dejar los gigantes al final!
    def get_size(f):
        if hasattr(f, 'totalBytes') and f.totalBytes is not None: return f.totalBytes
        if hasattr(f, 'size') and f.size is not None: return f.size
        if hasattr(f, '_raw_dict') and 'totalBytes' in f._raw_dict: return f._raw_dict['totalBytes']
        return 0
        
    print("Ordenando los archivos para descargar primero los livianos y dejar los gigantes para el final...")
    all_files.sort(key=get_size)
    
    print(f"Se descargarán uno por uno en: {TARGET_DIR}")
    
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    for i, f in enumerate(all_files, 1):
        # f es un objeto, sacamos su nombre real
        file_name = f.name if hasattr(f, 'name') else str(f)
        
        # Ignorar lo que no sea video
        if ".mp4" not in file_name:
            continue
            
        # Determinar la carpeta final (ej. Abuse, Arrest, etc.)
        category = file_name.split('/')[-2] if '/' in file_name else 'otros'
        video_name = file_name.split('/')[-1]
        
        cat_dir = os.path.join(TARGET_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)
        final_path = os.path.join(cat_dir, video_name)
        
        if os.path.exists(final_path):
            # Ya fue descargado
            continue
            
        print(f"[{i}/{len(all_files)}] Descargando {category}/{video_name} ...")
        
        # Loop de reintentos infinitos por si se va el internet
        while True:
            try:
                # Descargamos el archivo individual. Se bajará un .zip pequeño.
                api.dataset_download_file(DATASET, file_name, path=cat_dir, force=True, quiet=False)
                
                # Extraer el .zip individual si Kaggle lo comprimió
                posible_zip = os.path.join(cat_dir, video_name + ".zip")
                if os.path.exists(posible_zip):
                    import zipfile
                    with zipfile.ZipFile(posible_zip, 'r') as zip_ref:
                        zip_ref.extractall(cat_dir)
                    os.remove(posible_zip)
                    
                posible_zip2 = os.path.join(cat_dir, file_name.replace("/", "%2F") + ".zip")
                if os.path.exists(posible_zip2):
                    import zipfile
                    with zipfile.ZipFile(posible_zip2, 'r') as zip_ref:
                        zip_ref.extractall(cat_dir)
                    os.remove(posible_zip2)
                
                # Si llegamos aquí sin errores, rompemos el loop de reintento y pasamos al siguiente archivo
                break

            except Exception as e:
                print(f"    ⚠️ Error de conexión detectado: {e}")
                print("    ⏳ Reintentando en 10 segundos (esperando a que regrese el internet)...")
                
                # Limpiar cualquier zip a medio descargar para evitar archivos corruptos
                for pz in [os.path.join(cat_dir, video_name + ".zip"), os.path.join(cat_dir, file_name.replace("/", "%2F") + ".zip")]:
                    if os.path.exists(pz):
                        try:
                            os.remove(pz)
                        except:
                            pass
                            
                time.sleep(10)

    print("\n--- ¡DESCARGA INTELIGENTE FINALIZADA! ---")

if __name__ == "__main__":
    download_piecemeal()
