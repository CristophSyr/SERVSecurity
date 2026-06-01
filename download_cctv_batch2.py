"""
Descarga el resto de categorias del dataset CCTV (batch 2).
"""
import requests, json, struct, os, zlib

OUTPUT     = r"E:\SERVSecurity_datasets\cctv_anomaly"
CREDS_PATH = r"C:\Users\Usuario\.kaggle\kaggle.json"

TARGETS = [
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
    "data/normal/Normal_Videos_003_x264.mp4",
]

with open(CREDS_PATH) as f:
    creds = json.load(f)

print("Obteniendo URL firmada...")
r = requests.get(
    "https://www.kaggle.com/api/v1/datasets/download/webadvisor/real-time-anomaly-detection-in-cctv-surveillance",
    auth=(creds["username"], creds["key"]),
    allow_redirects=False, timeout=30
)
gcs_url    = r.headers["Location"]
total_size = int(requests.head(gcs_url, timeout=30).headers["Content-Length"])
print(f"Archivo total: {total_size/1e9:.2f} GB")

# Leer cola y parsear ZIP64 EOCD
tail_size = min(65536 + 56 + 20, total_size)
tail_data = requests.get(gcs_url, headers={"Range": f"bytes={total_size-tail_size}-{total_size-1}"}, timeout=60).content

loc_idx = tail_data.rfind(b"PK\x06\x07")
z64_eocd_offset = struct.unpack_from("<Q", tail_data, loc_idx + 8)[0]
z64_data = requests.get(gcs_url, headers={"Range": f"bytes={z64_eocd_offset}-{z64_eocd_offset+55}"}, timeout=30).content
cd_size   = struct.unpack_from("<Q", z64_data, 40)[0]
cd_offset = struct.unpack_from("<Q", z64_data, 48)[0]
print(f"CD offset: {cd_offset} | size: {cd_size/1e6:.2f} MB")

# Descargar y parsear Central Directory
cd_data = requests.get(gcs_url, headers={"Range": f"bytes={cd_offset}-{cd_offset+cd_size-1}"}, timeout=120).content
print(f"CD: {len(cd_data)} bytes")

file_map = {}
pos = 0
while pos + 46 <= len(cd_data):
    if cd_data[pos:pos+4] != b"PK\x01\x02":
        pos += 1; continue
    (_, _, flags, compression, _, _, _, comp_size, uncomp_size,
     fname_len, extra_len, comment_len, _, _, _, lh_offset) = struct.unpack_from("<HHHHHHIIIHHHHHIi", cd_data, pos+4)
    fname = cd_data[pos+46:pos+46+fname_len].decode("utf-8", errors="replace")
    extra = cd_data[pos+46+fname_len:pos+46+fname_len+extra_len]
    ep = 0
    while ep + 4 <= len(extra):
        tag, esz = struct.unpack_from("<HH", extra, ep)
        if tag == 0x0001:
            vals, vp = [], ep+4
            while vp+8 <= ep+4+esz:
                vals.append(struct.unpack_from("<Q", extra, vp)[0]); vp += 8
            vi = 0
            if uncomp_size == 0xFFFFFFFF and vi < len(vals): uncomp_size = vals[vi]; vi += 1
            if comp_size   == 0xFFFFFFFF and vi < len(vals): comp_size   = vals[vi]; vi += 1
            if lh_offset   == 0x7FFFFFFF and vi < len(vals): lh_offset   = vals[vi]; vi += 1
        ep += 4 + esz
    file_map[fname] = (lh_offset, comp_size, compression)
    pos += 46 + fname_len + extra_len + comment_len
print(f"Archivos indexados: {len(file_map)}")

def download_entry(gcs_url, lh_offset, comp_size, compression, dest_file):
    lh_raw     = requests.get(gcs_url, headers={"Range": f"bytes={lh_offset}-{lh_offset+29}"}, timeout=30).content
    lh_fn_len  = struct.unpack_from("<H", lh_raw, 26)[0]
    lh_ex_len  = struct.unpack_from("<H", lh_raw, 28)[0]
    data_start = lh_offset + 30 + lh_fn_len + lh_ex_len
    resp = requests.get(gcs_url, headers={"Range": f"bytes={data_start}-{data_start+comp_size-1}"}, stream=True, timeout=600)
    downloaded = 0
    if compression == 0:
        with open(dest_file, "wb") as f:
            for chunk in resp.iter_content(524288):
                f.write(chunk); downloaded += len(chunk)
                print(f"  {downloaded/1e6:.1f}/{comp_size/1e6:.1f} MB", end="\r", flush=True)
    elif compression == 8:
        decomp = zlib.decompressobj(-15)
        with open(dest_file, "wb") as f:
            for chunk in resp.iter_content(524288):
                f.write(decomp.decompress(chunk)); downloaded += len(chunk)
                print(f"  {downloaded/1e6:.1f}/{comp_size/1e6:.1f} MB", end="\r", flush=True)
            tail = decomp.flush()
            if tail: f.write(tail)
    print(f"\n  OK -> {dest_file}")

os.makedirs(OUTPUT, exist_ok=True)
ok, skip, fail = 0, 0, 0

for target in TARGETS:
    key = target if target in file_map else next((k for k in file_map if k.endswith(target.split("/")[-1])), None)
    if key is None:
        print(f"NO ENCONTRADO: {target}"); fail += 1; continue

    lh_offset, comp_size, compression = file_map[key]
    parts    = key.replace("\\","/").split("/")
    category = parts[-2] if len(parts) >= 2 else "misc"
    filename = parts[-1]
    dest_dir  = os.path.join(OUTPUT, category)
    dest_file = os.path.join(dest_dir, filename)
    os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(dest_file):
        print(f"Ya existe: {filename}"); skip += 1; continue

    print(f"\nDescargando {filename} ({comp_size/1e6:.1f} MB)...")
    try:
        download_entry(gcs_url, lh_offset, comp_size, compression, dest_file)
        ok += 1
    except Exception as e:
        print(f"  ERROR: {e}"); fail += 1

print(f"\n=== BATCH 2 COMPLETADO ===")
print(f"Descargados: {ok} | Ya existian: {skip} | Errores: {fail}")
