"""
Descarga robusta del dataset CCTV con reintentos y timeouts estrictos.
"""
import requests, json, struct, os, zlib, time

OUTPUT     = r"E:\SERVSecurity_datasets\cctv_anomaly"
CREDS_PATH = r"C:\Users\Usuario\.kaggle\kaggle.json"

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

def get_url():
    with open(CREDS_PATH) as f:
        creds = json.load(f)
    print("Obteniendo URL firmada...", flush=True)
    r = requests.get(
        "https://www.kaggle.com/api/v1/datasets/download/webadvisor/real-time-anomaly-detection-in-cctv-surveillance",
        auth=(creds["username"], creds["key"]),
        allow_redirects=False, timeout=30
    )
    return r.headers["Location"]

gcs_url = get_url()

total_size = int(requests.head(gcs_url, timeout=30).headers["Content-Length"])
tail_size = min(65536 + 56 + 20, total_size)
tail_data = requests.get(gcs_url, headers={"Range": f"bytes={total_size-tail_size}-{total_size-1}"}, timeout=60).content

loc_idx = tail_data.rfind(b"PK\x06\x07")
z64_eocd_offset = struct.unpack_from("<Q", tail_data, loc_idx + 8)[0]
z64_data = requests.get(gcs_url, headers={"Range": f"bytes={z64_eocd_offset}-{z64_eocd_offset+55}"}, timeout=30).content
cd_size   = struct.unpack_from("<Q", z64_data, 40)[0]
cd_offset = struct.unpack_from("<Q", z64_data, 48)[0]

cd_data = requests.get(gcs_url, headers={"Range": f"bytes={cd_offset}-{cd_offset+cd_size-1}"}, timeout=120).content

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

def download_entry(gcs_url, lh_offset, comp_size, compression, dest_file):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            lh_raw = requests.get(gcs_url, headers={"Range": f"bytes={lh_offset}-{lh_offset+29}"}, timeout=(5,15)).content
            lh_fn_len  = struct.unpack_from("<H", lh_raw, 26)[0]
            lh_ex_len  = struct.unpack_from("<H", lh_raw, 28)[0]
            data_start = lh_offset + 30 + lh_fn_len + lh_ex_len
            
            resp = requests.get(gcs_url, headers={"Range": f"bytes={data_start}-{data_start+comp_size-1}"}, stream=True, timeout=(5,15))
            resp.raise_for_status()
            
            downloaded = 0
            
            if compression == 0:
                with open(dest_file, "wb") as f:
                    for chunk in resp.iter_content(524288):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            print(f"  {downloaded/1e6:.1f}/{comp_size/1e6:.1f} MB", end="\r", flush=True)
            elif compression == 8:
                decomp = zlib.decompressobj(-15)
                with open(dest_file, "wb") as f:
                    for chunk in resp.iter_content(524288):
                        if chunk:
                            f.write(decomp.decompress(chunk))
                            downloaded += len(chunk)
                            print(f"  {downloaded/1e6:.1f}/{comp_size/1e6:.1f} MB", end="\r", flush=True)
                    tail = decomp.flush()
                    if tail: f.write(tail)
            print(f"\n  OK -> {dest_file}", flush=True)
            return True
        except requests.exceptions.RequestException as e:
            print(f"\n  Error (intento {attempt+1}/{max_retries}): {e}", flush=True)
            time.sleep(3)
            # Refrescar URL firmada en caso de que caducara
            gcs_url = get_url()
    return False

os.makedirs(OUTPUT, exist_ok=True)

for target in TARGETS:
    key = target if target in file_map else next((k for k in file_map if k.endswith(target.split("/")[-1])), None)
    if key is None: continue

    lh_offset, comp_size, compression = file_map[key]
    parts = key.replace("\\","/").split("/")
    dest_dir  = os.path.join(OUTPUT, parts[-2] if len(parts) >= 2 else "misc")
    dest_file = os.path.join(dest_dir, parts[-1])
    os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(dest_file): continue
    print(f"\nDescargando {parts[-1]} ({comp_size/1e6:.1f} MB)...", flush=True)
    download_entry(gcs_url, lh_offset, comp_size, compression, dest_file)

print("\n=== COMPLETADO ===", flush=True)
