"""
Descarga selectiva del dataset CCTV usando stream + range requests sobre el ZIP64 de GCS.
Versión corregida con soporte ZIP64 correcto.
"""
import requests
import json
import struct
import os

OUTPUT     = r"E:\SERVSecurity_datasets\cctv_anomaly"
CREDS_PATH = r"C:\Users\Usuario\.kaggle\kaggle.json"

TARGETS = [
    "data/abuse/Abuse002_x264.mp4",
    "data/abuse/Abuse005_x264.mp4",
    "data/arrest/Arrest001_x264.mp4",
    "data/arrest/Arrest002_x264.mp4",
]

# ── Paso 1: URL firmada ───────────────────────────────────────────────────────
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

# ── Paso 2: Leer cola para encontrar EOCD / ZIP64 EOCD ───────────────────────
tail_size = min(65536 + 56 + 20, total_size)
tail_data = requests.get(
    gcs_url,
    headers={"Range": f"bytes={total_size - tail_size}-{total_size - 1}"},
    timeout=60
).content
tail_base = total_size - tail_size       # offset absoluto del byte 0 de tail_data

def abs_offset(rel): return tail_base + rel

# Buscar ZIP64 EOCD Locator (PK\x06\x07)
LOC_SIG  = b"PK\x06\x07"
EOCD_SIG = b"PK\x05\x06"
Z64_SIG  = b"PK\x06\x06"

loc_idx = tail_data.rfind(LOC_SIG)
if loc_idx != -1:
    # ZIP64 path
    z64_eocd_offset = struct.unpack_from("<Q", tail_data, loc_idx + 8)[0]
    print(f"ZIP64 EOCD offset: {z64_eocd_offset}")

    z64_data = requests.get(
        gcs_url,
        headers={"Range": f"bytes={z64_eocd_offset}-{z64_eocd_offset + 55}"},
        timeout=30
    ).content
    assert z64_data[:4] == Z64_SIG, "No es ZIP64 EOCD"

    num_entries = struct.unpack_from("<Q", z64_data, 24)[0]
    cd_size     = struct.unpack_from("<Q", z64_data, 40)[0]
    cd_offset   = struct.unpack_from("<Q", z64_data, 48)[0]
else:
    # EOCD estándar
    eocd_idx  = tail_data.rfind(EOCD_SIG)
    num_entries = struct.unpack_from("<H", tail_data, eocd_idx +  8)[0]
    cd_size     = struct.unpack_from("<I", tail_data, eocd_idx + 12)[0]
    cd_offset   = struct.unpack_from("<I", tail_data, eocd_idx + 16)[0]

print(f"Entradas: {num_entries} | CD offset: {cd_offset} | CD size: {cd_size/1e6:.2f} MB")

# ── Paso 3: Descargar el Central Directory ────────────────────────────────────
print(f"Descargando Central Directory ({cd_size/1e6:.2f} MB)...")
cd_data = requests.get(
    gcs_url,
    headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"},
    timeout=120
).content
print(f"CD descargado: {len(cd_data)} bytes")

# ── Paso 4: Parsear el CD ─────────────────────────────────────────────────────
CD_SIG   = b"PK\x01\x02"
file_map = {}   # filename -> (local_header_offset, compressed_size, compression)
pos = 0

while pos + 46 <= len(cd_data):
    if cd_data[pos:pos+4] != CD_SIG:
        pos += 1
        continue

    (_, _, flags, compression,
     _, _, _,
     comp_size, uncomp_size,
     fname_len, extra_len, comment_len,
     _, _, _,
     lh_offset) = struct.unpack_from("<HHHHHHIIIHHHHHIi", cd_data, pos + 4)

    fname_bytes = cd_data[pos+46 : pos+46+fname_len]
    fname = fname_bytes.decode("utf-8", errors="replace")
    extra = cd_data[pos+46+fname_len : pos+46+fname_len+extra_len]

    # ZIP64 extended info (tag 0x0001)
    ep = 0
    while ep + 4 <= len(extra):
        tag, esz = struct.unpack_from("<HH", extra, ep)
        if tag == 0x0001:
            vals, vp = [], ep + 4
            while vp + 8 <= ep + 4 + esz:
                vals.append(struct.unpack_from("<Q", extra, vp)[0]); vp += 8
            vi = 0
            if uncomp_size == 0xFFFFFFFF and vi < len(vals): uncomp_size = vals[vi]; vi += 1
            if comp_size   == 0xFFFFFFFF and vi < len(vals): comp_size   = vals[vi]; vi += 1
            if lh_offset   == 0x7FFFFFFF and vi < len(vals): lh_offset   = vals[vi]; vi += 1
        ep += 4 + esz

    file_map[fname] = (lh_offset, comp_size, compression)
    pos += 46 + fname_len + extra_len + comment_len

print(f"Archivos en CD: {len(file_map)}")
# Muestra
sample = [k for k in list(file_map.keys())[:8]]
print("Muestra:", sample)

# ── Paso 5: Descargar archivos objetivo ───────────────────────────────────────
import zlib

def download_entry(gcs_url, target_key, lh_offset, comp_size, compression, dest_file):
    # Leer local file header para calcular offset real de datos
    lh_raw = requests.get(gcs_url,
                          headers={"Range": f"bytes={lh_offset}-{lh_offset+29}"},
                          timeout=30).content
    if lh_raw[:4] != b"PK\x03\x04":
        print(f"  ERROR: local header invalido (got {lh_raw[:4]})")
        return False

    lh_fname_len  = struct.unpack_from("<H", lh_raw, 26)[0]
    lh_extra_len  = struct.unpack_from("<H", lh_raw, 28)[0]
    data_start = lh_offset + 30 + lh_fname_len + lh_extra_len

    print(f"  datos en offset {data_start}, {comp_size/1e6:.1f} MB")

    resp = requests.get(gcs_url,
                        headers={"Range": f"bytes={data_start}-{data_start+comp_size-1}"},
                        stream=True, timeout=600)

    downloaded = 0
    chunk_sz   = 1024 * 512

    if compression == 0:   # STORED
        with open(dest_file, "wb") as f:
            for chunk in resp.iter_content(chunk_sz):
                f.write(chunk); downloaded += len(chunk)
                print(f"  {downloaded/1e6:.1f}/{comp_size/1e6:.1f} MB", end="\r", flush=True)
    elif compression == 8: # DEFLATE
        decomp = zlib.decompressobj(-15)
        with open(dest_file, "wb") as f:
            for chunk in resp.iter_content(chunk_sz):
                f.write(decomp.decompress(chunk)); downloaded += len(chunk)
                print(f"  {downloaded/1e6:.1f}/{comp_size/1e6:.1f} MB", end="\r", flush=True)
            remaining = decomp.flush()
            if remaining: f.write(remaining)
    else:
        print(f"  Compresion no soportada: {compression}")
        return False

    print(f"\n  OK -> {dest_file}")
    return True

os.makedirs(OUTPUT, exist_ok=True)

for target in TARGETS:
    # Buscar la clave exacta (con o sin prefijo)
    target_key = None
    if target in file_map:
        target_key = target
    else:
        fname_only = target.split("/")[-1]
        for k in file_map:
            if k.endswith(fname_only):
                target_key = k; break

    if target_key is None:
        print(f"\nNO ENCONTRADO: {target}")
        print("  Claves disponibles (muestra):", list(file_map.keys())[:10])
        continue

    lh_offset, comp_size, compression = file_map[target_key]
    parts    = target_key.replace("\\", "/").split("/")
    category = parts[-2] if len(parts) >= 2 else "misc"
    filename = parts[-1]
    dest_dir  = os.path.join(OUTPUT, category)
    dest_file = os.path.join(dest_dir, filename)

    os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(dest_file):
        print(f"Ya existe: {dest_file}"); continue

    print(f"\nDescargando {filename} ({comp_size/1e6:.1f} MB) ...")
    download_entry(gcs_url, target_key, lh_offset, comp_size, compression, dest_file)

print("\n=== COMPLETADO ===")
