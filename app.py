import os
import sys

# === CONFIGURACIÓN CRÍTICA DEL SISTEMA (Debe ir antes de cualquier otro import) ===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if sys.platform.startswith("linux"):
    # En la nube (Streamlit Cloud - Linux): Prevenir OOM y Segmentation Fault
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
else:
    # En local (Windows): Permitir el uso de la GPU (RTX 3060) y prevenir crash silencioso de OpenMP
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# CRÍTICO: Importar torch ANTES que streamlit, cv2 o tensorflow para evitar Segmentation Fault en Linux
import torch

if sys.platform.startswith("linux"):
    torch.set_num_threads(1)

import streamlit as st
import cv2
import numpy as np
import time
import tempfile

if sys.platform.startswith("linux"):
    cv2.setNumThreads(1)

def _configure_tf_gpu():
    """Configura TensorFlow para compartir la GPU. Se llama solo cuando se activa reconocimiento facial."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

_state = {"tf_configured": False}

from pathlib import Path
from datetime import datetime

import pandas as pd

import facial_auth



# Módulos propios
from database import (
    init_db,
    insert_event,
    get_all_events,
    get_event_counts_by_type,
    get_alert_count,
    clear_all_events,
)
from detector import PersonDetector, draw_detections
from rules import RulesEngine
from utils import (
    save_capture,
    frame_to_rgb,
    frame_to_jpeg_bytes,
    resize_frame,
    format_duration,
    get_status_emoji,
    get_event_color,
    normalize_zone,
    IS_CLOUD,
)

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="SERVSecurity – Control de Acceso",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado (importado de styles.py) ──────────────────────────────
from styles import get_custom_css
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ── Inicialización de BD ─────────────────────────────────────────────────────
init_db()


# ── Estado de sesión ──────────────────────────────────────────────────────────
# ── Alarma sonora (solo Windows) ──────────────────────────────────────────────
_HAS_WINSOUND = False
if sys.platform == "win32":
    try:
        import winsound
        _HAS_WINSOUND = True
    except ImportError:
        pass

ALARM_COOLDOWN_SEC = 10

def _play_alarm():
    """Reproduce un tono de alarma corto. Solo funciona en Windows."""
    if _HAS_WINSOUND:
        # Frecuencia 1000 Hz, duración 400ms — se ejecuta en el hilo principal
        # winsound.Beep es sincrónico pero 400ms es imperceptible en el loop
        winsound.Beep(1000, 400)


def _init_state():
    defaults = {
        "running":          False,
        "detector":         None,
        "rules_engine":     None,
        "last_capture_time": 0.0,
        "frame_count":      0,
        "alert_active":     False,
        "events_session":   [],
        "last_event_ids":   set(),
        "captures_count":   0,
        "last_alarm_time":  0.0,
        "fps_history":      [],
        "latency_history":  [],
        "current_incident_id": "",
        # Zona restringida — valores por defecto (porcentaje)
        # Inicializarlos aqui evita el warning:
        # "The widget with key 'zx1' was created with a default value
        #  but also had its value set via the Session State API."
        "zx1": 20, "zy1": 20, "zx2": 80, "zy2": 80,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Seleccionar fuente de video automaticamente segun el entorno.
    # En Hugging Face (Linux) no hay camara local, usamos Webcam en Nube.
    # En Windows usamos la Webcam Local.
    if "video_source" not in st.session_state:
        st.session_state.video_source = "Webcam en Nube" if IS_CLOUD else "Webcam Local"


_init_state()



# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR – Configuración
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom:1.5rem;'>
      <div style='font-size:2rem; font-weight:900; letter-spacing:1px; 
                  background: linear-gradient(90deg,#3b82f6,#06b6d4);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        SERVSecurity
      </div>
      <div style='font-size:0.7rem; color:#64748b; margin-top:2px;'>
        v1.0 // Control de Acceso IA
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Fuente de video ────────────────────────────────────────────────────
    st.markdown("#### Fuente de video")
    _source_options = ["Webcam Local", "Cámara IP (Seguridad)", "Subir video", "Webcam en Nube"]
    # En la nube no existe cámara local: pre-seleccionar y deshabilitar el selector
    # para evitar que el usuario escoja una fuente que causaría un error fatal.
    _default_source_idx = _source_options.index(st.session_state.video_source)
    video_source = st.radio(
        "Selecciona fuente",
        _source_options,
        index=_default_source_idx,
        disabled=IS_CLOUD,  # En HF Spaces solo hay Webcam en Nube
        label_visibility="collapsed",
        key="video_source",
    )

    ip_camera_url = None
    if "Cámara IP" in video_source:
        ip_camera_url = st.text_input(
            "URL RTSP/HTTP de la cámara", 
            placeholder="ej: rtsp://admin:12345@192.168.1.50/stream",
            help="Ingresa la dirección IP o RTSP de tu cámara de vigilancia."
        )

    uploaded_video = None
    if "Subir" in video_source:
        uploaded_video = st.file_uploader(
            "Sube un archivo de video (max 200 MB)",
            type=["mp4", "avi", "mov", "mkv"],
            help="Formatos soportados: MP4, AVI, MOV, MKV · Limite: 200 MB",
        )

    st.markdown("---")

    # ── Zona restringida ───────────────────────────────────────────────────
    st.markdown("#### Zona restringida")
    st.caption("Define el área protegida (% del ancho/alto del frame)")

    col_preset1, col_preset2, col_preset3 = st.columns(3)
    if col_preset1.button("Pantalla completa"):
        st.session_state.zx1, st.session_state.zy1 = 0, 0
        st.session_state.zx2, st.session_state.zy2 = 100, 100
    if col_preset2.button("Mitad inferior"):
        st.session_state.zx1, st.session_state.zy1 = 0, 50
        st.session_state.zx2, st.session_state.zy2 = 100, 100
    if col_preset3.button("Centro"):
        st.session_state.zx1, st.session_state.zy1 = 25, 25
        st.session_state.zx2, st.session_state.zy2 = 75, 75

    col_a, col_b = st.columns(2)
    with col_a:
        # NO pasamos el argumento 'value' a los sliders: Streamlit tomara el valor
        # de st.session_state[key] que ya inicializamos en _init_state().
        # Pasar 'value' Y tener la key en session_state genera el warning:
        # "widget with key 'zx1' was created with a default value but also had
        # its value set via the Session State API."
        zone_x1 = st.slider("X1 %", 0, 90, key="zx1")
        zone_y1 = st.slider("Y1 %", 0, 90, key="zy1")
    with col_b:
        zone_x2 = st.slider("X2 %", 10, 100, key="zx2")
        zone_y2 = st.slider("Y2 %", 10, 100, key="zy2")

    st.markdown("---")

    # ── Reglas ─────────────────────────────────────────────────────────────
    st.markdown("#### Parametros de deteccion")

    confidence = st.slider(
        "Confianza mínima (YOLO)", 0.20, 0.90, 0.45, 0.05,
        help="Umbral de confianza para aceptar una detección de persona",
    )
    max_perm = st.slider(
        "Permanencia máx. (seg)", 2, 30, 5,
        help="Segundos permitidos dentro de la zona antes de generar alerta",
    )

    st.markdown("##### Horario permitido")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        hour_start = st.text_input("Inicio", "08:00", key="hs")
    with col_h2:
        hour_end = st.text_input("Fin",   "20:00", key="he")

    use_overlap = st.checkbox(
        "Usar superposición de bbox", value=True,
        help="Si está activo, cualquier parte del cuerpo en la zona genera alerta. "
             "Si no, solo cuenta el centro.",
    )

    show_skeleton = st.checkbox(
        "Mostrar Esqueleto (Postura)", value=True,
        help="Muestra u oculta los puntos articulares en la imagen de la cámara.",
    )

    st.markdown("---")

    # ── Comportamiento (Behavioral Anomaly) ─────────────────────────────────
    st.markdown("#### Analisis de Comportamiento")
    st.caption("Deteccion de anomalías de movimiento y postura (Autorizados)")

    max_velocity = st.slider(
        "Umbral Movimiento Erratico (px/s)", 300, 1500, 800, 50,
        help="Velocidad que dispara alerta de 'Movimiento Erratico' (ej. carrera, forcejeo)."
    )
    max_aspect = st.slider(
        "Umbral Postura (Ancho/Alto)", 1.0, 2.5, 1.3, 0.1,
        help="Proporcion del cuerpo para detectar caídas o agachamientos inusuales."
    )

    st.markdown("---")
    
    # ── Biometría Facial ───────────────────────────────────────────────────
    st.markdown("#### Autenticación Biométrica")
    st.caption("Reconocimiento Facial mediante IA (One-Shot Learning)")

    use_face_auth = st.checkbox(
        "Activar Reconocimiento Facial", value=True,
        help="Si está activo, solo las personas en la carpeta 'authorized_faces' pueden entrar a la zona."
    )
    if IS_CLOUD and use_face_auth:
        st.warning(
            "💡 En la nube el reconocimiento facial tarda más (CPU limitada). "
            "El primer análisis puede demorar 3-8 segundos."
        )

    # ── Gestión de rostros autorizados ─────────────────────────────────────
    FACES_DIR = Path("authorized_faces")
    FACES_DIR.mkdir(exist_ok=True)

    with st.expander("👤 Gestionar Rostros Autorizados", expanded=False):

        # Contar caras existentes
        existing_faces = sorted([
            f for f in FACES_DIR.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        st.caption(f"{len(existing_faces)} persona(s) registrada(s)")

        # ── Subir foto desde PC ────────────────────────────────────────────
        st.markdown("**📤 Añadir por foto**")
        face_name = st.text_input(
            "Nombre completo / ID",
            placeholder="ej: GARCIA JUAN 12345678",
            key="face_name_input",
        )
        uploaded_face = st.file_uploader(
            "Selecciona la foto",
            type=["jpg", "jpeg", "png"],
            key="face_uploader",
            label_visibility="collapsed",
        )
        if st.button("💾 Guardar rostro", use_container_width=True, key="btn_save_face"):
            if not face_name.strip():
                st.warning("⚠️ Escribe el nombre antes de guardar.")
            elif uploaded_face is None:
                st.warning("⚠️ Selecciona una foto primero.")
            else:
                safe_name = face_name.strip().replace("/", "-").replace("\\", "-")
                ext = Path(uploaded_face.name).suffix.lower() or ".jpg"
                dest = FACES_DIR / f"{safe_name}{ext}"
                dest.write_bytes(uploaded_face.getvalue())
                # Borrar caché .pkl para que DeepFace re-indexe
                for pkl in FACES_DIR.glob("*.pkl"):
                    try:
                        pkl.unlink()
                    except OSError:
                        pass
                st.success(f"✅ **{safe_name}** guardado correctamente.")
                st.session_state.running = False
                st.rerun()

        st.markdown("---")

        # ── Capturar desde webcam ──────────────────────────────────────────
        st.markdown("**📸 Capturar desde cámara**")
        cam_name = st.text_input(
            "Nombre para la captura",
            placeholder="ej: PEREZ MARIA 98765432",
            key="cam_name_input",
        )
        cam_photo = st.camera_input(
            "Toma la foto",
            key="face_camera",
            label_visibility="collapsed",
        )
        if st.button("💾 Guardar captura", use_container_width=True, key="btn_save_cam"):
            if not cam_name.strip():
                st.warning("⚠️ Escribe el nombre antes de guardar.")
            elif cam_photo is None:
                st.warning("⚠️ Toma una foto primero.")
            else:
                safe_name = cam_name.strip().replace("/", "-").replace("\\", "-")
                dest = FACES_DIR / f"{safe_name}.jpg"
                dest.write_bytes(cam_photo.getvalue())
                for pkl in FACES_DIR.glob("*.pkl"):
                    try:
                        pkl.unlink()
                    except OSError:
                        pass
                st.success(f"✅ **{safe_name}** guardado correctamente.")
                st.session_state.running = False
                st.rerun()

        st.markdown("---")

        # ── Galería de rostros registrados ────────────────────────────────
        if existing_faces:
            st.markdown("**🗂️ Personas registradas**")
            for face_path in existing_faces:
                col_img, col_info, col_del = st.columns([1, 3, 1])
                with col_img:
                    try:
                        st.image(str(face_path), width=48)
                    except Exception:
                        st.write("🖼️")
                with col_info:
                    st.caption(face_path.stem)
                with col_del:
                    if st.button(
                        "🗑️",
                        key=f"del_{face_path.name}",
                        help=f"Eliminar {face_path.stem}",
                    ):
                        face_path.unlink(missing_ok=True)
                        for pkl in FACES_DIR.glob("*.pkl"):
                            try:
                                pkl.unlink()
                            except OSError:
                                pass
                        st.success(f"🗑️ **{face_path.stem}** eliminado.")
                        st.session_state.running = False
                        st.rerun()
        else:
            st.info("No hay personas registradas aún.")

    st.markdown("---")

    # ── Controles ──────────────────────────────────────────────────────────
    st.markdown("#### Controles")

    capture_interval = st.slider(
        "Intervalo de captura (seg)", 1, 30, 5,
        help="Cada cuántos segundos guardar una imagen en /captures",
    )

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        btn_start = st.button("Iniciar", use_container_width=True)
    with col_c2:
        btn_stop  = st.button("Detener", use_container_width=True)

    if st.button("Limpiar eventos", use_container_width=True):
        clear_all_events()
        st.session_state.events_session = []
        st.success("Eventos eliminados.")

    st.markdown("---")
    st.caption("Demo academica // YOLOv8 // SQLite")
    st.caption("github.com/CristophSyr/SERVSecurity")


# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
status_badge = (
    '<span class="badge-active">● SISTEMA ACTIVO</span>'
    if st.session_state.running
    else '<span class="badge-idle">○ EN ESPERA</span>'
)
st.markdown(f"""
<div class="header-banner">
  <div style="width:48px;height:48px;border-radius:12px;background:linear-gradient(135deg,#3b82f6,#06b6d4);display:flex;align-items:center;justify-content:center;font-size:1.4rem;font-weight:900;color:white;box-shadow:0 0 20px rgba(59,130,246,0.3);flex-shrink:0;">S</div>
  <div>
    <div class="header-title">SERVSecurity</div>
    <div style="color:#94a3b8; font-size:0.82rem; margin-top:4px; display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
      <span>Sistema Inteligente de Control de Acceso</span>
      <span style="color:#334155;">|</span>
      <span>Analisis de Video con IA</span>
      <span style="color:#334155;">|</span>
      {status_badge}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MÉTRICAS SUPERIORES
# ════════════════════════════════════════════════════════════════════════════
total_alerts    = get_alert_count()
all_events_data = get_all_events(limit=500)
total_events    = len(all_events_data)
total_normal    = sum(1 for e in all_events_data if e["estado"] == "normal")

st.markdown(f"""
<div class="stat-grid" style="grid-template-columns: repeat(4, 1fr); margin-bottom:1rem;">
  <div class="stat-card">
    <div class="stat-label">ALERTAS TOTALES</div>
    <div class="stat-value red">{total_alerts}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">EVENTOS REGISTRADOS</div>
    <div class="stat-value blue">{total_events}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">ESTADO NORMAL</div>
    <div class="stat-value green">{total_normal}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">FRAMES ANALIZADOS</div>
    <div class="stat-value amber">{st.session_state.frame_count}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL: VIDEO | PANEL DERECHO
# ════════════════════════════════════════════════════════════════════════════
vid_col, info_col = st.columns([3, 2], gap="large")


# ── Columna izquierda: video ──────────────────────────────────────────────
with vid_col:
    st.markdown("### Feed de Video")

    video_placeholder = st.empty()
    alert_placeholder = st.empty()

    # Imagen inicial con preview de la zona restringida
    if not st.session_state.running:
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)

        # Dibujar preview de la zona restringida
        zx1 = int(zone_x1 / 100 * 640)
        zy1 = int(zone_y1 / 100 * 360)
        zx2 = int(zone_x2 / 100 * 640)
        zy2 = int(zone_y2 / 100 * 360)
        # Relleno semitransparente naranja
        overlay = dummy.copy()
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 165, 255), -1)
        cv2.addWeighted(overlay, 0.15, dummy, 0.85, 0, dummy)
        # Borde de la zona
        cv2.rectangle(dummy, (zx1, zy1), (zx2, zy2), (0, 165, 255), 2)
        cv2.putText(dummy, "ZONA RESTRINGIDA", (zx1 + 8, zy1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # Texto de bienvenida
        cv2.putText(dummy, "SERVSecurity", (150, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (59, 130, 246), 3)
        cv2.putText(dummy, "Presiona  Iniciar  para comenzar", (100, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 116, 139), 2)
        cv2.putText(dummy, "Control de Acceso Inteligente", (130, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (6, 182, 212), 2)
        video_placeholder.image(
            frame_to_jpeg_bytes(dummy),
        )


# ── Columna derecha: info en tiempo real ─────────────────────────────────
with info_col:
    st.markdown("### Estado en tiempo real")
    live_status = st.empty()
    live_events = st.empty()


# ════════════════════════════════════════════════════════════════════════════
# ACCIONES DE CONTROL
# ════════════════════════════════════════════════════════════════════════════
if btn_start and not st.session_state.running:
    try:
        # Inicializar detector y reglas para cualquier fuente de video
        with st.spinner("Cargando modelo YOLOv8-Pose..."):
            st.session_state.detector = PersonDetector(
                model_name="yolov8n-pose.pt",
                confidence=confidence,
            )
        
        authenticator = None
        if use_face_auth:
            with st.spinner("Cargando motor de Autenticación Facial..."):
                if not _state["tf_configured"] and not sys.platform.startswith("linux"):
                    _configure_tf_gpu()
                    _state["tf_configured"] = True
                authenticator = facial_auth.FacialAuthenticator()

        st.session_state.rules_engine = RulesEngine(
            zone=(0, 0, 100, 100),
            max_permanence_sec=max_perm,
            allowed_start=hour_start,
            allowed_end=hour_end,
            use_overlap=use_overlap,
            max_velocity_px_sec=max_velocity,
            max_aspect_ratio=max_aspect,
            authenticator=authenticator,
        )

        st.session_state.running = True
        st.session_state.frame_count = 0
        st.session_state.alert_active = False
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error al iniciar el detector: {e}")

if btn_stop and st.session_state.running:
    st.session_state.running = False
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# LOOP DE PROCESAMIENTO DE VIDEO
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.running:

    # ── Webcam del navegador usando componente JS (sin WebRTC) ─────────────
    if "Nube" in video_source:
        from camera_input_live import camera_input_live

        with vid_col:
            st.info(
                "📹 **Cámara en vivo** — Acepta el permiso de cámara cuando aparezca."
            )

            # Usa el componente live (1000ms = 1 fps, seguro para la CPU gratuita)
            image_data = camera_input_live(show_controls=False, debounce=1000)

        if image_data is not None:
            # image_data es un objeto subido por st.camera_input (BytesIO / PIL Image wrappers)
            # Leer los bytes
            bytes_data = image_data.getvalue()
            img_array = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is not None:
                img = resize_frame(img, max_width=640)
                h, w = img.shape[:2]

                detector = st.session_state.detector
                rules_engine = st.session_state.rules_engine

                zone_px = normalize_zone(
                    (zone_x1, zone_y1, zone_x2, zone_y2), w, h
                )
                rules_engine.update_zone(zone_px)

                # Detección con IA
                detections, anomaly_info = detector.detect(img)
                alerts, events = rules_engine.evaluate(detections)

                has_alert = any(alerts) or anomaly_info is not None

                annotated = draw_detections(
                    img,
                    detections,
                    alerts,
                    zone=zone_px,
                    zone_active=True,
                    draw_skeleton=show_skeleton,
                    anomaly_info=anomaly_info,
                )

                # Guardar captura si hay alerta
                now_ts = time.time()
                cap_path = ""
                if (has_alert or len(detections) > 0) and (
                    now_ts - st.session_state.get("last_capture_time", 0) >= capture_interval
                ):
                    cap_path = save_capture(
                        annotated,
                        prefix="alerta" if has_alert else "presencia",
                    )
                    st.session_state["last_capture_time"] = now_ts

                # Registrar eventos
                for event in events:
                    insert_event(
                        tipo_evento=event["tipo_evento"],
                        estado=event["estado"],
                        duracion=event["duracion"],
                        captura=cap_path,
                        detalle=event.get("detalle", ""),
                    )

                if anomaly_info is not None:
                    insert_event(
                        tipo_evento="ANOMALIA_DETECTADA",
                        estado="Alerta Global",
                        duracion=0,
                        captura=cap_path,
                        detalle=f"Se detectó posible crimen: {anomaly_info['class']} (Conf: {anomaly_info['conf']:.0%})",
                    )

                # Mostrar frame anotado
                with vid_col:
                    st.image(
                        frame_to_jpeg_bytes(annotated),
                        caption=f"🔴 EN VIVO — {len(detections)} persona(s) detectada(s)",
                        use_container_width=True,
                    )

                with info_col:
                    n_det = len(detections)
                    st.metric("Personas", n_det)
                    if has_alert:
                        st.error("⚠️ ALERTA ACTIVA")
                    else:
                        st.success("✅ Sin alertas")

        with info_col:
            st.markdown("### 📹 Cámara en Vivo")
            st.write("La cámara captura frames continuamente y la IA los analiza en tiempo real.")

        st.stop()

    detector     = st.session_state.detector
    rules_engine = st.session_state.rules_engine

    # ── Abrir fuente de video subida o local ──────────────────────────────────────
    cap = None
    tmp_path = None

    if "Local" in video_source:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ No se pudo abrir la cámara web local.")
            st.session_state.running = False
            st.stop()
    elif "Cámara IP" in video_source:
        if not ip_camera_url:
            st.warning("⚠️ Por favor ingresa la URL de la cámara.")
            st.session_state.running = False
            st.stop()
        cap = cv2.VideoCapture(ip_camera_url)
        if not cap.isOpened():
            st.error("❌ No se pudo conectar a la cámara IP. Verifica la URL o la red.")
            st.session_state.running = False
            st.stop()
    elif uploaded_video is not None:
        # Guardar temporalmente el video subido
        suffix = Path(uploaded_video.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            st.error("No se pudo abrir el video. Intenta con otro archivo.")
            st.session_state.running = False
            st.stop()
    else:
        st.warning("Selecciona una fuente de video y presiona Iniciar.")
        st.session_state.running = False
        st.stop()

    # ── Loop principal de frames ──────────────────────────────────────────
    stop_btn_placeholder = st.empty()

    # Frame-skipping adaptativo:
    # - Nube (CPU limitada): procesar IA cada 8 frames
    # - Local (GPU): procesar IA cada 3 frames para mejor fluidez
    PROCESS_EVERY_N = 8 if IS_CLOUD else 3
    # Throttle del panel derecho: actualizar info cada N frames
    # (evita consultas BD y re-render de Streamlit en cada iteración)
    UPDATE_PANEL_EVERY_N = 25 if IS_CLOUD else 15

    last_detections = []
    last_anomaly = None
    last_alerts = []

    loop_start_time = time.time()

    try:
        while st.session_state.running:
            frame_start = time.time()

            ret, frame = cap.read()

            if not ret:
                if tmp_path:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    break

            frame = resize_frame(frame, max_width=640)
            h, w  = frame.shape[:2]

            zone_px = normalize_zone(
                (zone_x1, zone_y1, zone_x2, zone_y2), w, h
            )
            rules_engine.update_zone(zone_px)

            # Frame-skipping: solo ejecutar IA cada N frames para no saturar la CPU
            frame_num = st.session_state.frame_count
            run_ai = (frame_num % PROCESS_EVERY_N == 0)

            inference_ms = 0.0
            if run_ai:
                t0 = time.time()
                detections, anomaly_info = detector.detect(frame)
                inference_ms = (time.time() - t0) * 1000
                alerts, events = rules_engine.evaluate(detections)
                last_detections = detections
                last_anomaly = anomaly_info
                last_alerts = alerts
                # Registrar latencia
                lat_hist = st.session_state.latency_history
                lat_hist.append(inference_ms)
                if len(lat_hist) > 60:
                    lat_hist.pop(0)
            else:
                detections = last_detections
                anomaly_info = last_anomaly
                alerts = last_alerts
                events = []

            has_alert = any(alerts) or anomaly_info is not None

            # ── Agrupación de Incidentes ──────────────────────────────────
            if has_alert and not st.session_state.alert_active:
                # Inició una nueva alerta (nuevo incidente)
                st.session_state.current_incident_id = f"INC_{int(time.time())}"
            elif not has_alert and st.session_state.alert_active:
                # Terminó la alerta
                st.session_state.current_incident_id = ""

            st.session_state.alert_active = has_alert

            # ── Alarma sonora (Comentada temporalmente para pruebas) ──────
            # if has_alert:
            #     now_alarm = time.time()
            #     if now_alarm - st.session_state.last_alarm_time >= ALARM_COOLDOWN_SEC:
            #         _play_alarm()
            #         st.session_state.last_alarm_time = now_alarm

            # Dibujar anotaciones
            annotated = draw_detections(
                frame, detections, alerts,
                zone=zone_px, zone_active=True,
                draw_skeleton=show_skeleton,
                anomaly_info=anomaly_info
            )

            # ── Calcular FPS ──────────────────────────────────────────────
            frame_elapsed = time.time() - frame_start
            current_fps = 1.0 / max(frame_elapsed, 0.001)
            fps_hist = st.session_state.fps_history
            fps_hist.append(current_fps)
            if len(fps_hist) > 60:
                fps_hist.pop(0)

            # Mostrar frame
            avg_fps = sum(fps_hist) / len(fps_hist)
            video_placeholder.image(
                frame_to_jpeg_bytes(annotated),
                caption=f"Frame #{frame_num} · "
                        f"{len(detections)} persona(s) · "
                        f"{avg_fps:.1f} FPS",
            )

            # ── Guardar captura periódica ─────────────────────────────────
            now_ts = time.time()
            if (has_alert or len(detections) > 0) and \
               (now_ts - st.session_state.last_capture_time >= capture_interval):
                cap_path = save_capture(
                    annotated,
                    prefix="alerta" if has_alert else "presencia",
                )
                st.session_state.last_capture_time = now_ts
                st.session_state.captures_count += 1
            else:
                cap_path = ""

            # ── Registrar eventos en BD ───────────────────────────────────
            inc_id = st.session_state.current_incident_id
            for event in events:
                insert_event(
                    tipo_evento=event["tipo_evento"],
                    estado     =event["estado"],
                    duracion   =event["duracion"],
                    captura    =cap_path,
                    detalle    =event.get("detalle", ""),
                    incident_id=inc_id,
                )
            
            # Registrar evento de anomalía si se detecta
            if anomaly_info is not None:
                insert_event(
                    tipo_evento="ANOMALIA_DETECTADA",
                    estado="Alerta Global",
                    duracion=0,
                    captura=cap_path,
                    detalle=f"Se detectó posible crimen: {anomaly_info['class']} (Conf: {anomaly_info['conf']:.0%})",
                    incident_id=inc_id,
                )

            # ── Panel derecho: estado en vivo (throttled) ─────────────────
            # Solo actualizar cada N frames para evitar overhead de BD queries
            # y re-renders innecesarios de Streamlit
            if frame_num % UPDATE_PANEL_EVERY_N == 0:
                with live_status.container():
                    if has_alert:
                        # Buscar la razon de la alerta en los eventos actuales
                        alert_reason = "Anomalía Detectada"
                        if events:
                            # Tomar el último evento sospechoso
                            suspicious_events = [e for e in events if e["estado"] == "sospechoso"]
                            if suspicious_events:
                                alert_reason = suspicious_events[-1]["tipo_evento"]

                        alert_placeholder.markdown(f"""
                        <div class="alert-box">
                          <b>ALERTA ACTIVA</b> – {alert_reason}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        alert_placeholder.markdown("""
                        <div class="normal-box">
                          <b>Estado Normal</b> – Sin anomalías detectadas
                        </div>
                        """, unsafe_allow_html=True)

                    # Calcular métricas de rendimiento
                    avg_fps_panel = sum(st.session_state.fps_history) / max(len(st.session_state.fps_history), 1)
                    lat_hist = st.session_state.latency_history
                    avg_lat = sum(lat_hist) / max(len(lat_hist), 1)

                    st.markdown(f"""
                    | Parámetro | Valor |
                    |-----------|-------|
                    | 👤 Personas detectadas | **{len(detections)}** |
                    | 🚨 Alertas activas | **{sum(alerts)}** |
                    | 🎞 Frames procesados | **{frame_num}** |
                    | 📁 Capturas guardadas | **{st.session_state.captures_count}** |
                    | ⚡ FPS promedio | **{avg_fps_panel:.1f}** |
                    | 🧠 Latencia IA | **{avg_lat:.0f} ms** |
                    | 🕐 Hora actual | **{datetime.now().strftime('%H:%M:%S')}** |
                    """)

                # Últimos eventos en tiempo real
                recent = get_all_events(limit=10)
                with live_events.container():
                    st.markdown("**Últimos eventos**")
                    for ev in recent[:5]:
                        emoji = get_status_emoji(ev["estado"])
                        st.markdown(
                            f"{emoji} `{ev['hora']}` — {ev['tipo_evento']} "
                            f"(*{ev['duracion']:.1f}s*)"
                        )

            st.session_state.frame_count += 1

            # Ceder CPU al event loop de Streamlit en cada frame.
            # Sin este sleep, el while True monopoliza la CPU e impide que
            # Streamlit procese clics del botón Detener y otras actualizaciones de UI.
            time.sleep(0.01)

    finally:
        cap.release()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        st.session_state.running = False


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD (importado de dashboard.py)
# ════════════════════════════════════════════════════════════════════════════
from dashboard import render_dashboard
render_dashboard()
