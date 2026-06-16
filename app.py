"""
app.py – Aplicación principal SERVSecurity con Streamlit.

Sistema inteligente de control de acceso físico basado en análisis de video
para entornos de servidor.

Ejecutar con:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if sys.platform.startswith("linux"):
    # En la nube (Streamlit Cloud - Linux): Prevenir Segmentation Fault por choque de librerías CUDNN
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    # En local (Windows): Permitir el uso de la GPU (RTX 3060) y prevenir crash silencioso de OpenMP
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import tensorflow as tf
import torch

# Configurar TensorFlow para compartir la GPU de forma amigable con PyTorch (solo útil en local)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import facial_auth

# WebRTC es opcional – solo se usa para cámara remota en la nube
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except (ImportError, Exception):
    WEBRTC_AVAILABLE = False

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
    resize_frame,
    format_duration,
    get_status_emoji,
    get_event_color,
    normalize_zone,
)

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="SERVSecurity – Control de Acceso",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {
    --bg-primary:   #060a13;
    --bg-card:      rgba(17,24,39,0.7);
    --bg-glass:     rgba(15,23,42,0.6);
    --accent-blue:  #3b82f6;
    --accent-cyan:  #06b6d4;
    --accent-green: #10b981;
    --accent-red:   #ef4444;
    --accent-amber: #f59e0b;
    --accent-purple:#8b5cf6;
    --text-primary: #f1f5f9;
    --text-muted:   #64748b;
    --border:       rgba(56,68,90,0.5);
    --glow-blue:    rgba(59,130,246,0.25);
    --glow-red:     rgba(239,68,68,0.25);
    --glow-green:   rgba(16,185,129,0.25);
    --glow-purple:  rgba(139,92,246,0.2);
  }

  /* ── Background con gradiente animado ── */
  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    background-image:
      radial-gradient(ellipse 80% 50% at 20% 40%, rgba(59,130,246,0.06) 0%, transparent 50%),
      radial-gradient(ellipse 60% 40% at 80% 20%, rgba(139,92,246,0.05) 0%, transparent 50%),
      radial-gradient(ellipse 50% 60% at 50% 90%, rgba(6,182,212,0.04) 0%, transparent 50%) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
  }

  /* ── Sidebar Premium ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,14,26,0.95) 0%, rgba(15,23,42,0.95) 100%) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(20px) !important;
  }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {
    color: var(--accent-cyan) !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    margin-bottom: 0.5rem !important;
  }

  /* ── Glassmorphism Metric Cards ── */
  [data-testid="stMetric"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.4rem !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
  }
  [data-testid="stMetric"]:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0 20px var(--glow-blue) !important;
    border-color: rgba(59,130,246,0.3) !important;
  }
  [data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
  }
  [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
  }
  [data-testid="stMetricDelta"] > div {
    color: var(--accent-red) !important;
  }

  /* ── Headers ── */
  h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.02em !important;
  }

  /* ── Premium Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-cyan) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: 0 4px 15px var(--glow-blue), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    position: relative !important;
    overflow: hidden !important;
  }
  .stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 30px var(--glow-blue), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    filter: brightness(1.1) !important;
  }
  .stButton > button:active {
    transform: translateY(0) scale(0.98) !important;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
  }

  /* ── Alert Box (ALERTA ACTIVA) ── */
  .alert-box {
    background: linear-gradient(135deg, rgba(30,5,5,0.9), rgba(50,10,10,0.8));
    border: 1px solid rgba(239,68,68,0.5);
    border-left: 4px solid var(--accent-red);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    animation: pulse-red 2s ease-in-out infinite, shake-alert 0.5s ease-in-out;
    box-shadow: 0 0 30px var(--glow-red), inset 0 0 60px rgba(239,68,68,0.03);
    backdrop-filter: blur(10px);
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .alert-box b {
    font-size: 1.05rem;
    color: #ff6b6b;
  }
  @keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 20px var(--glow-red); border-color: rgba(239,68,68,0.5); }
    50%      { box-shadow: 0 0 50px var(--glow-red), 0 0 80px rgba(239,68,68,0.1); border-color: rgba(239,68,68,0.8); }
  }
  @keyframes shake-alert {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-3px); }
    75% { transform: translateX(3px); }
  }

  /* ── Normal Box ── */
  .normal-box {
    background: linear-gradient(135deg, rgba(5,30,20,0.8), rgba(10,40,28,0.7));
    border: 1px solid rgba(16,185,129,0.3);
    border-left: 4px solid var(--accent-green);
    border-radius: 14px;
    padding: 1rem 1.5rem;
    margin: 0.3rem 0;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px var(--glow-green);
    display: flex;
    align-items: center;
    gap: 12px;
    animation: breathe-green 4s ease-in-out infinite;
  }
  .normal-box b { color: #34d399; }
  @keyframes breathe-green {
    0%, 100% { box-shadow: 0 4px 20px var(--glow-green); }
    50%      { box-shadow: 0 4px 30px rgba(16,185,129,0.35); }
  }

  /* ── Header Banner Premium ── */
  .header-banner {
    background: linear-gradient(135deg, rgba(10,14,26,0.9) 0%, rgba(17,24,39,0.85) 50%, rgba(10,14,26,0.9) 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
    position: relative;
    overflow: hidden;
  }
  .header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-cyan), var(--accent-purple), transparent);
    animation: shimmer-top 3s ease-in-out infinite;
  }
  @keyframes shimmer-top {
    0%, 100% { opacity: 0.6; }
    50%      { opacity: 1; }
  }
  .header-title {
    font-size: 1.8rem;
    font-weight: 900;
    letter-spacing: 0.5px;
    background: linear-gradient(135deg, #60a5fa, #06b6d4, #8b5cf6);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-shift 4s ease infinite;
  }
  @keyframes gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
  }

  /* ── Status badges ── */
  .badge-active {
    background: linear-gradient(135deg, #10b981, #34d399);
    color: #000;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.72rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    box-shadow: 0 0 12px var(--glow-green);
    animation: pulse-badge 2s ease-in-out infinite;
  }
  @keyframes pulse-badge {
    0%, 100% { box-shadow: 0 0 12px var(--glow-green); }
    50%      { box-shadow: 0 0 20px rgba(16,185,129,0.5); }
  }
  .badge-idle {
    background: rgba(100,116,139,0.3);
    color: #94a3b8;
    font-weight: 600;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.72rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    border: 1px solid rgba(100,116,139,0.3);
  }

  /* ── Divider ── */
  hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
    margin: 1rem 0 !important;
  }

  /* ── Video Container ── */
  .video-container {
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 0 40px var(--glow-blue);
    position: relative;
  }

  /* ── Tabs Premium ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    background: var(--bg-glass) !important;
    border-radius: 14px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    color: white !important;
    box-shadow: 0 4px 15px var(--glow-blue) !important;
  }

  /* ── Section Titles ── */
  .section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }

  /* ── Stat Card (para panel derecho) ── */
  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 0.8rem 0;
  }
  .stat-card {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 14px;
    backdrop-filter: blur(8px);
    transition: all 0.25s ease;
  }
  .stat-card:hover {
    border-color: rgba(59,130,246,0.3);
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
  }
  .stat-card .stat-label {
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
  }
  .stat-card .stat-value {
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--text-primary);
  }
  .stat-card .stat-value.blue { color: var(--accent-blue); }
  .stat-card .stat-value.red { color: var(--accent-red); }
  .stat-card .stat-value.green { color: var(--accent-green); }
  .stat-card .stat-value.amber { color: var(--accent-amber); }

  /* ── Event Log Items ── */
  .event-item {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.82rem;
    transition: all 0.2s ease;
    backdrop-filter: blur(8px);
  }
  .event-item:hover {
    border-color: rgba(59,130,246,0.3);
    transform: translateX(4px);
  }
  .event-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .event-dot.red { background: var(--accent-red); box-shadow: 0 0 8px var(--glow-red); }
  .event-dot.green { background: var(--accent-green); box-shadow: 0 0 8px var(--glow-green); }
  .event-time { color: var(--text-muted); font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; }
  .event-text { color: var(--text-primary); flex: 1; }

  /* ── Scrollable log ── */
  .log-container {
    max-height: 300px;
    overflow-y: auto;
    padding-right: 8px;
  }
  .log-container::-webkit-scrollbar { width: 5px; }
  .log-container::-webkit-scrollbar-track { background: transparent; }
  .log-container::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.3); border-radius: 3px; }
  .log-container::-webkit-scrollbar-thumb:hover { background: rgba(59,130,246,0.5); }

  /* ── Slider styling ── */
  [data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    border-radius: 12px !important;
  }

  /* ── Radio buttons in sidebar ── */
  [data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Inicialización de BD ─────────────────────────────────────────────────────
init_db()


# ── Estado de sesión ──────────────────────────────────────────────────────────
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
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

if WEBRTC_AVAILABLE:
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )


if WEBRTC_AVAILABLE:
    def make_webrtc_processor(
        confidence,
        zone_percent,
        max_perm,
        hour_start,
        hour_end,
        use_overlap,
        max_velocity,
        max_aspect,
        use_face_auth,
        capture_interval,
        show_skeleton,
    ):
        class SERVSecurityWebRTCProcessor(VideoProcessorBase):
            def __init__(self):
                self.detector = PersonDetector(
                    model_name="yolov8n-pose.pt",
                    confidence=confidence,
                )
                
                authenticator = facial_auth.FacialAuthenticator() if use_face_auth else None
                
                self.rules_engine = RulesEngine(
                    zone=(0, 0, 100, 100),
                    max_permanence_sec=max_perm,
                    allowed_start=hour_start,
                    allowed_end=hour_end,
                    use_overlap=use_overlap,
                    max_velocity_px_sec=max_velocity,
                    max_aspect_ratio=max_aspect,
                    authenticator=authenticator,
                )
                self.last_capture_time = 0.0

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img = resize_frame(img, max_width=800)

                h, w = img.shape[:2]

                zone_px = normalize_zone(zone_percent, w, h)
                self.rules_engine.update_zone(zone_px)

                detections, anomaly_info = self.detector.detect(img)
                alerts, events = self.rules_engine.evaluate(detections)

                has_alert = any(alerts) or anomaly_info is not None

                annotated = draw_detections(
                    img,
                    detections,
                    alerts,
                    zone=zone_px,
                    zone_active=True,
                    draw_skeleton=show_skeleton,
                    anomaly_info=anomaly_info
                )

                now_ts = time.time()
                cap_path = ""

                if (has_alert or len(detections) > 0) and (
                    now_ts - self.last_capture_time >= capture_interval
                ):
                    cap_path = save_capture(
                        annotated,
                        prefix="alerta" if has_alert else "presencia",
                    )
                    self.last_capture_time = now_ts

                for event in events:
                    insert_event(
                        tipo_evento=event["tipo_evento"],
                        estado=event["estado"],
                        duracion=event["duracion"],
                        captura=cap_path,
                        detalle=event.get("detalle", ""),
                    )
                    
                # Registrar el evento global de anomalía si ocurrió uno nuevo
                if anomaly_info is not None:
                    # En un entorno real se haría un debounce (ej. no guardar si hace 5 seg se guardó otro igual)
                    insert_event(
                        tipo_evento="ANOMALIA_DETECTADA",
                        estado="Alerta Global",
                        duracion=0,
                        captura=cap_path,
                        detalle=f"Se detectó posible crimen: {anomaly_info['class']} (Conf: {anomaly_info['conf']:.0%})",
                    )

                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        return SERVSecurityWebRTCProcessor

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
    video_source = st.radio(
        "Selecciona fuente",
        ["Webcam Local", "Cámara IP (Seguridad)", "Subir video", "Webcam WebRTC (Nube)"],
        label_visibility="collapsed",
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

    col_a, col_b = st.columns(2)
    with col_a:
        zone_x1 = st.slider("X1 %", 0, 90, 20, key="zx1")
        zone_y1 = st.slider("Y1 %", 0, 90, 20, key="zy1")
    with col_b:
        zone_x2 = st.slider("X2 %", 10, 100, 80, key="zx2")
        zone_y2 = st.slider("Y2 %", 10, 100, 80, key="zy2")

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
    st.info("Coloca fotos en la carpeta `authorized_faces/`")

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

    # Imagen inicial cuando el sistema está inactivo
    if not st.session_state.running:
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        # Texto de bienvenida
        cv2.putText(dummy, "SERVSecurity", (150, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (59, 130, 246), 3)
        cv2.putText(dummy, "Presiona  Iniciar  para comenzar", (100, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 116, 139), 2)
        cv2.putText(dummy, "Control de Acceso Inteligente", (130, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (6, 182, 212), 2)
        video_placeholder.image(
        cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB),
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
        # Para subidas, local o IP: control manual
        if "Subir" in video_source or "Local" in video_source or "Cámara IP" in video_source:
            # Inicializar detector y reglas para video local o subido
            with st.spinner("Cargando modelo YOLOv8-Pose..."):
                st.session_state.detector = PersonDetector(
                    model_name="yolov8n-pose.pt",
                    confidence=confidence,
                )
            
            authenticator = None
            if use_face_auth:
                with st.spinner("Cargando motor de Autenticación Facial..."):
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
        else:
            # WebRTC maneja su propio detector internamente
            st.session_state.detector = None
            st.session_state.rules_engine = None

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

    # ── Webcam del navegador usando WebRTC ────────────────────────────────
    if "WebRTC" in video_source:
        if not WEBRTC_AVAILABLE:
            with vid_col:
                st.error(
                    "❌ **WebRTC no está disponible.**\n\n"
                    "El componente `streamlit-webrtc` no se pudo cargar correctamente. "
                    "Usa la opción **📸 Webcam Local** o **📁 Subir video** en su lugar."
                )
            st.session_state.running = False
            st.stop()

        with vid_col:
            st.info(
                "La cámara se abrirá desde tu navegador. "
                "Acepta el permiso de cámara cuando aparezca."
            )

            # Pre-cargar modelos para evitar timeout de WebRTC en la nube
            with st.spinner("Preparando cerebros de IA (puede tardar un minuto la primera vez que inicia el servidor)..."):
                from detector import PersonDetector
                _dummy_det = PersonDetector(model_name="yolov8n-pose.pt")
                if use_face_auth:
                    _dummy_fa = facial_auth.FacialAuthenticator()

            webrtc_streamer(
                key="servsecurity-webrtc",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=make_webrtc_processor(
                    confidence=confidence,
                    zone_percent=(zone_x1, zone_y1, zone_x2, zone_y2),
                    max_perm=max_perm,
                    hour_start=hour_start,
                    hour_end=hour_end,
                    use_overlap=use_overlap,
                    max_velocity=max_velocity,
                    max_aspect=max_aspect,
                    use_face_auth=use_face_auth,
                    capture_interval=capture_interval,
                    show_skeleton=show_skeleton,
                ),
                media_stream_constraints={
                    "video": True,
                    "audio": False,
                },
                async_processing=True,
            )

        with info_col:
            st.markdown("### Modo Webcam WebRTC")
            st.write("La deteccion se procesa directamente desde el video del navegador.")
            st.write("Si el navegador pide permisos, selecciona **Permitir camara**.")
            st.info("Nota: Si el componente no carga, asegurate de no usar bloqueadores de anuncios (como Brave Shields) que bloquean WebRTC. Para uso local, usa 'Webcam Local'.")

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

    try:
        while st.session_state.running:
            ret, frame = cap.read()

            if not ret:
                if tmp_path:
                    # Video terminó – reiniciar desde el principio
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    break

            frame = resize_frame(frame, max_width=800)
            h, w  = frame.shape[:2]

            # Calcular zona en píxeles a partir de porcentajes
            zone_px = normalize_zone(
                (zone_x1, zone_y1, zone_x2, zone_y2), w, h
            )
            rules_engine.update_zone(zone_px)
            if detector.model:
                pass   # la confianza se pasa en detect()

            # Detección de personas y anomalías
            detections, anomaly_info = detector.detect(frame)

            # Evaluación de reglas
            alerts, events = rules_engine.evaluate(detections)
            has_alert = any(alerts) or anomaly_info is not None
            st.session_state.alert_active = has_alert

            # Dibujar anotaciones
            annotated = draw_detections(
                frame, detections, alerts,
                zone=zone_px, zone_active=True,
                draw_skeleton=show_skeleton,
                anomaly_info=anomaly_info
            )

            # Mostrar frame
            video_placeholder.image(
                frame_to_rgb(annotated),
                caption=f"Frame #{st.session_state.frame_count} · "
                        f"{len(detections)} persona(s) detectada(s)",
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
            else:
                cap_path = ""

            # ── Registrar eventos en BD ───────────────────────────────────
            for event in events:
                insert_event(
                    tipo_evento=event["tipo_evento"],
                    estado     =event["estado"],
                    duracion   =event["duracion"],
                    captura    =cap_path,
                    detalle    =event.get("detalle", ""),
                )
            
            # Registrar evento de anomalía si se detecta
            if anomaly_info is not None:
                insert_event(
                    tipo_evento="ANOMALIA_DETECTADA",
                    estado="Alerta Global",
                    duracion=0,
                    captura=cap_path,
                    detalle=f"Se detectó posible crimen: {anomaly_info['class']} (Conf: {anomaly_info['conf']:.0%})",
                )

            # ── Panel derecho: estado en vivo ─────────────────────────────
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

                st.markdown(f"""
                | Parámetro | Valor |
                |-----------|-------|
                | 👤 Personas detectadas | **{len(detections)}** |
                | 🚨 Alertas activas | **{sum(alerts)}** |
                | 🎞 Frames procesados | **{st.session_state.frame_count}** |
                | 📁 Capturas guardadas | **{len(list(Path('captures').glob('*.jpg')))}** |
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

            # Pequeña pausa para no saturar CPU
            time.sleep(0.03)

    finally:
        cap.release()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        st.session_state.running = False


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD – Historial, gráficos y capturas
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Dashboard de Análisis")

tab1, tab2, tab3, tab4 = st.tabs([
    "Historial de Eventos",
    "Graficos",
    "Capturas",
    "Acerca de",
])


# ── Tab 1: Historial ──────────────────────────────────────────────────────
with tab1:
    all_events = get_all_events(limit=200)

    if not all_events:
        st.info("No hay eventos registrados aún. Inicia el sistema para comenzar.")
    else:
        df = pd.DataFrame(all_events)

        # Filtros
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            estados = ["Todos"] + df["estado"].unique().tolist()
            filtro_estado = st.selectbox("Filtrar por estado", estados)
        with col_f2:
            tipos = ["Todos"] + df["tipo_evento"].unique().tolist()
            filtro_tipo = st.selectbox("Filtrar por tipo", tipos)
        with col_f3:
            fechas = ["Todas"] + df["fecha"].unique().tolist()
            filtro_fecha = st.selectbox("Filtrar por fecha", fechas)

        df_filtrado = df.copy()
        if filtro_estado != "Todos":
            df_filtrado = df_filtrado[df_filtrado["estado"] == filtro_estado]
        if filtro_tipo != "Todos":
            df_filtrado = df_filtrado[df_filtrado["tipo_evento"] == filtro_tipo]
        if filtro_fecha != "Todas":
            df_filtrado = df_filtrado[df_filtrado["fecha"] == filtro_fecha]

        # Formatear columnas
        df_display = df_filtrado[["fecha", "hora", "tipo_evento", "estado", "duracion", "detalle"]].copy()
        df_display["estado"] = df_display["estado"].apply(
            lambda s: f"{'Sospechoso' if s == 'sospechoso' else 'Normal'}"
        )
        df_display["duracion"] = df_display["duracion"].apply(
            lambda d: format_duration(d)
        )
        df_display.columns = ["Fecha", "Hora", "Tipo de Evento", "Estado", "Duración", "Detalle"]

        st.dataframe(
            df_display,
            height=400,
        )
        st.caption(f"Mostrando {len(df_filtrado)} de {len(all_events)} eventos")

        # Exportar CSV
        csv = df_filtrado.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Exportar CSV",
            csv,
            "servsecurity_eventos.csv",
            "text/csv",
        )


# ── Tab 2: Gráficos ───────────────────────────────────────────────────────
with tab2:
    all_events_graph = get_all_events(limit=500)

    if not all_events_graph:
        st.info("Sin datos suficientes para mostrar gráficos.")
    else:
        df_g = pd.DataFrame(all_events_graph)

        gcol1, gcol2 = st.columns(2)

        with gcol1:
            # Gráfico de dona: eventos por tipo
            counts = get_event_counts_by_type()
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                hole=0.5,
                marker_colors=["#ef4444", "#f59e0b", "#a855f7", "#10b981"],
                textfont_size=12,
            )])
            fig_pie.update_layout(
                title="Distribución de Eventos por Tipo",
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font_color="#f1f5f9",
                legend=dict(bgcolor="#111827"),
                margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_pie)

        with gcol2:
            # Gráfico de barras: eventos por estado
            estado_counts = df_g["estado"].value_counts().reset_index()
            estado_counts.columns = ["Estado", "Cantidad"]
            color_map = {"sospechoso": "#ef4444", "normal": "#10b981"}
            fig_bar = px.bar(
                estado_counts,
                x="Estado",
                y="Cantidad",
                color="Estado",
                color_discrete_map=color_map,
                title="Eventos por Estado",
                template="plotly_dark",
            )
            fig_bar.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font_color="#f1f5f9",
                showlegend=False,
                margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_bar)

        # Línea temporal de alertas
        df_alerts = df_g[df_g["estado"] == "sospechoso"].copy()
        if not df_alerts.empty:
            df_alerts["datetime"] = pd.to_datetime(
                df_alerts["fecha"] + " " + df_alerts["hora"]
            )
            df_alerts_by_hour = (
                df_alerts.groupby(df_alerts["datetime"].dt.floor("h"))
                .size()
                .reset_index(name="Alertas")
            )
            df_alerts_by_hour.columns = ["Hora", "Alertas"]
            fig_line = px.line(
                df_alerts_by_hour,
                x="Hora",
                y="Alertas",
                title="Alertas a lo largo del tiempo",
                markers=True,
                template="plotly_dark",
                color_discrete_sequence=["#ef4444"],
            )
            fig_line.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font_color="#f1f5f9",
                margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_line)


# ── Tab 3: Capturas ───────────────────────────────────────────────────────
with tab3:
    captures_path = Path("captures")
    captures = sorted(captures_path.glob("*.jpg"), reverse=True) if captures_path.exists() else []

    if not captures:
        st.info("No hay capturas guardadas aún.")
    else:
        st.markdown(f"**{len(captures)} capturas** almacenadas en `/captures`")
        num_cols = 3
        rows = [captures[i:i+num_cols] for i in range(0, min(len(captures), 12), num_cols)]
        for row in rows:
            cols = st.columns(num_cols)
            for col, cap_file in zip(cols, row):
                with col:
                    img = cv2.imread(str(cap_file))
                    if img is not None:
                        st.image(
                            frame_to_rgb(img),
                            caption=cap_file.name,
                        )
                    else:
                        st.warning(f"No se pudo cargar {cap_file.name}")


# ── Tab 4: Acerca de ──────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    ## 🔐 SERVSecurity v1.0

    **Sistema Inteligente de Control de Acceso Físico basado en Análisis de Video**

    ---

    ### 🎯 Objetivo
    Demo académica de un sistema de vigilancia inteligente para entornos de servidores,
    que detecta personas y anomalías de comportamiento en tiempo real usando visión por computadora.

    ---

    ### 🧠 Tecnologías utilizadas

    | Componente | Tecnología |
    |-----------|-----------|
    | Detección de personas | YOLOv8n (Ultralytics) |
    | Motor de reglas | Python puro (reglas de comportamiento) |
    | Base de datos | SQLite3 |
    | Dashboard | Streamlit + Plotly |
    | Procesamiento de video | OpenCV |

    ---

    ### 📋 Reglas de anomalía implementadas

    | # | Regla | Severidad |
    |---|-------|-----------|
    | 1 | Persona dentro de la zona restringida | 🔴 Alto |
    | 2 | Permanencia mayor a X segundos en zona | 🔴 Alto |
    | 3 | Ingreso fuera del horario permitido | 🔴 Alto |
    | 4 | Presencia detectada (sin anomalía) | 🟢 Normal |

    ---

    ### 📁 Estructura del proyecto

    ```
    SERVSecurity/
    ├── app.py          # Dashboard Streamlit
    ├── detector.py     # Detección YOLOv8
    ├── rules.py        # Motor de reglas
    ├── database.py     # SQLite
    ├── utils.py        # Utilidades
    ├── requirements.txt
    ├── README.md
    ├── captures/       # Capturas de eventos
    └── data/           # Base de datos
    ```

    ---

    ### 👨‍💻 Autores
    Proyecto académico — CristophSyr · GitHub: [SERVSecurity](https://github.com/CristophSyr/SERVSecurity)
    """)
