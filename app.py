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
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av

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
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {
    --bg-primary:   #0a0e1a;
    --bg-card:      #111827;
    --bg-card2:     #1a2235;
    --accent-blue:  #3b82f6;
    --accent-cyan:  #06b6d4;
    --accent-green: #10b981;
    --accent-red:   #ef4444;
    --accent-amber: #f59e0b;
    --text-primary: #f1f5f9;
    --text-muted:   #64748b;
    --border:       #1e293b;
    --glow-blue:    rgba(59,130,246,0.3);
    --glow-red:     rgba(239,68,68,0.3);
  }

  html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4) !important;
  }
  [data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.78rem !important; }
  [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 700 !important; }

  /* Headers */
  h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 20px var(--glow-blue) !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 30px var(--glow-blue) !important;
  }

  /* Dataframe */
  [data-testid="stDataFrame"] { border-radius: 12px !important; }

  /* Alert box */
  .alert-box {
    background: linear-gradient(135deg, #1f0a0a, #2a1010);
    border: 1px solid var(--accent-red);
    border-left: 4px solid var(--accent-red);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    animation: pulse-red 2s ease-in-out infinite;
    box-shadow: 0 0 20px var(--glow-red);
  }
  @keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 20px var(--glow-red); }
    50%       { box-shadow: 0 0 40px var(--glow-red); }
  }

  .normal-box {
    background: linear-gradient(135deg, #0a1f14, #0f2a1c);
    border: 1px solid var(--accent-green);
    border-left: 4px solid var(--accent-green);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin: 0.3rem 0;
  }

  /* Header banner */
  .header-banner {
    background: linear-gradient(135deg, #0d1117 0%, #111827 50%, #0d1117 100%);
    border: 1px solid var(--border);
    border-bottom: 2px solid var(--accent-blue);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
  }

  /* Status badge */
  .badge-active {
    background: var(--accent-green);
    color: #000;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
  }
  .badge-idle {
    background: var(--text-muted);
    color: #fff;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
  }

  /* Divider */
  hr { border-color: var(--border) !important; }

  /* Video frame */
  .video-container {
    border: 2px solid var(--accent-blue);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 0 30px var(--glow-blue);
  }

  /* Scrollable log */
  .log-container {
    max-height: 300px;
    overflow-y: auto;
    padding-right: 8px;
  }
  .log-container::-webkit-scrollbar { width: 6px; }
  .log-container::-webkit-scrollbar-track { background: var(--bg-card); }
  .log-container::-webkit-scrollbar-thumb { background: var(--accent-blue); border-radius: 3px; }
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

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def make_webrtc_processor(
    confidence,
    zone_percent,
    max_perm,
    hour_start,
    hour_end,
    use_overlap,
    capture_interval,
):
    class SERVSecurityWebRTCProcessor(VideoProcessorBase):
        def __init__(self):
            self.detector = PersonDetector(
                model_name="yolov8n.pt",
                confidence=confidence,
            )
            self.rules_engine = RulesEngine(
                zone=(0, 0, 100, 100),
                max_permanence_sec=max_perm,
                allowed_start=hour_start,
                allowed_end=hour_end,
                use_overlap=use_overlap,
            )
            self.last_capture_time = 0.0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = resize_frame(img, max_width=800)

            h, w = img.shape[:2]

            zone_px = normalize_zone(zone_percent, w, h)
            self.rules_engine.update_zone(zone_px)

            detections = self.detector.detect(img)
            alerts, events = self.rules_engine.evaluate(detections)

            has_alert = any(alerts)

            annotated = draw_detections(
                img,
                detections,
                alerts,
                zone=zone_px,
                zone_active=True,
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

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    return SERVSecurityWebRTCProcessor

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR – Configuración
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom:1.5rem;'>
      <div style='font-size:2.5rem;'>🔐</div>
      <div style='font-size:1.3rem; font-weight:900; letter-spacing:1px; 
                  background: linear-gradient(90deg,#3b82f6,#06b6d4);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        SERVSecurity
      </div>
      <div style='font-size:0.7rem; color:#64748b; margin-top:2px;'>
        v1.0 · Control de Acceso IA
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Fuente de video ────────────────────────────────────────────────────
    st.markdown("#### 📹 Fuente de video")
    video_source = st.radio(
        "Selecciona fuente",
        ["🎥 Webcam en vivo", "📁 Subir video"],
        label_visibility="collapsed",
    )

    uploaded_video = None
    if "Subir" in video_source:
        uploaded_video = st.file_uploader(
            "Sube un archivo de video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Formatos soportados: MP4, AVI, MOV, MKV",
        )

    st.markdown("---")

    # ── Zona restringida ───────────────────────────────────────────────────
    st.markdown("#### 🔴 Zona restringida")
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
    st.markdown("#### ⚙️ Parámetros de detección")

    confidence = st.slider(
        "Confianza mínima (YOLO)", 0.20, 0.90, 0.45, 0.05,
        help="Umbral de confianza para aceptar una detección de persona",
    )
    max_perm = st.slider(
        "Permanencia máx. (seg)", 2, 30, 5,
        help="Segundos permitidos dentro de la zona antes de generar alerta",
    )

    st.markdown("##### ⏰ Horario permitido")
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

    st.markdown("---")

    # ── Controles ──────────────────────────────────────────────────────────
    st.markdown("#### 🎛 Controles")

    capture_interval = st.slider(
        "Intervalo de captura (seg)", 1, 30, 5,
        help="Cada cuántos segundos guardar una imagen en /captures",
    )

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        btn_start = st.button("▶ Iniciar", use_container_width=True)
    with col_c2:
        btn_stop  = st.button("⏹ Detener", use_container_width=True)

    if st.button("🗑 Limpiar eventos", use_container_width=True):
        clear_all_events()
        st.session_state.events_session = []
        st.success("Eventos eliminados.")

    st.markdown("---")
    st.caption("🎓 Demo académica · YOLOv8 · SQLite")
    st.caption("github.com/CristophSyr/SERVSecurity")


# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
status_badge = (
    '<span class="badge-active">● ACTIVO</span>'
    if st.session_state.running
    else '<span class="badge-idle">○ INACTIVO</span>'
)
st.markdown(f"""
<div class="header-banner">
  <span style="font-size:2.8rem;">🔐</span>
  <div>
    <div style="font-size:1.6rem; font-weight:900; letter-spacing:0.5px;
                background:linear-gradient(90deg,#3b82f6,#06b6d4);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
      SERVSecurity
    </div>
    <div style="color:#64748b; font-size:0.85rem; margin-top:2px;">
      Sistema Inteligente de Control de Acceso Físico &nbsp;·&nbsp;
      Análisis de Video en Tiempo Real &nbsp;·&nbsp;
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

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🚨 Total Alertas",    total_alerts,
              delta=None if total_alerts == 0 else f"+{total_alerts}")
with col2:
    st.metric("📋 Eventos Totales",  total_events)
with col3:
    st.metric("✅ Estado Normal",    total_normal)
with col4:
    st.metric("🎯 Frames analizados", st.session_state.frame_count)

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL: VIDEO | PANEL DERECHO
# ════════════════════════════════════════════════════════════════════════════
vid_col, info_col = st.columns([3, 2], gap="large")


# ── Columna izquierda: video ──────────────────────────────────────────────
with vid_col:
    st.markdown("### 📡 Feed de Video")

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
        use_column_width=True,
        )


# ── Columna derecha: info en tiempo real ─────────────────────────────────
with info_col:
    st.markdown("### 📊 Estado en tiempo real")
    live_status = st.empty()
    live_events = st.empty()


# ════════════════════════════════════════════════════════════════════════════
# ACCIONES DE CONTROL
# ════════════════════════════════════════════════════════════════════════════
if btn_start and not st.session_state.running:
    try:
        if "Subir" in video_source:
            with st.spinner("Cargando modelo YOLOv8…"):
                st.session_state.detector = PersonDetector(
                    model_name="yolov8n.pt",
                    confidence=confidence,
                )

            st.session_state.rules_engine = RulesEngine(
                zone=(0, 0, 100, 100),
                max_permanence_sec=max_perm,
                allowed_start=hour_start,
                allowed_end=hour_end,
                use_overlap=use_overlap,
            )
        else:
            # En Streamlit Cloud la webcam se maneja con WebRTC,
            # no con cv2.VideoCapture(0).
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
    if "Webcam" in video_source:
        with vid_col:
            st.info(
                "La cámara se abrirá desde tu navegador. "
                "Acepta el permiso de cámara cuando aparezca."
            )

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
                    capture_interval=capture_interval,
                ),
                media_stream_constraints={
                    "video": True,
                    "audio": False,
                },
                async_processing=True,
            )

        with info_col:
            st.markdown("### 📹 Modo Webcam WebRTC")
            st.write("La detección se procesa directamente desde el video del navegador.")
            st.write("Si el navegador pide permisos, selecciona **Permitir cámara**.")

        st.stop()

    detector     = st.session_state.detector
    rules_engine = st.session_state.rules_engine

    # ── Abrir fuente de video subida ──────────────────────────────────────
    cap = None
    tmp_path = None

    if uploaded_video is not None:
        # Guardar temporalmente el video subido
        suffix = Path(uploaded_video.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            st.error("❌ No se pudo abrir el video. Intenta con otro archivo.")
            st.session_state.running = False
            st.stop()
    else:
        st.warning("⚠️ Selecciona una fuente de video y presiona Iniciar.")
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

            # Detección de personas
            detections = detector.detect(frame)

            # Evaluación de reglas
            alerts, events = rules_engine.evaluate(detections)
            has_alert = any(alerts)
            st.session_state.alert_active = has_alert

            # Dibujar anotaciones
            annotated = draw_detections(
                frame, detections, alerts,
                zone=zone_px, zone_active=True,
            )

            # Mostrar frame
            video_placeholder.image(
                frame_to_rgb(annotated),
                use_column_width=True,
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

            # ── Panel derecho: estado en vivo ─────────────────────────────
            with live_status.container():
                if has_alert:
                    alert_placeholder.markdown("""
                    <div class="alert-box">
                      <b>⚠️ ALERTA ACTIVA</b> – Persona detectada en zona restringida
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    alert_placeholder.markdown("""
                    <div class="normal-box">
                      <b>✅ Estado Normal</b> – Sin anomalías detectadas
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
                st.markdown("**📋 Últimos eventos**")
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
st.markdown("## 📊 Dashboard de Análisis")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Historial de Eventos",
    "📈 Gráficos",
    "🖼 Capturas",
    "ℹ️ Acerca de",
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
            lambda s: f"{'🔴 Sospechoso' if s == 'sospechoso' else '🟢 Normal'}"
        )
        df_display["duracion"] = df_display["duracion"].apply(
            lambda d: format_duration(d)
        )
        df_display.columns = ["Fecha", "Hora", "Tipo de Evento", "Estado", "Duración", "Detalle"]

        st.dataframe(
            df_display,
            use_column_width=True,
            height=400,
        )
        st.caption(f"Mostrando {len(df_filtrado)} de {len(all_events)} eventos")

        # Exportar CSV
        csv = df_filtrado.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Exportar CSV",
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
            st.plotly_chart(fig_pie, use_column_width=True)

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
            st.plotly_chart(fig_bar, use_column_width=True)

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
            st.plotly_chart(fig_line, use_column_width=True)


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
                            use_column_width=True,
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
