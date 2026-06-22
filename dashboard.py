import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
from pathlib import Path

from database import get_all_events, get_event_counts_by_type
from utils import frame_to_rgb, format_duration, get_status_emoji


def render_dashboard():
    """Renderiza la sección completa del dashboard con sus 4 tabs."""
    st.markdown("---")
    st.markdown("## Dashboard de Análisis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Historial de Eventos",
        "Graficos",
        "Capturas",
        "Acerca de",
    ])

    with tab1:
        _render_history()
    with tab2:
        _render_charts()
    with tab3:
        _render_captures()
    with tab4:
        _render_about()


def _render_history():
    """Tab 1: Tabla de eventos con filtros y exportación CSV."""
    all_events = get_all_events(limit=200)

    if not all_events:
        st.info("No hay eventos registrados aún. Inicia el sistema para comenzar.")
        return

    df = pd.DataFrame(all_events)

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

    df_display = df_filtrado[["fecha", "hora", "tipo_evento", "estado", "duracion", "detalle"]].copy()
    df_display["estado"] = df_display["estado"].apply(
        lambda s: "Sospechoso" if s == "sospechoso" else "Normal"
    )
    df_display["duracion"] = df_display["duracion"].apply(format_duration)
    df_display.columns = ["Fecha", "Hora", "Tipo de Evento", "Estado", "Duración", "Detalle"]

    st.dataframe(df_display, height=400)
    st.caption(f"Mostrando {len(df_filtrado)} de {len(all_events)} eventos")

    csv = df_filtrado.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Exportar CSV",
        csv,
        "servsecurity_eventos.csv",
        "text/csv",
    )


def _render_charts():
    """Tab 2: Gráficos de dona, barras y línea temporal."""
    all_events_graph = get_all_events(limit=500)

    if not all_events_graph:
        st.info("Sin datos suficientes para mostrar gráficos.")
        return

    df_g = pd.DataFrame(all_events_graph)

    gcol1, gcol2 = st.columns(2)

    with gcol1:
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


def _render_captures():
    """Tab 3: Galería de capturas guardadas."""
    captures_path = Path("captures")
    captures = sorted(captures_path.glob("*.jpg"), reverse=True) if captures_path.exists() else []

    if not captures:
        st.info("No hay capturas guardadas aún.")
        return

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


def _render_about():
    """Tab 4: Información del proyecto."""
    st.markdown("""
    ## 🔐 SERVSecurity v1.0

    **Sistema Inteligente de Control de Acceso Físico basado en Análisis de Video con IA**

    ---

    ### 🎯 Objetivo
    Sistema de videovigilancia inteligente para entornos de servidores que detecta personas,
    analiza comportamientos anómalos y autentica identidades mediante reconocimiento facial
    en tiempo real usando visión por computadora.

    ---

    ### 🧠 Tecnologías utilizadas

    | Componente | Tecnología |
    |---|---|
    | Detección de personas y postura | YOLOv8n-Pose (Ultralytics) |
    | Clasificación de anomalías | YOLOv8n-cls (Transfer Learning) |
    | Reconocimiento facial | DeepFace (Facenet) |
    | Motor de reglas | Python (tracking espaciotemporal) |
    | Base de datos | SQLite3 |
    | Dashboard | Streamlit + Plotly |
    | Procesamiento de video | OpenCV |

    ---

    ### 📋 Reglas de anomalía implementadas

    | # | Regla | Severidad |
    |---|---|---|
    | 1 | Persona dentro de la zona restringida | 🔴 Alto |
    | 2 | Permanencia mayor a X segundos en zona | 🔴 Alto |
    | 3 | Ingreso fuera del horario permitido | 🔴 Alto |
    | 4 | Rostro no reconocido en base biométrica | 🔴 Alto |
    | 5 | Movimiento errático (carrera/forcejeo) | 🔴 Alto |
    | 6 | Postura inusual (caída/agachamiento) | 🔴 Alto |
    | 7 | Anomalía global (clasificador IA) | 🔴 Alto |
    | 8 | Presencia detectada (sin anomalía) | 🟢 Normal |

    ---

    ### 📁 Estructura del proyecto

    ```
    SERVSecurity/
    ├── app.py              # Aplicación principal (Streamlit)
    ├── detector.py         # Detección YOLOv8 + Modelo de anomalías
    ├── rules.py            # Motor de reglas espaciotemporal
    ├── database.py         # Gestión SQLite (WAL mode)
    ├── utils.py            # Utilidades (frames, zona, capturas)
    ├── facial_auth.py      # Autenticación biométrica (DeepFace)
    ├── styles.py           # CSS del dashboard
    ├── dashboard.py        # Visualización (historial, gráficos)
    ├── yolov8n-pose.pt     # Modelo de postura
    ├── authorized_faces/   # Fotos de personas autorizadas
    ├── captures/           # Capturas de eventos
    ├── data/               # Base de datos SQLite
    ├── runs/               # Modelo de anomalías (best.pt)
    └── training_scripts/   # Scripts de entrenamiento
    ```

    ---

    ### 👨‍💻 Autores
    Proyecto académico — CristophSyr · GitHub: [SERVSecurity](https://github.com/CristophSyr/SERVSecurity)
    """)
