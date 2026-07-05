---
title: ServSecurityHost
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.58.0
python_version: 3.12
app_file: app.py
pinned: false
---
# SERVSecurity

**Sistema Inteligente de Control de Acceso Físico basado en Análisis de Video con IA**

Sistema de videovigilancia en tiempo real que utiliza visión por computadora para detectar personas, analizar comportamientos anómalos y autenticar identidades mediante reconocimiento facial.

## Tecnologías

| Componente | Tecnología |
|---|---|
| Detección de personas y postura | YOLOv8n-Pose (Ultralytics) |
| Clasificación de anomalías | YOLOv8n-cls (Transfer Learning personalizado) |
| Reconocimiento facial | DeepFace (Facenet) |
| Motor de reglas | Python (tracking espaciotemporal) |
| Base de datos | SQLite3 |
| Dashboard | Streamlit + Plotly |
| Procesamiento de video | OpenCV |

## Requisitos

- Python 3.10+
- Webcam USB o cámara IP (para modo local)

## Instalación y ejecución

```bash
pip install -r requirements.txt
streamlit run app.py
```

O simplemente haz doble clic en `iniciar.bat` (Windows).

## Estructura del proyecto

```
SERVSecurity/
├── app.py              # Dashboard principal (Streamlit)
├── detector.py         # Detección YOLOv8 + Modelo de anomalías
├── rules.py            # Motor de reglas espaciotemporal
├── database.py         # Gestión SQLite
├── utils.py            # Utilidades (frames, zona, capturas)
├── facial_auth.py      # Autenticación biométrica con DeepFace
├── requirements.txt    # Dependencias Python
├── packages.txt        # Dependencias del sistema (Linux/HF Spaces)
├── iniciar.bat         # Script de arranque rápido (Windows)
├── yolov8n-pose.pt     # Modelo de detección de postura
├── authorized_faces/   # Fotos de personas autorizadas
├── captures/           # Capturas de eventos guardadas
├── data/               # Base de datos SQLite
├── runs/               # Modelo de anomalías entrenado (best.pt)
│   └── classify/servsecurity_anomaly_model/weights/
└── training_scripts/   # Scripts de entrenamiento de la IA
    ├── download_smart.py
    ├── prepare_dataset.py
    └── train_anomaly.py
```

## Modelo de anomalías

El archivo `runs/classify/servsecurity_anomaly_model/weights/best.pt` contiene un modelo de clasificación entrenado mediante Transfer Learning sobre YOLOv8n-cls, usando un dataset propio de ~100 GB. Este modelo clasifica escenas como "Normal" o "Anomalía" (peleas, caídas, etc.).

**Si este archivo no existe**, el sistema funciona con normalidad pero solo detecta personas y aplica reglas de zona/horario, sin clasificar anomalías de comportamiento global.

Para re-entrenarlo, consulta los scripts en `training_scripts/`.

## Reconocimiento facial

Para habilitar la autenticación biométrica:
1. Coloca fotos de las personas autorizadas en la carpeta `authorized_faces/` (formato JPG/PNG).
2. Activa la casilla "Reconocimiento Facial" en la barra lateral de la app.

El sistema recortará automáticamente el rostro detectado por YOLO y lo comparará con la base de datos usando DeepFace (modelo Facenet).

## Reglas de anomalía

| Regla | Descripción | Severidad |
|---|---|---|
| Intrusión en zona | Persona detectada dentro del área restringida | 🔴 Alto |
| Permanencia prolongada | Excede el tiempo máximo en zona | 🔴 Alto |
| Acceso fuera de horario | Ingreso fuera del rango horario configurado | 🔴 Alto |
| Rostro desconocido | Persona no reconocida por el sistema biométrico | 🔴 Alto |
| Movimiento errático | Velocidad de desplazamiento anormalmente alta | 🔴 Alto |
| Postura inusual | Proporción corporal indica caída o agachamiento | 🔴 Alto |
| Presencia normal | Persona detectada sin anomalías | 🟢 Normal |

## Autores

Proyecto académico — [CristophSyr/SERVSecurity](https://github.com/CristophSyr/SERVSecurity)
