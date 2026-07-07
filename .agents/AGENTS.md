# Contexto del Proyecto: SERVSecurity

**SERVSecurity** es un sistema inteligente de control de acceso físico basado en análisis de video para mejorar la seguridad de activos TI. Desarrollado como proyecto de Tesis (UPAO).

## Entornos y Despliegue
- **Local:** Windows con GPU (opcional). Permite procesos pesados a buenos FPS.
- **Nube (Live):** Hugging Face Spaces (Ubuntu Linux). 
  - *Restricción crítica:* Solo cuenta con 2 vCPUs compartidas y muy poca RAM (~16GB compartida, suele haber OOM si no se cuida).
  - *Consecuencia:* `PROCESS_EVERY_N` debe ser alto (ej. 8) en la nube.
  - *Modelo Facial:* DeepFace en la nube debe correr con `detector_backend="skip"` y `max_workers=1` para no agotar la RAM ni fallar con los crops pequeños de YOLO.

## Arquitectura de Código
- **`app.py`**: Interfaz principal en Streamlit. Maneja el bucle de video. 
  - *Regla de UI:* Streamlit se bloquea si el bucle de video es infinito sin ceder I/O. Siempre incluir `time.sleep(0.01)` al final del loop.
  - *Regla de UI 2:* Evitar llamar a `st.image()` o renderizar el panel lateral en TODOS los frames. Usar variables como `UPDATE_PANEL_EVERY_N`.
- **`detector.py`**: Detección de personas usando `ultralytics/YOLOv8n-pose`. 
  - *Regla:* Extraemos los keypoints y el Bounding Box, y recortamos (crop) la cabeza para enviarla a `facial_auth.py`.
  - *Modelo 2:* También contiene el modelo de clasificación de anomalías (Cerebro 2), el cual usa frame-skipping para no quemar la CPU.
- **`facial_auth.py`**: Usa DeepFace y Facenet. 
  - *Regla:* Maneja la autenticación asíncrona mediante hilos para no trabar el stream de video.
- **`rules.py`**: Motor de reglas lógicas.
  - *Regla:* Compara centroide vs. zona restringida. Evalúa eventos de intrusión, permanencia, fuera de horario y anomalías (velocidad alta / postura inusual).

## Reglas Generales al Modificar Código
- **Preferir git CLI:** El usuario prefiere usar comandos `git` nativos (`git add`, `git commit`, `git push`) en lugar de wrappers de IDE.
- **Git LFS:** Los modelos pesados (`.pt`, `.pkl`) y bases de datos deben estar trackeados en Git LFS para poder hacer push a Hugging Face sin ser rechazados por el límite de 10MB.
- **Codificación UTF-8:** En Windows, forzar UTF-8 en `sys.stdout` y `sys.stderr` para evitar problemas con emojis o caracteres especiales de los prints.
- **Comentarios:** Mantener los comentarios en el código que explican el "por qué" de las decisiones (como por qué usamos frame-skipping o locks).
