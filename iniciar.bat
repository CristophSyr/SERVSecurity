@echo off
echo ===========================================
echo   Iniciando SERVSecurity (Control de Acceso)
echo ===========================================
echo.

REM Verifica si Python esta instalado en la maquina destino
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] No se encontro Python en este equipo.
    echo Por favor, instala Python ^(version 3.10 o superior^) desde python.org.
    echo MUY IMPORTANTE: Asegurate de marcar la casilla "Add Python to PATH" durante la instalacion.
    echo.
    pause
    exit /b
)

echo Instalando/Actualizando dependencias (puede tardar un poco la primera vez)...
pip install -r requirements.txt

echo.
echo Arrancando el sistema de vigilancia...
streamlit run app.py
pause
