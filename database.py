"""
database.py – Gestión de la base de datos SQLite para SERVSecurity.
Registra todos los eventos de acceso y anomalías detectadas.
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path
import threading

# Ruta a la base de datos
DB_PATH = Path("data") / "servsecurity.db"

# Conexión persistente (singleton) — evita abrir/cerrar en cada operación
_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    """Retorna la conexión persistente a la BD, creándola si no existe."""
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        # WAL mode para mejor rendimiento en escrituras concurrentes
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
    return _conn


def init_db():
    """Inicializa la base de datos y crea las tablas si no existen."""
    conn = _get_conn()
    with _lock:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha       TEXT    NOT NULL,
                hora        TEXT    NOT NULL,
                tipo_evento TEXT    NOT NULL,
                estado      TEXT    NOT NULL DEFAULT 'normal',
                duracion    REAL    DEFAULT 0.0,
                captura     TEXT    DEFAULT '',
                detalle     TEXT    DEFAULT '',
                incident_id TEXT    DEFAULT '',
                created_at  TEXT    DEFAULT (datetime('now'))
            )
        """)

        # Intentar añadir la columna por si la BD ya existe (migración simple)
        try:
            cursor.execute("ALTER TABLE events ADD COLUMN incident_id TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass

        # Índice para acelerar consultas por fecha y estado
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_fecha
            ON events (fecha)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_estado
            ON events (estado)
        """)

        conn.commit()


def insert_event(
    tipo_evento: str,
    estado: str = "normal",
    duracion: float = 0.0,
    captura: str = "",
    detalle: str = "",
    incident_id: str = "",
) -> int:
    """
    Inserta un nuevo evento en la base de datos.

    Args:
        tipo_evento: Categoría del evento (ej. 'Persona en zona restringida').
        estado: 'normal' o 'sospechoso'.
        duracion: Tiempo en segundos que duró el evento.
        captura: Ruta relativa al archivo de captura de imagen.
        detalle: Información adicional.
        incident_id: ID que agrupa varios eventos relacionados.

    Returns:
        ID del evento insertado.
    """
    now = datetime.now()
    fecha = now.strftime("%Y-%m-%d")
    hora = now.strftime("%H:%M:%S")

    conn = _get_conn()
    with _lock:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO events (fecha, hora, tipo_evento, estado, duracion, captura, detalle, incident_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (fecha, hora, tipo_evento, estado, duracion, captura, detalle, incident_id))
        event_id = cursor.lastrowid
        conn.commit()
    return event_id


def get_all_events(limit: int = 200) -> list[dict]:
    """
    Recupera los eventos más recientes.

    Args:
        limit: Número máximo de eventos a retornar.

    Returns:
        Lista de diccionarios con los datos de cada evento.
    """
    conn = _get_conn()
    with _lock:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, fecha, hora, tipo_evento, estado, duracion, captura, detalle, incident_id
            FROM events
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
    return rows


def get_event_counts_by_type() -> dict:
    """
    Cuenta eventos agrupados por tipo_evento.

    Returns:
        Diccionario {tipo_evento: cantidad}.
    """
    conn = _get_conn()
    with _lock:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tipo_evento, COUNT(*) as total
            FROM events
            GROUP BY tipo_evento
            ORDER BY total DESC
        """)
        result = {row[0]: row[1] for row in cursor.fetchall()}
    return result


def get_alert_count() -> int:
    """Retorna el número total de eventos con estado 'sospechoso'."""
    conn = _get_conn()
    with _lock:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events WHERE estado = 'sospechoso'")
        count = cursor.fetchone()[0]
    return count


def get_events_today() -> list[dict]:
    """Retorna todos los eventos del día actual."""
    today = datetime.now().strftime("%Y-%m-%d")
    conn = _get_conn()
    with _lock:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, fecha, hora, tipo_evento, estado, duracion, captura, detalle, incident_id
            FROM events
            WHERE fecha = ?
            ORDER BY id DESC
        """, (today,))
        rows = [dict(r) for r in cursor.fetchall()]
    return rows


def clear_all_events():
    """Elimina todos los eventos de la base de datos (útil para demos)."""
    conn = _get_conn()
    with _lock:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM events")
        conn.commit()
