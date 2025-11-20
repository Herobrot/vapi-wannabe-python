import sqlite3
import json
from typing import List, Dict, Optional

DB_NAME = "vapi_history.db"

class DatabaseManager:
    def __init__(self, db_path: str = DB_NAME):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Inicializa el esquema de la base de datos"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Tabla de historial conversacional
        # options_json: Almacena las opciones detectadas (A, B, C...)
        # selected_option: Para uso futuro, si el usuario elige una
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                options_json TEXT,
                selected_option TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Índices para mejorar velocidad de búsqueda por sesión
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_session_id 
            ON conversation_history(session_id)
        ''')
        
        conn.commit()
        conn.close()

    def add_message(self, session_id: str, role: str, content: str, options: Optional[List[str]] = None):
        """Guarda un mensaje en el historial"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        options_str = json.dumps(options) if options else None
        
        cursor.execute('''
            INSERT INTO conversation_history (session_id, role, content, options_json)
            VALUES (?, ?, ?, ?)
        ''', (session_id, role, content, options_str))
        
        conn.commit()
        conn.close()

    def get_recent_context(self, session_id: str, limit: int = 5) -> List[Dict]:
        """Recupera las últimas N interacciones para el contexto del LLM"""
        conn = self._get_connection()
        cursor = conn.cursor() # Habilita acceso por nombre de columna
        
        # Obtenemos los últimos N mensajes (orden descendente por tiempo)
        cursor.execute('''
            SELECT role, content 
            FROM conversation_history 
            WHERE session_id = ? 
            ORDER BY id DESC 
            LIMIT ?
        ''', (session_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Invertimos la lista para que esté en orden cronológico (antiguo -> nuevo)
        # Formato esperado por Ollama: {'role': '...', 'content': '...'}
        history = [{"role": r[0], "content": r[1]} for r in rows]
        return history[::-1]

# Instancia global
db = DatabaseManager()