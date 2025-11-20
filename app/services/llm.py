import ollama
from typing import List, Optional
from fastapi import HTTPException
from app.config import config
from app.database import db
from app.prompts import HEALTH_SYSTEM_PROMPT, extract_options_from_text

class LLMService:
    """Servicio de Lenguaje Local con Gestión de Contexto y Persistencia"""
    
    @staticmethod
    def process_user_interaction(session_id: str, user_text: str, temperature: float = 0.7) -> str:
        """
        Flujo completo: 
        1. Guarda input usuario 
        2. Recupera contexto 
        3. Genera respuesta 
        4. Parsea opciones 
        5. Guarda respuesta
        """
        try:
            # 1. Guardar mensaje del usuario en BD
            db.add_message(session_id=session_id, role="user", content=user_text)
            
            # 2. Recuperar contexto (últimos 5 mensajes)
            history = db.get_recent_context(session_id, limit=5)
            
            # 3. Construir payload para Ollama con System Prompt
            messages_payload = [
                {"role": "system", "content": HEALTH_SYSTEM_PROMPT}
            ] + history
            
            # 4. Llamada a Ollama
            response_raw = ollama.chat(
                model=config.LLM_MODEL,
                messages=messages_payload,
                stream=False, # Stream es básicamente la capacidad de recibir datos parciales; aquí no lo usamos
                options={
                    'temperature': temperature,
                    'num_predict': 400 # Limitar output para voz
                }
            )
            
            assistant_text = response_raw['message']['content']
            
            # 5. Extraer opciones estructuradas (si las hay)
            detected_options = extract_options_from_text(assistant_text)
            
            # 6. Guardar respuesta del asistente en BD con las opciones parseadas
            db.add_message(
                session_id=session_id, 
                role="assistant", 
                content=assistant_text,
                options=detected_options if detected_options else None
            )
            
            return assistant_text

        except Exception as e:
            print(f"❌ Error en LLM Service: {e}")
            raise HTTPException(status_code=500, detail=f"Error procesando interacción: {str(e)}")

    # Mantenemos el método antiguo por compatibilidad si se usa en otros endpoints, 
    # pero idealmente deberíamos migrar todo a process_user_interaction
    @staticmethod
    def chat_completion_legacy(messages: List[dict], temperature: float = 0.7):
        response = ollama.chat(model=config.LLM_MODEL, messages=messages)
        return response['message']['content']