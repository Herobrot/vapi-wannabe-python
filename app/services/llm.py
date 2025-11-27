import ollama
from typing import List, Optional
from fastapi import HTTPException
from app.config import config
from app.database import db
# Importamos AMBOS prompts por si necesitamos valores por defecto
from app.prompts import HEALTH_SYSTEM_PROMPT, PROACTIVE_NUDGE_PROMPT, extract_options_from_text

class LLMService:
    """Servicio de Lenguaje Local con Gestión de Contexto y Persistencia"""
    
    @staticmethod
    def process_user_interaction(
        session_id: str, 
        user_text: str, 
        temperature: float = 0.7,
        # CAMBIO 1: Agregamos este argumento opcional.
        # Si no se envía nada, usa el de salud por defecto (para no romper código viejo).
        system_prompt: str = HEALTH_SYSTEM_PROMPT 
    ) -> str:
        """
        Flujo estándar reactivo (Usuario habla -> IA responde)
        Soporta personalidades dinámicas (VitalBot o ExaBot).
        """
        try:
            # 1. Guardar mensaje del usuario en BD
            db.add_message(session_id=session_id, role="user", content=user_text)
            
            # 2. Recuperar contexto
            history = db.get_recent_context(session_id, limit=5)
            
            # 3. Construir payload
            # CAMBIO 2: Usamos la variable 'system_prompt' en lugar de la constante fija
            messages_payload = [
                {"role": "system", "content": system_prompt}
            ] + history
            
            # 4. Llamada a Ollama
            response_raw = ollama.chat(
                model=config.LLM_MODEL,
                messages=messages_payload,
                stream=False,
                options={'temperature': temperature, 'num_predict': 400}
            )
            
            assistant_text = response_raw['message']['content']
            
            # 5. Extraer y guardar (Lógica común para ambos bots)
            detected_options = extract_options_from_text(assistant_text)
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

    # ==========================================
    # NUEVO MÉTODO: Para las inyecciones del Timer
    # ==========================================
    @staticmethod
    def process_injection(session_id: str, system_msg: str, system_prompt: str) -> str:
        """
        Procesa un mensaje automático del sistema (ej: Timer de examen)
        haciéndolo pasar por contexto para que el LLM reaccione.
        """
        try:
            # 1. Recuperamos historial para que el bot sepa qué estaba preguntando
            history = db.get_recent_context(session_id, limit=5)
            
            # 2. Construimos el payload
            messages_payload = [
                {"role": "system", "content": system_prompt} # Personalidad ExaBot
            ] + history
            
            # 3. Inyectamos el aviso del timer como si fuera un mensaje de 'user'
            # Esto fuerza al LLM a responder al aviso (ej: "Quedan 30 seg").
            messages_payload.append({
                "role": "user", 
                "content": system_msg
            })
            
            # 4. Llamada a Ollama
            response = ollama.chat(
                model=config.LLM_MODEL,
                messages=messages_payload,
                options={'temperature': 0.7, 'num_predict': 150}
            )
            
            assistant_text = response['message']['content']
            
            # 5. Guardamos la respuesta del asistente en BD para mantener el hilo
            # (Opcional: No guardamos el mensaje del timer en BD para no ensuciar el historial visual)
            db.add_message(session_id, "assistant", assistant_text)
            
            return assistant_text
            
        except Exception as e:
            print(f"❌ Error en Injection: {e}")
            return "¡Vamos, tú puedes!" # Fallback de emergencia

    @staticmethod
    def generate_proactive_followup(session_id: str) -> str:
        """
        (Mantenemos este método igual para VitalBot)
        Genera un mensaje proactivo cuando el temporizador expira.
        """
        try:
            history = db.get_recent_context(session_id, limit=5)
            if not history:
                return "Hola, estoy aquí si necesitas ayuda para empezar."

            # Usamos trigger message para VitalBot
            trigger_message = {
                "role": "user", 
                "content": "(Sistema: El usuario ha guardado silencio por 45 segundos...)"
            }

            messages_payload = [
                {"role": "system", "content": PROACTIVE_NUDGE_PROMPT}
            ] + history + [trigger_message]
            
            response_raw = ollama.chat(
                model=config.LLM_MODEL,
                messages=messages_payload,
                options={'temperature': 0.8, 'num_predict': 60}
            )
            
            proactive_text = response_raw['message']['content']
            
            if not proactive_text or not proactive_text.strip():
                # Reintento simple
                rescue_payload = [{"role": "user", "content": "Genera una pregunta corta."}]
                retry = ollama.chat(model=config.LLM_MODEL, messages=rescue_payload)
                proactive_text = retry['message']['content']

            db.add_message(session_id, "assistant", proactive_text)
            return proactive_text

        except Exception as e:
            print(f"❌ Error nudge: {e}")
            return "¿Hola? ¿Sigues ahí?"