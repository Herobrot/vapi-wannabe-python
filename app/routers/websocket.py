import json
import base64
import uuid
import asyncio
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from app.services.stt import STTService
from app.services.llm import LLMService
from app.services.tts import TTSService
from app.services.idle_monitor import IdleMonitor
from app.services.exam_timer import ExamTimer, TimerState
from app.config import config
from app.prompts import HEALTH_SYSTEM_PROMPT, EXABOT_SYSTEM_PROMPT

router = APIRouter()

@router.websocket("/ws/voice")
async def websocket_voice_endpoint(
    websocket: WebSocket, 
    client_id: Optional[str] = Query(None),
    bot_mode: str = Query("vitalbot")
):
    await websocket.accept()
    session_id = client_id if client_id else str(uuid.uuid4())
    
    current_system_prompt = EXABOT_SYSTEM_PROMPT if bot_mode == "exabot" else HEALTH_SYSTEM_PROMPT
    
    print(f"🔌 Cliente conectado ({bot_mode}). ID: {session_id}")
    await websocket.send_json({'type': 'status', 'message': f'Modo: {bot_mode.upper()}'})

    # ==========================================
    # 1. CALLBACKS
    # ==========================================

    async def send_exam_stats(timer: ExamTimer):
        stats = timer.get_stats()
        await websocket.send_json({'type': 'exam_update', 'data': stats})

    async def on_idle_timeout():
        """Callback VitalBot (Timeout 45s)"""
        try:
            await websocket.send_json({'type': 'status', 'message': '🤔 Pensando sugerencia...'})
            text_nudge = await asyncio.to_thread(LLMService.generate_proactive_followup, session_id)
            if not text_nudge or not text_nudge.strip(): return

            await websocket.send_json({'type': 'response', 'text': text_nudge})
            await websocket.send_json({'type': 'status', 'message': '🗣️ Generando voz...'})
            audio_bytes, mime = await asyncio.to_thread(TTSService.synthesize, text_nudge)
            b64 = base64.b64encode(audio_bytes).decode('ascii')
            await websocket.send_json({'type': 'audio', 'data': b64, 'format': mime})
        except Exception as e:
            print(f"❌ Error callback idle: {e}")

    async def on_exam_event(system_msg: str):
        """Callback ExaBot - CRÍTICO: Inyección como System Message"""
        try:
            print(f"📥 Timer Event: {system_msg}")
            
            # Pausar timer durante procesamiento
            if exam_timer:
                exam_timer.pause()
            
            # PARSEAR el mensaje del timer
            if "30s_elapsed" in system_msg:
                # Extraer datos reales del mensaje
                parts = system_msg.split("|")
                remaining = next((p.split("=")[1].strip() for p in parts if "remaining=" in p), "N/A")
                question = next((p.split("=")[1].strip() for p in parts if "question=" in p), "N/A")
                elapsed_q = next((p.split("=")[1].strip() for p in parts if "elapsed_q=" in p), "30s")
                
                # INSTRUCCIÓN EXPLÍCITA PARA EL LLM (separada del contexto conversacional)
                llm_instruction = (
                    f"[SYSTEM DIRECTIVE - NOT USER INPUT]\n"
                    f"TIMER ALERT: The student has NOT answered yet. "
                    f"They have spent {elapsed_q} on question {question}. "
                    f"Total time remaining: {remaining}.\n"
                    f"YOUR TASK: Gently remind them of the time and encourage them to answer. "
                    f"DO NOT evaluate any answer (there is none). "
                    f"Keep it brief (1-2 sentences)."
                )
                
                await websocket.send_json({'type': 'status', 'message': '⏰ Recordatorio de tiempo...'})
                
            elif "time_up" in system_msg:
                llm_instruction = (
                    f"[SYSTEM DIRECTIVE]\n"
                    f"EXAM ENDED: Time is up. "
                    f"Politely inform the student the exam has concluded and thank them."
                )
                await websocket.send_json({'type': 'status', 'message': '⏰ ¡Tiempo terminado!'})
                
            elif "INICIO EXAMEN" in system_msg:
                # Mensaje de bienvenida INICIAL (antes de iniciar timer real)
                llm_instruction = (
                    f"[SYSTEM DIRECTIVE]\n"
                    f"WELCOME MESSAGE: Greet the student warmly and present Question 1. "
                    f"Total questions: {config.EXAM_TOTAL_QUESTIONS}. "
                    f"Total time: {config.EXAM_TOTAL_TIME // 60} minutes. "
                    f"Keep it concise (2-3 sentences)."
                )
            else:
                llm_instruction = system_msg

            # Generar respuesta usando inyección de sistema
            response_text = await asyncio.to_thread(
                LLMService.process_injection,
                session_id,
                llm_instruction,
                current_system_prompt
            )
            
            await websocket.send_json({'type': 'response', 'text': response_text})
            
            # Sintetizar audio
            await websocket.send_json({'type': 'status', 'message': '🗣️ Generando audio...'})
            audio_bytes, mime = await asyncio.to_thread(TTSService.synthesize, response_text)
            b64 = base64.b64encode(audio_bytes).decode('ascii')
            await websocket.send_json({'type': 'audio', 'data': b64, 'format': mime})
            
        except Exception as e:
            print(f"❌ Error callback examen: {e}")
            if exam_timer: exam_timer.resume()

    # ==========================================
    # 2. INICIALIZACIÓN
    # ==========================================
    idle_monitor = None
    exam_timer = None
    welcome_pending = False  # Bandera para saber si esperamos el primer playback_complete

    if bot_mode == "exabot":
        exam_timer = ExamTimer(callback=on_exam_event)
        exam_timer.prepare_exam()  # Prepara PERO NO inicia conteo
        await send_exam_stats(exam_timer)
        welcome_pending = True
        
        # Enviar mensaje de bienvenida (el timer NO corre aún)
        asyncio.create_task(on_exam_event("[SYSTEM] INICIO EXAMEN: Saluda y lanza Pregunta 1."))
    else:
        idle_monitor = IdleMonitor(config.IDLE_TIMEOUT_SECONDS, on_idle_timeout)

    # ==========================================
    # 3. BUCLE PRINCIPAL
    # ==========================================
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # --- PLAYBACK COMPLETE ---
            if message['type'] == 'playback_complete':
                # CASO ESPECIAL: Primer playback_complete después del welcome
                if welcome_pending and exam_timer:
                    welcome_pending = False
                    exam_timer.start_counting()  # AHORA SÍ inicia el conteo real
                    await send_exam_stats(exam_timer)
                    print("✅ Welcome completado → Timer iniciado")
                    continue
                
                # Caso normal: reanudar después de audio
                if idle_monitor: 
                    idle_monitor.start()
                if exam_timer and exam_timer.state == TimerState.PAUSED: 
                    exam_timer.resume()
                    await send_exam_stats(exam_timer)
                continue

            # --- CLEAR CHAT ---
            if message['type'] == 'clear_chat':
                if idle_monitor: idle_monitor.cancel()
                continue
                
            # --- AUDIO RECIBIDO ---
            if message['type'] == 'audio':
                # Validar estado (no procesar si aún no terminó el welcome)
                if welcome_pending:
                    await websocket.send_json({
                        'type': 'error', 
                        'message': '⚠️ Espera a que termine la introducción'
                    })
                    continue
                
                if idle_monitor: idle_monitor.cancel()
                if exam_timer: exam_timer.pause()
                
                try:
                    # Transcribir
                    audio_data = base64.b64decode(message['data'])
                    await websocket.send_json({'type': 'status', 'message': '🎤 Transcribiendo...'})
                    transcription = await asyncio.to_thread(STTService.transcribe, audio_data)
                    
                    if not transcription or not transcription.strip():
                        await websocket.send_json({'type': 'status', 'message': '⚠️ No se detectó voz.'})
                        await websocket.send_json({'type': 'playback_complete'})
                        if exam_timer: exam_timer.resume()
                        continue

                    await websocket.send_json({'type': 'transcription', 'text': transcription})
                    
                    # PROCESAMIENTO
                    response_text = ""
                    
                    if bot_mode == "exabot":
                        stats = exam_timer.get_stats()
                        
                        # Enriquecer mensaje del usuario con metadata
                        enriched_user_msg = (
                            f"STUDENT ANSWER: '{transcription}'\n"
                            f"[METADATA: Question {stats['current_q']}/{stats['total_q']} | "
                            f"Time spent: {stats['elapsed_question']}s | "
                            f"Total remaining: {stats['remaining_total']}]\n"
                            f"Evaluate if correct and move to next question."
                        )
                        
                        await websocket.send_json({'type': 'status', 'message': '📝 Evaluando respuesta...'})
                        
                        response_text = await asyncio.to_thread(
                            LLMService.process_user_interaction, 
                            session_id,
                            enriched_user_msg,
                            0.7,
                            current_system_prompt
                        )
                        
                        # IMPORTANTE: Avanzar pregunta DESPUÉS de la respuesta del LLM
                        exam_timer.next_question()
                        await send_exam_stats(exam_timer)
                        
                    else:
                        response_text = await asyncio.to_thread(
                            LLMService.process_user_interaction,
                            session_id,
                            transcription
                        )
                    
                    await websocket.send_json({'type': 'response', 'text': response_text})
                    
                    await websocket.send_json({'type': 'status', 'message': '🗣️ Sintetizando...'})
                    audio_bytes, mime = await asyncio.to_thread(TTSService.synthesize, response_text)
                    b64 = base64.b64encode(audio_bytes).decode('ascii')
                    
                    await websocket.send_json({'type': 'audio', 'data': b64, 'format': mime})
                    
                except Exception as e:
                    print(f"❌ Error procesando audio: {str(e)}")
                    await websocket.send_json({'type': 'error', 'message': str(e)})
                    if exam_timer: exam_timer.resume()

    except WebSocketDisconnect:
        print("🔌 Cliente desconectado")
    finally:
        if exam_timer: exam_timer.stop()
        if idle_monitor: idle_monitor.cancel()