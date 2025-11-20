import json
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from app.services.stt import STTService
from app.services.llm import LLMService
from app.services.tts import TTSService

router = APIRouter()

@router.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Cliente conectado al WebSocket")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'audio':
                try:
                    # 1. Decodificar audio
                    audio_data = base64.b64decode(message['data'])
                    
                    # 2. Transcribir
                    await websocket.send_json({'type': 'status', 'message': 'Transcribiendo audio...'})
                    transcription = STTService.transcribe(audio_data)
                    
                    # --- NUEVA VALIDACIÓN ---
                    # Si Whisper devuelve vacío o espacios en blanco, ignoramos y no llamamos al LLM
                    if not transcription or not transcription.strip():
                        await websocket.send_json({
                            'type': 'status', 
                            'message': '⚠️ No se detectó voz. Intenta hablar un poco más fuerte.'
                        })
                        continue  # Salta al inicio del bucle y espera el siguiente mensaje
                    # ------------------------

                    await websocket.send_json({'type': 'transcription', 'text': transcription})
                    
                    # 3. Generar respuesta
                    await websocket.send_json({'type': 'status', 'message': 'Generando respuesta...'})
                    
                    # Recuperar el historial enviado por el cliente
                    messages = message.get('history', [])
                    
                    # Inyectar System Prompt si no existe
                    if not any(m.get('role') == 'system' for m in messages):
                        messages.insert(0, {
                            'role': 'system',
                            'content': 'Eres un asistente de voz útil y conciso. Responde de forma breve y clara.'
                        })
                    
                    # --- CORRECCIÓN AQUÍ ---
                    # Agregamos la transcripción actual a los mensajes para el LLM
                    messages.append({"role": "user", "content": transcription})
                    # -----------------------

                    response = LLMService.chat_completion(messages, temperature=0.7, max_tokens=300)
                    await websocket.send_json({'type': 'response', 'text': response})
                    
                    # 4. Generar TTS
                    try:
                        audio_bytes, mime = TTSService.synthesize(response)
                        b64 = base64.b64encode(audio_bytes).decode('ascii')
                        await websocket.send_json({
                            'type': 'audio',
                            'data': b64,
                            'format': mime
                        })
                    except HTTPException as tts_err:
                        await websocket.send_json({'type': 'error', 'message': f"TTS error: {tts_err.detail}"})
                    
                except Exception as e:
                    print(f"❌ Error procesando audio: {str(e)}")
                    await websocket.send_json({'type': 'error', 'message': str(e)})
            
    except WebSocketDisconnect:
        print("🔌 Cliente desconectado")
    except Exception as e:
        print(f"❌ Error en WebSocket: {str(e)}")