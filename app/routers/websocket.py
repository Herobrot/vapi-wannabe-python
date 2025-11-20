import json
import base64
import uuid 
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from app.services.stt import STTService
from app.services.llm import LLMService
from app.services.tts import TTSService

router = APIRouter()

@router.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket, client_id: Optional[str] = Query(None)):
    """
    Endpoint WebSocket que acepta un client_id opcional.
    Si se provee client_id (John o Anabel), se usa ese para mantener historial.
    Si no (Anónimo), se genera uno nuevo.
    """
    await websocket.accept()
    
    # Determinar el Session ID
    if client_id:
        session_id = client_id
        is_new_session = False
    else:
        session_id = str(uuid.uuid4())
        is_new_session = True
        
    print(f"🔌 Cliente conectado. Session ID: {session_id} (Nuevo: {is_new_session})")
    
    # Mensaje de bienvenida técnica
    await websocket.send_json({
        'type': 'status',
        'message': f'Conectado como: {session_id}'
    })
    
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
                    
                    # Validación de silencio/vacío
                    if not transcription or not transcription.strip():
                        await websocket.send_json({
                            'type': 'status', 
                            'message': '⚠️ No se detectó voz. Intenta hablar un poco más fuerte.'
                        })
                        continue

                    await websocket.send_json({'type': 'transcription', 'text': transcription})
                    
                    # 3. Generar respuesta
                    await websocket.send_json({'type': 'status', 'message': 'Generando respuesta...'})
                    
                    # LLAMADA AL SERVICIO LLM CON EL ID DE SESIÓN
                    response_text = LLMService.process_user_interaction(
                        session_id=session_id,
                        user_text=transcription
                    )
                    
                    await websocket.send_json({'type': 'response', 'text': response_text})
                    
                    # 4. Generar TTS
                    try:
                        audio_bytes, mime = TTSService.synthesize(response_text)
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
        print(f"🔌 Cliente {session_id} desconectado")
    except Exception as e:
        print(f"❌ Error en WebSocket: {str(e)}")