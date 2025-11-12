"""
VAPI Server - Voice API con conversación en tiempo real
Sistema completo de voz con LLM local y micrófono
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import ollama
import io
import json
import time
import whisper
import tempfile
import os
import base64
from pathlib import Path
import requests  # <- para ElevenLabs TTS
from dotenv import load_dotenv
load_dotenv()

# Inicializar FastAPI
app = FastAPI(title="VAPI - Voice API Real-Time", version="2.0.0")

# ======================
# MODELOS DE DATOS
# ======================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "llama3.2:3b"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    stream: Optional[bool] = False

# ======================
# CONFIGURACIÓN
# ======================

class VAPIConfig:
    LLM_MODEL = "llama3.2:3b"
    STT_MODEL = "base"  # tiny, base, small, medium, large
    TTS_ENGINE = "elevenlabs"  # elevenlabs o simple
    
    # ElevenLabs Configuration
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "LnGOA2SxH2fX1e1iNzEp")
    ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

config = VAPIConfig()

# Cargar modelo Whisper al inicio
print(f"🎤 Cargando modelo Whisper '{config.STT_MODEL}'...")
whisper_model = whisper.load_model(config.STT_MODEL)
print("✅ Modelo Whisper cargado")

# ======================
# COMPONENTE: LLM
# ======================

class LLMService:
    """Servicio de Lenguaje Local con Ollama"""
    
    @staticmethod
    def chat_completion(messages: List[dict], temperature: float = 0.7, 
                       max_tokens: int = 500, stream: bool = False):
        """Generar respuesta del LLM"""
        try:
            response = ollama.chat(
                model=config.LLM_MODEL,
                messages=messages,
                stream=stream,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens
                }
            )
            
            if stream:
                return response
            else:
                return response['message']['content']
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error LLM: {str(e)}")

# ======================
# COMPONENTE: STT (Whisper Local)
# ======================

class STTService:
    """Servicio de Speech-to-Text con Whisper"""
    
    @staticmethod
    def transcribe(audio_data: bytes, language: str = "es"):
        """Transcribir audio a texto usando Whisper"""
        try:
            # Crear archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            # Transcribir con Whisper
            result = whisper_model.transcribe(tmp_path, language=language)
            
            # Limpiar archivo temporal
            os.unlink(tmp_path)
            
            return result["text"]
            
        except Exception as e:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(status_code=500, detail=f"Error STT: {str(e)}")

# ======================
# COMPONENTE: TTS (ElevenLabs)
# ======================

class TTSService:
    """Servicio de Text-to-Speech. Actualmente ElevenLabs (MP3)."""
    
    @staticmethod
    def elevenlabs_tts(text: str, voice_id: Optional[str] = None) -> bytes:
        """Llama a la API de ElevenLabs y devuelve audio (bytes, MP3)."""
        if not config.ELEVENLABS_API_KEY:
            raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY no configurada en el entorno.")
        
        voice = voice_id or config.ELEVENLABS_VOICE_ID
        url = f"{config.ELEVENLABS_API_URL}/{voice}"
        headers = {
            "xi-api-key": config.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        payload = {
            "text": text,
            # Opciones de voz, se pueden ajustar
            "voice_settings": {
                "stability": 0.7,
                "similarity_boost": 0.7
            }
        }
        
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30, stream=True)
            if resp.status_code not in (200, 201):
                # Intentar leer mensaje de error si existe
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                raise HTTPException(status_code=502, detail=f"ElevenLabs TTS error: {resp.status_code} - {err}")
            
            # Leer contenido binario (MP3)
            audio_bytes = resp.content
            if not audio_bytes:
                raise HTTPException(status_code=502, detail="ElevenLabs devolvió audio vacío.")
            return audio_bytes
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Error conectando a ElevenLabs: {str(e)}")
    
    @staticmethod
    def synthesize(text: str) -> tuple[bytes, str]:
        """Genera audio y devuelve bytes + mime-type"""
        # Ahora mismo ElevenLabs devuelve MP3
        audio = TTSService.elevenlabs_tts(text)
        mime = "audio/mpeg"
        return audio, mime

# ======================
# ENDPOINTS API
# ======================

@app.get("/")
async def root():
    """Información del servidor"""
    return {
        "service": "VAPI - Voice API Real-Time",
        "version": "2.0.0",
        "features": ["real-time-voice", "microphone-input", "websocket", "tts-elevenlabs"],
        "components": {
            "llm": config.LLM_MODEL,
            "stt": f"Whisper-{config.STT_MODEL}",
            "tts": config.TTS_ENGINE
        }
    }

@app.get("/voice-chat", response_class=HTMLResponse)
async def voice_chat_interface():
    """Interfaz web para chat por voz en tiempo real (HTML actualizado con reproducción de audio)"""
    # Nota: incluí JS para reproducir mensajes tipo 'audio' recibidos por WS
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>VAPI - Chat de Voz en Tiempo Real</title>
        <meta charset="UTF-8">
        <style>
            /* (mismos estilos que antes - omitidos por brevedad en esta respuesta) */
            /* Copia los estilos completos del original si lo deseas */
            * { margin:0; padding:0; box-sizing:border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); min-height:100vh; display:flex; justify-content:center; align-items:center; padding:20px;}
            .container { background:white; border-radius:20px; box-shadow:0 20px 60px rgba(0,0,0,0.3); max-width:800px; width:100%; padding:40px;}
            h1{ color:#667eea; text-align:center; margin-bottom:10px; font-size:2.5em;}
            .subtitle{ text-align:center; color:#666; margin-bottom:30px;}
            .chat-container{ border:2px solid #e0e0e0; border-radius:15px; height:400px; overflow-y:auto; padding:20px; margin-bottom:20px; background:#f8f9fa;}
            .message{ margin-bottom:15px; padding:12px 18px; border-radius:18px; max-width:80%; animation:slideIn .3s ease;}
            .user-message{ background:#667eea; color:white; margin-left:auto; text-align:right;}
            .assistant-message{ background:#e9ecef; color:#333;}
            .system-message{ background:#fff3cd; color:#856404; text-align:center; margin:10px auto; font-size:.9em;}
            .controls{ display:flex; gap:15px; justify-content:center; flex-wrap:wrap;}
            button{ padding:15px 30px; border:none; border-radius:25px; font-size:16px; font-weight:600; cursor:pointer; transition:all .3s; display:flex; align-items:center; gap:10px;}
            #recordBtn{ background:#28a745; color:white;}
            #recordBtn.recording{ background:#dc3545; animation:pulse 1.5s infinite;}
            #stopBtn{ background:#dc3545; color:white;}
            #clearBtn{ background:#6c757d; color:white;}
            .status{ text-align:center; padding:15px; border-radius:10px; margin-top:20px; font-weight:500;}
            .status.connected{ background:#d4edda; color:#155724;}
            .status.disconnected{ background:#f8d7da; color:#721c24;}
            .status.processing{ background:#d1ecf1; color:#0c5460;}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ VAPI Chat de Voz</h1>
            <p class="subtitle">Habla con tu asistente AI en tiempo real</p>
            
            <div class="chat-container" id="chatContainer">
                <div class="message system-message">
                    👋 ¡Bienvenido! Presiona "Iniciar Grabación" y habla con el asistente
                </div>
            </div>
            
            <div class="controls">
                <button id="recordBtn" onclick="startRecording()" disabled>
                    🎤 Iniciar Grabación
                </button>
                <button id="stopBtn" onclick="stopRecording()" disabled>
                    ⏹️ Detener
                </button>
                <button id="clearBtn" onclick="clearChat()">
                    🗑️ Limpiar Chat
                </button>
            </div>
            
            <div id="status" class="status disconnected">
                ⚠️ Conectando al servidor...
            </div>
        </div>

        <script>
            let ws = null;
            let mediaRecorder = null;
            let audioChunks = [];
            let conversationHistory = [];
            
            // Conectar al WebSocket
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws/voice`);
                
                ws.onopen = () => {
                    updateStatus('connected', '✅ Conectado - Listo para hablar');
                    document.getElementById('recordBtn').disabled = false;
                };
                
                ws.onclose = () => {
                    updateStatus('disconnected', '⚠️ Desconectado - Intentando reconectar...');
                    document.getElementById('recordBtn').disabled = true;
                    setTimeout(connectWebSocket, 3000);
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    updateStatus('disconnected', '❌ Error de conexión');
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    handleServerMessage(data);
                };
            }
            
            // Manejar mensajes del servidor
            function handleServerMessage(data) {
                if (data.type === 'transcription') {
                    addMessage('user', data.text);
                } else if (data.type === 'response') {
                    addMessage('assistant', data.text);
                    updateStatus('connected', '✅ Listo para hablar');
                } else if (data.type === 'audio') {
                    // Reproducir audio enviado por el servidor (base64)
                    try {
                        const src = `data:${data.format};base64,${data.data}`;
                        const audio = new Audio(src);
                        // Forzar reproducción tras interacción del usuario (debería permitirse porque el usuario ya interactuó)
                        audio.play().catch(err => {
                            console.warn('No se pudo reproducir automáticamente:', err);
                        });
                    } catch (e) {
                        console.error('Error reproduciendo audio:', e);
                    }
                } else if (data.type === 'status') {
                    updateStatus('processing', data.message);
                } else if (data.type === 'error') {
                    addMessage('system', '❌ Error: ' + data.message);
                    updateStatus('connected', '✅ Listo para hablar');
                }
            }
            
            // Iniciar grabación
            async function startRecording() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        audio: {
                            sampleRate: 16000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true
                        } 
                    });
                    
                    mediaRecorder = new MediaRecorder(stream, {
                        mimeType: 'audio/webm'
                    });
                    
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            audioChunks.push(event.data);
                        }
                    };
                    
                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        await sendAudio(audioBlob);
                        
                        // Detener el stream
                        stream.getTracks().forEach(track => track.stop());
                    };
                    
                    mediaRecorder.start();
                    
                    // UI Updates
                    document.getElementById('recordBtn').disabled = true;
                    document.getElementById('recordBtn').classList.add('recording');
                    document.getElementById('stopBtn').disabled = false;
                    updateStatus('processing', '🔴 Grabando... Habla ahora');
                    
                } catch (error) {
                    console.error('Error accessing microphone:', error);
                    addMessage('system', '❌ No se pudo acceder al micrófono. Verifica los permisos.');
                }
            }
            
            // Detener grabación
            function stopRecording() {
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                    
                    document.getElementById('recordBtn').disabled = false;
                    document.getElementById('recordBtn').classList.remove('recording');
                    document.getElementById('stopBtn').disabled = true;
                    updateStatus('processing', '⏳ Procesando tu mensaje...');
                }
            }
            
            // Enviar audio al servidor
            async function sendAudio(audioBlob) {
                try {
                    const reader = new FileReader();
                    reader.onload = () => {
                        const base64Audio = reader.result.split(',')[1];
                        ws.send(JSON.stringify({
                            type: 'audio',
                            data: base64Audio,
                            history: conversationHistory
                        }));
                    };
                    reader.readAsDataURL(audioBlob);
                } catch (error) {
                    console.error('Error sending audio:', error);
                    addMessage('system', '❌ Error al enviar audio');
                    updateStatus('connected', '✅ Listo para hablar');
                }
            }
            
            // Agregar mensaje al chat
            function addMessage(role, text) {
                const chatContainer = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                
                if (role === 'user') {
                    messageDiv.className = 'message user-message';
                    conversationHistory.push({role: 'user', content: text});
                } else if (role === 'assistant') {
                    messageDiv.className = 'message assistant-message';
                    conversationHistory.push({role: 'assistant', content: text});
                } else {
                    messageDiv.className = 'message system-message';
                }
                
                messageDiv.textContent = text;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            // Actualizar estado
            function updateStatus(type, message) {
                const statusDiv = document.getElementById('status');
                statusDiv.className = `status ${type}`;
                statusDiv.textContent = message;
            }
            
            // Limpiar chat
            function clearChat() {
                document.getElementById('chatContainer').innerHTML = `
                    <div class="message system-message">
                        🗑️ Chat limpiado - Conversación reiniciada
                    </div>
                `;
                conversationHistory = [];
            }
            
            // Conectar al cargar la página
            connectWebSocket();
        </script>
    </body>
    </html>
    """

# ======================
# WEBSOCKET PARA VOZ EN TIEMPO REAL
# ======================

@app.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    """WebSocket para conversación de voz en tiempo real"""
    await websocket.accept()
    print("🔌 Cliente conectado al WebSocket")
    
    try:
        while True:
            # Recibir mensaje del cliente
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'audio':
                try:
                    # 1. Decodificar audio base64
                    audio_data = base64.b64decode(message['data'])
                    
                    # 2. Transcribir audio con Whisper
                    await websocket.send_json({
                        'type': 'status',
                        'message': 'Transcribiendo audio...'
                    })
                    
                    transcription = STTService.transcribe(audio_data)
                    
                    # Enviar transcripción al cliente
                    await websocket.send_json({
                        'type': 'transcription',
                        'text': transcription
                    })
                    
                    # 3. Obtener respuesta del LLM
                    await websocket.send_json({
                        'type': 'status',
                        'message': 'Generando respuesta...'
                    })
                    
                    # Construir historial de conversación
                    messages = message.get('history', [])
                    if not any(m.get('role') == 'system' for m in messages):
                        messages.insert(0, {
                            'role': 'system',
                            'content': 'Eres un asistente de voz útil y conciso. Responde de forma breve y clara.'
                        })
                    
                    # Generar respuesta
                    response = LLMService.chat_completion(messages, temperature=0.7, max_tokens=300)
                    
                    # Enviar texto de respuesta al cliente
                    await websocket.send_json({
                        'type': 'response',
                        'text': response
                    })
                    
                    # 4. Generar TTS (ElevenLabs) y enviar audio (base64)
                    try:
                        audio_bytes, mime = TTSService.synthesize(response)
                        b64 = base64.b64encode(audio_bytes).decode('ascii')
                        await websocket.send_json({
                            'type': 'audio',
                            'data': b64,
                            'format': mime
                        })
                    except HTTPException as tts_err:
                        # Enviar error pero no detener la conversación
                        await websocket.send_json({
                            'type': 'error',
                            'message': f"TTS error: {tts_err.detail}"
                        })
                    
                except Exception as e:
                    print(f"❌ Error procesando audio: {str(e)}")
                    await websocket.send_json({
                        'type': 'error',
                        'message': str(e)
                    })
            
    except WebSocketDisconnect:
        print("🔌 Cliente desconectado del WebSocket")
    except Exception as e:
        print(f"❌ Error en WebSocket: {str(e)}")
        try:
            await websocket.send_json({
                'type': 'error',
                'message': str(e)
            })
        except WebSocketDisconnect:
            pass

# ======================
# ENDPOINTS TRADICIONALES (Mantener compatibilidad)
# ======================

@app.get("/v1/models")
async def list_models():
    """Listar modelos disponibles"""
    return {
        "object": "list",
        "data": [
            {
                "id": config.LLM_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Endpoint de chat completions (compatible con OpenAI)"""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    if request.stream:
        async def generate_stream():
            for chunk in LLMService.chat_completion(
                messages, 
                request.temperature, 
                request.max_tokens,
                stream=True
            ):
                data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk['message']['content']},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        content = LLMService.chat_completion(
            messages,
            request.temperature,
            request.max_tokens
        )
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = "whisper-1",
    language: Optional[str] = "es"
):
    """Transcribir audio a texto"""
    audio_data = await file.read()
    text = STTService.transcribe(audio_data, language)
    
    return {"text": text}

# ======================
# INICIAR SERVIDOR
# ======================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Iniciando VAPI Server - Conversación en Tiempo Real")
    print("=" * 60)
    print(f"📦 Modelo LLM: {config.LLM_MODEL}")
    print(f"🎤 Modelo STT: Whisper-{config.STT_MODEL}")
    print(f"🔊 Motor TTS: {config.TTS_ENGINE}")
    print()
    print("🌐 Servidor disponible en:")
    print("   - API: http://localhost:8000")
    print("   - Interfaz de Voz: http://localhost:8000/voice-chat")
    print("   - Documentación: http://localhost:8000/docs")
    print()
    print("💡 Abre http://localhost:8000/voice-chat en tu navegador")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)