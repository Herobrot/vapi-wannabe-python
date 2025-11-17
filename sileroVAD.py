"""
VAPI Server - Voice API con Silero VAD + WhisperX
Detección profesional de voz con pre-roll buffer y transcripción precisa
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState
from typing import Optional
import uuid
import wave
import datetime
from collections import deque
import numpy as np
import torch
import whisperx
import ollama
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="VAPI - Silero VAD + WhisperX", version="3.0.0")

# ======================
# CONFIGURACIÓN
# ======================

class VAPIConfig:
    # LLM
    LLM_MODEL = "llama3.2:3b"
    
    # WhisperX
    WHISPERX_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WHISPERX_BATCH_SIZE = 16
    WHISPERX_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
    WHISPERX_MODEL = "large-v3"
    WHISPERX_LANGUAGE = "es"
    
    # Silero VAD
    VAD_SAMPLE_RATE = 16000
    VAD_CHUNK_SIZE = 512  # ~32ms chunks
    VAD_PRE_ROLL_MS = 100  # Incluir 100ms antes del inicio
    
    # ElevenLabs
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "LnGOA2SxH2fX1e1iNzEp")
    ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

config = VAPIConfig()

# ======================
# CARGAR MODELOS
# ======================

print("🔊 Cargando Silero VAD...")
model_vad, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
print("✅ Silero VAD cargado")

print(f"🎤 Cargando WhisperX modelo '{config.WHISPERX_MODEL}'...")
model_whisper = whisperx.load_model(
    config.WHISPERX_MODEL,
    device=config.WHISPERX_DEVICE,
    compute_type=config.WHISPERX_COMPUTE_TYPE,
    language=config.WHISPERX_LANGUAGE
)
print("✅ WhisperX cargado")

# ======================
# SERVICIOS
# ======================

class LLMService:
    """Servicio LLM con Ollama"""
    
    @staticmethod
    def chat_completion(messages: list, temperature: float = 0.7, max_tokens: int = 300):
        try:
            response = ollama.chat(
                model=config.LLM_MODEL,
                messages=messages,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Error LLM: {str(e)}"

class TTSService:
    """Servicio TTS con ElevenLabs"""
    audio_mpeg = "audio/mpeg"
    
    @staticmethod
    def synthesize(text: str) -> tuple[bytes, str]:
        if not config.ELEVENLABS_API_KEY:
            return b"", TTSService.audio_mpeg
        
        url = f"{config.ELEVENLABS_API_URL}/{config.ELEVENLABS_VOICE_ID}"
        headers = {
            "xi-api-key": config.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": TTSService.audio_mpeg
        }
        payload = {
            "text": text,
            "voice_settings": {
                "stability": 0.7,
                "similarity_boost": 0.7
            }
        }
        
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code in (200, 201):
                return resp.content, TTSService.audio_mpeg
        except Exception as e:
            print(f"TTS Error: {e}")
        
        return b"", TTSService.audio_mpeg

# ======================
# ENDPOINTS
# ======================

@app.get("/")
async def root():
    return {
        "service": "VAPI - Silero VAD + WhisperX",
        "version": "3.0.0",
        "features": ["silero-vad", "whisperx", "pre-roll-buffer", "llm", "tts"],
        "components": {
            "vad": "Silero VAD",
            "stt": f"WhisperX-{config.WHISPERX_MODEL}",
            "llm": config.LLM_MODEL,
            "device": config.WHISPERX_DEVICE
        }
    }

@app.get("/voice-chat", response_class=HTMLResponse)
async def voice_chat_interface():
    """Interfaz web optimizada para Silero VAD"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>VAPI - Silero VAD</title>
        <meta charset="UTF-8">
        <style>
            * { margin:0; padding:0; box-sizing:border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 800px;
                width: 100%;
                padding: 40px;
            }
            h1 {
                color: #667eea;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }
            .badge {
                background: #667eea;
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.8em;
                display: inline-block;
                margin: 5px;
            }
            .chat-container {
                border: 2px solid #e0e0e0;
                border-radius: 15px;
                height: 400px;
                overflow-y: auto;
                padding: 20px;
                margin-bottom: 20px;
                background: #f8f9fa;
            }
            .message {
                margin-bottom: 15px;
                padding: 12px 18px;
                border-radius: 18px;
                max-width: 80%;
                animation: slideIn 0.3s ease;
            }
            .user-message {
                background: #667eea;
                color: white;
                margin-left: auto;
                text-align: right;
            }
            .assistant-message {
                background: #e9ecef;
                color: #333;
            }
            .system-message {
                background: #fff3cd;
                color: #856404;
                text-align: center;
                margin: 10px auto;
                font-size: 0.9em;
            }
            .controls {
                display: flex;
                gap: 15px;
                justify-content: center;
            }
            button {
                padding: 15px 30px;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }
            #recordBtn {
                background: #28a745;
                color: white;
            }
            #recordBtn.recording {
                background: #dc3545;
                animation: pulse 1.5s infinite;
            }
            #recordBtn:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            #clearBtn {
                background: #6c757d;
                color: white;
            }
            .status {
                text-align: center;
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
                font-weight: 500;
            }
            .status.connected { background: #d4edda; color: #155724; }
            .status.disconnected { background: #f8d7da; color: #721c24; }
            .status.processing { background: #d1ecf1; color: #0c5460; }
            .status.listening { background: #cfe2ff; color: #084298; }
            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }
            @keyframes slideIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ VAPI Silero VAD</h1>
            <p class="subtitle">
                Detección profesional de voz con pre-roll buffer
            </p>
            <div style="text-align: center; margin-bottom: 20px;">
                <span class="badge">Silero VAD</span>
                <span class="badge">WhisperX</span>
                <span class="badge">16kHz Streaming</span>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="message system-message">
                    👋 Sistema Silero VAD activo. Presiona "Iniciar" y habla naturalmente.
                </div>
            </div>
            
            <div class="controls">
                <button id="recordBtn" onclick="toggleRecording()" disabled>
                    🎤 Iniciar
                </button>
                <button id="clearBtn" onclick="clearChat()">
                    🗑️ Limpiar
                </button>
            </div>
            
            <div id="status" class="status disconnected">
                ⚠️ Conectando al servidor...
            </div>
        </div>

        <script>
            let ws = null;
            let audioContext = null;
            let mediaStream = null;
            let scriptNode = null;
            let isRecording = false;
            let conversationHistory = [];
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/audio`);
                
                ws.onopen = () => {
                    updateStatus('connected', '✅ Conectado - Sistema VAD listo');
                    document.getElementById('recordBtn').disabled = false;
                };
                
                ws.onclose = () => {
                    updateStatus('disconnected', '⚠️ Desconectado - Reconectando...');
                    document.getElementById('recordBtn').disabled = true;
                    setTimeout(connectWebSocket, 3000);
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    updateStatus('disconnected', '❌ Error de conexión');
                };
                
                ws.onmessage = (event) => {
                    try {
                        const data = eval('(' + event.data + ')');  // Parse Python dict format
                        handleServerMessage(data);
                    } catch (e) {
                        console.error('Error parsing message:', e);
                    }
                };
            }
            
            function handleServerMessage(data) {
                if (data.detection === 'proper_speech_start') {
                    updateStatus('listening', '🎤 Voz detectada - Escuchando...');
                } else if (data.detection === 'speech_false_detection') {
                    addMessage('system', '⚠️ Audio muy corto - habla más tiempo');
                    updateStatus('connected', '✅ Listo para hablar');
                } else if (data.text) {
                    addMessage('user', data.text);
                    processLLMResponse(data.text);
                }
            }
            
            async function processLLMResponse(userText) {
                updateStatus('processing', '🤔 Generando respuesta...');
                
                conversationHistory.push({role: 'user', content: userText});
                
                if (!conversationHistory.some(m => m.role === 'system')) {
                    conversationHistory.unshift({
                        role: 'system',
                        content: 'Eres un asistente de voz útil y conciso. Responde de forma breve y clara.'
                    });
                }
                
                try {
                    const response = await fetch('/v1/chat/completions', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model: 'llama3.2:3b',
                            messages: conversationHistory,
                            temperature: 0.7,
                            max_tokens: 300
                        })
                    });
                    
                    const result = await response.json();
                    const assistantText = result.choices[0].message.content;
                    
                    addMessage('assistant', assistantText);
                    conversationHistory.push({role: 'assistant', content: assistantText});
                    
                    // Generar TTS
                    generateTTS(assistantText);
                    
                } catch (error) {
                    console.error('Error con LLM:', error);
                    addMessage('system', '❌ Error generando respuesta');
                    updateStatus('connected', '✅ Listo para hablar');
                }
            }
            
            async function generateTTS(text) {
                try {
                    const response = await fetch('/v1/audio/tts', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text })
                    });
                    
                    if (response.ok) {
                        const audioBlob = await response.blob();
                        const audioUrl = URL.createObjectURL(audioBlob);
                        const audio = new Audio(audioUrl);
                        audio.play();
                    }
                } catch (error) {
                    console.error('Error con TTS:', error);
                } finally {
                    updateStatus('connected', '✅ Listo para hablar');
                }
            }
            
            async function toggleRecording() {
                if (isRecording) {
                    stopRecording();
                } else {
                    await startRecording();
                }
            }
            
            async function startRecording() {
                try {
                    mediaStream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            sampleRate: 16000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        }
                    });
                    
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({
                        sampleRate: 16000
                    });
                    
                    const source = audioContext.createMediaStreamSource(mediaStream);
                    scriptNode = audioContext.createScriptProcessor(4096, 1, 1);
                    
                    scriptNode.onaudioprocess = (event) => {
                        if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;
                        
                        const inputData = event.inputBuffer.getChannelData(0);
                        const int16Data = new Int16Array(inputData.length);
                        
                        for (let i = 0; i < inputData.length; i++) {
                            int16Data[i] = Math.max(-32768, Math.min(32767, Math.floor(inputData[i] * 32768)));
                        }
                        
                        ws.send(int16Data.buffer);
                    };
                    
                    source.connect(scriptNode);
                    scriptNode.connect(audioContext.destination);
                    
                    isRecording = true;
                    
                    const btn = document.getElementById('recordBtn');
                    btn.textContent = '🔴 Grabando...';
                    btn.classList.add('recording');
                    updateStatus('listening', '🎤 Esperando voz... (Silero VAD activo)');
                    
                } catch (error) {
                    console.error('Error accediendo al micrófono:', error);
                    addMessage('system', '❌ No se pudo acceder al micrófono');
                }
            }
            
            function stopRecording() {
                isRecording = false;
                
                if (scriptNode) {
                    scriptNode.disconnect();
                    scriptNode = null;
                }
                
                if (audioContext) {
                    audioContext.close();
                    audioContext = null;
                }
                
                if (mediaStream) {
                    mediaStream.getTracks().forEach(track => track.stop());
                    mediaStream = null;
                }
                
                const btn = document.getElementById('recordBtn');
                btn.textContent = '🎤 Iniciar';
                btn.classList.remove('recording');
                updateStatus('connected', '✅ Listo para hablar');
            }
            
            function addMessage(role, text) {
                const chatContainer = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                
                if (role === 'user') {
                    messageDiv.className = 'message user-message';
                } else if (role === 'assistant') {
                    messageDiv.className = 'message assistant-message';
                } else {
                    messageDiv.className = 'message system-message';
                }
                
                messageDiv.textContent = text;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            function updateStatus(type, message) {
                const statusDiv = document.getElementById('status');
                statusDiv.className = `status ${type}`;
                statusDiv.textContent = message;
            }
            
            function clearChat() {
                document.getElementById('chatContainer').innerHTML = `
                    <div class="message system-message">
                        🗑️ Chat limpiado - Conversación reiniciada
                    </div>
                `;
                conversationHistory = [];
            }
            
            connectWebSocket();
            
            window.addEventListener('beforeunload', () => {
                if (isRecording) stopRecording();
            });
        </script>
    </body>
    </html>
    """

# ======================
# WEBSOCKET CON SILERO VAD
# ======================

@app.websocket("/audio")
async def websocket_audio(websocket: WebSocket):
    """WebSocket con Silero VAD para detección profesional"""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    utterance_count = 0
    proper_start_sent = False
    
    # VAD Iterator
    vad_iterator = VADIterator(model_vad)
    
    # Buffers
    audio_buffer = bytearray()
    triggered = False
    
    # Ring buffer para pre-roll
    num_pre_roll_frames = int(config.VAD_PRE_ROLL_MS // ((config.VAD_CHUNK_SIZE / config.VAD_SAMPLE_RATE) * 1000))
    ring_buffer = deque(maxlen=num_pre_roll_frames)
    
    # Buffer de acumulación
    vad_buffer = np.array([], dtype=np.float32)
    
    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                break
            
            # Convertir PCM a float32
            pcm_samples = np.frombuffer(data, dtype=np.int16)
            if pcm_samples.size == 0:
                continue
            
            audio_float32 = pcm_samples.astype(np.float32) / 32768.0
            vad_buffer = np.concatenate((vad_buffer, audio_float32))
            
            # Procesar chunks de tamaño fijo
            while len(vad_buffer) >= config.VAD_CHUNK_SIZE:
                current_chunk = vad_buffer[:config.VAD_CHUNK_SIZE]
                vad_buffer = vad_buffer[config.VAD_CHUNK_SIZE:]
                
                # Detección VAD
                speech_segments = vad_iterator(current_chunk, return_seconds=False)
                is_speech_start = (speech_segments is not None and 'start' in speech_segments)
                is_speech_end = (speech_segments is not None and 'end' in speech_segments)
                
                chunk_int16 = (current_chunk * 32768.0).astype(np.int16).tobytes()
                
                # INICIO de voz detectado
                if is_speech_start and not triggered:
                    print("🎤 Voz detectada (start)")
                    # Agregar pre-roll
                    for rb_chunk in ring_buffer:
                        audio_buffer.extend(rb_chunk)
                    ring_buffer.clear()
                    triggered = True
                    proper_start_sent = False
                
                # Acumular audio mientras hay voz
                if triggered:
                    audio_buffer.extend(chunk_int16)
                    
                    # Enviar señal de "voz confirmada" después de 0.75s
                    if len(audio_buffer) >= 24000 and not proper_start_sent:
                        await websocket.send_text(str({'detection': 'proper_speech_start'}))
                        proper_start_sent = True
                else:
                    # Guardar en ring buffer (pre-roll)
                    ring_buffer.append(chunk_int16)
                
                # FIN de voz detectado
                if is_speech_end and triggered:
                    triggered = False
                    
                    # Solo transcribir si hay suficiente audio (>0.75s)
                    if len(audio_buffer) >= 24000:
                        # Guardar audio temporalmente
                        utterance_filename = f"/tmp/{session_id}_utterance_{utterance_count}.wav"
                        with wave.open(utterance_filename, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(config.VAD_SAMPLE_RATE)
                            wf.writeframes(audio_buffer)
                        
                        # Transcribir con WhisperX
                        print(f"📝 Transcribiendo {utterance_filename}...")
                        start_time = datetime.datetime.now()
                        
                        audio_data = whisperx.load_audio(utterance_filename)
                        result = model_whisper.transcribe(audio_data, batch_size=config.WHISPERX_BATCH_SIZE)
                        
                        segments = result.get("segments", [])
                        full_text = "".join(segment["text"] for segment in segments).strip()
                        
                        elapsed = datetime.datetime.now() - start_time
                        print(f"✅ Transcrito en {elapsed}: {full_text}")
                        
                        # Enviar transcripción
                        await websocket.send_text(str({'text': full_text}))
                        
                        # Limpiar archivo temporal
                        try:
                            os.unlink(utterance_filename)
                        except OSError:
                            pass
                        
                        utterance_count += 1
                    else:
                        # Audio muy corto
                        await websocket.send_text(str({'detection': 'speech_false_detection'}))
                    
                    # Reset buffers
                    audio_buffer = bytearray()
                    proper_start_sent = False
    
    except WebSocketDisconnect:
        print("🔌 Cliente desconectado")
    except Exception as e:
        print(f"❌ Error en WebSocket: {str(e)}")
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()

# ======================
# ENDPOINTS LLM Y TTS
# ======================

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """Endpoint de chat compatible con OpenAI"""
    messages = request.get("messages", [])
    temperature = request.get("temperature", 0.7)
    max_tokens = request.get("max_tokens", 300)
    
    content = LLMService.chat_completion(messages, temperature, max_tokens)
    
    return {
        "id": f"chatcmpl-{int(datetime.datetime.now().timestamp())}",
        "object": "chat.completion",
        "created": int(datetime.datetime.now().timestamp()),
        "model": config.LLM_MODEL,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }]
    }

@app.post("/v1/audio/tts")
async def text_to_speech(request: dict):
    """Endpoint de TTS"""
    text = request.get("text", "")
    
    if not text:
        return {"error": "No text provided"}
    
    audio_bytes, mime = TTSService.synthesize(text)
    
    if not audio_bytes:
        return {"error": "TTS failed"}
    
    from fastapi.responses import Response
    return Response(content=audio_bytes, media_type=mime)

# ======================
# INICIO
# ======================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🚀 VAPI Server v3.0 - Silero VAD + WhisperX")
    print("=" * 70)
    print(f"🔊 VAD: Silero (16kHz, pre-roll: {config.VAD_PRE_ROLL_MS}ms)")
    print(f"🎤 STT: WhisperX-{config.WHISPERX_MODEL} ({config.WHISPERX_DEVICE})")
    print(f"🤖 LLM: {config.LLM_MODEL}")
    print("🔊 TTS: ElevenLabs")
    print()
    print("🌐 Interfaz: http://localhost:8000/voice-chat")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)