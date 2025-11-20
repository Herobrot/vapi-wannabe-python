import uvicorn
from fastapi import FastAPI
from app.config import config
from app.routers import api, websocket, web

# Inicializar FastAPI
app = FastAPI(title="VAPI - Voice API Real-Time", version="2.1.0")

# Incluir Routers
app.include_router(web.router)
app.include_router(websocket.router)
app.include_router(api.router)

@app.get("/")
async def root():
    """Información del servidor"""
    return {
        "service": "VAPI - Voice API Real-Time",
        "version": "2.1.0",
        "features": ["auto-silence-detection", "real-time-voice", "websocket", "tts-elevenlabs"],
        "components": {
            "llm": config.LLM_MODEL,
            "stt": f"Whisper-{config.STT_MODEL}",
            "tts": config.TTS_ENGINE                            
        }
    }

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 VAPI Server v2.1 - Modularizado")
    print("=" * 60)
    print("🌐 Servidor disponible en: http://localhost:8000/voice-chat")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)