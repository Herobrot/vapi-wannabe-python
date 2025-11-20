import whisper
import tempfile
import os
from fastapi import HTTPException
from app.config import config

print(f"🎤 Cargando modelo Whisper '{config.STT_MODEL}'...")
try:
    whisper_model = whisper.load_model(config.STT_MODEL)
    print("✅ Modelo Whisper cargado")
except Exception as e:
    print(f"❌ Error cargando Whisper: {e}")
    whisper_model = None

class STTService:
    """Servicio de Speech-to-Text con Whisper"""
    
    @staticmethod
    def transcribe(audio_data: bytes, language: str = "es"):
        if not whisper_model:
            raise HTTPException(status_code=500, detail="Modelo Whisper no inicializado")
            
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            result = whisper_model.transcribe(
                audio=tmp_path,
                language=language,
                fp16=False 
            )
            
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            return result["text"]
            
        except Exception as e:
            if tmp_path and os.path.exists(tmp_path):
                try: os.unlink(tmp_path)
                except OSError: 
                    pass
            raise HTTPException(status_code=500, detail=f"Error STT: {str(e)}")