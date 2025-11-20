import requests
from typing import Optional, Tuple
from fastapi import HTTPException
from app.config import config

class TTSService:
    """Servicio de Text-to-Speech"""
    
    @staticmethod
    def elevenlabs_tts(text: str, voice_id: Optional[str] = None) -> bytes:
        if not config.ELEVENLABS_API_KEY:
            raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY no configurada")
        
        voice = voice_id or config.ELEVENLABS_VOICE_ID
        url = f"{config.ELEVENLABS_API_URL}/{voice}"
        headers = {
            "xi-api-key": config.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        payload = {
            "text": text,
            "voice_settings": {
                "stability": 0.7,
                "similarity_boost": 0.7
            }
        }
        
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30, stream=True)
            if resp.status_code not in (200, 201):
                try: err = resp.json()
                except ValueError: err = resp.text
                raise HTTPException(status_code=502, detail=f"ElevenLabs error: {resp.status_code} - {err}")
            
            audio_bytes = resp.content
            if not audio_bytes:
                raise HTTPException(status_code=502, detail="ElevenLabs devolvió audio vacío")
            return audio_bytes
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Error conectando a ElevenLabs: {str(e)}")
    
    @staticmethod
    def synthesize(text: str) -> Tuple[bytes, str]:
        audio = TTSService.elevenlabs_tts(text)
        mime = "audio/mpeg"
        return audio, mime