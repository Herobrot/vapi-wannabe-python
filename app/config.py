import os
from dotenv import load_dotenv

load_dotenv()

class VAPIConfig:
    LLM_MODEL = "llama3.2:3b"
    STT_MODEL = "base"
    TTS_ENGINE = "elevenlabs"
    
    # ElevenLabs Configuration
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "LnGOA2SxH2fX1e1iNzEp")
    ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

config = VAPIConfig()