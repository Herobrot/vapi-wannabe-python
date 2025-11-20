import os
import aiofiles
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/voice-chat", response_class=HTMLResponse)
async def voice_chat_interface():
    """Interfaz web cargada desde archivo"""
    # Ruta relativa asumiendo que se ejecuta desde root
    template_path = os.path.join("app", "templates", "index.html")
    async with aiofiles.open(template_path, "r", encoding="utf-8") as f:
        return await f.read()