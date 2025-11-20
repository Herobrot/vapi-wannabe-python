import time
import json
from typing import Optional
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse

from app.models import ChatCompletionRequest
from app.services.llm import LLMService
from app.services.stt import STTService
from app.config import config

router = APIRouter()

@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": config.LLM_MODEL,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local"
        }]
    }

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    if request.stream:
        async def generate_stream():
            for chunk in LLMService.chat_completion(messages, request.temperature, request.max_tokens, stream=True):
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
        content = LLMService.chat_completion(messages, request.temperature, request.max_tokens)
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

@router.post("/v1/audio/transcriptions")
async def create_transcription(file: UploadFile = File(...), model: str = "whisper-1", language: Optional[str] = "es"):
    audio_data = await file.read()
    text = STTService.transcribe(audio_data, language)
    return {"text": text}