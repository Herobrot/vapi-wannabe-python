import ollama
from typing import List
from fastapi import HTTPException
from app.config import config

class LLMService:
    """Servicio de Lenguaje Local con Ollama"""
    
    @staticmethod
    def chat_completion(messages: List[dict], temperature: float = 0.7, 
                       max_tokens: int = 500, stream: bool = False):
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