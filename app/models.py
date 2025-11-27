from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "llama3.2:3b"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    stream: Optional[bool] = False

# Nuevo modelo para ExaBot
class ExamState(BaseModel):
    session_id: str
    start_time: float
    current_question: int = 1
    questions_answered: int = 0
    elapsed_time: float = 0.0
    remaining_time: float = 0.0
    is_active: bool = False