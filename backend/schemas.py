from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Message]] = []
    user_id: Optional[str] = None

class Source(BaseModel):
    title: str
    url: str
    relevance_score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float
    processing_time: float
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    llm_provider: str
    vectordb_status: str

class ErrorResponse(BaseModel):
    detail: str
    timestamp: datetime