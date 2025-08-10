from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    doc_id: str
    question: str
    top_k: int = 6

class ChatResponse(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    doc_id: str
    session_id: Optional[str] = None
    question: str
    top_k: int = 6