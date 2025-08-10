from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from .utils import save_upload_file_bytes
from .data_ingestion import ingest_pdf
from .query import answer_query, start_session, conversational_query, get_history
from .schemas import QueryRequest, ChatResponse, ChatRequest
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG LangChain FastAPI (conversational + table rows)")

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF allowed")
    save_path = save_upload_file_bytes(file)
    doc_id = ingest_pdf(save_path)
    return {"doc_id": doc_id, "filename": Path(save_path).name}

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    if not req.doc_id:
        raise HTTPException(status_code=400, detail="doc_id required")
    result = answer_query(doc_id=req.doc_id, question=req.question,
                          top_k=req.top_k
                          )
    return result

@app.post("/chat/start", response_model=ChatResponse)
async def chat_start():
    session_id = start_session()
    return {"session_id": session_id}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.doc_id:
        raise HTTPException(status_code=400, detail="doc_id required")
    # if no session_id provided, start one
    sid = req.session_id or start_session()
    result = conversational_query(doc_id=req.doc_id, session_id=sid, question=req.question,
                                  top_k=req.top_k)
    return result

@app.get("/chat/history/{session_id}")
async def chat_history(session_id: str):
    return {"session_id": session_id, "chat_history": get_history(session_id)}

@app.get("/health")
def health():
    return {"status": "ok"}
