from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from .llm import get_llm, get_embedding_model
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

VECTORDIR = "./vectorstore"

DEFAULT_PROMPT = """You are an expert assistant. Use ONLY the provided context to answer the question.
Cite sources inline like [page 3] or [page 3, table 1] when possible.
If the answer is not present in the context, respond: "I don't know — the document does not contain the answer."

Context:
{context}

Question: {question}
Answer:"""

def make_qa_chain(llm, retriever):
    prompt = PromptTemplate(
        template=DEFAULT_PROMPT,
        input_variables=["context", "question"]
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # for short contexts; map_reduce for huge docs
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa

def make_conversational_chain(llm, retriever):
    """
    Create a ConversationalRetrievalChain. The chain expects input like:
    {"question": "...", "chat_history": [(user, assistant), ...]}
    and returns {"answer": ..., "source_documents": [...]}
    """
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return conv_chain

def get_vectordb_for_doc(doc_id: str):
    embeddings = get_embedding_model()
    vectordb = Chroma(persist_directory=VECTORDIR, embedding_function=embeddings, collection_name=doc_id)
    return vectordb

def answer_query(doc_id: str, question: str, top_k: int = 6, embedding_provider="hf"):
    vectordb = get_vectordb_for_doc(doc_id, embedding_provider=embedding_provider)
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    llm = get_llm()
    qa = make_qa_chain(llm, retriever)
    result = qa({"query": question})
    # normalize results
    answer = result.get("result") or result.get("answer")
    source_docs = result.get("source_documents", [])
    sources = []
    for d in source_docs:
        md = d.metadata or {}
        sources.append({
            "page": md.get("page"),
            "type": md.get("type"),
            "source": md.get("source"),
            "snippet": d.page_content
        })
    return {"answer": answer, "sources": sources}

# Temporary in-memory session store: session_id -> chat_history (list of tuples (user_msg, assistant_msg))
_session_store = {}

def start_session(session_id: str = None):
    import uuid
    if session_id is None:
        session_id = str(uuid.uuid4())
    _session_store[session_id] = []  # empty history
    return session_id

def get_history(session_id: str):
    return _session_store.get(session_id, [])

def append_turn(session_id: str, user_msg: str, assistant_msg: str):
    hist = _session_store.setdefault(session_id, [])
    hist.append((user_msg, assistant_msg))
    _session_store[session_id] = hist

def conversational_query(doc_id: str, session_id: str, question: str, top_k: int = 6):
    """
    Run a conversational retrieval chain using the stored chat_history (list of (u,a)).
    Returns answer + source documents and updates session history.
    """
    vectordb = get_vectordb_for_doc(doc_id)
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    llm = get_llm()
    conv_chain = make_conversational_chain(llm, retriever)

    # prepare chat_history as list of tuples (user, assistant)
    chat_history = get_history(session_id)

    # chain expects {"question": question, "chat_history": chat_history}
    result = conv_chain({"question": question, "chat_history": chat_history})
    # result keys: 'answer' and 'source_documents' (langchain may use 'answer' or 'result')
    answer = result.get("answer") or result.get("result")
    source_docs = result.get("source_documents", [])

    # append turn to store
    append_turn(session_id, question, answer)

    # prepare simplified sources
    sources = []
    for d in source_docs:
        md = d.metadata or {}
        sources.append({
            "page": md.get("page"),
            "type": md.get("type"),
            "source": md.get("source"),
            "snippet": d.page_content
        })

    return {"answer": answer, "sources": sources, "session_id": session_id, "chat_history": get_history(session_id)}
