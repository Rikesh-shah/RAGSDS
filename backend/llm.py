import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_embedding_model(hf_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Huggingface embedding model
    """
    # HF embedding model
    return HuggingFaceEmbeddings(model_name=hf_model_name)

def get_llm(model_name="gemini-2.5-flash", temperature=0.8):
    """
    Google LLM model
    """
    # google LLM
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
