import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def get_chat_model() -> BaseChatModel:
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key or api_key == "...":
        raise ValueError("GROQ_API_KEY is not set in environment")
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.1,
        groq_api_key=api_key,
    )
