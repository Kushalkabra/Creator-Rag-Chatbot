from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "video_chunks"

# chunk_size=300 keeps each chunk small enough for precise retrieval (roughly a
# short paragraph) without diluting relevance. chunk_overlap=50 preserves
# context across boundaries so sentences split between chunks are not lost.
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

_chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            client=_chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=_embeddings,
        )
    return _vectorstore


def embed_video(video_data: dict) -> int:
    """
    Chunk a video transcript, embed with OpenAI, and persist in ChromaDB.

    Accepts the dict returned by get_youtube_data or get_instagram_data.
    Returns the number of chunks stored.
    """
    transcript = (video_data.get("transcript") or "").strip()
    if not transcript:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(transcript)
    if not chunks:
        return 0

    video_label = video_data["video_label"]
    creator = video_data.get("creator") or video_data.get("channel_title", "")
    engagement_rate = video_data.get("engagement_rate", 0.0)
    source_url = video_data["url"]

    texts: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for i, chunk in enumerate(chunks):
        texts.append(chunk)
        metadatas.append(
            {
                "video_id": video_label,
                "source_url": source_url,
                "creator": creator,
                "engagement_rate": float(engagement_rate),
                "chunk_index": i,
            }
        )
        ids.append(f"{video_label}_chunk_{i}")

    _get_vectorstore().add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(chunks)


def get_retriever(video_filter: str | None = None):
    """
    Return a LangChain retriever over stored video chunks.

    video_filter: optional "A" or "B" to restrict results to one video label.
    """
    search_kwargs: dict = {"k": 4}
    if video_filter in ("A", "B"):
        search_kwargs["filter"] = {"video_id": video_filter}

    return _get_vectorstore().as_retriever(search_kwargs=search_kwargs)
