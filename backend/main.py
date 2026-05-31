import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ingestion.instagram import get_instagram_data
from ingestion.youtube import get_youtube_data
from rag.embedder import embed_video
from rag.graph import build_graph

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_sessions: dict[str, Any] = {}
cached_videos: dict[str, dict[str, Any]] = {"A": {}, "B": {}}


class IngestRequest(BaseModel):
    youtube_url: str
    instagram_url: str


class ChatRequest(BaseModel):
    message: str
    session_id: str


def _public_video_stats(data: dict[str, Any]) -> dict[str, Any]:
    """Return ingest stats without transcript fields."""
    comments = data.get("comments")
    if comments is None:
        comments = data.get("comment_count", 0)

    return {
        "video_label": data.get("video_label"),
        "url": data.get("url"),
        "views": data.get("views"),
        "likes": data.get("likes"),
        "comments": comments,
        "engagement_rate": data.get("engagement_rate"),
        "creator": data.get("creator") or data.get("channel_title"),
        "hashtags": data.get("hashtags", []),
        "upload_date": data.get("upload_date") or data.get("published_at"),
        "duration": data.get("duration") or data.get("duration_seconds"),
    }


def _graph_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Metadata shape expected by build_graph's system prompt."""
    return {
        "creator": data.get("creator") or data.get("channel_title"),
        "channel_title": data.get("channel_title"),
        "creator_followers": data.get("creator_followers"),
        "engagement_rate": data.get("engagement_rate"),
        "views": data.get("views"),
        "likes": data.get("likes"),
        "comments": data.get("comments") or data.get("comment_count", 0),
        "hashtags": data.get("hashtags", []),
        "url": data.get("url"),
    }


def _docs_to_sources(docs: list[Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for doc in docs:
        meta = getattr(doc, "metadata", None) or {}
        video_id = meta.get("video_id", "?")
        tag = "[Video A]" if video_id == "A" else "[Video B]" if video_id == "B" else f"[Video {video_id}]"
        sources.append(
            {
                "video_id": video_id,
                "tag": tag,
                "chunk_index": meta.get("chunk_index"),
                "source_url": meta.get("source_url"),
                "creator": meta.get("creator"),
                "engagement_rate": meta.get("engagement_rate"),
                "excerpt": (getattr(doc, "page_content", "") or "")[:200],
            }
        )
    return sources


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(body: IngestRequest):
    youtube_data = get_youtube_data(body.youtube_url, "A")
    instagram_data = get_instagram_data(body.instagram_url, "B")

    embed_video(youtube_data)
    embed_video(instagram_data)

    video_metadata = {
        "A": _graph_metadata(youtube_data),
        "B": _graph_metadata(instagram_data),
    }
    active_sessions["main_graph"] = build_graph(video_metadata)

    cached_videos["A"] = _public_video_stats(youtube_data)
    cached_videos["B"] = _public_video_stats(instagram_data)

    return {
        "A": cached_videos["A"],
        "B": cached_videos["B"],
    }


@app.get("/videos")
def get_videos():
    if not cached_videos.get("A") and not cached_videos.get("B"):
        raise HTTPException(status_code=404, detail="No videos ingested yet. Call POST /ingest first.")
    return cached_videos


@app.post("/chat")
async def chat(body: ChatRequest):
    graph = active_sessions.get("main_graph")
    if graph is None:
        raise HTTPException(status_code=400, detail="No graph available. Call POST /ingest first.")

    config = {"configurable": {"thread_id": body.session_id}}
    input_state = {"messages": [HumanMessage(content=body.message)]}

    async def event_stream():
        sources: list[dict[str, Any]] = []

        async for event in graph.astream_events(input_state, config=config, version="v2"):
            event_type = event.get("event")
            metadata = event.get("metadata") or {}
            node_name = metadata.get("langgraph_node")

            if event_type == "on_chain_end" and node_name == "retrieve":
                output = event.get("data", {}).get("output") or {}
                retrieved = output.get("retrieved_docs") or []
                sources = _docs_to_sources(retrieved)

            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                token = ""
                if chunk is not None:
                    token = chunk.content if isinstance(chunk.content, str) else str(chunk.content or "")
                if token:
                    payload = {"token": token, "sources": sources}
                    yield f"data: {json.dumps(payload)}\n\n"

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

