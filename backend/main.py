from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ingestion.errors import (
    IngestionError,
    InstagramIngestionError,
    InstagramTimeoutError,
    InvalidInstagramURLError,
    InvalidYouTubeURLError,
    YouTubeAPIError,
    YouTubeTranscriptError,
)
from ingestion.instagram import get_instagram_data
from ingestion.youtube import get_youtube_data
from rag.embedder import clear_video_chunks, embed_video
from rag.eval import run_eval
from rag.graph import build_graph

INSTAGRAM_INGEST_TIMEOUT_SECONDS = 120

GROQ_INPUT_COST_PER_1M = 0.05
GROQ_OUTPUT_COST_PER_1M = 0.08
WHISPER_COST_PER_MINUTE = 0.006
GPT4O_INPUT_COST_PER_1M = 2.50
GPT4O_OUTPUT_COST_PER_1M = 10.00
OPENAI_EMBEDDING_COST_PER_1M = 0.02

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

session_metrics: dict[str, Any] = {
    "embedding_tokens_estimated": 0,
    "chat_input_tokens_estimated": 0,
    "chat_output_tokens_estimated": 0,
    "whisper_minutes": 0.0,
    "chunks_stored": 0,
    "chat_turns": 0,
    "ingestion_count": 0,
}


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


def _run_instagram_ingest(url: str, video_label: str) -> dict[str, Any]:
    """Run Instagram ingestion in a worker thread with a hard timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(get_instagram_data, url, video_label)
        try:
            return future.result(timeout=INSTAGRAM_INGEST_TIMEOUT_SECONDS)
        except FuturesTimeoutError as exc:
            raise InstagramTimeoutError(
                "Instagram ingestion timed out after 120 seconds (Whisper transcription is slow)"
            ) from exc


@app.get("/health")
def health():
    return {"status": "ok"}


def _usd_from_tokens(tokens: int, cost_per_1m: float) -> float:
    return tokens * cost_per_1m / 1_000_000


def _build_metrics_payload() -> dict[str, Any]:
    embedding_tokens = int(session_metrics["embedding_tokens_estimated"])
    chat_input_tokens = int(session_metrics["chat_input_tokens_estimated"])
    chat_output_tokens = int(session_metrics["chat_output_tokens_estimated"])
    whisper_minutes = float(session_metrics["whisper_minutes"])
    ingestion_count = int(session_metrics["ingestion_count"])

    chat_usd = _usd_from_tokens(chat_input_tokens, GROQ_INPUT_COST_PER_1M) + _usd_from_tokens(
        chat_output_tokens, GROQ_OUTPUT_COST_PER_1M
    )
    whisper_usd = whisper_minutes * WHISPER_COST_PER_MINUTE
    embeddings_usd = 0.0
    total_session_usd = embeddings_usd + chat_usd + whisper_usd
    cost_per_creator_usd = (
        total_session_usd / ingestion_count if ingestion_count > 0 else 0.0
    )

    gpt4o_estimated_usd = (
        _usd_from_tokens(embedding_tokens, OPENAI_EMBEDDING_COST_PER_1M)
        + _usd_from_tokens(chat_input_tokens, GPT4O_INPUT_COST_PER_1M)
        + _usd_from_tokens(chat_output_tokens, GPT4O_OUTPUT_COST_PER_1M)
        + whisper_usd
    )
    gpt4o_per_creator = (
        gpt4o_estimated_usd / ingestion_count if ingestion_count > 0 else 0.0
    )
    savings_usd = gpt4o_estimated_usd - total_session_usd
    savings_percent = (
        round((savings_usd / gpt4o_estimated_usd) * 100, 1)
        if gpt4o_estimated_usd > 0
        else 0.0
    )

    scale_keys = (10, 100, 1000, 10000)
    daily = {str(key): cost_per_creator_usd * key for key in scale_keys}
    monthly = {str(key): cost_per_creator_usd * key * 30 for key in scale_keys}

    return {
        "session": {
            "chunks_stored": int(session_metrics["chunks_stored"]),
            "chat_turns": int(session_metrics["chat_turns"]),
            "ingestion_count": ingestion_count,
            "embedding_tokens_estimated": embedding_tokens,
            "chat_input_tokens_estimated": chat_input_tokens,
            "chat_output_tokens_estimated": chat_output_tokens,
            "whisper_minutes": whisper_minutes,
        },
        "cost": {
            "embeddings_usd": embeddings_usd,
            "chat_usd": chat_usd,
            "whisper_usd": whisper_usd,
            "total_session_usd": total_session_usd,
            "cost_per_creator_usd": cost_per_creator_usd,
        },
        "model_info": {
            "llm": "llama-3.1-8b-instant (Groq)",
            "embeddings": "BAAI/bge-small-en-v1.5 (local)",
            "whisper": "base (local)",
            "llm_input_cost_per_1m": GROQ_INPUT_COST_PER_1M,
            "llm_output_cost_per_1m": GROQ_OUTPUT_COST_PER_1M,
            "embedding_cost_per_1m": 0.0,
        },
        "vs_openai_gpt4o": {
            "estimated_cost_usd": gpt4o_estimated_usd,
            "our_cost_usd": total_session_usd,
            "savings_usd": savings_usd,
            "savings_percent": savings_percent,
        },
        "scale_projections": {
            "cost_per_creator": cost_per_creator_usd,
            "daily": daily,
            "monthly": monthly,
            "vs_gpt4o_monthly_1000_creators": gpt4o_per_creator * 1000 * 30,
        },
    }


@app.get("/metrics", response_model=None)
def get_metrics():
    return _build_metrics_payload()


def _format_ingest_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "groq_api_key" in message or ("groq" in message and "api" in message):
        return (
            "Groq API key missing. Set GROQ_API_KEY in backend/.env. "
            "Get a key at https://console.groq.com"
        )
    return f"Ingestion failed: {exc}"


@app.post("/ingest")
def ingest(body: IngestRequest):
    try:
        # Parallel ingest cuts wall-clock time ~40-50% — YouTube API and Whisper are
        # both I/O/compute bound and completely independent so there's no reason to
        # run them sequentially.
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_yt = executor.submit(get_youtube_data, body.youtube_url, "A")
            future_ig = executor.submit(_run_instagram_ingest, body.instagram_url, "B")
            try:
                youtube_data = future_yt.result()
                instagram_data = future_ig.result()
            except Exception:
                for future in (future_yt, future_ig):
                    if not future.done():
                        future.cancel()
                raise

        clear_video_chunks("A")
        clear_video_chunks("B")

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(embed_video, youtube_data)
            future_b = executor.submit(embed_video, instagram_data)
            try:
                chunks_a = future_a.result()
                chunks_b = future_b.result()
            except Exception:
                for future in (future_a, future_b):
                    if not future.done():
                        future.cancel()
                raise

        session_metrics["chunks_stored"] = chunks_a + chunks_b
        session_metrics["embedding_tokens_estimated"] = (chunks_a + chunks_b) * 300
        session_metrics["whisper_minutes"] = round(
            float(instagram_data.get("duration") or 0) / 60, 2
        )
        session_metrics["ingestion_count"] += 1
        session_metrics["chat_turns"] = 0
        session_metrics["chat_input_tokens_estimated"] = 0
        session_metrics["chat_output_tokens_estimated"] = 0

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
    except (
        YouTubeAPIError,
        YouTubeTranscriptError,
        InvalidYouTubeURLError,
        InvalidInstagramURLError,
        InstagramIngestionError,
        InstagramTimeoutError,
        IngestionError,
    ) as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=_format_ingest_error(exc),
        ) from exc


@app.get("/videos")
def get_videos():
    if not cached_videos.get("A") and not cached_videos.get("B"):
        raise HTTPException(status_code=404, detail="No videos ingested yet. Call POST /ingest first.")
    return cached_videos


# Retrieval eval runs after ingestion to verify chunk quality.
# A failing eval (pass_rate < 50%) signals chunking or embedding
# issues before they silently degrade chat answer quality.
def _require_videos_for_eval() -> None:
    if not cached_videos.get("A") or not cached_videos.get("B"):
        raise HTTPException(
            status_code=400,
            detail="Ingest two videos before running eval",
        )


@app.get("/eval", response_model=None)
def run_retrieval_eval():
    _require_videos_for_eval()
    return run_eval(k=4)


@app.get("/eval/k/{k}", response_model=None)
def run_retrieval_eval_with_k(k: int = Path(..., ge=1, le=10)):
    _require_videos_for_eval()
    return run_eval(k=k)


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

        session_metrics["chat_turns"] += 1
        # estimate: system prompt ~500 + 4 chunks * 80 tokens + message ~50
        session_metrics["chat_input_tokens_estimated"] += 630
        # estimate: average response length
        session_metrics["chat_output_tokens_estimated"] += 250

    return StreamingResponse(event_stream(), media_type="text/event-stream")

