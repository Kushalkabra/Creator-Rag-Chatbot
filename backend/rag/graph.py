from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from rag.embedder import get_retriever
from rag.llm import get_chat_model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# LangGraph vs a basic LangChain chain:
# - Chains are linear pipelines; this agent is an explicit graph (retrieve → generate)
#   so each step is visible, testable, and easy to extend (e.g. re-rank, rewrite query).
# - MessagesState + MemorySaver gives per-session conversation memory across turns
#   without hand-rolling history in every invoke.
# - State carries retrieved_docs between nodes instead of stuffing context into ad-hoc globals.


class RAGState(MessagesState):
    retrieved_docs: list[Document]


def _video_tag(video_label: str) -> str:
    if video_label == "A":
        return "[Video A]"
    if video_label == "B":
        return "[Video B]"
    return f"[Video {video_label}]"


def _format_retrieved_context(docs: list[Document]) -> str:
    if not docs:
        return "No relevant transcript chunks were retrieved."

    lines: list[str] = []
    for doc in docs:
        meta = doc.metadata or {}
        tag = _video_tag(str(meta.get("video_id", "?")))
        chunk_index = meta.get("chunk_index", "?")
        lines.append(f"{tag} (chunk {chunk_index}):\n{doc.page_content}")
    return "\n\n".join(lines)


def _format_video_stats(label: str, stats: dict[str, Any]) -> str:
    creator = stats.get("creator") or stats.get("channel_title", "Unknown")
    followers = stats.get("creator_followers")
    if followers is None:
        followers = stats.get("followers", "N/A")
    comments = stats.get("comments")
    if comments is None:
        comments = stats.get("comment_count", "N/A")

    return (
        f"{_video_tag(label)} — Creator: {creator} | "
        f"Followers: {followers} | "
        f"Engagement rate: {stats.get('engagement_rate', 'N/A')}% | "
        f"Views: {stats.get('views', 'N/A')} | "
        f"Likes: {stats.get('likes', 'N/A')} | "
        f"Comments: {comments}"
    )


def _metric_float(stats: dict[str, Any], key: str) -> float | None:
    value = stats.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metrics_comparison(stats_a: dict[str, Any], stats_b: dict[str, Any]) -> str:
    """Pre-computed facts so the LLM does not invert numeric comparisons."""
    lines: list[str] = []

    er_a = _metric_float(stats_a, "engagement_rate")
    er_b = _metric_float(stats_b, "engagement_rate")
    if er_a is not None and er_b is not None:
        if er_a > er_b:
            lines.append(
                f"Engagement rate: [Video A] is HIGHER ({er_a}% > {er_b}% on [Video B])."
            )
        elif er_b > er_a:
            lines.append(
                f"Engagement rate: [Video B] is HIGHER ({er_b}% > {er_a}% on [Video A])."
            )
        else:
            lines.append(f"Engagement rate: tied at {er_a}%.")

    views_a = _metric_float(stats_a, "views")
    views_b = _metric_float(stats_b, "views")
    if views_a is not None and views_b is not None:
        if views_a > views_b:
            lines.append(f"Views: [Video A] has MORE ({int(views_a):,} vs {int(views_b):,}).")
        elif views_b > views_a:
            lines.append(f"Views: [Video B] has MORE ({int(views_b):,} vs {int(views_a):,}).")

    likes_a = _metric_float(stats_a, "likes")
    likes_b = _metric_float(stats_b, "likes")
    if likes_a is not None and likes_b is not None:
        if likes_a > likes_b:
            lines.append(f"Likes: [Video A] has MORE ({int(likes_a):,} vs {int(likes_b):,}).")
        elif likes_b > likes_a:
            lines.append(f"Likes: [Video B] has MORE ({int(likes_b):,} vs {int(likes_a):,}).")

    lines.append(
        "Engagement rate = (likes + comments) / views × 100 (from ingestion metadata). "
        "Do not recalculate or contradict these figures."
    )
    return "\n".join(lines)


def _build_system_prompt(
    retrieved_docs: list[Document],
    video_metadata: dict[str, Any],
) -> str:
    context = _format_retrieved_context(retrieved_docs)

    stats_a = video_metadata.get("A", {})
    stats_b = video_metadata.get("B", {})
    stats_block = "\n".join(
        [
            _format_video_stats("A", stats_a),
            _format_video_stats("B", stats_b),
        ]
    )
    comparison_block = _format_metrics_comparison(stats_a, stats_b)

    return f"""You are an analytical video comparison assistant for two pieces of content (Video A and Video B).

## Retrieved transcript excerpts
{context}

## Video metrics (authoritative numbers — use exactly as written)
{stats_block}

## Metric comparison (ground truth — do not contradict)
{comparison_block}

## Instructions
- For views, likes, comments, and engagement rate, use ONLY the Video metrics and Metric comparison sections above. Never claim a lower number is higher.
- If the user asks why one video got "more engagement", check Metric comparison first: higher engagement rate wins; more views alone is reach, not engagement rate.
- Always cite sources as [Video A] or [Video B], including chunk context when referencing transcript text (e.g. [Video A] chunk 2).
- Use transcript excerpts for hooks, pacing, quotes, and content themes — not for inventing metrics.
- Suggest concrete improvements (hooks, pacing, CTAs, hashtags) grounded in retrieved chunks and metrics.
- If retrieved context is empty, say so and answer from metrics only where possible.
- If the question assumes the wrong winner (e.g. "why did A outperform B on engagement" when B has the higher rate), correct the premise briefly, then explain using the data.
"""


def build_graph(video_metadata: dict[str, Any]):
    """
    Build and compile a conversational RAG agent.

    video_metadata: dict keyed by "A" and "B" with each video's ingestion stats
    (creator/channel_title, engagement_rate, views, likes, creator_followers, etc.).
    """
    retriever = get_retriever()
    llm = get_chat_model()

    def retrieve(state: RAGState) -> dict[str, list[Document]]:
        query = ""
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                query = message.content if isinstance(message.content, str) else str(message.content)
                break

        if not query.strip():
            return {"retrieved_docs": []}

        docs = retriever.invoke(query)
        return {"retrieved_docs": docs}

    def generate(state: RAGState, config: RunnableConfig) -> dict[str, list[AIMessage]]:
        retrieved_docs = state.get("retrieved_docs") or []
        system_prompt = _build_system_prompt(retrieved_docs, video_metadata)

        all_messages = list(state["messages"])
        # Trimming to last 6 messages — MemorySaver still holds full history, but
        # sending everything to the LLM on every turn grows cost linearly with
        # conversation length. 6 turns covers follow-up context without runaway
        # cost at scale.
        recent_messages = all_messages[-6:] if len(all_messages) > 6 else all_messages

        conversation = [SystemMessage(content=system_prompt)] + recent_messages

        content = ""
        for chunk in llm.stream(conversation, config=config):
            if chunk.content:
                content += chunk.content

        return {"messages": [AIMessage(content=content)]}

    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
