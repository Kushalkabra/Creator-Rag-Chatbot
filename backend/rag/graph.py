from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from rag.embedder import get_retriever

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

    return (
        f"{_video_tag(label)} — Creator: {creator} | "
        f"Followers: {followers} | "
        f"Engagement rate: {stats.get('engagement_rate', 'N/A')}% | "
        f"Views: {stats.get('views', 'N/A')} | "
        f"Likes: {stats.get('likes', 'N/A')}"
    )


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

    return f"""You are an analytical video comparison assistant for two pieces of content (Video A and Video B).

## Retrieved transcript excerpts
{context}

## Video metrics
{stats_block}

## Instructions
- Always cite sources as [Video A] or [Video B], including chunk context when referencing transcript text (e.g. [Video A] chunk 2).
- Be analytical: compare engagement metrics, creators, and content themes across both videos.
- Suggest concrete improvements (hooks, pacing, CTAs, hashtags) grounded in the retrieved chunks and metrics.
- If retrieved context is empty, say so and answer from metrics only where possible.
"""


def build_graph(video_metadata: dict[str, Any]):
    """
    Build and compile a conversational RAG agent.

    video_metadata: dict keyed by "A" and "B" with each video's ingestion stats
    (creator/channel_title, engagement_rate, views, likes, creator_followers, etc.).
    """
    retriever = get_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

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

        conversation = [SystemMessage(content=system_prompt)] + list(state["messages"])

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
