# Retrieval evaluation for the RAG pipeline.
#
# recall@k: fraction of expected keywords found in the top-k retrieved chunks.
# Example: 3 of 4 expected keywords appear anywhere in the k chunks → recall@k = 0.75.
#
# Why this matters: retrieval quality drives most of RAG answer quality — if the
# wrong chunks (or none) are returned, the LLM cannot cite or reason correctly.
#
# Score guide: >70% pass rate is acceptable for a demo after ingest; >90% targets
# production with tuned chunking, embeddings, and eval cases grounded in your videos.

from __future__ import annotations

from dataclasses import asdict, dataclass

from rag.embedder import get_retriever

CROSS_VIDEO_NOTES_MARKER = "cross-video"


@dataclass
class EvalCase:
    question: str
    expected_keywords: list[str]
    video_filter: str | None
    notes: str


@dataclass
class EvalResult:
    question: str
    passed: bool
    recall_at_k: float
    retrieved_excerpts: list[str]
    matched_keywords: list[str]
    missing_keywords: list[str]
    video_ids_retrieved: list[str]
    notes: str


DEFAULT_EVAL_CASES: list[EvalCase] = [
    EvalCase(
        question="What does the creator say at the beginning of the video?",
        expected_keywords=["hey", "welcome", "intro", "hi", "hello", "what", "today"],
        video_filter=None,
        notes="Tests retrieval of opening hook content",
    ),
    EvalCase(
        question="What is the main topic or theme of video A?",
        expected_keywords=[],
        video_filter="A",
        notes="Tests video-specific filtering works correctly",
    ),
    EvalCase(
        question="What is the main topic or theme of video B?",
        expected_keywords=[],
        video_filter="B",
        notes="Tests video-specific filtering works correctly",
    ),
    EvalCase(
        question="What call to action does the creator give?",
        expected_keywords=[
            "follow",
            "subscribe",
            "like",
            "comment",
            "share",
            "click",
            "check",
        ],
        video_filter=None,
        notes="Tests retrieval of CTA content typically near end of video",
    ),
    EvalCase(
        question="How does the creator describe their product or service?",
        expected_keywords=[],
        video_filter=None,
        notes="Tests mid-video content retrieval",
    ),
    EvalCase(
        question="What hashtags or topics does this content relate to?",
        expected_keywords=[],
        video_filter=None,
        notes="Tests metadata-adjacent retrieval",
    ),
    EvalCase(
        question="Compare the opening lines of both videos",
        expected_keywords=[],
        video_filter=None,
        notes="Tests cross-video retrieval — should return chunks from both A and B",
    ),
    EvalCase(
        question="What advice or tips does the creator share?",
        expected_keywords=[
            "tip",
            "advice",
            "should",
            "make sure",
            "important",
            "remember",
            "key",
        ],
        video_filter=None,
        notes="Tests instructional content retrieval",
    ),
]


def _is_cross_video_case(case: EvalCase) -> bool:
    return CROSS_VIDEO_NOTES_MARKER in case.notes.lower()


def _evaluate_case(case: EvalCase, k: int) -> EvalResult:
    retriever = get_retriever(case.video_filter)
    retriever.search_kwargs["k"] = k

    docs = retriever.invoke(case.question)
    combined_text = " ".join(
        (getattr(doc, "page_content", "") or "") for doc in docs
    ).lower()

    video_ids_retrieved = []
    for doc in docs:
        meta = getattr(doc, "metadata", None) or {}
        video_id = str(meta.get("video_id", ""))
        if video_id and video_id not in video_ids_retrieved:
            video_ids_retrieved.append(video_id)

    retrieved_excerpts = [
        (getattr(doc, "page_content", "") or "")[:150] for doc in docs
    ]

    expected = [kw.strip().lower() for kw in case.expected_keywords if kw.strip()]
    matched_keywords = [kw for kw in expected if kw in combined_text]
    missing_keywords = [kw for kw in expected if kw not in combined_text]

    if expected:
        recall_at_k = len(matched_keywords) / len(expected)
    else:
        recall_at_k = 1.0 if docs else 0.0

    if _is_cross_video_case(case):
        passed = "A" in video_ids_retrieved and "B" in video_ids_retrieved
    elif expected:
        passed = len(matched_keywords) > 0
    elif case.video_filter in ("A", "B"):
        passed = bool(docs) and all(vid == case.video_filter for vid in video_ids_retrieved)
    else:
        passed = bool(docs)

    return EvalResult(
        question=case.question,
        passed=passed,
        recall_at_k=round(recall_at_k, 3),
        retrieved_excerpts=retrieved_excerpts,
        matched_keywords=matched_keywords,
        missing_keywords=missing_keywords,
        video_ids_retrieved=video_ids_retrieved,
        notes=case.notes,
    )


def run_eval(k: int = 4) -> dict:
    """Run DEFAULT_EVAL_CASES against the vector retriever and aggregate metrics."""
    results: list[EvalResult] = []

    for case in DEFAULT_EVAL_CASES:
        results.append(_evaluate_case(case, k))

    passed_count = sum(1 for result in results if result.passed)
    failed_count = len(results) - passed_count
    pass_rate = round((passed_count / len(results)) * 100, 1) if results else 0.0
    average_recall_at_k = round(
        sum(result.recall_at_k for result in results) / len(results), 3
    ) if results else 0.0

    summary = (
        f"{passed_count}/{len(results)} cases passed ({pass_rate}%) — "
        f"avg recall@{k}: {average_recall_at_k}"
    )

    return {
        "total_cases": len(results),
        "passed": passed_count,
        "failed": failed_count,
        "pass_rate": pass_rate,
        "average_recall_at_k": average_recall_at_k,
        "k": k,
        "results": [asdict(result) for result in results],
        "summary": summary,
    }
