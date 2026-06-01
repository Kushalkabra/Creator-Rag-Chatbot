# Video RAG Chatbot

Compare two social media videos and chat with their content.
Paste a YouTube and Instagram URL, get engagement stats for both,
then ask questions in plain English — why did one perform better,
what was the hook, suggest improvements.

Built for a real use case: creator agencies manually watch videos
and take notes to compare performance. This automates that.

---

## What it does

1. Takes a YouTube URL and Instagram Reel URL as input
2. Fetches transcripts and metadata for both (views, likes, 
   comments, engagement rate, creator, followers, hashtags)
3. Chunks and embeds both transcripts into ChromaDB using 
   BAAI/bge-small-en-v1.5 (local, no API cost)
4. Builds a LangGraph conversational agent that retrieves 
   relevant chunks and answers questions with citations
5. Streams responses token by token with source attribution
6. Tracks real session cost and projects at scale

---

## Architecture
Browser (Next.js 14)
│
▼
FastAPI backend
│
├── /ingest (parallel)
│     ├── YouTube: transcript API + Data API v3
│     └── Instagram: yt-dlp audio + faster-whisper + instaloader
│           ↓
│     ChromaDB (chunks + BGE embeddings)
│
└── /chat
└── LangGraph agent
├── retrieve node → ChromaDB k=4
└── generate node → Groq llama-3.1-8b-instant (streaming)

---

## Key decisions and why

**LangGraph over a basic LangChain chain**
Chains are linear and opaque. LangGraph makes the retrieve → generate
flow explicit — each node is isolated, testable, and extensible. Adding
a reranker or query rewriter later is just adding a node. State carries
retrieved docs between nodes so there are no globals.

**BAAI/bge-small-en-v1.5 over OpenAI embeddings**
Runs locally. Zero per-token cost. Downloads once (~130MB) and caches.
Specifically optimized for retrieval tasks. Quality is comparable to
text-embedding-3-small for English conversational text, which is exactly
what video transcripts are. Eliminates an entire cost line.

**Groq llama-3.1-8b-instant over GPT-4o**
For RAG, retrieval quality drives 80% of answer quality — the LLM just
synthesizes retrieved text. 8b handles synthesis well at a fraction of
the cost. At 1000 creators/day the difference is ~$8/day vs ~$480/day.

**faster-whisper over openai-whisper**
Same model weights, int8 quantization via CTranslate2, 4-8x faster on
CPU. Instagram has no transcript API so audio download + transcription
is unavoidable — this is the biggest free optimization in that pipeline.

**chunk_size=300, overlap=50**
Tried 100 — too fragmented, retrieval returned half-sentences.
Tried 600 — too broad, relevant content buried in noise.
300 maps roughly to one idea in conversational speech.
50-token overlap prevents losing context at chunk boundaries.

**ChromaDB for dev, Qdrant for production**
ChromaDB is zero config, runs in-process, persists to disk.
Right call for development — no cloud setup slowing down iteration
on the actual RAG logic. Qdrant has the same LangChain retriever
interface so the swap is three lines of code.

**YouTube disk cache**
YouTube Data API has a 10K unit daily free quota. Caching by video_id
means repeat ingests hit disk instead of the API. At 1K creators/day
with any URL overlap this is the difference between staying in quota
vs not.

**Parallel ingest**
YouTube API and Whisper transcription are completely independent.
Running them sequentially was leaving ~40-50% of wall-clock time
on the table. ThreadPoolExecutor runs both simultaneously.

---

## Cost analysis

Per creator (2 videos, ~5 chat turns):

| Operation | Cost |
|---|---|
| BGE embeddings (local) | $0.00 |
| Groq llama-3.1-8b chat | ~$0.0003 |
| faster-whisper (local CPU) | ~$0.006 compute |
| **Total** | **~$0.006** |

At 1000 creators/day: ~$6/day → ~$180/month

Compared to GPT-4o + OpenAI embeddings: ~$480/day → $14,400/month

The cost dashboard in the app tracks actual session token usage and
projects these numbers in real time.

---

## What breaks at scale and the fix

**Up to 1,000/day — watch these:**
- YouTube API quota (10K units/day): cache hits prevent burn,
  but monitor usage. Fix: Postgres cache with TTL.
- MemorySaver session history in RAM: fine for demo, dies on restart.
  Fix: langgraph-checkpoint-redis, same interface, 3-line swap.
- Whisper sequential on CPU: faster-whisper helps but still ~15s/reel.
  Fix: Celery background task, return job_id immediately.

**Up to 10,000/day — requires infra changes:**
- ChromaDB single-node: no replication, no horizontal scale.
  Fix: Qdrant Cloud, identical LangChain interface.
- Single FastAPI process: concurrent ingestion queue builds up.
  Fix: Gunicorn + multiple workers behind nginx.
- BGE on CPU becomes bottleneck at volume.
  Fix: dedicated embedding service or Cohere API ($0.0001/1K tokens).

**10,000+/day:**
- Groq rate limits, storage grows linearly, cost ~$60-180/day.
  Fix: Groq Batch API (50% discount), TTL on ChromaDB collections,
  request queuing with multiple API keys.

---

## Retrieval quality

After every ingest, `/eval` runs 8 retrieval test cases and returns
recall@4 — what fraction of expected keywords appear in the top 4
chunks. This was added because retrieval quality is the primary driver
of answer quality in RAG. A visually impressive chat interface can mask
silently poor retrieval. The eval score surfaces that signal immediately.

Score guide: >70% pass rate acceptable for demo, >90% for production.

---

## Setup

**Prerequisites**
- Python 3.11+
- Node.js 18+
- ffmpeg installed and on PATH
- Groq API key (free at console.groq.com)
- YouTube Data API v3 key (free at console.developers.google.com)

**Backend**
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
cp .env.example .env
# fill in GROQ_API_KEY and YOUTUBE_API_KEY
uvicorn main:app --reload
```

**Frontend**
```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open http://localhost:3000

---

## Demo

Tested with:
- YouTube: https://www.youtube.com/shorts/cpP-oXmNZXc
  Jay Shetty — "Money habits that changed my life"
- Instagram: https://www.instagram.com/reels/CmMPMRHLNq5/
  Same content, different platform

Same video cross-posted to both platforms. The engagement rate
difference is purely platform-driven — interesting case for the
chatbot to reason about.

---

## Trade-offs I'd revisit

- **Query rewriting**: vague questions retrieve poorly. A rewrite node
  that reformulates the query before hitting ChromaDB would improve
  recall on comparative questions. LangGraph makes this a one-node addition.

- **Semantic chunking**: current splitting is character-based, not
  topic-aware. LangChain's SemanticChunker splits on meaning changes
  instead. Better for long videos where topics shift mid-transcript.

- **Eval cases are generic**: the 8 test cases in `/eval` don't know
  your specific videos. A production eval set would be grounded in
  actual transcript content with verified answers.

- **MemorySaver**: fine for demo, wrong for production. Sessions die
  on restart. Redis checkpointer is the fix.
