"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { fetchMetrics, type MetricsResponse } from "@/lib/api";

const POLL_INTERVAL_MS = 30_000;
const SCALE_TIERS = [10, 100, 1000, 10000] as const;
const SLIDER_MIN = 10;
const SLIDER_MAX = 10_000;
const OPENAI_EMBEDDING_COST_PER_1M = 0.02;
const GPT4O_MINI_CHAT_MULTIPLIER = 6;

const EMPTY_METRICS: MetricsResponse = {
  session: {
    chunks_stored: 0,
    chat_turns: 0,
    ingestion_count: 0,
    embedding_tokens_estimated: 0,
    chat_input_tokens_estimated: 0,
    chat_output_tokens_estimated: 0,
    whisper_minutes: 0,
  },
  cost: {
    embeddings_usd: 0,
    chat_usd: 0,
    whisper_usd: 0,
    total_session_usd: 0,
    cost_per_creator_usd: 0,
  },
  model_info: {
    llm: "llama-3.1-8b-instant (Groq)",
    embeddings: "BAAI/bge-small-en-v1.5 (local)",
    whisper: "base (local)",
    llm_input_cost_per_1m: 0.05,
    llm_output_cost_per_1m: 0.08,
    embedding_cost_per_1m: 0,
  },
  vs_openai_gpt4o: {
    estimated_cost_usd: 0,
    our_cost_usd: 0,
    savings_usd: 0,
    savings_percent: 0,
  },
  scale_projections: {
    cost_per_creator: 0,
    daily: { "10": 0, "100": 0, "1000": 0, "10000": 0 },
    monthly: { "10": 0, "100": 0, "1000": 0, "10000": 0 },
    vs_gpt4o_monthly_1000_creators: 0,
  },
};

function formatUsd(value: number): string {
  if (value === 0) return "$0.0000";
  if (value >= 1000) return `$${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  if (value >= 1) return `$${value.toFixed(2)}`;
  return `$${value.toFixed(4)}`;
}

function formatScaleLabel(tier: number): string {
  if (tier >= 1000) return `${tier / 1000}K`;
  return String(tier);
}

function sliderToCreators(sliderValue: number): number {
  const minLog = Math.log10(SLIDER_MIN);
  const maxLog = Math.log10(SLIDER_MAX);
  const log = minLog + (sliderValue / 100) * (maxLog - minLog);
  return Math.round(10 ** log);
}

function creatorsToSlider(creators: number): number {
  const minLog = Math.log10(SLIDER_MIN);
  const maxLog = Math.log10(SLIDER_MAX);
  const log = Math.log10(Math.max(SLIDER_MIN, Math.min(SLIDER_MAX, creators)));
  return ((log - minLog) / (maxLog - minLog)) * 100;
}

function nearestTier(creators: number): number {
  return SCALE_TIERS.reduce((prev, curr) =>
    Math.abs(curr - creators) < Math.abs(prev - creators) ? curr : prev
  );
}

function gpt4oPerCreator(metrics: MetricsResponse): number {
  const count = metrics.session.ingestion_count;
  if (count <= 0) return 0;
  return metrics.vs_openai_gpt4o.estimated_cost_usd / count;
}

function gpt4oMiniPerCreator(metrics: MetricsResponse): number {
  const count = metrics.session.ingestion_count;
  if (count <= 0) return 0;
  const whisper = metrics.cost.whisper_usd / count;
  const embedding =
    (metrics.session.embedding_tokens_estimated * OPENAI_EMBEDDING_COST_PER_1M) /
    1_000_000 /
    count;
  const chatMini = (metrics.cost.chat_usd / count) * GPT4O_MINI_CHAT_MULTIPLIER;
  return whisper + embedding + chatMini;
}

type StatCardProps = {
  title: string;
  value: string;
  subtitle: string;
  valueClassName?: string;
};

function StatCard({ title, value, subtitle, valueClassName }: StatCardProps) {
  return (
    <div className="rounded-lg border border-white/10 bg-[#1a1a1a] p-4">
      <p className="text-xs text-white/50">{title}</p>
      <p className={`mt-1 text-2xl font-semibold text-white ${valueClassName ?? ""}`}>
        {value}
      </p>
      <p className="mt-1 text-xs text-white/40">{subtitle}</p>
    </div>
  );
}

type BottleneckRowProps = {
  badge: string;
  badgeClass: string;
  scale: string;
  risks: { label: string; fix: string }[];
  summary?: string;
};

function BottleneckRow({ badge, badgeClass, scale, risks, summary }: BottleneckRowProps) {
  return (
    <div className="rounded-lg border border-white/10 bg-[#1a1a1a] p-4">
      <div className="flex flex-wrap items-start gap-3">
        <span className={`shrink-0 rounded-full border px-2.5 py-0.5 text-xs font-medium ${badgeClass}`}>
          {badge}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-white">{scale}</p>
          {summary && <p className="mt-1 text-xs text-white/50">{summary}</p>}
          {risks.length > 0 && (
            <ul className="mt-2 space-y-2">
              {risks.map((risk) => (
                <li key={risk.label} className="text-xs text-white/60">
                  <span className="text-white/80">• {risk.label}</span>
                  <br />
                  <span className="ml-3 text-white/40">Fix: {risk.fix}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

type ModelCardProps = {
  title: string;
  chosen: string;
  why: string;
  tradeoff: string;
  upgrade: string;
};

function ModelCard({ title, chosen, why, tradeoff, upgrade }: ModelCardProps) {
  return (
    <div className="rounded-lg border border-white/10 bg-[#1a1a1a] p-4">
      <h3 className="text-sm font-semibold text-white">{title}</h3>
      <p className="mt-1 text-xs font-medium text-white/70">{chosen}</p>
      <p className="mt-3 text-xs leading-relaxed text-white/60">{why}</p>
      <p className="mt-2 text-xs text-white/40">
        <span className="text-white/50">Trade-off:</span> {tradeoff}
      </p>
      <p className="mt-1 text-xs text-white/40">
        <span className="text-white/50">Upgrade path:</span> {upgrade}
      </p>
    </div>
  );
}

export default function ScalePage() {
  const [metrics, setMetrics] = useState<MetricsResponse>(EMPTY_METRICS);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [secondsAgo, setSecondsAgo] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sliderValue, setSliderValue] = useState(() => creatorsToSlider(100));

  const loadMetrics = useCallback(async () => {
    try {
      const data = await fetchMetrics();
      setMetrics(data);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadMetrics();
    const pollId = window.setInterval(loadMetrics, POLL_INTERVAL_MS);
    return () => window.clearInterval(pollId);
  }, [loadMetrics]);

  useEffect(() => {
    const tickId = window.setInterval(() => {
      if (!lastUpdated) {
        setSecondsAgo(0);
        return;
      }
      setSecondsAgo(Math.floor((Date.now() - lastUpdated.getTime()) / 1000));
    }, 1000);
    return () => window.clearInterval(tickId);
  }, [lastUpdated]);

  const creatorsPerDay = useMemo(() => sliderToCreators(sliderValue), [sliderValue]);
  const highlightedTier = useMemo(() => nearestTier(creatorsPerDay), [creatorsPerDay]);

  const ourPerCreator = metrics.scale_projections.cost_per_creator;
  const gpt4oPerCreatorCost = gpt4oPerCreator(metrics);
  const miniPerCreator = gpt4oMiniPerCreator(metrics);

  const sliderCosts = useMemo(
    () => ({
      our: ourPerCreator * creatorsPerDay,
      mini: miniPerCreator * creatorsPerDay,
      gpt4o: gpt4oPerCreatorCost * creatorsPerDay,
    }),
    [creatorsPerDay, ourPerCreator, miniPerCreator, gpt4oPerCreatorCost]
  );

  const savingsPercent = metrics.vs_openai_gpt4o.savings_percent;

  return (
    <div className="min-h-screen bg-[#0f0f0f] text-white">
      <header className="sticky top-0 z-10 flex flex-wrap items-center justify-between gap-3 border-b border-white/10 bg-[#0f0f0f]/95 px-6 py-4 backdrop-blur">
        <div className="flex items-center gap-4">
          <Link href="/" className="text-xs text-white/50 transition hover:text-white">
            ← RAG Chatbot
          </Link>
          <h1 className="text-lg font-semibold">Cost &amp; Scale</h1>
        </div>
        <p className="text-xs text-white/40">
          {loading && !lastUpdated
            ? "Loading metrics…"
            : error
              ? `Error: ${error}`
              : `Last updated ${secondsAgo}s ago`}
        </p>
      </header>

      <main className="mx-auto max-w-6xl space-y-10 px-6 py-8">
        {/* Section 1 */}
        <section>
          <h2 className="mb-4 text-sm font-medium uppercase tracking-wide text-white/50">
            Current Session Cost
          </h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              title="Session Cost"
              value={formatUsd(metrics.cost.total_session_usd)}
              subtitle={`this ingestion + ${metrics.session.chat_turns} chat turns`}
            />
            <StatCard
              title="Cost Per Creator"
              value={formatUsd(metrics.cost.cost_per_creator_usd)}
              subtitle="at current usage"
            />
            <StatCard
              title="vs GPT-4o"
              value={`${savingsPercent.toFixed(1)}% cheaper`}
              subtitle={`saved ${formatUsd(metrics.vs_openai_gpt4o.savings_usd)}`}
              valueClassName="text-green-400"
            />
            <StatCard
              title="Model Stack"
              value="Groq + BGE"
              subtitle="llama-3.1-8b + local embeddings"
            />
          </div>
        </section>

        {/* Section 2 */}
        <section>
          <h2 className="text-sm font-medium uppercase tracking-wide text-white/50">
            Projected Daily Cost
          </h2>
          <p className="mt-1 text-xs text-white/40">
            Based on actual cost per creator from this session
          </p>

          <div className="mt-6 rounded-lg border border-white/10 bg-[#1a1a1a] p-4">
            <label htmlFor="creators-slider" className="text-sm text-white/70">
              Creators per day:{" "}
              <span className="font-semibold text-white">
                {creatorsPerDay.toLocaleString()}
              </span>
            </label>
            <input
              id="creators-slider"
              type="range"
              min={0}
              max={100}
              step={0.1}
              value={sliderValue}
              onChange={(event) => setSliderValue(Number(event.target.value))}
              className="mt-3 w-full accent-green-500"
            />
            <div className="mt-1 flex justify-between text-[10px] text-white/30">
              <span>10</span>
              <span>100</span>
              <span>1K</span>
              <span>10K</span>
            </div>

            <div className="mt-4 grid gap-2 text-xs sm:grid-cols-3">
              <p>
                <span className="text-green-400">Our stack:</span>{" "}
                <span className="text-white">{formatUsd(sliderCosts.our)}/day</span>
              </p>
              <p>
                <span className="text-yellow-400">GPT-4o-mini:</span>{" "}
                <span className="text-white">{formatUsd(sliderCosts.mini)}/day</span>
              </p>
              <p>
                <span className="text-red-400">GPT-4o:</span>{" "}
                <span className="text-white">{formatUsd(sliderCosts.gpt4o)}/day</span>
              </p>
            </div>
          </div>

          <div className="mt-4 overflow-x-auto rounded-lg border border-white/10">
            <table className="w-full min-w-[640px] text-left text-sm">
              <thead>
                <tr className="border-b border-white/10 bg-[#1a1a1a] text-xs text-white/50">
                  <th className="px-4 py-3 font-medium">Scale</th>
                  <th className="px-4 py-3 font-medium">Our Stack (Groq + BGE)</th>
                  <th className="px-4 py-3 font-medium">OpenAI GPT-4o-mini</th>
                  <th className="px-4 py-3 font-medium">OpenAI GPT-4o</th>
                </tr>
              </thead>
              <tbody>
                {SCALE_TIERS.map((tier) => {
                  const ourDaily = ourPerCreator * tier;
                  const miniDaily = miniPerCreator * tier;
                  const gpt4oDaily = gpt4oPerCreatorCost * tier;
                  const isHighlighted = tier === highlightedTier;

                  return (
                    <tr
                      key={tier}
                      className={`border-b border-white/5 transition ${
                        isHighlighted ? "bg-white/10 ring-1 ring-inset ring-white/20" : "bg-[#141414]"
                      }`}
                    >
                      <td className="px-4 py-3 font-medium text-white">
                        {formatScaleLabel(tier)} creators/day
                      </td>
                      <td className="px-4 py-3 text-green-400">
                        <div>{formatUsd(ourDaily)}</div>
                        <div className="text-[10px] text-green-400/60">
                          {formatUsd(ourDaily * 30)}/mo
                        </div>
                      </td>
                      <td className="px-4 py-3 text-yellow-400">
                        <div>{formatUsd(miniDaily)}</div>
                        <div className="text-[10px] text-yellow-400/60">
                          {formatUsd(miniDaily * 30)}/mo
                        </div>
                      </td>
                      <td className="px-4 py-3 text-red-400">
                        <div>{formatUsd(gpt4oDaily)}</div>
                        <div className="text-[10px] text-red-400/60">
                          {formatUsd(gpt4oDaily * 30)}/mo
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        {/* Section 3 */}
        <section>
          <h2 className="mb-4 text-sm font-medium uppercase tracking-wide text-white/50">
            Bottleneck Analysis
          </h2>
          <div className="space-y-3">
            <BottleneckRow
              badge="Stable"
              badgeClass="border-green-500/30 bg-green-500/15 text-green-400"
              scale="Up to 100/day"
              summary="Current stack handles this comfortably. No action needed."
              risks={[]}
            />
            <BottleneckRow
              badge="Watch"
              badgeClass="border-yellow-500/30 bg-yellow-500/15 text-yellow-400"
              scale="Up to 1,000/day"
              risks={[
                {
                  label: "YouTube Data API: 10K unit quota, using ~2K/day (20%)",
                  fix: "cache API responses in Postgres by video_id",
                },
                {
                  label: "MemorySaver: session history in RAM",
                  fix: "swap to langgraph-checkpoint-redis (3-line change)",
                },
                {
                  label: "Whisper on CPU: ~60s per reel, sequential",
                  fix: "Celery background task + async job polling",
                },
              ]}
            />
            <BottleneckRow
              badge="Upgrade Needed"
              badgeClass="border-orange-500/30 bg-orange-500/15 text-orange-400"
              scale="Up to 10,000/day"
              risks={[
                {
                  label: "ChromaDB: single-node, no horizontal scaling",
                  fix: "Qdrant Cloud — same LangChain interface, 3-line swap",
                },
                {
                  label: "YouTube API quota: exceeded at this volume",
                  fix: "paid quota increase or aggressive response caching",
                },
                {
                  label: "CPU embedding bottleneck: BGE-small at volume",
                  fix: "dedicated embedding service or Cohere API ($0.0001/1K)",
                },
                {
                  label: "Single FastAPI process becomes bottleneck",
                  fix: "Gunicorn + multiple workers behind nginx",
                },
              ]}
            />
            <BottleneckRow
              badge="Full Rebuild"
              badgeClass="border-red-500/30 bg-red-500/15 text-red-400"
              scale="10,000+/day"
              risks={[
                {
                  label: "OpenAI/Groq rate limits hit",
                  fix: "request queuing + multiple API keys",
                },
                {
                  label: "Storage costs grow linearly with creators",
                  fix: "TTL on ChromaDB collections, archive old sessions",
                },
                {
                  label: "Cost becomes $80–320/day depending on chat volume",
                  fix: "Groq Batch API for non-realtime inference (50% discount)",
                },
              ]}
              summary="Everything above plus:"
            />
          </div>
        </section>

        {/* Section 4 */}
        <section>
          <h2 className="mb-4 text-sm font-medium uppercase tracking-wide text-white/50">
            Why These Models
          </h2>
          <div className="grid gap-4 lg:grid-cols-3">
            <ModelCard
              title="LLM"
              chosen="Groq llama-3.1-8b-instant"
              why="Fastest streaming latency via Groq LPU hardware. For RAG the LLM just synthesizes retrieved text — 8b handles this well. 94% cheaper than GPT-4o at this task."
              tradeoff="Weaker at complex multi-step reasoning vs 70b models."
              upgrade="llama-3.3-70b-versatile on Groq ($0.59/1M input)"
            />
            <ModelCard
              title="Embeddings"
              chosen="BAAI/bge-small-en-v1.5"
              why="Runs locally, zero API cost, downloads once (~130MB). Specifically optimized for retrieval tasks (unlike general models). normalize_embeddings=True required for correct cosine similarity."
              tradeoff="CPU bottleneck at high ingestion volume."
              upgrade="dedicated embedding service or Cohere embed-v3"
            />
            <ModelCard
              title="Vector DB"
              chosen="ChromaDB"
              why="Zero config, runs in-process, persists to disk. Right choice for development — no cloud setup slowing iteration."
              tradeoff="Single-node only, no replication, no horizontal scale."
              upgrade="Qdrant Cloud — identical LangChain interface."
            />
          </div>
        </section>
      </main>
    </div>
  );
}
