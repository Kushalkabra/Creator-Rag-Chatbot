"use client";

import { useState } from "react";

import type { EvalSummary } from "@/lib/api";

type EvalCardProps = {
  evalData: EvalSummary;
};

function passRateBadgeClass(passRate: number): string {
  if (passRate >= 75) return "bg-green-500/15 text-green-400 border-green-500/30";
  if (passRate >= 50) return "bg-yellow-500/15 text-yellow-400 border-yellow-500/30";
  return "bg-red-500/15 text-red-400 border-red-500/30";
}

export default function EvalCard({ evalData }: EvalCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="rounded-lg border border-white/10 bg-[#1a1a1a] p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-medium text-white">Retrieval Quality Check</h3>
        <span
          className={`rounded-full border px-2 py-0.5 text-xs font-medium ${passRateBadgeClass(evalData.pass_rate)}`}
        >
          {evalData.pass_rate.toFixed(1)}% pass
        </span>
      </div>

      <p className="mt-2 text-xs text-white/70">
        {evalData.passed}/{evalData.total_cases} cases passed | avg recall@4:{" "}
        {evalData.average_recall_at_k.toFixed(3)}
      </p>

      <button
        type="button"
        onClick={() => setExpanded((value) => !value)}
        className="mt-2 text-xs text-white/50 underline-offset-2 hover:text-white/80 hover:underline"
      >
        {expanded ? "Hide details" : "Show case details"}
      </button>

      {expanded && (
        <ul className="mt-2 max-h-48 space-y-2 overflow-y-auto border-t border-white/10 pt-2">
          {evalData.results.map((result) => (
            <li key={result.question} className="text-xs text-white/70">
              <div className="flex items-start gap-2">
                <span className={result.passed ? "text-green-400" : "text-red-400"}>
                  {result.passed ? "✓" : "✗"}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-white/90">{result.question}</p>
                  {result.matched_keywords.length > 0 && (
                    <p className="mt-0.5 text-green-400/80">
                      matched: {result.matched_keywords.join(", ")}
                    </p>
                  )}
                  {result.missing_keywords.length > 0 && (
                    <p className="text-red-400/80">
                      missing: {result.missing_keywords.join(", ")}
                    </p>
                  )}
                  {result.matched_keywords.length === 0 &&
                    result.missing_keywords.length === 0 && (
                      <p className="text-white/40">
                        videos: {result.video_ids_retrieved.join(", ") || "none"}
                      </p>
                    )}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}

      <p className="mt-2 border-t border-white/5 pt-2 text-[10px] text-white/40">
        Higher recall = more relevant context retrieved per question
      </p>
    </div>
  );
}
