"use client";

import { FormEvent, useState } from "react";

import VideoCard from "@/components/VideoCard";
import { ingestVideos, type IngestResponse } from "@/lib/api";

type VideoPanelProps = {
  videosData: IngestResponse | null;
  setVideosData: (data: IngestResponse | null) => void;
  setChatEnabled: (enabled: boolean) => void;
};

export default function VideoPanel({
  videosData,
  setVideosData,
  setChatEnabled,
}: VideoPanelProps) {
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [instagramUrl, setInstagramUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAnalyze(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setLoading(true);
    setChatEnabled(false);

    try {
      const data = await ingestVideos(youtubeUrl, instagramUrl);
      setVideosData(data);
      setChatEnabled(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to ingest videos");
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setVideosData(null);
    setChatEnabled(false);
    setError(null);
  }

  return (
    <div className="flex h-full flex-col overflow-hidden p-6">
      <div className="mb-6 flex items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-white">Video Analysis</h2>
        {videosData !== null && !loading && (
          <button
            type="button"
            onClick={handleReset}
            className="rounded-lg border border-white/15 px-3 py-1.5 text-xs text-white/70 transition hover:border-white/30 hover:text-white"
          >
            Reset
          </button>
        )}
      </div>

      {loading && (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 text-white/60">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/20 border-t-white" />
          <p className="text-sm">Ingesting videos...</p>
        </div>
      )}

      {!loading && videosData === null && (
        <form onSubmit={handleAnalyze} className="flex flex-1 flex-col gap-4">
          <div>
            <label htmlFor="youtube-url" className="mb-1.5 block text-xs text-white/50">
              YouTube URL
            </label>
            <input
              id="youtube-url"
              type="url"
              value={youtubeUrl}
              onChange={(event) => setYoutubeUrl(event.target.value)}
              placeholder="YouTube URL"
              required
              className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2.5 text-sm text-white placeholder:text-white/30 outline-none focus:border-white/30"
            />
          </div>

          <div>
            <label htmlFor="instagram-url" className="mb-1.5 block text-xs text-white/50">
              Instagram Reel URL
            </label>
            <input
              id="instagram-url"
              type="url"
              value={instagramUrl}
              onChange={(event) => setInstagramUrl(event.target.value)}
              placeholder="Instagram Reel URL"
              required
              className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2.5 text-sm text-white placeholder:text-white/30 outline-none focus:border-white/30"
            />
          </div>

          {error && (
            <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={!youtubeUrl.trim() || !instagramUrl.trim()}
            className="mt-auto rounded-lg bg-white px-4 py-2.5 text-sm font-medium text-black transition hover:bg-white/90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Analyze Videos
          </button>
        </form>
      )}

      {!loading && videosData !== null && (
        <div className="flex flex-1 flex-col gap-4 overflow-y-auto lg:flex-row">
          <VideoCard video={videosData.A} label="A" />
          <VideoCard video={videosData.B} label="B" />
        </div>
      )}
    </div>
  );
}
