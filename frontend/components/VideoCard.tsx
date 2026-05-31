import type { VideoMeta } from "@/lib/api";

type VideoCardProps = {
  video: VideoMeta;
  label: "A" | "B";
};

function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatDuration(seconds: number): string {
  if (!seconds) return "—";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function formatDate(value: string): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function engagementColor(rate: number): string {
  if (rate > 3) return "text-green-400";
  if (rate >= 1) return "text-yellow-400";
  return "text-red-400";
}

function ExternalLinkIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      className="h-3.5 w-3.5"
      aria-hidden="true"
    >
      <path
        fillRule="evenodd"
        d="M4.25 5.5a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h8.5a.75.75 0 0 0 .75-.75v-4a.75.75 0 0 1 1.5 0v4A2.25 2.25 0 0 1 12.75 17h-8.5A2.25 2.25 0 0 1 2 14.75v-8.5A2.25 2.25 0 0 1 4.25 4h5a.75.75 0 0 1 0 1.5h-5Z"
        clipRule="evenodd"
      />
      <path
        fillRule="evenodd"
        d="M6.194 12.753a.75.75 0 0 0 1.06.053L16.5 4.44v2.81a.75.75 0 0 0 1.5 0v-4.5a.75.75 0 0 0-.75-.75h-4.5a.75.75 0 0 0 0 1.5h2.553l-9.056 8.194a.75.75 0 0 0-.053 1.06Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export default function VideoCard({ video, label }: VideoCardProps) {
  const badgeClass =
    label === "A"
      ? "bg-blue-500/20 text-blue-300"
      : "bg-orange-500/20 text-orange-300";

  const visibleHashtags = video.hashtags.slice(0, 5);

  return (
    <article className="flex flex-1 flex-col rounded-lg bg-[#1a1a1a] p-3">
      <div className="mb-2 flex items-start justify-between gap-2">
        <span className={`rounded px-2 py-0.5 text-xs font-semibold ${badgeClass}`}>
          Video {label}
        </span>
        {video.url && (
          <a
            href={video.url}
            target="_blank"
            rel="noopener noreferrer"
            aria-label={`Open Video ${label} source`}
            className="text-white/40 transition hover:text-white/70"
          >
            <ExternalLinkIcon />
          </a>
        )}
      </div>

      <h3 className="truncate text-sm font-bold text-white">{video.creator}</h3>

      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-white/70">
        <span>
          <span className="text-white/40">Views </span>
          {formatNumber(video.views)}
        </span>
        <span className="text-white/20">|</span>
        <span>
          <span className="text-white/40">Likes </span>
          {formatNumber(video.likes)}
        </span>
        <span className="text-white/20">|</span>
        <span>
          <span className="text-white/40">Comments </span>
          {formatNumber(video.comments)}
        </span>
      </div>

      <p className={`mt-3 text-xl font-semibold ${engagementColor(video.engagement_rate)}`}>
        {video.engagement_rate}% engagement
      </p>

      {visibleHashtags.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {visibleHashtags.map((tag) => (
            <span
              key={tag}
              className="rounded bg-white/10 px-1.5 py-0.5 text-[10px] text-white/60"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      <div className="mt-auto pt-3 text-[11px] text-white/40">
        {formatDate(video.upload_date)} · {formatDuration(video.duration)}
      </div>
    </article>
  );
}
