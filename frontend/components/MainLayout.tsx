import ChatPanel from "@/components/ChatPanel";
import VideoPanel from "@/components/VideoPanel";
import type { IngestResponse } from "@/lib/api";

type MainLayoutProps = {
  videosData: IngestResponse | null;
  setVideosData: (data: IngestResponse | null) => void;
  setChatEnabled: (enabled: boolean) => void;
  chatEnabled: boolean;
};

export default function MainLayout({
  videosData,
  setVideosData,
  setChatEnabled,
  chatEnabled,
}: MainLayoutProps) {
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#0f0f0f]">
      <header className="flex shrink-0 items-center justify-between border-b border-white/10 px-6 py-3">
        <h1 className="text-sm font-semibold text-white">RAG Chatbot</h1>
        <a
          href="/scale"
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-white/60 transition hover:text-white"
        >
          Cost &amp; Scale ↗
        </a>
      </header>
      <div className="flex min-h-0 flex-1 overflow-hidden">
        <aside className="w-[35%] shrink-0 border-r border-white/10">
          <VideoPanel
            videosData={videosData}
            setVideosData={setVideosData}
            setChatEnabled={setChatEnabled}
          />
        </aside>
        <main className="relative w-[65%] shrink-0">
          <ChatPanel chatEnabled={chatEnabled} />
        </main>
      </div>
    </div>
  );
}
