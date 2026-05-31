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
    <div className="flex h-screen overflow-hidden bg-[#0f0f0f]">
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
  );
}
