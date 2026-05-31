"use client";

import { useState } from "react";

import MainLayout from "@/components/MainLayout";
import type { IngestResponse } from "@/lib/api";

export default function Home() {
  const [videosData, setVideosData] = useState<IngestResponse | null>(null);
  const [chatEnabled, setChatEnabled] = useState(false);

  return (
    <MainLayout
      videosData={videosData}
      setVideosData={setVideosData}
      setChatEnabled={setChatEnabled}
      chatEnabled={chatEnabled}
    />
  );
}
