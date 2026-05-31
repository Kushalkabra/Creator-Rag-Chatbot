"use client";

import {
  FormEvent,
  KeyboardEvent,
  useEffect,
  useRef,
  useState,
} from "react";

import { streamChat } from "@/lib/api";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
};

const SUGGESTED_QUESTIONS = [
  "Why did Video A get more engagement than Video B?",
  "Compare the hooks in the first 5 seconds",
  "What's the engagement rate of each video?",
  "Suggest improvements for Video B based on Video A",
] as const;

type ChatPanelProps = {
  chatEnabled: boolean;
};

function LockIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="currentColor"
      className="h-6 w-6"
      aria-hidden="true"
    >
      <path
        fillRule="evenodd"
        d="M12 1.5a5.25 5.25 0 0 0-5.25 5.25v3a3 3 0 0 0-3 3v6.75a3 3 0 0 0 3 3h10.5a3 3 0 0 0 3-3v-6.75a3 3 0 0 0-3-3v-3c0-2.9-2.35-5.25-5.25-5.25Zm3.75 8.25v-3a3.75 3.75 0 1 0-7.5 0v3h7.5Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export default function ChatPanel({ chatEnabled }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [networkError, setNetworkError] = useState<string | null>(null);
  const [sessionId] = useState(() => crypto.randomUUID());

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const historyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage(messageText?: string) {
    const trimmed = (messageText ?? input).trim();
    if (!trimmed || isStreaming || !chatEnabled) return;

    setInput("");
    setIsStreaming(true);
    setNetworkError(null);

    setMessages((prev) => [
      ...prev,
      { role: "user", content: trimmed },
      { role: "assistant", content: "", sources: [] },
    ]);

    try {
      await streamChat(
        trimmed,
        sessionId,
        (token) => {
          setMessages((prev) => {
            const next = [...prev];
            const lastIndex = next.length - 1;
            const last = next[lastIndex];
            if (last?.role !== "assistant") return prev;

            next[lastIndex] = {
              ...last,
              content: last.content + token,
            };
            return next;
          });
        },
        (sources) => {
          setMessages((prev) => {
            const next = [...prev];
            const lastIndex = next.length - 1;
            const last = next[lastIndex];
            if (last?.role !== "assistant") return prev;

            next[lastIndex] = {
              ...last,
              sources,
            };
            return next;
          });
          setIsStreaming(false);
        }
      );
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Network error while streaming response";

      setMessages((prev) => {
        const next = [...prev];
        const lastIndex = next.length - 1;
        const last = next[lastIndex];
        if (last?.role === "assistant") {
          const interruptedSuffix = " (response interrupted)";
          const baseContent = last.content || "";
          next[lastIndex] = {
            ...last,
            content: baseContent
              ? `${baseContent}${interruptedSuffix}`
              : interruptedSuffix.trim(),
          };
        }
        return next;
      });
      setNetworkError(errorMessage);
      setIsStreaming(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendMessage();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  function handleSuggestionClick(question: string) {
    void sendMessage(question);
  }

  return (
    <div className="relative flex h-full flex-col bg-[#0f0f0f]">
      {!chatEnabled && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-[#0f0f0f]/90 px-6 text-center">
          <LockIcon />
          <p className="text-sm text-white/60">
            Ingest two videos first to start chatting
          </p>
        </div>
      )}

      {networkError && (
        <div className="absolute left-4 right-4 top-4 z-20 flex items-start justify-between gap-3 rounded-lg border border-red-500/30 bg-red-950/90 px-3 py-2 text-sm text-red-200">
          <span>{networkError}</span>
          <button
            type="button"
            onClick={() => setNetworkError(null)}
            className="shrink-0 text-red-300/80 hover:text-red-100"
            aria-label="Dismiss error"
          >
            ✕
          </button>
        </div>
      )}

      <div
        ref={historyRef}
        className="flex-1 overflow-y-auto px-4 py-4"
      >
        {messages.length === 0 && (
          <p className="text-center text-sm text-white/30">
            Pick a suggestion below or type your own question.
          </p>
        )}

        <div className="flex flex-col gap-3">
          {messages.map((message, index) => (
            <div
              key={`${message.role}-${index}`}
              className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-lg px-3 py-2 ${
                  message.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-[#1a1a1a] text-white/90"
                }`}
              >
                <p
                  className={`whitespace-pre-wrap text-sm leading-relaxed ${
                    message.role === "assistant" ? "font-mono" : ""
                  }`}
                >
                  {message.content}
                  {message.role === "assistant" &&
                    isStreaming &&
                    index === messages.length - 1 &&
                    !message.content && (
                      <span className="text-white/40">Thinking…</span>
                    )}
                </p>

                {message.role === "assistant" &&
                  message.sources &&
                  message.sources.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {message.sources.map((source) => (
                        <span
                          key={source}
                          className="rounded bg-white/10 px-1.5 py-0.5 text-[10px] text-white/50"
                        >
                          {source}
                        </span>
                      ))}
                    </div>
                  )}
              </div>
            </div>
          ))}
        </div>

        <div ref={messagesEndRef} />
      </div>

      {messages.length === 0 && chatEnabled && (
        <div className="shrink-0 border-t border-white/10 px-4 pt-4">
          <div className="grid grid-cols-2 gap-2">
            {SUGGESTED_QUESTIONS.map((question) => (
              <button
                key={question}
                type="button"
                onClick={() => handleSuggestionClick(question)}
                disabled={isStreaming || !chatEnabled}
                className="rounded-full border border-white/15 bg-[#1a1a1a] px-3 py-2 text-left text-xs leading-snug text-white/70 transition hover:border-white/30 hover:bg-white/5 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      <form
        onSubmit={handleSubmit}
        className="flex shrink-0 gap-2 border-t border-white/10 bg-[#0f0f0f] p-4"
      >
        <input
          type="text"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about the videos…"
          disabled={isStreaming || !chatEnabled}
          className="flex-1 rounded-lg border border-white/10 bg-[#1a1a1a] px-3 py-2.5 text-sm text-white placeholder:text-white/30 outline-none focus:border-white/30 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={isStreaming || !input.trim() || !chatEnabled}
          className="rounded-lg bg-white px-4 py-2.5 text-sm font-medium text-black transition hover:bg-white/90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Send
        </button>
      </form>
    </div>
  );
}
