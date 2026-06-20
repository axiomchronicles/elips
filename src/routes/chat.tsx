import { createFileRoute, Link } from "@tanstack/react-router";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Lightbulb, Send, Square, RotateCcw, BookOpen, Sparkle } from "lucide-react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { docs as DOC_ENTRIES } from "../lib/content";

export const Route = createFileRoute("/chat")({
  validateSearch: (s: Record<string, unknown>) => ({
    q: typeof s.q === "string" ? s.q : undefined,
  }),
  head: () => ({
    meta: [
      { title: "Ask AI — ELIPS Docs" },
      { name: "description", content: "Have a full conversation with the ELIPS documentation assistant. Ask about the planner, persistence, EQL, or any API." },
      { property: "og:title", content: "Ask AI — ELIPS" },
      { property: "og:description", content: "Conversational assistant for the ELIPS docs." },
      { property: "og:url", content: "/chat" },
    ],
    links: [{ rel: "canonical", href: "/chat" }],
  }),
  component: ChatPage,
});

const SUGGESTIONS = [
  { q: "Explain the hybrid retrieval pipeline end-to-end.", tag: "concepts", to: "/docs/architecture" },
  { q: "Show me a minimal Python example with document attachments.", tag: "python", to: "/docs/python-sdk" },
  { q: "How does the WAL recover from a crash mid-write?", tag: "storage", to: "/docs/storage" },
  { q: "What strategies can the planner emit, and when?", tag: "planner", to: "/docs/core-concepts" },
  { q: "Write an EQL query that finds the 10 nearest design notes from 2024.", tag: "eql", to: "/docs/eql" },
  { q: "Compare HNSW and the exact index — when should I pick which?", tag: "algorithms", to: "/docs/algorithms" },
  { q: "How does the GPU engine fall back to CPU on device-lost?", tag: "gpu", to: "/docs/gpu-engine" },
  { q: "Walk me through opening a database in read-only mode.", tag: "ops", to: "/docs/cli" },
];

const KEYWORD_MAP: Array<{ keys: RegExp; path: string }> = [
  { keys: /\b(wal|recovery|crash|checkpoint|snapshot|segment)\b/i, path: "/docs/storage" },
  { keys: /\b(hnsw|ann|graph|ef_search|recall|hybrid|metric)\b/i, path: "/docs/algorithms" },
  { keys: /\b(gpu|cuda|hip|metal|cagra|ivf|pq|batcher)\b/i, path: "/docs/gpu-engine" },
  { keys: /\b(eql|seek|query language|grammar|where|nearest)\b/i, path: "/docs/eql" },
  { keys: /\b(python|engine|arena|pybind)\b/i, path: "/docs/python-sdk" },
  { keys: /\b(c\+\+|cpp|elipsinstance|vault|cmake)\b/i, path: "/docs/cpp-sdk" },
  { keys: /\b(lock|flock|writer|reader|concurrency)\b/i, path: "/docs/internals/lock-manager" },
  { keys: /\b(transaction|commit|rollback|atomic)\b/i, path: "/docs/internals/transaction-engine" },
  { keys: /\b(planner|plan_seek|strategy|candidate)\b/i, path: "/docs/core-concepts" },
  { keys: /\b(cli|elips command|repl)\b/i, path: "/docs/cli" },
];

function relatedDocs(text: string, limit = 3) {
  const seen = new Set<string>();
  const hits: typeof DOC_ENTRIES = [];
  for (const { keys, path } of KEYWORD_MAP) {
    if (keys.test(text) && !seen.has(path)) {
      const doc = DOC_ENTRIES.find((d) => d.path === path);
      if (doc) {
        hits.push(doc);
        seen.add(path);
      }
      if (hits.length >= limit) break;
    }
  }
  return hits;
}

function ChatPage() {
  const search = Route.useSearch();
  const [input, setInput] = useState("");
  const taRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const { messages, sendMessage, status, error, stop, setMessages } = useChat({
    transport: new DefaultChatTransport({ api: "/api/chat" }),
  });

  useEffect(() => {
    if (search.q) {
      sendMessage({ text: search.q });
      window.history.replaceState({}, "", "/chat");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, status]);

  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 220) + "px";
  }, [input]);

  useEffect(() => {
    if (status === "ready" || status === "error") taRef.current?.focus();
  }, [status]);

  const busy = status === "submitted" || status === "streaming";

  function submit() {
    const text = input.trim();
    if (!text || busy) return;
    sendMessage({ text });
    setInput("");
  }

  return (
    <div className="min-h-[calc(100vh-64px)] px-4 sm:px-6">
      <div className="max-w-[860px] mx-auto w-full pt-10 sm:pt-14 pb-16">
        {/* Header */}
        <header>
          <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-4 sm:flex sm:flex-wrap sm:justify-between">
            <div className="min-w-0">
              <div className="eyebrow mb-3 flex items-center gap-2">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                Assistant · live
              </div>
              <h1 className="display-lg text-ink">
                Ask the docs{" "}
                <span className="hand-underline handwritten text-primary" style={{ fontSize: "1.15em" }}>
                  anything
                </span>
                .
              </h1>
              <p className="lede handwritten-lede mt-4 max-w-[560px]">
                A conversational layer over the ELIPS documentation. Knows
                the planner, persistence model, EQL grammar, both SDKs, and
                the GPU engine — and links back to the canonical page.
              </p>
            </div>
            {messages.length > 0 && (
              <button
                onClick={() => setMessages([])}
                className="btn btn-ghost text-[13px] h-9 shrink-0"
                aria-label="Start a new conversation"
              >
                <RotateCcw size={14} aria-hidden /> New chat
              </button>
            )}
          </div>
        </header>

        {/* Transcript */}
        <div className="mt-10 space-y-6">
          {messages.length === 0 && (
            <div>
              <div className="eyebrow mb-4 flex items-center gap-2">
                <Sparkle size={12} className="text-primary" /> Try asking
              </div>
              <div className="grid sm:grid-cols-2 gap-3">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s.q}
                    onClick={() => sendMessage({ text: s.q })}
                    className="group text-left rounded-xl border border-hairline hover:border-ink hover:-translate-y-0.5 bg-surface p-4 transition"
                  >
                    <div className="flex items-start gap-3">
                      <Lightbulb size={18} className="text-primary mt-0.5 shrink-0 group-hover:scale-110 transition" aria-hidden />
                      <div className="min-w-0">
                        <div className="text-[14.5px] text-ink leading-snug">{s.q}</div>
                        <div className="text-[10.5px] uppercase tracking-wider text-muted mt-1.5 flex items-center gap-2">
                          <span>{s.tag}</span>
                          <span className="opacity-50">·</span>
                          <span className="opacity-70 group-hover:text-primary transition">{s.to.replace("/docs/", "")}</span>
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
              <div className="mt-8 text-[13px] text-muted">
                Tip: press{" "}
                <kbd className="font-mono text-[11px] px-1.5 py-0.5 border border-hairline rounded bg-canvas-soft">⌘</kbd>{" "}+{" "}
                <kbd className="font-mono text-[11px] px-1.5 py-0.5 border border-hairline rounded bg-canvas-soft">↵</kbd>{" "}
                to send. Answers are generated and may contain mistakes — verify against the{" "}
                <Link to="/docs" className="text-ink underline underline-offset-2">docs</Link>.
              </div>
            </div>
          )}

          {messages.map((m: typeof messages[number]) => {
            const text = m.parts
              .filter((p): p is Extract<typeof p, { type: "text" }> => p.type === "text")
              .map((p) => p.text)
              .join("");
            const isUser = m.role === "user";
            const refs = !isUser ? relatedDocs(text) : [];
            return (
              <div key={m.id} className={`chat-row ${isUser ? "chat-row-user" : ""}`}>
                {!isUser && (
                  <span className="chat-avatar chat-avatar-ai" aria-hidden>
                    E
                  </span>
                )}
                <div className={`chat-msg ${isUser ? "chat-msg-user" : "chat-msg-asst"} min-w-0`}>
                  {isUser ? (
                    <p className="m-0 whitespace-pre-wrap break-words">{text}</p>
                  ) : (
                    <>
                      <ReactMarkdown
                        components={{
                          code({ inline, className, children, ...props }: any) {
                            const match = /language-(\w+)/.exec(className || "");
                            const code = String(children).replace(/\n$/, "");
                            if (!inline && match) {
                              return (
                                <div className="my-3 rounded-xl overflow-hidden border border-hairline-strong">
                                  <div className="flex items-center justify-between px-3 py-1.5 text-[10.5px] uppercase tracking-wider text-muted bg-canvas-soft">
                                    <span>{match[1]}</span>
                                    <button
                                      type="button"
                                      onClick={() => navigator.clipboard?.writeText(code)}
                                      className="hover:text-ink transition"
                                    >
                                      copy
                                    </button>
                                  </div>
                                  <SyntaxHighlighter
                                    language={match[1]}
                                    style={oneDark}
                                    customStyle={{ margin: 0, padding: "14px 16px", fontSize: 13, background: "oklch(0.18 0.012 260)" }}
                                    PreTag="div"
                                  >
                                    {code}
                                  </SyntaxHighlighter>
                                </div>
                              );
                            }
                            return <code className={className} {...props}>{children}</code>;
                          },
                          a({ href, children, ...props }: any) {
                            const isInternal = href?.startsWith("/");
                            if (isInternal) {
                              return <Link to={href as never} {...props}>{children}</Link>;
                            }
                            return <a href={href} target="_blank" rel="noreferrer" {...props}>{children}</a>;
                          },
                        }}
                      >
                        {text || "…"}
                      </ReactMarkdown>
                      {refs.length > 0 && (
                        <div className="mt-4 pt-3 border-t border-hairline-soft">
                          <div className="eyebrow mb-2 flex items-center gap-1.5">
                            <BookOpen size={11} /> References
                          </div>
                          <ul className="flex flex-wrap gap-2 list-none p-0 m-0">
                            {refs.map((r) => (
                              <li key={r.path}>
                                <Link
                                  to={r.path as never}
                                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-hairline hover:border-ink bg-surface text-[12px] text-ink no-underline transition"
                                >
                                  <span className="text-muted text-[10px]">{r.group.toLowerCase()}/</span>
                                  {r.title}
                                </Link>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </>
                  )}
                </div>
                {isUser && (
                  <span className="chat-avatar handwritten" aria-hidden>
                    you
                  </span>
                )}
              </div>
            );
          })}

          {busy && (
            <div className="chat-row">
              <span className="chat-avatar chat-avatar-ai" aria-hidden>E</span>
              <div className="chat-msg chat-msg-asst inline-flex items-center" aria-live="polite">
                <span className="typing-dot" />
                <span className="typing-dot" />
                <span className="typing-dot" />
                <span className="ml-3 text-[14px] handwritten text-muted">scribbling a reply…</span>
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-xl border border-hairline-strong bg-surface p-4 text-[14px] text-ink">
              <strong>Couldn't reach the assistant.</strong>
              <p className="text-muted mt-1 text-[13px]">{error.message}</p>
            </div>
          )}

          <div ref={endRef} />
        </div>

        {/* Inline composer — sits in flow, not floating */}
        <form
          className="mt-10"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <div className="askai-bar askai-bar-inline">
            <Lightbulb className="askai-spark" size={18} aria-hidden />
            <textarea
              ref={taRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit();
                }
              }}
              placeholder="Ask about ELIPS — vaults, planner, EQL, persistence, GPU…"
              rows={1}
              autoFocus
              aria-label="Message"
            />
            {busy ? (
              <button
                type="button"
                onClick={() => stop()}
                className="askai-send"
                style={{ background: "var(--color-primary)" }}
                aria-label="Stop generating"
              >
                <Square size={12} fill="currentColor" /> Stop
              </button>
            ) : (
              <button
                type="submit"
                className="askai-send"
                disabled={!input.trim()}
                aria-label="Send message"
              >
                <Send size={13} /> Send
              </button>
            )}
          </div>
          <div className="text-center text-[11px] text-muted mt-3 handwritten" style={{ fontSize: 15 }}>
            ELIPS Assistant · grounded in the open-source docs · powered by Lovable AI
          </div>
        </form>
      </div>
    </div>
  );
}
