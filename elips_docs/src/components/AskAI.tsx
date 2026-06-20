import { useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { Lightbulb } from "lucide-react";

/**
 * Floating Ask-AI bar. Routes the question into /chat with a pre-seeded
 * prompt, where the full conversation renders.
 */
export function AskAI({ context }: { context?: string }) {
  const [q, setQ] = useState("");
  const navigate = useNavigate();
  function send() {
    const value = q.trim();
    if (!value) return;
    const prefixed = context ? `[Context: ${context}]\n\n${value}` : value;
    navigate({ to: "/chat", search: { q: prefixed } as never });
  }
  return (
    <form
      className="askai-bar"
      onSubmit={(e) => {
        e.preventDefault();
        send();
      }}
      role="search"
      aria-label="Ask AI about the docs"
    >
      <Lightbulb className="askai-spark" size={16} aria-hidden />
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Ask anything about ELIPS — APIs, planner, EQL, persistence…"
        aria-label="Ask the ELIPS docs assistant"
      />
      <button type="submit" className="askai-send" disabled={!q.trim()}>
        Ask AI <span aria-hidden>→</span>
      </button>
    </form>
  );
}
