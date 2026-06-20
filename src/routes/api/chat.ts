import { createFileRoute } from "@tanstack/react-router";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { streamText, convertToModelMessages, type UIMessage } from "ai";
import { docs } from "@/lib/content";

const SYSTEM = `You are the ELIPS documentation assistant. ELIPS is an
embedded, in-process vector and document retrieval engine written in C++23
with native Python bindings. It exposes:

- An ElipsInstance / Database / Engine handle (open / connect)
- Named vaults (partitions) per database
- Records with id, vector, payload, optional document attachment, chunk info, and embedding lineage
- A planner that emits a QueryPlan with strategies: ann_index, exact_candidates, full_scan, text_probe, hybrid_fusion
- A Filter predicate tree shared by the fluent builder and the EQL parser
- Hybrid retrieval (seek_text / seek_hybrid) that fuses lexical overlap with vector distance
- Two index families behind IndexPort: HNSW graph and exact, plus a GPU port
- Write-ahead log with CRC32C framing, segmented persistence, snapshot fallback, advisory file locking (single-writer, multi-reader)
- A Python "modern" wrapper (Engine / Arena / RecordInput / Row / Hit) on top of the low-level bindings
- An elips CLI: info, vaults, stats, verify, query, checkpoint, import, export, bench
- EQL — ELIPS Query Language — that parses to the same Filter / QueryPlan

Be concise and technical. Prefer short paragraphs and code samples in
Python or C++. When you suggest a documentation page, use the relative
path (e.g. /docs/architecture). Never invent APIs that are not listed
above; if you are not sure, say so and point to the closest doc page.

Available doc pages:
${docs.map((d) => `- ${d.path} — ${d.title}: ${d.description}`).join("\n")}
`;

export const Route = createFileRoute("/api/chat")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const key = process.env.LOVABLE_API_KEY;
        if (!key) {
          return new Response("Missing LOVABLE_API_KEY", { status: 500 });
        }
        let body: { messages: UIMessage[] };
        try {
          body = (await request.json()) as { messages: UIMessage[] };
        } catch {
          return new Response("Invalid JSON", { status: 400 });
        }

        const gateway = createOpenAICompatible({
          name: "lovable",
          baseURL: "https://ai.gateway.lovable.dev/v1",
          headers: { "Lovable-API-Key": key },
        });

        try {
          const result = streamText({
            model: gateway("google/gemini-3-flash-preview"),
            system: SYSTEM,
            messages: await convertToModelMessages(body.messages ?? []),
          });
          return result.toUIMessageStreamResponse();
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          return new Response(`AI error: ${msg}`, { status: 500 });
        }
      },
    },
  },
});
