import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { SendHorizonal, Sparkles, Square } from "lucide-react";
import { toast } from "sonner";
import { askStream, listDocuments, type ChatTurn } from "@/lib/api";
import { useActiveMessages, useStore } from "@/store";
import { Button } from "@/components/ui/button";
import { MessageBubble } from "./MessageBubble";

let idSeq = 0;
const nextId = () => `m${++idSeq}-${Date.now()}`;

const SAMPLES = [
  "Summarize the key points across these documents.",
  "What does error code E-4021 mean?",
  "List every deadline mentioned and its page.",
];

export function ChatPanel() {
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const { addMessage, updateMessage, appendToken, selectedDocIds } = useStore();
  const messages = useActiveMessages();

  // Suggested questions generated per-document at ingest; fall back to the
  // static samples when no docs (or none with suggestions) are present.
  const { data: docs = [] } = useQuery({
    queryKey: ["documents"],
    queryFn: listDocuments,
  });
  const samples = useMemo(() => {
    const pool = (
      selectedDocIds.length
        ? docs.filter((d) => selectedDocIds.includes(d.doc_id))
        : docs
    ).flatMap((d) => d.suggested_questions ?? []);
    const unique = [...new Set(pool)];
    return unique.length ? unique.slice(0, 4) : SAMPLES;
  }, [docs, selectedDocIds]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send(question: string) {
    const q = question.trim();
    if (!q || busy) return;
    setInput("");
    setBusy(true);

    // Last few completed turns give the backend context to rewrite follow-ups
    // ("how do I fix it?") into standalone search queries.
    const history: ChatTurn[] = messages
      .filter((m) => m.content && !m.streaming)
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.content.slice(0, 2000) }));

    addMessage({ id: nextId(), role: "user", content: q });
    const assistantId = nextId();
    addMessage({ id: assistantId, role: "assistant", content: "", streaming: true });

    abortRef.current = new AbortController();
    await askStream(
      {
        question: q,
        doc_ids: selectedDocIds.length ? selectedDocIds : null,
        history: history.length ? history : undefined,
        include_trace: true,
      },
      {
        onMeta: (meta) => updateMessage(assistantId, { meta }),
        onSources: (sources) => updateMessage(assistantId, { sources }),
        onTrace: (trace) => updateMessage(assistantId, { trace }),
        onToken: (t) => appendToken(assistantId, t),
        onDone: ({ grounded, citations }) =>
          updateMessage(assistantId, { grounded, citations, streaming: false }),
        onError: (err) => {
          toast.error(err.message);
          updateMessage(assistantId, {
            content: `⚠️ ${err.message}`,
            streaming: false,
            grounded: false,
          });
        },
      },
      abortRef.current.signal
    );
    // Covers aborts ("Stop") where neither onDone nor onError fires.
    updateMessage(assistantId, { streaming: false });
    setBusy(false);
  }

  return (
    <main className="flex h-full min-w-0 flex-1 flex-col">
      <div className="scroll-thin flex-1 space-y-4 overflow-y-auto px-6 py-6">
        {messages.length === 0 ? (
          <div className="mx-auto mt-16 max-w-md text-center">
            <Sparkles className="mx-auto mb-3 h-8 w-8 text-primary/60" />
            <h2 className="text-lg font-semibold">Ask your documents anything</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Answers are grounded in your PDFs with page-level citations.
            </p>
            <div className="mt-5 flex flex-col gap-2">
              {samples.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="rounded-md border border-border px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) => <MessageBubble key={m.id} msg={m} />)
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-border px-6 py-4">
        <form
          className="flex items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <textarea
            id="composer"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
            rows={1}
            placeholder="Ask a question about your documents…  (press / to focus)"
            className="max-h-40 min-h-[44px] flex-1 resize-none rounded-md border border-border bg-background px-3 py-2.5 text-sm outline-none focus:ring-2 focus:ring-ring"
          />
          {busy ? (
            <Button
              type="button"
              size="icon"
              variant="outline"
              title="Stop generating"
              onClick={() => abortRef.current?.abort()}
            >
              <Square className="h-3.5 w-3.5" />
            </Button>
          ) : (
            <Button type="submit" size="icon" disabled={!input.trim()}>
              <SendHorizonal className="h-4 w-4" />
            </Button>
          )}
        </form>
        <p className="mt-1.5 text-center text-[11px] text-muted-foreground">
          DocuMind answers only from your documents — it will say so when an
          answer isn't there.
        </p>
      </div>
    </main>
  );
}
