import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { FileText, ShieldCheck, ShieldAlert } from "lucide-react";
import { useStore, type ChatMessage } from "@/store";
import { Card } from "@/components/ui/card";
import { CitationChip } from "./CitationChip";
import { RetrievalInspector } from "./RetrievalInspector";
import { remarkCitations } from "@/lib/remarkCitations";
import { cn } from "@/lib/utils";

// While tokens stream in, an unclosed ``` fence would swallow the rest of the
// answer into a code block; close it provisionally.
function stableMarkdown(text: string, streaming?: boolean): string {
  if (!streaming) return text;
  const fences = (text.match(/```/g) ?? []).length;
  return fences % 2 === 1 ? `${text}\n\`\`\`` : text;
}

export function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";
  const openViewer = useStore((s) => s.openViewer);

  // Chip click: open the cited page in the source viewer (PDFs only — other
  // formats show their snippet in the citation card); fall back to scrolling
  // to the card while citations are still streaming in.
  const onCite = (n: number) => {
    const c = msg.citations?.find((c) => c.marker === n);
    if (c && c.source_type === "pdf") {
      openViewer({
        docId: c.doc_id,
        docName: c.doc_name,
        page: c.page,
        snippet: c.snippet,
      });
    } else {
      document
        .getElementById(`citation-${msg.id}-${n}`)
        ?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  const components = useMemo(
    () => ({
      cite: (props: any) => (
        <CitationChip marker={Number(props["data-marker"])} onCite={onCite} />
      ),
      a: (props: any) => (
        <a {...props} target="_blank" rel="noreferrer" className="text-primary underline" />
      ),
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [msg.citations, msg.id]
  );

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-lg rounded-br-sm bg-primary px-4 py-2.5 text-sm text-primary-foreground">
          {msg.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <Card className="max-w-[85%] px-4 py-3">
        <div className="prose-chat text-sm leading-relaxed">
          {msg.content ? (
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkCitations]}
              components={components}
            >
              {stableMarkdown(msg.content, msg.streaming)}
            </ReactMarkdown>
          ) : (
            msg.streaming && (
              <span className="text-muted-foreground">Thinking…</span>
            )
          )}
        </div>

        {!msg.streaming && msg.content && (
          <div className="mt-2 flex items-center gap-1.5 text-xs">
            {msg.grounded ? (
              <span className="flex items-center gap-1 text-emerald-600">
                <ShieldCheck className="h-3.5 w-3.5" /> Grounded in sources
              </span>
            ) : (
              <span className="flex items-center gap-1 text-amber-600">
                <ShieldAlert className="h-3.5 w-3.5" /> Not found in documents
              </span>
            )}
          </div>
        )}
      </Card>

      {!msg.streaming && msg.trace && <RetrievalInspector trace={msg.trace} />}

      {msg.citations && msg.citations.length > 0 && (
        <div className="flex max-w-[85%] flex-col gap-1.5">
          {msg.citations.map((c) => (
            <div
              key={c.marker}
              id={`citation-${msg.id}-${c.marker}`}
              role="button"
              tabIndex={0}
              onClick={() =>
                c.source_type === "pdf" &&
                openViewer({
                  docId: c.doc_id,
                  docName: c.doc_name,
                  page: c.page,
                  snippet: c.snippet,
                })
              }
              onKeyDown={(e) => e.key === "Enter" && (e.currentTarget as HTMLElement).click()}
              className={cn(
                "rounded-md border border-border bg-muted/40 px-3 py-2 text-xs transition-colors",
                c.source_type === "pdf" && "cursor-pointer hover:bg-accent/70"
              )}
            >
              <div className="flex items-center gap-1.5 font-medium">
                <span className="inline-flex h-4 min-w-4 items-center justify-center rounded bg-primary/15 px-1 text-[10px] font-semibold text-primary">
                  {c.marker}
                </span>
                <FileText className="h-3 w-3 text-muted-foreground" />
                <span className="truncate">{c.doc_name}</span>
                <span className="text-muted-foreground">
                  · {c.source_type === "pdf" ? "page" : "part"} {c.page}
                </span>
                {c.section && (
                  <span className="truncate text-muted-foreground">
                    · {c.section}
                  </span>
                )}
              </div>
              <p className="mt-1 text-muted-foreground">{c.snippet}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
