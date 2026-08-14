import { useMemo, useState } from "react";
import { ChevronDown, ChevronRight, SearchCode, Timer } from "lucide-react";
import type { RetrievalTrace, TraceEntry } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

const STAGES: { key: keyof Pick<RetrievalTrace, "dense" | "bm25" | "fused" | "reranked">; label: string; hint: string }[] = [
  { key: "reranked", label: "Reranked", hint: "cross-encoder relevance (final answer sources)" },
  { key: "fused", label: "RRF fused", hint: "reciprocal-rank fusion of dense + BM25" },
  { key: "dense", label: "Dense", hint: "bge-large cosine similarity" },
  { key: "bm25", label: "BM25", hint: "lexical match (exact terms, codes, IDs)" },
];

function StageList({
  entries,
  finalists,
  rescued,
}: {
  entries: TraceEntry[];
  finalists: Set<string>;
  rescued: Set<string>;
}) {
  if (!entries.length) {
    return <p className="px-1 py-2 text-muted-foreground">No candidates at this stage.</p>;
  }
  return (
    <ul className="space-y-0.5">
      {entries.map((e) => (
        <li
          key={`${e.rank}-${e.chunk_id}`}
          className={cn(
            "flex items-center gap-2 rounded px-1.5 py-1",
            finalists.has(e.chunk_id) && "bg-primary/10"
          )}
        >
          <span className="w-5 shrink-0 text-right font-mono text-muted-foreground">
            {e.rank}.
          </span>
          <span className="min-w-0 flex-1 truncate">
            {e.doc_name} <span className="text-muted-foreground">· p.{e.page}</span>
          </span>
          {rescued.has(e.chunk_id) && (
            <Badge variant="outline" className="border-amber-500/50 px-1.5 py-0 text-[10px] text-amber-600 dark:text-amber-400">
              rescued by BM25
            </Badge>
          )}
          <span className="shrink-0 font-mono text-muted-foreground">
            {e.score.toFixed(3)}
          </span>
        </li>
      ))}
    </ul>
  );
}

/**
 * "How this answer was found" — per-stage view of the hybrid retrieval
 * pipeline (dense + BM25 → RRF → rerank), including which final sources only
 * lexical search caught ("rescued by BM25").
 */
export function RetrievalInspector({ trace }: { trace: RetrievalTrace }) {
  const [open, setOpen] = useState(false);

  const { finalists, rescued } = useMemo(() => {
    const finalists = new Set(trace.reranked.map((e) => e.chunk_id));
    const denseIds = new Set(trace.dense.map((e) => e.chunk_id));
    const bm25Ids = new Set(trace.bm25.map((e) => e.chunk_id));
    const rescued = new Set(
      [...finalists].filter((id) => bm25Ids.has(id) && !denseIds.has(id))
    );
    return { finalists, rescued };
  }, [trace]);

  const totalMs = Object.values(trace.timings_ms).reduce((a, b) => a + b, 0);

  return (
    <div className="max-w-[85%] text-xs">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 text-muted-foreground transition-colors hover:text-foreground"
      >
        {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        <SearchCode className="h-3.5 w-3.5" />
        How this answer was found
        {rescued.size > 0 && !open && (
          <Badge variant="outline" className="ml-1 border-amber-500/50 px-1.5 py-0 text-[10px] text-amber-600 dark:text-amber-400">
            {rescued.size} rescued by BM25
          </Badge>
        )}
      </button>

      {open && (
        <div className="mt-1.5 rounded-md border border-border bg-muted/30 p-2.5">
          <div className="mb-2 flex flex-wrap items-center gap-1.5">
            <span className="text-muted-foreground">Searched:</span>
            <span className="font-medium">“{trace.query_used}”</span>
            {trace.rewritten && (
              <Badge variant="secondary" className="px-1.5 py-0 text-[10px]">
                rewritten from follow-up
              </Badge>
            )}
            <span className="ml-auto flex items-center gap-1 text-muted-foreground">
              <Timer className="h-3 w-3" />
              {Object.entries(trace.timings_ms)
                .map(([k, v]) => `${k} ${v}ms`)
                .join(" · ")}{" "}
              · total {totalMs}ms
            </span>
          </div>

          <Tabs defaultValue="reranked">
            <TabsList className="h-7">
              {STAGES.map((s) => (
                <TabsTrigger key={s.key} value={s.key} className="px-2 py-0.5 text-xs">
                  {s.label}
                </TabsTrigger>
              ))}
            </TabsList>
            {STAGES.map((s) => (
              <TabsContent key={s.key} value={s.key} className="mt-1.5">
                <p className="mb-1 px-1 text-[10px] text-muted-foreground">{s.hint}</p>
                <StageList
                  entries={trace[s.key]}
                  finalists={finalists}
                  rescued={s.key === "reranked" || s.key === "bm25" ? rescued : new Set()}
                />
              </TabsContent>
            ))}
          </Tabs>
        </div>
      )}
    </div>
  );
}
