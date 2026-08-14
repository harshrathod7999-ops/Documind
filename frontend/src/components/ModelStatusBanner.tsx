import { useQuery } from "@tanstack/react-query";
import { Loader2, Sparkles } from "lucide-react";
import { health } from "@/lib/api";

// One-time "warming up" banner. The bge embedder + cross-encoder reranker
// lazy-load on the backend (~20–60s after server start). We poll /health and
// show this strip until models_ready flips true, so the first slow answer
// reads as expected setup rather than a broken app. Then it disappears.
export function ModelStatusBanner() {
  const { data, isError } = useQuery({
    queryKey: ["health"],
    queryFn: health,
    // Poll every 2s until models are ready, then stop.
    refetchInterval: (query) =>
      query.state.data?.models_ready ? false : 2000,
  });

  // Backend up AND models loaded → nothing to show.
  if (data?.models_ready) return null;

  const backendDown = isError || !data;

  return (
    <div className="flex items-center justify-center gap-2 border-b border-amber-300/60 bg-amber-50 px-4 py-2 text-sm text-amber-900 dark:border-amber-500/30 dark:bg-amber-950/40 dark:text-amber-200">
      <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
      {backendDown ? (
        <span>Connecting to the DocuMind backend…</span>
      ) : (
        <span className="flex items-center gap-1.5">
          <Sparkles className="h-3.5 w-3.5" />
          First-time setup: loading the retrieval &amp; reranking models
          (one-time, ~20–60s). Your first answer may be slower — everything is
          fast after this.
        </span>
      )}
    </div>
  );
}
