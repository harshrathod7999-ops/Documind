import { useQuery } from "@tanstack/react-query";
import { PanelLeft, RotateCcw, Sparkles } from "lucide-react";
import { health } from "@/lib/api";
import { useStore } from "@/store";
import { Button } from "@/components/ui/button";
import { SessionsMenu } from "./SessionsMenu";
import { ThemeToggle } from "./ThemeToggle";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

function HealthDot() {
  const { data, isError } = useQuery({
    queryKey: ["health"],
    queryFn: health,
    refetchInterval: (query) =>
      query.state.data?.models_ready ? 30_000 : 2_000,
  });

  const down = isError || !data;
  const ready = data?.models_ready && data?.qdrant_ok;
  const label = down
    ? "Backend unreachable"
    : ready
      ? `Online · ${data.indexed_chunks} chunks indexed`
      : !data.qdrant_ok
        ? "Vector store (Qdrant) unreachable"
        : "Warming up retrieval models…";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className="flex cursor-default items-center gap-1.5 rounded-full border border-border px-2.5 py-1 text-xs text-muted-foreground"
          aria-label={label}
        >
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              down ? "bg-red-500" : ready ? "bg-emerald-500" : "bg-amber-500 animate-pulse"
            )}
          />
          {down ? "Offline" : ready ? "Online" : "Warming up"}
        </span>
      </TooltipTrigger>
      <TooltipContent side="bottom">{label}</TooltipContent>
    </Tooltip>
  );
}

export function AppHeader({ children }: { children?: React.ReactNode }) {
  const newSession = useStore((s) => s.newSession);
  const hasMessages = useStore(
    (s) =>
      !!s.activeSessionId &&
      (s.sessions[s.activeSessionId]?.messages.length ?? 0) > 0
  );

  const setDocsOpen = useStore((s) => s.setDocsOpen);

  return (
    <header className="flex items-center gap-3 border-b border-border bg-card/40 px-4 py-2.5">
      <Button
        variant="ghost"
        size="icon"
        className="lg:hidden"
        onClick={() => setDocsOpen(true)}
        title="Documents"
      >
        <PanelLeft className="h-4 w-4" />
      </Button>
      <div className="flex items-center gap-2">
        <Sparkles className="h-5 w-5 text-primary" />
        <h1 className="text-base font-semibold tracking-tight">DocuMind</h1>
      </div>
      <span className="hidden text-xs text-muted-foreground md:inline">
        hybrid retrieval · reranked · page-cited
      </span>
      <div className="ml-auto flex items-center gap-1.5">
        <HealthDot />
        {children}
        {hasMessages && (
          <Button variant="ghost" size="sm" onClick={newSession}>
            <RotateCcw className="h-3.5 w-3.5" /> New chat
          </Button>
        )}
        <SessionsMenu />
        <ThemeToggle />
      </div>
    </header>
  );
}
