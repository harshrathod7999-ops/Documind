import { useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Loader2, Upload, XCircle } from "lucide-react";
import { toast } from "sonner";
import { uploadDocuments } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const ACCEPT = ".pdf,.docx,.txt,.md";
const ACCEPT_RE = /\.(pdf|docx|txt|md)$/i;

type Status = "queued" | "processing" | "done" | "error";
interface QueueItem {
  name: string;
  status: Status;
  error?: string;
}

/**
 * Drag-and-drop / click upload with sequential per-file progress. Files are
 * sent one request each so a bad file fails alone instead of aborting the
 * whole batch, and the queue shows exactly which file is being indexed.
 */
export function UploadDropzone() {
  const qc = useQueryClient();
  const fileInput = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const busyRef = useRef(false);

  async function processFiles(files: File[]) {
    const accepted = files.filter((f) => ACCEPT_RE.test(f.name));
    const rejected = files.length - accepted.length;
    if (rejected > 0) {
      toast.error(
        `${rejected} file${rejected > 1 ? "s" : ""} skipped — supported: PDF, DOCX, TXT, MD.`
      );
    }
    if (!accepted.length || busyRef.current) return;
    busyRef.current = true;

    setQueue(accepted.map((f) => ({ name: f.name, status: "queued" })));
    for (const file of accepted) {
      setQueue((q) =>
        q.map((i) => (i.name === file.name ? { ...i, status: "processing" } : i))
      );
      try {
        await uploadDocuments([file]);
        setQueue((q) =>
          q.map((i) => (i.name === file.name ? { ...i, status: "done" } : i))
        );
        toast.success(`Indexed ${file.name}`);
        qc.invalidateQueries({ queryKey: ["documents"] });
        qc.invalidateQueries({ queryKey: ["health"] });
      } catch (err) {
        const msg = (err as Error).message;
        setQueue((q) =>
          q.map((i) =>
            i.name === file.name ? { ...i, status: "error", error: msg } : i
          )
        );
        toast.error(msg);
      }
    }
    busyRef.current = false;
    // Clear the transient queue once everything settled (leave errors visible).
    setTimeout(
      () => setQueue((q) => q.filter((i) => i.status === "error")),
      4000
    );
  }

  const busy = queue.some((i) => i.status === "queued" || i.status === "processing");

  return (
    <div className="px-4 py-3">
      <input
        ref={fileInput}
        type="file"
        accept={ACCEPT}
        multiple
        hidden
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length) processFiles(files);
          e.target.value = "";
        }}
      />
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          processFiles(Array.from(e.dataTransfer.files));
        }}
        className={cn(
          "rounded-md border border-dashed p-3 text-center transition-colors",
          dragOver ? "border-primary bg-primary/5" : "border-border"
        )}
      >
        <Button
          className="w-full"
          onClick={() => fileInput.current?.click()}
          disabled={busy}
        >
          {busy ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Upload className="h-4 w-4" />
          )}
          {busy ? "Indexing…" : "Upload documents"}
        </Button>
        <p className="mt-1.5 text-[11px] text-muted-foreground">
          or drop PDF, DOCX, TXT, MD here
        </p>
      </div>

      {queue.length > 0 && (
        <ul className="mt-2 space-y-1">
          {queue.map((i) => (
            <li key={i.name} className="flex items-center gap-1.5 text-xs">
              {i.status === "processing" && (
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-muted-foreground" />
              )}
              {i.status === "queued" && (
                <span className="h-3.5 w-3.5 shrink-0 rounded-full border border-border" />
              )}
              {i.status === "done" && (
                <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-500" />
              )}
              {i.status === "error" && (
                <XCircle className="h-3.5 w-3.5 shrink-0 text-destructive" />
              )}
              <span className="min-w-0 flex-1 truncate" title={i.error ?? i.name}>
                {i.name}
              </span>
              <span className="shrink-0 text-muted-foreground">
                {i.status === "processing"
                  ? "indexing"
                  : i.status === "error"
                    ? "failed"
                    : i.status}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
