import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Document, Page } from "react-pdf";
import {
  ChevronLeft,
  ChevronRight,
  FileWarning,
  Loader2,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import "@/lib/pdf";
import { documentFileUrl } from "@/lib/api";
import { useStore } from "@/store";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

function escapeHtml(t: string) {
  return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// The cited page opens with the snippet highlighted (best-effort): we try
// progressively shorter prefixes of the snippet because pdf.js text items are
// short line fragments that rarely contain the whole snippet.
function useSnippetNeedles(snippet?: string) {
  return useMemo(() => {
    if (!snippet) return [];
    const words = snippet.replace(/\s+/g, " ").trim().split(" ");
    return [8, 5, 3]
      .map((n) => words.slice(0, n).join(" "))
      .filter((s, i, arr) => s.length >= 12 && arr.indexOf(s) === i);
  }, [snippet]);
}

export function PdfViewerPanel() {
  const viewer = useStore((s) => s.viewer);
  const closeViewer = useStore((s) => s.closeViewer);

  const [numPages, setNumPages] = useState<number>(0);
  const [page, setPage] = useState(viewer?.page ?? 1);
  const [zoom, setZoom] = useState(1);
  const [failed, setFailed] = useState(false);
  const [width, setWidth] = useState(480);
  const bodyRef = useRef<HTMLDivElement>(null);

  // Follow citation clicks while the panel stays open.
  useEffect(() => {
    if (viewer) {
      setPage(viewer.page);
      setFailed(false);
    }
  }, [viewer]);

  // Fit the page to the panel width.
  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() =>
      setWidth(Math.max(240, el.clientWidth - 24))
    );
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const needles = useSnippetNeedles(viewer?.snippet);
  const textRenderer = useCallback(
    ({ str }: { str: string }) => {
      for (const needle of needles) {
        const idx = str.toLowerCase().indexOf(needle.toLowerCase());
        if (idx >= 0) {
          return (
            escapeHtml(str.slice(0, idx)) +
            `<mark class="pdf-hl">${escapeHtml(str.slice(idx, idx + needle.length))}</mark>` +
            escapeHtml(str.slice(idx + needle.length))
          );
        }
      }
      return escapeHtml(str);
    },
    [needles]
  );

  const fileUrl = useMemo(
    () => (viewer ? documentFileUrl(viewer.docId) : null),
    [viewer?.docId]
  );

  if (!viewer || !fileUrl) return null;

  return (
    <div className="flex h-full min-h-0 flex-col bg-muted/30">
      <div className="flex items-center gap-1 border-b border-border bg-card/60 px-3 py-2">
        <p className="min-w-0 flex-1 truncate text-sm font-medium" title={viewer.docName}>
          {viewer.docName}
        </p>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => setZoom((z) => Math.max(0.6, z - 0.2))}
          title="Zoom out"
        >
          <ZoomOut className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => setZoom((z) => Math.min(2.4, z + 0.2))}
          title="Zoom in"
        >
          <ZoomIn className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={closeViewer}
          title="Close viewer"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      <div ref={bodyRef} className="scroll-thin flex-1 overflow-auto p-3">
        {failed ? (
          <div className="mt-16 flex flex-col items-center gap-2 text-center text-sm text-muted-foreground">
            <FileWarning className="h-8 w-8" />
            <p>Couldn't load the original file.</p>
            <p className="text-xs">
              It may have been uploaded before file storage was enabled — re-upload it to enable the viewer.
            </p>
          </div>
        ) : (
          <Document
            file={fileUrl}
            onLoadSuccess={({ numPages }) => setNumPages(numPages)}
            onLoadError={() => setFailed(true)}
            loading={
              <div className="space-y-3">
                <Skeleton className="h-[500px] w-full" />
                <p className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading PDF…
                </p>
              </div>
            }
          >
            <Page
              pageNumber={Math.min(Math.max(page, 1), numPages || page)}
              width={width * zoom}
              customTextRenderer={textRenderer}
              className="mx-auto shadow-md"
              loading={<Skeleton className="h-[500px] w-full" />}
            />
          </Document>
        )}
      </div>

      {!failed && numPages > 0 && (
        <div className="flex items-center justify-center gap-2 border-t border-border bg-card/60 px-3 py-1.5">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            disabled={page <= 1}
            onClick={() => setPage((p) => p - 1)}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-xs text-muted-foreground">
            Page {page} / {numPages}
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            disabled={page >= numPages}
            onClick={() => setPage((p) => p + 1)}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  );
}
