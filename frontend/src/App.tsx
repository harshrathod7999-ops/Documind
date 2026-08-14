import { Suspense, lazy } from "react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { useApplyTheme } from "@/lib/theme";
import { useKeyboardShortcuts } from "@/lib/useKeyboardShortcuts";
import { useMediaQuery } from "@/lib/useMediaQuery";
import { useStore } from "@/store";
import { AppHeader } from "./components/AppHeader";
import { DocumentsPanel } from "./components/DocumentsPanel";
import { ChatPanel } from "./components/ChatPanel";
import { ModelStatusBanner } from "./components/ModelStatusBanner";

// pdf.js is ~1 MB — load it only when a citation is first opened.
const PdfViewerPanel = lazy(() =>
  import("./components/PdfViewerPanel").then((m) => ({
    default: m.PdfViewerPanel,
  }))
);

const ViewerFallback = () => (
  <div className="space-y-3 p-3">
    <Skeleton className="h-[500px] w-full" />
  </div>
);

export default function App() {
  useApplyTheme();
  useKeyboardShortcuts();
  const viewer = useStore((s) => s.viewer);
  const closeViewer = useStore((s) => s.closeViewer);
  const docsOpen = useStore((s) => s.docsOpen);
  const setDocsOpen = useStore((s) => s.setDocsOpen);
  const isWide = useMediaQuery("(min-width: 1280px)");
  const isDesktop = useMediaQuery("(min-width: 1024px)");

  return (
    <TooltipProvider delayDuration={200}>
      <div className="flex h-screen w-screen flex-col overflow-hidden bg-background text-foreground">
        <AppHeader />
        <ModelStatusBanner />
        <div className="flex min-h-0 flex-1">
          {isDesktop && <DocumentsPanel />}
          <ChatPanel />
          {/* Side-by-side source viewer on wide screens */}
          {viewer && isWide && (
            <div className="w-[520px] shrink-0 border-l border-border">
              <Suspense fallback={<ViewerFallback />}>
                <PdfViewerPanel />
              </Suspense>
            </div>
          )}
        </div>
      </div>

      {/* Documents sidebar as an overlay on small screens */}
      <Sheet open={docsOpen && !isDesktop} onOpenChange={setDocsOpen}>
        <SheetContent side="left" className="w-80 gap-0 p-0">
          <SheetTitle className="sr-only">Documents</SheetTitle>
          <DocumentsPanel />
        </SheetContent>
      </Sheet>

      {/* Overlay viewer on narrower screens */}
      <Sheet
        open={!!viewer && !isWide}
        onOpenChange={(open) => !open && closeViewer()}
      >
        <SheetContent side="right" className="w-full gap-0 p-0 sm:max-w-xl">
          <SheetTitle className="sr-only">Source document</SheetTitle>
          <Suspense fallback={<ViewerFallback />}>
            <PdfViewerPanel />
          </Suspense>
        </SheetContent>
      </Sheet>

      <Toaster position="bottom-right" />
    </TooltipProvider>
  );
}
