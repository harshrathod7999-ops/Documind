import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileText, FileType2, FileCode2, Trash2, Files } from "lucide-react";
import { toast } from "sonner";
import {
  deleteDocument,
  listDocuments,
  type DocumentInfo,
  type SourceType,
} from "@/lib/api";
import { useStore } from "@/store";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { UploadDropzone } from "./UploadDropzone";
import { cn } from "@/lib/utils";

const TYPE_ICON: Record<SourceType, typeof FileText> = {
  pdf: FileText,
  docx: FileType2,
  text: FileCode2,
};

export function DocumentsPanel() {
  const qc = useQueryClient();
  const { selectedDocIds, toggleDoc, clearSelection } = useStore();
  const [toDelete, setToDelete] = useState<DocumentInfo | null>(null);

  const { data: docs = [], isLoading } = useQuery({
    queryKey: ["documents"],
    queryFn: listDocuments,
  });

  const remove = useMutation({
    mutationFn: deleteDocument,
    onSuccess: (_d, docId) => {
      qc.invalidateQueries({ queryKey: ["documents"] });
      if (selectedDocIds.includes(docId)) toggleDoc(docId);
      toast.success("Document deleted.");
    },
    onError: (err) => toast.error(`Delete failed: ${(err as Error).message}`),
  });

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col border-r border-border bg-card/40">
      <div className="flex items-center gap-2 border-b border-border px-4 py-3">
        <Files className="h-4 w-4 text-muted-foreground" />
        <h2 className="text-sm font-semibold">Documents</h2>
        {selectedDocIds.length > 0 && (
          <button
            onClick={clearSelection}
            className="ml-auto text-xs text-muted-foreground hover:text-foreground"
          >
            Clear filter ({selectedDocIds.length})
          </button>
        )}
      </div>

      <UploadDropzone />

      <div className="scroll-thin flex-1 overflow-y-auto px-2 pb-4">
        {isLoading && (
          <div className="space-y-2 px-2">
            <Skeleton className="h-14 w-full" />
            <Skeleton className="h-14 w-full" />
            <Skeleton className="h-14 w-full" />
          </div>
        )}
        {!isLoading && docs.length === 0 && (
          <p className="px-2 text-sm text-muted-foreground">
            No documents yet. Upload a PDF, DOCX, TXT or MD file to start
            asking questions.
          </p>
        )}
        <ul className="space-y-1">
          {docs.map((d) => {
            const active = selectedDocIds.includes(d.doc_id);
            const Icon = TYPE_ICON[d.source_type] ?? FileText;
            const unit = d.source_type === "pdf" ? "pages" : "parts";
            return (
              <li
                key={d.doc_id}
                onClick={() => toggleDoc(d.doc_id)}
                className={cn(
                  "group cursor-pointer rounded-md border px-3 py-2 text-sm transition-colors",
                  active
                    ? "border-primary bg-accent"
                    : "border-transparent hover:bg-accent/60"
                )}
              >
                <div className="flex items-start gap-2">
                  <Icon className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-medium" title={d.doc_name}>
                      {d.doc_name}
                    </p>
                    {d.summary && (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <p className="mt-0.5 line-clamp-2 text-xs text-muted-foreground">
                            {d.summary}
                          </p>
                        </TooltipTrigger>
                        <TooltipContent side="right" className="max-w-64">
                          {d.summary}
                        </TooltipContent>
                      </Tooltip>
                    )}
                    <div className="mt-1 flex items-center gap-2">
                      <Badge variant="secondary">
                        {d.pages} {unit}
                      </Badge>
                      <Badge variant="secondary">{d.chunks} chunks</Badge>
                    </div>
                  </div>
                  <button
                    title="Delete"
                    onClick={(e) => {
                      e.stopPropagation();
                      setToDelete(d);
                    }}
                    className="opacity-0 transition-opacity group-hover:opacity-100"
                  >
                    <Trash2 className="h-4 w-4 text-muted-foreground hover:text-destructive" />
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      </div>

      {selectedDocIds.length > 0 && (
        <p className="border-t border-border px-4 py-2 text-xs text-muted-foreground">
          Answers scoped to {selectedDocIds.length} selected document
          {selectedDocIds.length > 1 ? "s" : ""}.
        </p>
      )}

      <AlertDialog open={!!toDelete} onOpenChange={(o) => !o && setToDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete “{toDelete?.doc_name}”?</AlertDialogTitle>
            <AlertDialogDescription>
              Its {toDelete?.chunks} indexed chunks and the stored file will be
              removed. This can't be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => toDelete && remove.mutate(toDelete.doc_id)}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </aside>
  );
}
