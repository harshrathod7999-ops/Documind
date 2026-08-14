import { useState } from "react";
import { Check, History, Pencil, Plus, Trash2 } from "lucide-react";
import { useStore } from "@/store";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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

export function SessionsMenu() {
  const { sessions, activeSessionId, newSession, switchSession, renameSession, deleteSession } =
    useStore();
  const [renameOpen, setRenameOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [title, setTitle] = useState("");

  const list = Object.values(sessions).sort((a, b) => b.updatedAt - a.updatedAt);
  const active = activeSessionId ? sessions[activeSessionId] : null;

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="sm" title="Chat history">
            <History className="h-4 w-4" />
            <span className="hidden sm:inline">Chats</span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-64">
          <DropdownMenuItem onClick={newSession}>
            <Plus className="h-4 w-4" /> New chat
          </DropdownMenuItem>
          {list.length > 0 && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuLabel className="text-xs text-muted-foreground">
                Recent chats
              </DropdownMenuLabel>
              {list.slice(0, 12).map((s) => (
                <DropdownMenuItem key={s.id} onClick={() => switchSession(s.id)}>
                  {s.id === activeSessionId ? (
                    <Check className="h-4 w-4 shrink-0" />
                  ) : (
                    <span className="w-4 shrink-0" />
                  )}
                  <span className="truncate">{s.title}</span>
                </DropdownMenuItem>
              ))}
            </>
          )}
          {active && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={() => {
                  setTitle(active.title);
                  setRenameOpen(true);
                }}
              >
                <Pencil className="h-4 w-4" /> Rename current chat
              </DropdownMenuItem>
              <DropdownMenuItem
                variant="destructive"
                onClick={() => setDeleteOpen(true)}
              >
                <Trash2 className="h-4 w-4" /> Delete current chat
              </DropdownMenuItem>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Rename chat</DialogTitle>
          </DialogHeader>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (active) renameSession(active.id, title);
              setRenameOpen(false);
            }}
          >
            <input
              autoFocus
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring"
            />
            <DialogFooter className="mt-4">
              <Button type="submit" size="sm">
                Save
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <AlertDialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this chat?</AlertDialogTitle>
            <AlertDialogDescription>
              “{active?.title}” and its messages will be removed. Your documents
              are not affected.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => active && deleteSession(active.id)}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
