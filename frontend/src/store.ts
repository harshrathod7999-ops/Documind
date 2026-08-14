import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AskMeta, Citation, RetrievalTrace, Source } from "@/lib/api";

export type Theme = "light" | "dark" | "system";

// What the citation PDF viewer is currently showing (null = closed).
export interface ViewerTarget {
  docId: string;
  docName: string;
  page: number;
  snippet?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  citations?: Citation[];
  grounded?: boolean;
  streaming?: boolean;
  meta?: AskMeta;
  trace?: RetrievalTrace; // not persisted (bulky) — see partialize
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
}

const MAX_SESSIONS = 30;
const MAX_MESSAGES = 200;

const newId = () =>
  typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

function makeSession(): ChatSession {
  const now = Date.now();
  return { id: newId(), title: "New chat", createdAt: now, updatedAt: now, messages: [] };
}

interface AppState {
  theme: Theme;
  setTheme: (t: Theme) => void;

  viewer: ViewerTarget | null;
  openViewer: (v: ViewerTarget) => void;
  closeViewer: () => void;

  // Documents sidebar visibility on small screens (rendered as a Sheet).
  docsOpen: boolean;
  setDocsOpen: (open: boolean) => void;

  // Which documents retrieval is scoped to (empty = all docs).
  selectedDocIds: string[];
  toggleDoc: (id: string) => void;
  clearSelection: () => void;

  // Chat sessions (persisted to localStorage).
  sessions: Record<string, ChatSession>;
  activeSessionId: string | null;
  newSession: () => void;
  switchSession: (id: string) => void;
  renameSession: (id: string, title: string) => void;
  deleteSession: (id: string) => void;

  // Message ops target the active session (created lazily on first message).
  addMessage: (m: ChatMessage) => void;
  updateMessage: (id: string, patch: Partial<ChatMessage>) => void;
  appendToken: (id: string, token: string) => void;
}

function mutateActive(
  s: Pick<AppState, "sessions" | "activeSessionId">,
  fn: (session: ChatSession) => ChatSession
): Partial<AppState> {
  let id = s.activeSessionId;
  let sessions = s.sessions;
  if (!id || !sessions[id]) {
    const fresh = makeSession();
    id = fresh.id;
    sessions = { ...sessions, [fresh.id]: fresh };
    // Evict oldest sessions beyond the cap (localStorage quota guard).
    const all = Object.values(sessions).sort((a, b) => b.updatedAt - a.updatedAt);
    if (all.length > MAX_SESSIONS) {
      sessions = Object.fromEntries(all.slice(0, MAX_SESSIONS).map((x) => [x.id, x]));
    }
  }
  const updated = fn(sessions[id]);
  return {
    activeSessionId: id,
    sessions: { ...sessions, [id]: { ...updated, updatedAt: Date.now() } },
  };
}

export const useStore = create<AppState>()(
  persist(
    (set) => ({
      theme: "system",
      setTheme: (t) => set({ theme: t }),

      viewer: null,
      openViewer: (v) => set({ viewer: v }),
      closeViewer: () => set({ viewer: null }),

      docsOpen: false,
      setDocsOpen: (open) => set({ docsOpen: open }),

      selectedDocIds: [],
      toggleDoc: (id) =>
        set((s) => ({
          selectedDocIds: s.selectedDocIds.includes(id)
            ? s.selectedDocIds.filter((d) => d !== id)
            : [...s.selectedDocIds, id],
        })),
      clearSelection: () => set({ selectedDocIds: [] }),

      sessions: {},
      activeSessionId: null,
      newSession: () => {
        const fresh = makeSession();
        set((s) => ({
          sessions: { ...s.sessions, [fresh.id]: fresh },
          activeSessionId: fresh.id,
        }));
      },
      switchSession: (id) =>
        set((s) => (s.sessions[id] ? { activeSessionId: id } : {})),
      renameSession: (id, title) =>
        set((s) => {
          const sess = s.sessions[id];
          if (!sess) return {};
          return {
            sessions: { ...s.sessions, [id]: { ...sess, title: title.trim() || sess.title } },
          };
        }),
      deleteSession: (id) =>
        set((s) => {
          const { [id]: _gone, ...rest } = s.sessions;
          return {
            sessions: rest,
            activeSessionId: s.activeSessionId === id ? null : s.activeSessionId,
          };
        }),

      addMessage: (m) =>
        set((s) =>
          mutateActive(s, (sess) => {
            const messages = [...sess.messages, m].slice(-MAX_MESSAGES);
            // First user message names the session.
            const title =
              sess.title === "New chat" && m.role === "user"
                ? m.content.slice(0, 40) + (m.content.length > 40 ? "…" : "")
                : sess.title;
            return { ...sess, messages, title };
          })
        ),
      updateMessage: (id, patch) =>
        set((s) =>
          mutateActive(s, (sess) => ({
            ...sess,
            messages: sess.messages.map((m) => (m.id === id ? { ...m, ...patch } : m)),
          }))
        ),
      appendToken: (id, token) =>
        set((s) =>
          mutateActive(s, (sess) => ({
            ...sess,
            messages: sess.messages.map((m) =>
              m.id === id ? { ...m, content: m.content + token } : m
            ),
          }))
        ),
    }),
    {
      name: "documind-app",
      version: 2,
      migrate: (state: any) => state, // v1 only persisted {theme}; shape is compatible
      partialize: (s) => ({
        theme: s.theme,
        activeSessionId: s.activeSessionId,
        // Strip bulky traces and never persist a "streaming" flag.
        sessions: Object.fromEntries(
          Object.entries(s.sessions).map(([id, sess]) => [
            id,
            {
              ...sess,
              messages: sess.messages.map(
                ({ trace: _t, streaming: _s, ...m }) => m
              ),
            },
          ])
        ),
      }),
    }
  )
);

// Convenience selector: messages of the active session.
export const useActiveMessages = () =>
  useStore((s) =>
    s.activeSessionId ? (s.sessions[s.activeSessionId]?.messages ?? EMPTY) : EMPTY
  );
const EMPTY: ChatMessage[] = [];
