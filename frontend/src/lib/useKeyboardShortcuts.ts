import { useEffect } from "react";
import { useStore } from "@/store";

function isTypingTarget(el: EventTarget | null): boolean {
  const node = el as HTMLElement | null;
  if (!node) return false;
  return (
    node.tagName === "INPUT" ||
    node.tagName === "TEXTAREA" ||
    node.isContentEditable
  );
}

/** Global shortcuts: "/" focuses the composer, Escape closes the viewer. */
export function useKeyboardShortcuts() {
  const closeViewer = useStore((s) => s.closeViewer);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "/" && !isTypingTarget(e.target)) {
        e.preventDefault();
        document.getElementById("composer")?.focus();
      } else if (e.key === "Escape") {
        closeViewer();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [closeViewer]);
}
