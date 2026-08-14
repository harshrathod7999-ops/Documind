import { visit, SKIP } from "unist-util-visit";

/**
 * Remark plugin that turns inline [n] citation markers into custom `cite`
 * elements so react-markdown can render them as interactive chips.
 *
 * Operating on mdast text nodes (post-parse) means markers survive inside
 * bold/italic/list items, and code blocks are naturally skipped because
 * their content is not a `text` node.
 */
const MARKER = /\[(\d{1,2})\]/g;

export function remarkCitations() {
  return (tree: any) => {
    visit(tree, "text", (node: any, index: number | undefined, parent: any) => {
      if (!parent || index === undefined) return;
      const value: string = node.value;
      if (!MARKER.test(value)) {
        MARKER.lastIndex = 0;
        return;
      }
      MARKER.lastIndex = 0;

      const children: any[] = [];
      let last = 0;
      for (const m of value.matchAll(MARKER)) {
        const at = m.index ?? 0;
        if (at > last) children.push({ type: "text", value: value.slice(last, at) });
        children.push({
          type: "citationChip",
          data: {
            hName: "cite",
            hProperties: { "data-marker": m[1] },
          },
          children: [],
        });
        last = at + m[0].length;
      }
      if (last < value.length) {
        children.push({ type: "text", value: value.slice(last) });
      }

      parent.children.splice(index, 1, ...children);
      return [SKIP, index + children.length];
    });
  };
}
