export function CitationChip({
  marker,
  onCite,
}: {
  marker: number;
  onCite: (n: number) => void;
}) {
  return (
    <button
      onClick={() => onCite(marker)}
      title={`Open source [${marker}]`}
      className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-primary/15 px-1 align-text-top text-[11px] font-semibold text-primary hover:bg-primary/25"
    >
      {marker}
    </button>
  );
}
