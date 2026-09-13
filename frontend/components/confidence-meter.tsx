import { formatConfidence } from "@/lib/format";
import clsx from "clsx";

const TICKS = 10;

export default function ConfidenceMeter({
  confidence,
  source,
  className,
}: {
  confidence: number;
  source: "native" | "ocr";
  className?: string;
}) {
  const filled = Math.round(confidence * TICKS);
  return (
    <div className={clsx("flex items-center gap-1.5", className)}>
      <div className="flex items-center gap-[2px]" aria-hidden>
        {Array.from({ length: TICKS }, (_, i) => (
          <span
            key={i}
            className={clsx(
              "h-2.5 w-[3px] rounded-full transition-colors",
              i < filled
                ? source === "native"
                  ? "bg-brass"
                  : "bg-teal"
                : "bg-ink-600",
            )}
          />
        ))}
      </div>
      <span className="font-mono text-[11px] tabular-nums text-muted">
        {formatConfidence(confidence)}
      </span>
    </div>
  );
}
