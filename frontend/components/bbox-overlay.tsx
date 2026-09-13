"use client";

import { useMemo } from "react";
import clsx from "clsx";
import type { BoundingBox } from "@/lib/types";
import { projectBBox, type ProjectionContext } from "@/lib/bbox";

export interface HighlightBox {
  key: string;
  bbox: BoundingBox;
  /** true = the exact cited/selected object (bright); false = another
   * object on the same page, shown dimmer for context. */
  active: boolean;
  label?: string;
  onClick?: () => void;
}

/**
 * Draws one or more bbox highlights over a rendered page image.
 * Absolutely positioned within a `relative` parent sized to the image —
 * see app/source/page.tsx for how it's composed with the <img>.
 *
 * Never guesses a placement: any box `projectBBox` can't confidently
 * project (unspecified coordinate space, missing native size) is simply
 * not drawn — see lib/bbox.ts's docstring on why an absent highlight is
 * preferable to a wrong one for this product.
 */
export default function BBoxOverlay({
  boxes,
  context,
}: {
  boxes: HighlightBox[];
  context: ProjectionContext;
}) {
  const projected = useMemo(
    () =>
      boxes
        .map((b) => ({ ...b, rect: projectBBox(b.bbox, context) }))
        .filter((b): b is HighlightBox & { rect: NonNullable<ReturnType<typeof projectBBox>> } => b.rect !== null),
    [boxes, context],
  );

  if (projected.length === 0) return null;

  return (
    <div className="pointer-events-none absolute inset-0">
      {projected.map((b) => (
        <div
          key={b.key}
          onClick={b.onClick}
          className={clsx(
            "absolute rounded-sm border-2 transition-all",
            b.active
              ? "z-10 border-brass bg-brass/15 shadow-[0_0_0_3px_rgba(199,145,62,0.25)]"
              : "z-0 border-teal-dim/70 bg-teal-dim/5 hover:border-teal",
            b.onClick && "pointer-events-auto cursor-pointer",
          )}
          style={{
            left: b.rect.left,
            top: b.rect.top,
            width: Math.max(b.rect.width, 4),
            height: Math.max(b.rect.height, 4),
          }}
          title={b.label}
        >
          {b.active && b.label && (
            <span className="absolute -top-6 left-0 whitespace-nowrap rounded-sm bg-brass px-1.5 py-[1px] font-mono text-[10px] font-medium text-ink-950">
              {b.label}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
