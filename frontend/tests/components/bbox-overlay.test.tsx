import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import BBoxOverlay, { type HighlightBox } from "@/components/bbox-overlay";
import type { BoundingBox } from "@/lib/types";
import type { ProjectionContext } from "@/lib/bbox";

// A well-formed context: rendered image is 1000x1000px, native (PDF
// point) page is 500x500 -> projectBBox should scale everything by 2x.
const goodContext: ProjectionContext = {
  renderedWidth: 1000,
  renderedHeight: 1000,
  nativeWidth: 500,
  nativeHeight: 500,
};

function pdfBox(overrides: Partial<BoundingBox> = {}): BoundingBox {
  return { x0: 10, y0: 20, x1: 60, y1: 70, coordinate_space: "pdf_points", ...overrides };
}

describe("BBoxOverlay", () => {
  it("renders nothing when boxes is empty", () => {
    const { container } = render(<BBoxOverlay boxes={[]} context={goodContext} />);
    expect(container.firstChild).toBeNull();
  });

  it("projects a valid pdf_points bbox onto the rendered image using the native/rendered scale factor", () => {
    const boxes: HighlightBox[] = [{ key: "a", bbox: pdfBox(), active: true, label: "Fig. 1" }];
    render(<BBoxOverlay boxes={boxes} context={goodContext} />);

    // scaleX = scaleY = 1000/500 = 2 -> left=20, top=40, width=100, height=100
    const label = screen.getByText("Fig. 1");
    const box = label.parentElement as HTMLElement;
    expect(box.style.left).toBe("20px");
    expect(box.style.top).toBe("40px");
    expect(box.style.width).toBe("100px");
    expect(box.style.height).toBe("100px");
  });

  it("projects an image_pixels bbox using the same scale-factor mechanism", () => {
    const imageContext: ProjectionContext = {
      renderedWidth: 200,
      renderedHeight: 100,
      nativeWidth: 400,
      nativeHeight: 200,
    };
    const boxes: HighlightBox[] = [
      {
        key: "scan",
        bbox: { x0: 40, y0: 20, x1: 80, y1: 60, coordinate_space: "image_pixels" },
        active: false,
        label: "Region",
      },
    ];
    const { container } = render(<BBoxOverlay boxes={boxes} context={imageContext} />);

    // scaleX = scaleY = 0.5 -> left=20, top=10, width=20, height=20
    const box = container.querySelector('[title="Region"]') as HTMLElement;
    expect(box.style.left).toBe("20px");
    expect(box.style.top).toBe("10px");
    expect(box.style.width).toBe("20px");
    expect(box.style.height).toBe("20px");
  });

  it("does not draw a box whose coordinate_space is 'unspecified' (never guesses a placement)", () => {
    const boxes: HighlightBox[] = [
      { key: "unspec", bbox: { x0: 0, y0: 0, x1: 10, y1: 10, coordinate_space: "unspecified" }, active: true },
    ];
    const { container } = render(<BBoxOverlay boxes={boxes} context={goodContext} />);
    // No boxes could be projected -> the component renders nothing at all.
    expect(container.firstChild).toBeNull();
  });

  it("does not draw a box when the projection context is missing native dimensions", () => {
    const incompleteContext: ProjectionContext = {
      renderedWidth: 1000,
      renderedHeight: 1000,
      nativeWidth: null,
      nativeHeight: null,
    };
    const boxes: HighlightBox[] = [{ key: "a", bbox: pdfBox(), active: true, label: "Fig. 1" }];
    const { container } = render(<BBoxOverlay boxes={boxes} context={incompleteContext} />);
    expect(container.firstChild).toBeNull();
  });

  it("drops only the unprojectable boxes, keeping valid ones alongside them", () => {
    const boxes: HighlightBox[] = [
      { key: "good", bbox: pdfBox(), active: true, label: "Good" },
      { key: "bad", bbox: { x0: 0, y0: 0, x1: 5, y1: 5, coordinate_space: "unspecified" }, active: false, label: "Bad" },
    ];
    render(<BBoxOverlay boxes={boxes} context={goodContext} />);
    expect(screen.getByText("Good")).toBeInTheDocument();
    expect(screen.queryByText("Bad")).not.toBeInTheDocument();
  });

  it("shows the label overlay only for active boxes, not inactive ones", () => {
    const boxes: HighlightBox[] = [
      { key: "active", bbox: pdfBox(), active: true, label: "Active label" },
      { key: "inactive", bbox: pdfBox({ x0: 100, x1: 150 }), active: false, label: "Inactive label" },
    ];
    render(<BBoxOverlay boxes={boxes} context={goodContext} />);
    expect(screen.getByText("Active label")).toBeInTheDocument();
    // Inactive boxes still render (dimmer styling) but never show the
    // floating label span — only their `title` attribute carries it.
    expect(screen.queryByText("Inactive label")).not.toBeInTheDocument();
  });

  it("enforces a 4px minimum width/height so a degenerate (zero-area) bbox is still visible/clickable", () => {
    const zeroBox: BoundingBox = { x0: 10, y0: 10, x1: 10, y1: 10, coordinate_space: "pdf_points" };
    const boxes: HighlightBox[] = [{ key: "zero", bbox: zeroBox, active: true, label: "Zero" }];
    const { container } = render(<BBoxOverlay boxes={boxes} context={goodContext} />);
    const box = container.querySelector('[title="Zero"]') as HTMLElement;
    expect(box.style.width).toBe("4px");
    expect(box.style.height).toBe("4px");
  });

  it("invokes onClick when a box is clicked, and applies the clickable/pointer-events styling", async () => {
    const onClick = vi.fn();
    const boxes: HighlightBox[] = [{ key: "clickable", bbox: pdfBox(), active: true, label: "Click me", onClick }];
    const { container } = render(<BBoxOverlay boxes={boxes} context={goodContext} />);

    const box = container.querySelector('[title="Click me"]') as HTMLElement;
    expect(box.className).toContain("pointer-events-auto");
    await userEvent.setup().click(box);
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("does not apply pointer/cursor styling to a box with no onClick handler", () => {
    const boxes: HighlightBox[] = [{ key: "static", bbox: pdfBox(), active: true, label: "Static" }];
    const { container } = render(<BBoxOverlay boxes={boxes} context={goodContext} />);
    const box = container.querySelector('[title="Static"]') as HTMLElement;
    expect(box.className).not.toContain("pointer-events-auto");
  });
});