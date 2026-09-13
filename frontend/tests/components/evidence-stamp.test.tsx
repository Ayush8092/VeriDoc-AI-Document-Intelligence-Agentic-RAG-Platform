import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import EvidenceStamp from "@/components/evidence-stamp";
import type { Citation } from "@/lib/types";

// `EvidenceStamp` calls useRouter().push() for its "View source"
// click-through — mocked here so this suite exercises the STAMP's own
// rendering/interaction logic, not Next's router itself.
const pushMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
}));

// Realistic backend shape — mirrors app/schemas/documents.py's Citation.
function makeCitation(overrides: Partial<Citation> = {}): Citation {
  return {
    chunk_id: "corpus::policy.pdf::leave::3",
    source_file: "policy.pdf",
    section: "Leave policy",
    snippet: "Employees accrue 1.5 days of paid leave per month worked.",
    score: 0.87,
    block_type: "text",
    page_start: 4,
    page_end: 4,
    source: "native",
    confidence: 1,
    table_rows: null,
    object_id: null,
    bbox: null,
    coordinate_space: null,
    caption: null,
    visual_type: null,
    chart_type: null,
    related_text: null,
    ...overrides,
  };
}

describe("EvidenceStamp", () => {
  beforeEach(() => {
    pushMock.mockReset();
  });

  it("renders the source file and page range in the collapsed (closed) state", () => {
    render(<EvidenceStamp citation={makeCitation()} index={0} />);
    expect(screen.getByText("policy.pdf")).toBeInTheDocument();
    expect(screen.getByText("p. 4")).toBeInTheDocument();
    expect(screen.getByText("Leave policy")).toBeInTheDocument();
  });

  it("shows an OCR badge only for OCR-sourced citations, not native ones", () => {
    const { rerender } = render(<EvidenceStamp citation={makeCitation({ source: "native" })} index={0} />);
    expect(screen.queryByText("OCR")).not.toBeInTheDocument();

    rerender(<EvidenceStamp citation={makeCitation({ source: "ocr" })} index={0} />);
    expect(screen.getByText("OCR")).toBeInTheDocument();
  });

  it("shows a block-type badge for table citations, not for plain text", () => {
    const { rerender } = render(<EvidenceStamp citation={makeCitation({ block_type: "text" })} index={0} />);
    expect(screen.queryByText("Table")).not.toBeInTheDocument();

    rerender(<EvidenceStamp citation={makeCitation({ block_type: "table" })} index={0} />);
    expect(screen.getByText("Table")).toBeInTheDocument();
  });

  it("shows the visual_type badge for a figure citation, but omits the 'unknown_visual' placeholder", () => {
    const { rerender } = render(
      <EvidenceStamp citation={makeCitation({ block_type: "figure", visual_type: "photo" })} index={0} />,
    );
    expect(screen.getByText("photo")).toBeInTheDocument();

    rerender(
      <EvidenceStamp citation={makeCitation({ block_type: "figure", visual_type: "unknown_visual" })} index={0} />,
    );
    expect(screen.queryByText("unknown_visual")).not.toBeInTheDocument();
  });

  it("shows the chart_type badge for a chart citation, but omits the 'unknown_chart' placeholder", () => {
    const { rerender } = render(
      <EvidenceStamp citation={makeCitation({ block_type: "chart", chart_type: "bar" })} index={0} />,
    );
    expect(screen.getByText("bar chart")).toBeInTheDocument();

    rerender(<EvidenceStamp citation={makeCitation({ block_type: "chart", chart_type: "unknown_chart" })} index={0} />);
    expect(screen.queryByText("unknown_chart chart")).not.toBeInTheDocument();
  });

  it("prefers the caption over the section as the secondary line when both are present", () => {
    render(<EvidenceStamp citation={makeCitation({ caption: "Figure 2: revenue by region", section: "Appendix" })} index={0} />);
    expect(screen.getByText("Figure 2: revenue by region")).toBeInTheDocument();
    expect(screen.queryByText("Appendix")).not.toBeInTheDocument();
  });

  it("falls back to the section when caption is null", () => {
    render(<EvidenceStamp citation={makeCitation({ caption: null, section: "Appendix" })} index={0} />);
    expect(screen.getByText("Appendix")).toBeInTheDocument();
  });

  it("expands on click to show the snippet, chunk id, and relevance score for a text citation", async () => {
    const user = userEvent.setup();
    render(<EvidenceStamp citation={makeCitation({ score: 0.87 })} index={0} />);

    // Snippet/chunk id are only rendered once expanded.
    expect(screen.queryByText(/Employees accrue/)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button"));

    expect(screen.getByText(/Employees accrue 1\.5 days/)).toBeInTheDocument();
    expect(screen.getByText(/chunk corpus::policy\.pdf::leave::3/)).toBeInTheDocument();
    expect(screen.getByText(/relevance 87%/)).toBeInTheDocument();
  });

  it("collapses again on a second click", async () => {
    const user = userEvent.setup();
    render(<EvidenceStamp citation={makeCitation()} index={0} />);
    const toggle = screen.getByRole("button");

    await user.click(toggle);
    expect(screen.getByText(/Employees accrue/)).toBeInTheDocument();

    await user.click(toggle);
    expect(screen.queryByText(/Employees accrue/)).not.toBeInTheDocument();
  });

  it("renders the table preview when expanded for a table citation with table_rows", async () => {
    const user = userEvent.setup();
    const citation = makeCitation({
      block_type: "table",
      table_rows: [
        ["Region", "Revenue"],
        ["West", "120000"],
      ],
    });
    render(<EvidenceStamp citation={citation} index={0} />);
    await user.click(screen.getByRole("button"));

    expect(screen.getByText("Region")).toBeInTheDocument();
    expect(screen.getByText("West")).toBeInTheDocument();
    expect(screen.getByText("120000")).toBeInTheDocument();
  });

  it("shows an explanatory placeholder for a visual citation with neither caption nor related_text", async () => {
    const user = userEvent.setup();
    const citation = makeCitation({ block_type: "figure", caption: null, related_text: null });
    render(<EvidenceStamp citation={citation} index={0} />);
    await user.click(screen.getByRole("button"));

    expect(screen.getByText(/No caption or extracted text was associated with this figure/)).toBeInTheDocument();
  });

  it("shows the 'View source' action when page_start is present, and navigates to /source on click", async () => {
    const user = userEvent.setup();
    render(<EvidenceStamp citation={makeCitation({ page_start: 4 })} index={0} />);
    await user.click(screen.getByRole("button", { name: /policy\.pdf/ }));

    const viewSourceButton = screen.getByRole("button", { name: /View source/ });
    await user.click(viewSourceButton);

    expect(pushMock).toHaveBeenCalledTimes(1);
    expect(pushMock).toHaveBeenCalledWith(expect.stringMatching(/^\/source\?session=/));
  });

  it("does not show the 'View source' action when page_start is null (nothing to navigate to)", async () => {
    const user = userEvent.setup();
    render(<EvidenceStamp citation={makeCitation({ page_start: null, page_end: null })} index={0} />);
    await user.click(screen.getByRole("button", { name: /policy\.pdf/ }));

    expect(screen.queryByRole("button", { name: /View source/ })).not.toBeInTheDocument();
  });

  it("does not crash and simply omits the score display for a malformed citation missing confidence-affecting fields", () => {
    // Incomplete/malformed shape: score is present but object_id/bbox/etc
    // are all null (the normal "not applicable" case per the Citation
    // type's own docstring) -- must render without throwing.
    const citation = makeCitation({ object_id: null, bbox: null, coordinate_space: null });
    expect(() => render(<EvidenceStamp citation={citation} index={0} />)).not.toThrow();
  });
});