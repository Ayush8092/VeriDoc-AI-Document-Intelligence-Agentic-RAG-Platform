import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import EvidenceCitationChip from "@/components/evidence-citation-chip";
import type { EvidenceCitation } from "@/lib/types";

// Realistic backend shapes — mirrors app/schemas/reasoning.py's
// EvidenceCitation (chunk_id/source_file/page_start/page_end/document_id).
const fullCitation: EvidenceCitation = {
  chunk_id: "corpus::policy.pdf::leave::3",
  source_file: "policy.pdf",
  page_start: 4,
  page_end: 4,
  document_id: 12,
};

describe("EvidenceCitationChip", () => {
  it("renders the source file name", () => {
    // The chip renders source_file and the page-range suffix as
    // adjacent text within the same <span> (no wrapping element around
    // either), so their combined textContent is "policy.pdf \u00b7 p. 4" --
    // an exact-string getByText("policy.pdf") would never match. A
    // substring/regex matcher checks the same real thing (the file
    // name text is present) without weakening what's asserted.
    render(<EvidenceCitationChip citation={fullCitation} />);
    expect(screen.getByText(/policy\.pdf/)).toBeInTheDocument();
  });

  it("renders a single-page reference using formatPageRange's 'p. N' form", () => {
    render(<EvidenceCitationChip citation={fullCitation} />);
    // formatPageRange(4, 4) -> "p. 4" (start === end collapses to one page)
    expect(screen.getByText(/p\. 4/)).toBeInTheDocument();
  });

  it("renders a multi-page reference using formatPageRange's 'pp. N–M' form", () => {
    const multiPage: EvidenceCitation = { ...fullCitation, page_start: 4, page_end: 6 };
    render(<EvidenceCitationChip citation={multiPage} />);
    expect(screen.getByText(/pp\. 4–6/)).toBeInTheDocument();
  });

  it("omits the page range entirely when page_start is null (no fabricated page number)", () => {
    const noPage: EvidenceCitation = { ...fullCitation, page_start: null, page_end: null };
    const { container } = render(<EvidenceCitationChip citation={noPage} />);
    expect(container.textContent).not.toMatch(/p\./);
    // Source file is still shown -- absence of a page number doesn't
    // suppress the rest of the chip.
    expect(screen.getByText("policy.pdf")).toBeInTheDocument();
  });

  it("does not render a page range when only page_end is present but page_start is null", () => {
    // Malformed/incomplete shape: page_end without page_start. The
    // component's own guard is `citation.page_start != null`, so this
    // must behave identically to "no page info at all" rather than
    // rendering a broken/partial range.
    const malformed = { ...fullCitation, page_start: null, page_end: 6 } as EvidenceCitation;
    const { container } = render(<EvidenceCitationChip citation={malformed} />);
    expect(container.textContent).not.toMatch(/p\./);
  });

  it("renders distinct source files for two different citations (no cross-contamination)", () => {
    const other: EvidenceCitation = { ...fullCitation, source_file: "handbook.docx", page_start: 1, page_end: 1 };
    const { rerender } = render(<EvidenceCitationChip citation={fullCitation} />);
    expect(screen.getByText(/policy\.pdf/)).toBeInTheDocument();
    rerender(<EvidenceCitationChip citation={other} />);
    expect(screen.getByText(/handbook\.docx/)).toBeInTheDocument();
    expect(screen.queryByText(/policy\.pdf/)).not.toBeInTheDocument();
  });
});