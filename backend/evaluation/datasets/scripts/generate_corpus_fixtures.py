"""Regenerate `data/corpus/08_regional_sales_chart.png`,
`09_system_architecture_diagram.png`, `12_data_security_policy.pdf`, and
`13_monthly_active_users_chart.png` deterministically.

Why this script exists: these are the ONLY real chart/figure/multi-page
content in the sample corpus (see `data/corpus/README.txt`) — every
chart/figure-modality and multi-page-evidence row in
`evaluation/datasets/phase4_v1.jsonl` is grounded in their EXACT drawn
values/written text, not a described-but-nonexistent file. If these
files are ever lost from a packaging/export step (as happened once — see
MANIFEST.md history), this script reproduces them byte-for-byte
-equivalent (same values, same layout, same text) rather than leaving
those benchmark rows pointing at nothing.

**Do not change the drawn values/written text without also updating
every benchmark row that references them** (search
`evaluation/datasets/*.jsonl` for `08_regional_sales_chart` /
`09_system_architecture` / `12_data_security_policy` /
`13_monthly_active_users`) — the whole point of these fixtures is that
their ground truth is exact and inspectable.

Usage (from `backend/`):
    python3 evaluation/datasets/scripts/generate_corpus_fixtures.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "corpus"

# Ground truth — referenced by id in evaluation/datasets/phase4_v1.jsonl
# (q048-q051 for the chart, q052-q054 for the diagram). Keep these two
# dicts as the single source of truth for those rows' "notes" fields.
CHART_TITLE = "Regional Sales - FY2025 (in $000s)"
CHART_VALUES = {"Region A": 42, "Region B": 67, "Region C": 29, "Region D": 51}
DIAGRAM_EDGES = [
    ("Client", "API Gateway"),
    ("API Gateway", "Application\nServer"),
    ("Application\nServer", "Database"),
    ("Application\nServer", "Cache"),
]

# Ground truth for 13_monthly_active_users_chart.png (Phase 7 benchmark
# -expansion pass) — a second, independent chart fixture with a genuine
# month-over-month trend (mostly increasing, one real dip in March),
# used for trend/timeline-flavored chart questions 08's single-snapshot
# regional chart cannot support. Values fixed here per
# data/corpus/README.txt — do not change without updating every
# benchmark row that references them (search
# `13_monthly_active_users` across evaluation/datasets/*.jsonl).
MAU_CHART_TITLE = "Monthly Active Users - H1 2025 (in thousands)"
MAU_CHART_VALUES = {"Jan": 12, "Feb": 15, "Mar": 14, "Apr": 19, "May": 23, "Jun": 27}

# Ground truth for 12_data_security_policy.pdf (Phase 7 benchmark
# -expansion pass) — the only genuinely multi-page fixture in this
# corpus (every other document/image is single-page), added specifically
# to close the "multi-page evidence" gap earlier evaluation passes
# flagged. Page 1: encryption, access control, incident response. Page
# 2: retention, penetration testing, breach notification, plus a real
# table. Kept here as plain data (not scattered reportlab calls) so a
# benchmark author can read the exact ground truth without re-rendering
# the PDF — do not change without updating every benchmark row that
# references `12_data_security_policy` across
# evaluation/datasets/*.jsonl.
SECURITY_POLICY_TITLE = "Data Security Policy Excerpt — Northfield Analytics (fictional)"
SECURITY_POLICY_INTRO = (
    "This is a fictional policy excerpt used as sample corpus content. All figures below are "
    "made up for demonstration purposes only."
)
SECURITY_POLICY_PAGE1_SECTIONS = [
    (
        "Encryption",
        "All customer data is encrypted at rest using AES-256. Data in transit is encrypted "
        "using TLS 1.3 or higher. Encryption keys are rotated every 90 days.",
    ),
    (
        "Access Control",
        "Access to production systems requires multi-factor authentication (MFA). Role-based "
        "access control (RBAC) limits each employee to the minimum permissions required for "
        "their role. Access reviews are conducted quarterly.",
    ),
    (
        "Incident Response",
        "Security incidents are classified into three severity tiers: Critical, High, and "
        "Moderate. Critical incidents must be triaged within 1 hour of detection. The incident "
        "response team is led by the Security Operations Manager.",
    ),
]
SECURITY_POLICY_PAGE2_SECTIONS = [
    (
        "Data Retention",
        "Customer data is retained for 24 months after account closure, after which it is "
        "permanently deleted. Backup copies are retained for an additional 30 days beyond the "
        "primary retention period.",
    ),
    (
        "Penetration Testing",
        "An independent third-party penetration test is conducted annually. The most recent "
        "test was completed in September and identified 2 medium-severity findings, both "
        "remediated within 30 days.",
    ),
    (
        "Breach Notification",
        "In the event of a confirmed data breach, affected customers are notified within 72 "
        "hours. Regulatory authorities are notified within 72 hours where required by "
        "applicable law.",
    ),
]
SECURITY_POLICY_TABLE_HEADER = ["Severity Tier", "Response Time", "Escalation Contact"]
SECURITY_POLICY_TABLE_ROWS = [
    ["Critical", "1 hour", "Security Operations Manager"],
    ["High", "4 hours", "Security Team Lead"],
    ["Moderate", "1 business day", "On-call Engineer"],
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001 - fall through to the next candidate
            continue
    return ImageFont.load_default()


def generate_chart(out_path: Path) -> None:
    width, height = 800, 500
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, label_font, value_font = _font(22), _font(16), _font(16)

    draw.text((width / 2, 30), CHART_TITLE, fill="black", font=title_font, anchor="mm")

    regions = list(CHART_VALUES.keys())
    values = list(CHART_VALUES.values())
    max_val = 80
    chart_left, chart_right = 100, 700
    chart_top, chart_bottom = 80, 420
    bar_width = 90
    gap = (chart_right - chart_left - len(regions) * bar_width) / (len(regions) + 1)

    draw.line([(chart_left, chart_top), (chart_left, chart_bottom)], fill="black", width=2)
    draw.line([(chart_left, chart_bottom), (chart_right, chart_bottom)], fill="black", width=2)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    x = chart_left + gap
    for region, val, color in zip(regions, values, colors):
        bar_h = (val / max_val) * (chart_bottom - chart_top)
        y0 = chart_bottom - bar_h
        draw.rectangle([x, y0, x + bar_width, chart_bottom], fill=color, outline="black")
        draw.text((x + bar_width / 2, y0 - 12), str(val), fill="black", font=value_font, anchor="mm")
        draw.text((x + bar_width / 2, chart_bottom + 20), region, fill="black", font=label_font, anchor="mm")
        x += bar_width + gap

    draw.text((40, (chart_top + chart_bottom) / 2), "$000s", fill="black", font=label_font, anchor="mm")
    img.save(out_path)


def generate_diagram(out_path: Path) -> None:
    width, height = 700, 500
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    box_font = _font(18)

    positions = {
        "Client": (100, 80),
        "API Gateway": (320, 80),
        "Application\nServer": (560, 80),
        "Database": (560, 280),
        "Cache": (320, 280),
    }
    box_size = {"Client": (140, 60), "API Gateway": (160, 60), "Application\nServer": (170, 60), "Database": (150, 60), "Cache": (150, 60)}

    boxes = {}
    for label, (cx, cy) in positions.items():
        w, h = box_size[label]
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        draw.rectangle([x0, y0, x1, y1], outline="black", width=2, fill="#EEF3FB")
        draw.text((cx, cy), label, fill="black", font=box_font, anchor="mm")
        boxes[label] = (x0, y0, x1, y1)

    def arrow(p0: tuple[float, float], p1: tuple[float, float]) -> None:
        draw.line([p0, p1], fill="black", width=2)
        angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        for da in (0.4, -0.4):
            ax = p1[0] - 12 * math.cos(angle - da)
            ay = p1[1] - 12 * math.sin(angle - da)
            draw.line([p1, (ax, ay)], fill="black", width=2)

    arrow((boxes["Client"][2], 80), (boxes["API Gateway"][0], 80))
    arrow((boxes["API Gateway"][2], 80), (boxes["Application\nServer"][0], 80))
    arrow((560, boxes["Application\nServer"][3]), (560, boxes["Database"][1]))
    arrow((boxes["Application\nServer"][0], 90), (boxes["Cache"][2], 280))

    draw.text((width / 2, 20), "System Architecture", fill="black", font=_font(22), anchor="mm")
    img.save(out_path)


def generate_mau_chart(out_path: Path) -> None:
    """Second, independent chart fixture — same drawing approach as
    `generate_chart` (a simple PIL bar chart with values labeled directly
    on each bar), but with a genuine month-over-month trend rather than a
    single snapshot, per `MAU_CHART_VALUES`.
    """
    width, height = 800, 500
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, label_font, value_font = _font(22), _font(16), _font(16)

    draw.text((width / 2, 30), MAU_CHART_TITLE, fill="black", font=title_font, anchor="mm")

    months = list(MAU_CHART_VALUES.keys())
    values = list(MAU_CHART_VALUES.values())
    max_val = 30
    chart_left, chart_right = 100, 760
    chart_top, chart_bottom = 80, 420
    bar_width = 70
    gap = (chart_right - chart_left - len(months) * bar_width) / (len(months) + 1)

    draw.line([(chart_left, chart_top), (chart_left, chart_bottom)], fill="black", width=2)
    draw.line([(chart_left, chart_bottom), (chart_right, chart_bottom)], fill="black", width=2)

    color = "#4C72B0"
    x = chart_left + gap
    for month, val in zip(months, values):
        bar_h = (val / max_val) * (chart_bottom - chart_top)
        y0 = chart_bottom - bar_h
        draw.rectangle([x, y0, x + bar_width, chart_bottom], fill=color, outline="black")
        draw.text((x + bar_width / 2, y0 - 12), str(val), fill="black", font=value_font, anchor="mm")
        draw.text((x + bar_width / 2, chart_bottom + 20), month, fill="black", font=label_font, anchor="mm")
        x += bar_width + gap

    draw.text((40, (chart_top + chart_bottom) / 2), "000s", fill="black", font=label_font, anchor="mm")
    img.save(out_path)


def generate_security_policy_pdf(out_path: Path) -> None:
    """The corpus's only genuinely multi-page fixture — two real PDF
    pages of body text plus a real table on page 2, via reportlab (the
    same library `04_quarterly_report_excerpt.pdf` was originally built
    with). Ground truth is `SECURITY_POLICY_*` above; kept as data, not
    scattered through this function, so a benchmark author can read it
    without re-rendering the PDF.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out_path), pagesize=LETTER, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    story = [Paragraph(SECURITY_POLICY_TITLE, styles["Title"]), Spacer(1, 12), Paragraph(SECURITY_POLICY_INTRO, styles["Italic"]), Spacer(1, 16)]

    for heading, body in SECURITY_POLICY_PAGE1_SECTIONS:
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Paragraph(body, styles["BodyText"]))
        story.append(Spacer(1, 10))

    story.append(PageBreak())

    for heading, body in SECURITY_POLICY_PAGE2_SECTIONS:
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Paragraph(body, styles["BodyText"]))
        story.append(Spacer(1, 10))

    story.append(Paragraph("Incident Response Time by Severity", styles["Heading2"]))
    table_data = [SECURITY_POLICY_TABLE_HEADER, *SECURITY_POLICY_TABLE_ROWS]
    table = Table(table_data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)

    doc.build(story)


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = CORPUS_DIR / "08_regional_sales_chart.png"
    diagram_path = CORPUS_DIR / "09_system_architecture_diagram.png"
    security_policy_path = CORPUS_DIR / "12_data_security_policy.pdf"
    mau_chart_path = CORPUS_DIR / "13_monthly_active_users_chart.png"
    generate_chart(chart_path)
    generate_diagram(diagram_path)
    generate_security_policy_pdf(security_policy_path)
    generate_mau_chart(mau_chart_path)
    print(f"wrote {chart_path}")
    print(f"wrote {diagram_path}")
    print(f"wrote {security_policy_path}")
    print(f"wrote {mau_chart_path}")


if __name__ == "__main__":
    main()