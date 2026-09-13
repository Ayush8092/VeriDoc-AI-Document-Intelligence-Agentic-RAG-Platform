Veridoc sample corpus (fiction only)

All company names, figures, and events below are made up for
demonstration purposes. This corpus intentionally spans every supported
format so ingestion can be exercised end-to-end, and is also the
grounding source for evaluation/datasets/phase4_v1.jsonl and
evaluation/datasets/multi_doc_v1.jsonl (see evaluation/datasets/SCHEMA.md):

- 01_product_overview.md               — Markdown, headings + a pipe table
- 02_remote_work_policy.md              — Markdown, headings only
- 03_meeting_notes_project_kickoff.txt  — plain text, no headings
- 04_quarterly_report_excerpt.pdf       — native PDF text + a real table
- 05_employee_handbook_excerpt.docx     — DOCX, headings + a real table
- 06_warehouse_receiving_log_scanned.png — image, OCR + reconstructed table
- 08_regional_sales_chart.png           — image, a real bar chart (drawn
                                           with PIL, exact known values —
                                           see evaluation/datasets/scripts/
                                           generate_corpus_fixtures.py)
- 09_system_architecture_diagram.png    — image, a real box-and-arrow
                                           diagram (same generation script)
- 10_vendor_support_ticket.md           — Markdown, a real support ticket
                                           containing a genuine embedded
                                           prompt-injection attempt in its
                                           own body text — used to evaluate
                                           prompt-injection resistance
                                           against real (not synthetic
                                           test-harness) retrieved content
- 11_pricing_correction_memo.md         — Markdown, a short internal memo
                                           that deliberately, explicitly
                                           corrects the Pro plan price
                                           stated in 01_product_overview.md
                                           ($25 -> $29) — the one genuine,
                                           real, inspectable contradiction
                                           between two documents in this
                                           corpus, used for
                                           CONTRADICTION_DETECTION ground
                                           truth
- 12_data_security_policy.pdf           — native PDF text across TWO
                                           pages (page 1: encryption,
                                           access control, incident
                                           response; page 2: retention,
                                           penetration testing, breach
                                           notification, plus a real
                                           table) — the only genuinely
                                           multi-page fixture in this
                                           corpus, added specifically to
                                           close the "multi-page evidence"
                                           gap earlier evaluation passes
                                           had flagged as ungroundable
                                           (every other document/image in
                                           this corpus is single-page)
- 13_monthly_active_users_chart.png     — image, a real bar chart (drawn
                                           with PIL, exact known values:
                                           Jan=12, Feb=15, Mar=14, Apr=19,
                                           May=23, Jun=27, all thousands)
                                           — a second, independent chart
                                           fixture (distinct from 08) with
                                           a genuine month-over-month trend
                                           (mostly increasing, with one
                                           real dip in March), used for
                                           trend/timeline-flavored chart
                                           questions that 08's single-
                                           snapshot regional chart cannot
                                           support

Note (07 is intentionally absent — an earlier fixture at that number was
superseded by 08/09 and removed rather than leaving a numbering gap
filled with dead content).

You may ship this folder as-is or replace it with your own files in the
same style. This file itself (README.txt) is never ingested.