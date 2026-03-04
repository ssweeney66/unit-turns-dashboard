# Project Instructions

## Core Guardrails

1. **Only touch files explicitly mentioned or directly relevant to the task.** Do not wander into unrelated files, refactor code that wasn't discussed, or make "improvements" that weren't requested.
2. **Never delete data.** Do not remove rows, columns, or files from the Excel data source or any production data without explicit confirmation.
3. **Always syntax check after editing app.py.** Run `python3 -c "import py_compile; py_compile.compile('app.py', doraise=True)"` before declaring any code change complete.
4. **Do not push to GitHub unless asked.** Commit locally when requested, but only push when explicitly told to.
5. **Do not create documentation files** (README, .md files) unless explicitly requested.

## Project Context

- **App:** Full Turn Analytics Dashboard (Streamlit)
- **File:** `app.py` — single-file dashboard (~2,500 lines)
- **Data:** `Unit Turns - AI Clean - 2.26.2026.xlsx` — 19,257 invoice line items across 14 multifamily properties
- **Repo:** `ssweeney66/unit-turns-dashboard` (public, main branch)
- **Stack:** Streamlit 1.54.0, Pandas 2.3.3, Plotly 6.5.2, OpenAI, Anthropic, Google GenAI

## Dashboard Structure (5 Tabs)

1. **Executive Summary** — Board-level KPIs, cost trajectory, property benchmarks, vendor risk, capital forecast, risk flags
2. **Portfolio Overview** — Property × Year cost/volume matrices, Property × Floor Plan table, budget category trends (Core Labor, Core Materials, Other avg per turn)
3. **Property Summary** — Single-property deep dive: volume, floor plans, expense group trend, category expenses, last 5 turns with floor plan comparison, outliers
4. **Unit Search** — Unit-level: turn history, work history table with export (Excel/PDF), projected scope with comp columns and export (Excel/PDF)
5. **AI Data Review** — Multi-provider LLM Q&A (Claude, GPT, Gemini) with expanded portfolio data context

## Data Model

- **Turn Types:** Full Turn, Make Ready, Partial Turn
- **17 Budget Categories:** Core Labor (7), Core Materials (4), Other (6)
- **Cost Types:** Materials, Labor, Mixed, Fee
- **14 Properties:** Monterey Park, Woodman, Collins, Lindley, El Rancho, 51 at the Village, Alta Vista, Roscoe, Topanga, Darby, Fruitland, Dickens, Garfield, Woodbridge

## Coding Standards

- Use `fmt()` for all currency formatting, `pct()` for percentages
- Use `section()` for section headers, `insight()` for narrative boxes, `banner()` for page headers
- Duration always formatted as "X days" (not "Xd")
- Property sort order uses `_PROP_RANK` / `PROPERTY_ORDER`
- All tables use standardized category grouping: Core Labor → Core Materials → Other
