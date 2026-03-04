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
- **Live URL:** `https://unit-turns-dashboard-t2yhaxw6dfvmrqixmdxprz.streamlit.app/`
- **Auth:** Password gate via `st.secrets["password"]` — set in Streamlit Cloud Settings → Secrets
- **Stack:** Streamlit 1.54.0, Pandas 2.3.3, Plotly 6.5.2, OpenAI, Anthropic, Google GenAI

## Dashboard Structure (7 Tabs)

1. **Executive Summary** — Board-level KPIs, cost trajectory, property benchmarks, vendor risk, capital forecast, risk flags
2. **Portfolio Overview** — Property × Year cost/volume matrices, Property × Floor Plan table, budget category trends (Core Labor, Core Materials, Other avg per turn)
3. **Property Summary** — Single-property deep dive: volume, floor plans, expense group trend, category expenses, last 5 turns with floor plan comparison
4. **Unit Search** — Unit-level: turn history, work history table with export (Excel/PDF), projected scope with comp columns and export (Excel/PDF)
5. **Rent Roll** — (In Development) Property-level rent roll view from external Excel files
6. **Vacancy** — (In Development) Unit vacancy analysis from external Excel file
7. **AI Data Review** — Multi-provider LLM Q&A (Claude, GPT, Gemini) with expanded portfolio data context

## Data Sources

### Turn Data
- **File:** `Unit Turns - AI Clean - 2.26.2026.xlsx` — 19,257 invoice line items across 14 multifamily properties
- **Key columns:** Property Name, Building Code, Unit Number, Floor Plan, Move-Out Date, Turn Type, Vendor Name, Invoice Amount, Budget Category, Cost Type

### Rent Rolls (folder: `Rent Rolls/`)
- **Files:** `Rent Roll - Woodman.xlsx`, `Rent Roll - Monterey Park.xlsx`, `Rent Roll - Collins.xlsx`
- **Format:** 8 metadata header rows (skip), header at row 8, 2 footer/summary rows to filter out
- **Columns (8):** Unit, BD/BA, Status, Market Rent, Rent, Lease From, Lease To, Move-out
- **Cleanup:** `header=8`, drop all-NaN rows, filter out rows where Unit contains "Units" or "Total", filter out property address rows
- **Collins quirk:** Has an extra property address row after the header that must be filtered (contains "COLLINS, LLC")

### Rent Roll ↔ Turn Data Unit Mapping
- **Woodman:** Direct match — rent roll Unit ("101") = turn data Unit Number ("101"), no Building Code
- **Monterey Park:** Rent roll "505 Pomona, Unit A" → turn data Building Code "505 Pomona" + Unit Number "A"
- **Collins:** Rent roll prefix encodes building address — "39-xx" → Building Code "18339 Collins" + Unit Number "xx"; "47-xx" → Building Code "18347 Collins" + Unit Number "xx"

### Vacancy (folder: `Vacancy/`)
- **File:** `Unit Vacancy.xlsx` — (Not yet analyzed)

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
