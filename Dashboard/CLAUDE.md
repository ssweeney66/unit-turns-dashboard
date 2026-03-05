# Project Instructions

## Core Guardrails

1. **Only touch files explicitly mentioned or directly relevant to the task.** Do not wander into unrelated files, refactor code that wasn't discussed, or make "improvements" that weren't requested.
2. **Never delete data.** Do not remove rows, columns, or files from the Excel data source or any production data without explicit confirmation.
3. **Always syntax check after editing app.py.** Run `python3 -c "import py_compile; py_compile.compile('app.py', doraise=True)"` before declaring any code change complete.
4. **Do not push to GitHub unless asked.** Commit locally when requested, but only push when explicitly told to.
5. **Do not create documentation files** (README, .md files) unless explicitly requested.

## Project Context

- **App:** Full Turn Analytics Dashboard (Streamlit)
- **File:** `Dashboard/app.py` — single-file dashboard (~2,896 lines)
- **Data:** `Data/Unit Turns - AI Clean - 2.26.2026.xlsx` — 19,257 invoice line items across 14 multifamily properties
- **Repo:** `ssweeney66/unit-turns-dashboard` (public, main branch)
- **Live URL:** `https://unit-turns-dashboard-mbt5kfbyktdvpuebrcxtss.streamlit.app/`
- **Auth:** Password gate via `st.secrets["password"]` — set in Streamlit Cloud Settings → Secrets
- **Stack:** Streamlit 1.54.0, Pandas 2.3.3, Plotly 6.5.2, OpenAI, Anthropic, Google GenAI, fpdf2

## Folder Structure
```
Claude Test/
├── Dashboard/          ← app code lives here
│   ├── app.py
│   ├── analyze_ft.py   (standalone analysis script)
│   ├── CLAUDE.md
│   └── requirements.txt
├── Data/               ← all data files live here
│   ├── Unit Turns - AI Clean - 2.26.2026.xlsx
│   ├── Rent Rolls/     (16 property rent roll files)
│   └── Vacancy/        (Unit Vacancy.xlsx)
└── requirements.txt    ← root copy for Streamlit Cloud
```
- All data paths in app.py use `Path(__file__).parent.parent / "Data" / ...` to reach the sibling Data folder

## Dashboard Structure (7 Tabs)

1. **Executive Summary** — Board-level KPIs, cost trajectory, property benchmarks, vendor risk, capital forecast, risk flags
2. **Portfolio Overview** — Property x Year cost/volume matrices, Property x Floor Plan table, budget category trends (Core Labor, Core Materials, Other avg per turn)
3. **Property Summary** — Single-property deep dive: volume, floor plans, category expenses (Core Labor/Materials/Other avg per turn), last 5 turns with floor plan comparison
4. **Unit Search** — Unit-level: turn history, work history table with export (Excel/PDF), projected scope with comp columns and export (Excel/PDF)
5. **Rent Roll** — Portfolio Renovated vs Classic summary (% columns), floor plan summary (avg reno/classic rent, premium), property-level rent roll with Loss to Lease, % Upside, Move-In, Status (green/red), FT Budget (trailing 3-yr avg per floor plan), ROI %, KPIs, high-frequency turn outliers (5+ turns since 2019)
6. **Data Health** — File timestamp monitoring with 90-day freshness threshold (green/red/missing signals)
7. **AI Data Review** — Multi-provider LLM Q&A (Claude, GPT, Gemini) with expanded portfolio data context

## Data Sources

### Turn Data
- **File:** `Unit Turns - AI Clean - 2.26.2026.xlsx` — 19,257 invoice line items across 14 multifamily properties
- **Key columns:** Property Name, Building Code, Unit Number, Floor Plan, Move-Out Date, Turn Type, Vendor Name, Invoice Amount, Budget Category, Cost Type

### Rent Rolls (folder: `Data/Rent Rolls/`)
- **Active files (16):** Woodman, MontereyPark, Collins, 51Village, Lindley, ElRancho, Alta Vista, Roscoe, Woodbridge, Darby, Dickens, Fruitland, Garfield, Topanga, 12756Moorpark, 12800Moorpark
- **No turn data (2):** 12756 Moorpark (10 units), 12800 Moorpark (18 units) — all Classic, no matching property in turn data
- **All properties covered** — Topanga rent roll now active (40 units, direct match, "301 HUD" suffix stripped)
- **Format:** 8 metadata header rows (skip), header at row 8, 2 footer/summary rows to filter out
- **Columns (5):** Unit, BD/BA, Market Rent, Rent, Move-in
- **Cleanup:** `header=8`, drop all-NaN rows, filter out rows where Unit contains "Units" or "Total", filter out rows with "LLC", "Properties", or ", LP"
- **Unit suffixes:** Some rent rolls include suffixes like "HUD" (subsidized), "MGR" (manager unit), "BC" — these are stripped by `_norm_unit()` before matching

### Rent Roll <> Turn Data Unit Mapping

All mapping is handled by `rr_to_turn_key()` (rent roll side) and `_norm_unit()` (both sides). Both sides normalize leading zeros for purely numeric units ("01" -> "1") and strip suffixes.

**Direct match (no Building Code):**
- **Woodman:** "101" -> "101"
- **51 at the Village:** "101" -> "101"
- **Lindley:** "01" -> "1" (leading zero stripped; turn data uses "1")
- **El Rancho:** "01" -> "1" (both sides normalized)
- **Alta Vista:** "01" -> "1" (leading zero stripped; turn data uses "1")
- **Roscoe:** "101" -> "101" (3-digit, no leading zero issue)
- **Woodbridge:** "01" -> "1" (leading zero stripped; turn data uses "1")
- **Darby:** "101" -> "101" (3-digit, direct match)
- **Dickens:** "12-A" -> "12A" (hyphen removed by `_norm_unit`); "07 MGR" -> "7" (suffix stripped + leading zero)
- **Fruitland:** "01" -> "1" (leading zero stripped); "09 MGR" -> "9" (suffix + leading zero)
- **Garfield:** "616" -> "616", "616A" -> "616A", "618 1/2" -> "618 1/2" (kept as-is)
- **Topanga:** "101" -> "101" (3-digit, direct match); "301 HUD" -> "301" (suffix stripped)
- **12756 Moorpark:** "101" -> "101" (3-digit, no turn data — all Classic)
- **12800 Moorpark:** "1" -> "1", "12A" -> "12A" (no turn data — all Classic)

**Compound key (Building Code + Unit Number):**
- **Monterey Park:** "505 Pomona, Unit A" -> "505 Pomona|A". Address normalization strips directional prefixes (W, S) and suffixes (Ave, Blvd, Dr). Spelling correction: rent roll "Hendricks" = turn data "Hendericks". Unit suffixes like "- ASST. MGR" or "- HUD" stripped to single letter.
- **Collins:** "39-xx" -> "18339 Collins|xx"; "47-xx" -> "18347 Collins|xx". Suffixes like "-MGR" are stripped.

### Vacancy (folder: `Data/Vacancy/`)
- **File:** `Unit Vacancy.xlsx` — not yet analyzed or integrated

## Data Model

- **Turn Types:** Full Turn, Make Ready, Partial Turn
- **Turn Abbreviations:** FT (Full Turn), MR (Make Ready), PT (Partial Turn)
- **17 Budget Categories:** Core Labor (7), Core Materials (4), Other (6)
- **Cost Types:** Materials, Labor, Mixed, Fee
- **16 Properties:** Monterey Park, Woodman, Collins, Lindley, El Rancho, 51 at the Village, Alta Vista, Roscoe, Topanga, Darby, Fruitland, Dickens, Garfield, Woodbridge, 12756 Moorpark, 12800 Moorpark
- **Classic unit:** A unit on the rent roll with no Full Turn on record in turn data

## Key Functions (Rent Roll Tab)

- `load_rent_roll(path)` — loads/cleans a rent roll Excel file (5-column format)
- `_norm_unit(u)` — normalizes a unit ID: strips suffixes (HUD/MGR/BC/ASST. MGR), removes formatting hyphens ("12-A" → "12A"), normalizes leading zeros
- `_normalize_mp_bldg(addr)` — normalizes Monterey Park building addresses to match turn data
- `rr_to_turn_key(prop_name, rr_unit_str)` — maps a rent roll unit to the compound key used in turn data
- `_bdba_to_fp(bdba)` — converts rent roll BD/BA (e.g. "2/1.00") to turn data Floor Plan format (e.g. "2x1")
- `get_ft_units(prop_name, _df_all)` — returns set of compound keys for units with at least one Full Turn
- `get_avg_ft_cost(prop_name, _df_all)` — returns average Full Turn cost for a property (all-time)
- `get_ft_cost_by_fp(prop_name, _df_all)` — returns dict of Floor Plan → trailing 3-year avg FT cost

## Coding Standards

- Use `fmt()` for all currency formatting, `pct()` for YoY percentage deltas
- Use `section()` for section headers, `insight()` for narrative boxes, `banner()` for page headers
- Duration always formatted as "X days" (not "Xd")
- Property sort order uses `_PROP_RANK` / `PROPERTY_ORDER`
- All tables use standardized category grouping: Core Labor -> Core Materials -> Other
- Classic % uses direct `f"{value * 100:.1f}%"` — do NOT use `pct()` (which is for YoY deltas with "+" prefix)
- File paths must use `Path(__file__).parent.parent / "Data"` to reach the sibling Data folder — never bare relative paths
- When writing narrative insights with `pct()`, use `abs()` with directional words ("above"/"below") to avoid double-negatives like "-12.3% below"
- Benchmark aggregations: use `median` for duration (named `med_duration`), `mean` for costs
- Export button labels: use "📥 Excel" / "📥 PDF" (compact, consistent)
- AI system prompt category grouping must match dashboard: Core Labor (7) / Core Materials (4) / Other (6)
