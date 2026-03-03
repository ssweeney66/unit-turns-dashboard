import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Full Turn Analytics | Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
TREND_YEARS = [2021, 2022, 2023, 2024, 2025]
CHART_TEMPLATE = "plotly_white"

# ── Materials vs Labor category classification ──
MATERIALS_CATS = [
    "Supplies", "Appliances", "Flooring Materials",
    "Cabinets Materials", "Countertops Materials", "Windows",
]
LABOR_CATS = [
    "Labor General", "Flooring Labor", "Electric General",
    "Countertops Labor", "Plumbing", "Powerwash and Demo",
    "Management Fee", "Scrape Ceiling", "Glaze", "Cabinets Labor", "Paint",
]

# ── Expense analysis constants ──
EXPENSE_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
CORE_LABOR = [
    "Flooring Labor", "Countertops Labor", "Cabinets Labor",
    "Paint", "Glaze", "Labor General", "Management Fee",
]
CORE_MATERIALS = [
    "Flooring Materials", "Countertops Materials",
    "Cabinets Materials", "Appliances",
]
OTHER_CATS = [
    "Electric General", "Plumbing", "Scrape Ceiling",
    "Supplies", "Windows", "Powerwash and Demo",
]
COST_TYPE_COLORS = {
    "Materials": "#2563eb",
    "Labor":     "#f59e0b",
    "Mixed":     "#10b981",
    "Fee":       "#6366f1",
}
COST_TYPES = ["Materials", "Labor", "Mixed", "Fee"]

MAT_COLORS = {
    "Supplies":              "#2563eb",
    "Appliances":            "#ef4444",
    "Flooring Materials":    "#f59e0b",
    "Cabinets Materials":    "#8b5cf6",
    "Countertops Materials": "#10b981",
    "Windows":               "#06b6d4",
}
LAB_COLORS = {
    "Labor General":    "#0ea5e9",
    "Flooring Labor":   "#f97316",
    "Electric General": "#eab308",
    "Countertops Labor":"#14b8a6",
    "Plumbing":         "#3b82f6",
    "Powerwash and Demo":"#a855f7",
    "Management Fee":   "#64748b",
    "Scrape Ceiling":   "#ec4899",
    "Glaze":            "#84cc16",
    "Cabinets Labor":   "#f43f5e",
    "Paint":            "#6366f1",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STYLING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h3 { color: #ffffff !important; }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        border-left: 4px solid #2563eb;
    }
    div[data-testid="stMetric"] label {
        font-size: 11px !important; text-transform: uppercase;
        letter-spacing: 0.6px; color: #64748b !important; font-weight: 600 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 26px !important; font-weight: 700 !important; color: #0f172a !important;
    }

    /* Tables */
    .stDataFrame th {
        background: #f1f5f9 !important; font-weight: 600 !important;
        text-transform: uppercase; font-size: 11px !important;
        letter-spacing: 0.5px; color: #475569 !important;
    }
    .stDataFrame td { font-size: 13px !important; }

    /* Headers */
    .page-banner {
        background: linear-gradient(135deg, #0f172a, #1e40af);
        color: white; padding: 28px 32px; border-radius: 12px; margin-bottom: 24px;
    }
    .page-banner h1 { color: white !important; margin: 0 !important; font-size: 26px !important; font-weight: 700; }
    .page-banner p  { color: #93c5fd; margin: 4px 0 0 0; font-size: 13px; }

    .section-bar {
        border-left: 4px solid #2563eb; background: #f8fafc;
        padding: 10px 16px; margin: 28px 0 16px 0; border-radius: 0 8px 8px 0;
    }
    .section-bar h3 { margin: 0 !important; font-size: 15px !important; color: #1e293b !important; font-weight: 600; }

    .insight-box {
        background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px;
        padding: 16px 20px; margin: 8px 0 16px 0; font-size: 13.5px; line-height: 1.7; color: #1e293b;
    }
    .insight-box strong { color: #1e40af; }

    .outlier-flag {
        background: #fef2f2; border: 1px solid #fecaca; border-left: 4px solid #dc2626;
        border-radius: 8px; padding: 12px 16px; margin: 6px 0; font-size: 13px; color: #7f1d1d;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 14px; }
    .stTabs [aria-selected="true"] { border-bottom-color: #2563eb !important; }

    /* Footer */
    .dashboard-footer {
        margin-top: 60px; padding: 20px 0; border-top: 1px solid #e2e8f0;
        text-align: center; font-size: 11px; color: #94a3b8; letter-spacing: 0.3px;
    }

    /* Sidebar active view accent */
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
        background: rgba(37,99,235,0.15) !important; border-radius: 6px;
    }

</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def banner(title, subtitle):
    st.markdown(f'<div class="page-banner"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-bar"><h3>{title}</h3></div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def fmt(val, decimals=0):
    if pd.isna(val):
        return "—"
    if val < 0:
        return f"-${abs(val):,.{decimals}f}"
    return f"${val:,.{decimals}f}"

def pct(val):
    if pd.isna(val):
        return "—"
    return f"{val:+.1f}%"

def expense_year_label(yr):
    return "2026 YTD" if yr == 2026 else str(yr)

EXPENSE_YEAR_LABELS = [expense_year_label(y) for y in EXPENSE_YEARS]

def footer():
    st.markdown(
        '<div class="dashboard-footer">'
        'CONFIDENTIAL — Full Turn Analytics Dashboard &nbsp;|&nbsp; '
        f'Data as of Feb 2026 &nbsp;|&nbsp; '
        'Prepared for Executive Review'
        '</div>',
        unsafe_allow_html=True,
    )


def render_category_table(title, categories, data, years=None, year_labels=None, yoy_pairs=None):
    """Render a pivot table for a set of budget categories with YoY change column(s).

    Parameters
    ----------
    title : str – section title
    categories : list – budget category names to include
    data : DataFrame – must have columns: Budget Category, Year, avg_per_turn
    years : list[int] – year columns (default EXPENSE_YEARS)
    year_labels : list[str] – display labels for year columns (default EXPENSE_YEAR_LABELS)
    yoy_pairs : list[tuple] – explicit (prior_year, current_year) pairs for YoY columns.
                               If None, auto-generates a single column from the last two years.
    """
    if years is None:
        years = EXPENSE_YEARS
    if year_labels is None:
        year_labels = EXPENSE_YEAR_LABELS

    st.markdown(f"**{title}**")
    subset = data[data["Budget Category"].isin(categories)]
    if len(subset) == 0:
        st.info(f"No data for {title.split(' (')[0]} categories.")
        return
    pivot = subset.pivot_table(
        index="Budget Category", columns="Year",
        values="avg_per_turn", fill_value=0
    ).reindex(columns=years, fill_value=0)
    pivot = pivot.reindex([c for c in categories if c in pivot.index])
    pivot = pivot.fillna(0)
    if len(pivot) == 0:
        st.info(f"No data for {title.split(' (')[0]} categories.")
        return
    pivot = pivot.loc[pivot.sum(axis=1) > 0]
    if len(pivot) == 0:
        st.info(f"No data for {title.split(' (')[0]} categories.")
        return
    # Add Total row
    pivot.loc["Total"] = pivot.sum()

    # Build YoY pairs list
    if yoy_pairs is None and len(years) >= 2:
        yoy_pairs = [(years[-2], years[-1])]
    elif yoy_pairs is None:
        yoy_pairs = []

    # Compute YoY columns
    yoy_columns = {}
    for prior, current in yoy_pairs:
        if prior in pivot.columns and current in pivot.columns:
            yoy_vals = pivot[current] / pivot[prior].replace(0, np.nan) - 1
            label = f"'{str(prior)[-2:]}→'{str(current)[-2:]}"
            yoy_columns[label] = yoy_vals

    # Format
    pivot_display = pivot.copy()
    pivot_display.columns = year_labels
    for col in pivot_display.columns:
        pivot_display[col] = pivot_display[col].apply(fmt)
    for label, yoy_vals in yoy_columns.items():
        pivot_display[label] = yoy_vals.apply(
            lambda x: f"{x:+.0%}" if pd.notna(x) and np.isfinite(x) else "—"
        )
    st.dataframe(pivot_display, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data
def load_data():
    path = Path(__file__).parent / "Unit Turns - AI Clean - 2.26.2026.xlsx"
    df = pd.read_excel(path)

    df["Building Code"] = df["Building Code"].fillna("").astype(str).str.strip()
    df["Unit Number"]   = df["Unit Number"].fillna("").astype(str).str.strip()
    df["Property ID"]   = df["Property ID"].fillna("").astype(str).str.strip()
    df["Invoice Amount"] = pd.to_numeric(df["Invoice Amount"], errors="coerce").fillna(0)
    df["Move-Out Date"]  = pd.to_datetime(df["Move-Out Date"], errors="coerce")
    df["Invoice Date"]   = pd.to_datetime(df["Invoice Date"], errors="coerce")

    # Cap obviously wrong dates
    cap = pd.Timestamp(datetime.now().year + 1, 12, 31)
    df.loc[df["Invoice Date"] > cap, "Invoice Date"] = pd.NaT

    df = df[df["Property ID"] != ""].copy()

    # ── Unique Unit Key ──
    def uid(r):
        if r["Building Code"] != "":
            return f"{r['Property ID']}|{r['Building Code']}|{r['Unit Number']}"
        return f"{r['Property ID']}|{r['Unit Number']}"

    def uid_display(r):
        if r["Building Code"] != "":
            return f"{r['Building Code']} | {r['Unit Number']}"
        return r["Unit Number"]

    df["UID"]         = df.apply(uid, axis=1)
    df["Unit Label"]  = df.apply(uid_display, axis=1)
    df["Turn Key"]    = df["UID"] + "|" + df["Move-Out Date"].astype(str)

    # ── Full Turn only ──
    ft = df[df["Turn Type"] == "Full Turn"].copy()
    ft["Year"] = ft["Move-Out Date"].dt.year

    # ── Turn-level summary ──
    turns = (
        ft.groupby(["Property ID", "Property Name", "UID", "Unit Label",
                     "Move-Out Date", "Turn Key", "Floor Plan",
                     "Bedrooms", "Bathrooms", "Year"])
        .agg(total_cost=("Invoice Amount", "sum"),
             line_items=("Invoice Amount", "count"),
             completion_date=("Invoice Date", "max"))
        .reset_index()
    )
    turns["Duration"] = (turns["completion_date"] - turns["Move-Out Date"]).dt.days
    turns.loc[turns["Duration"] < 0, "Duration"] = pd.NA

    return df, ft, turns


_df_all, ft_lines, ft_turns = load_data()
PROPERTY_ORDER = [
    "Monterey Park", "Woodman", "Collins", "Lindley", "El Rancho",
    "51 at the Village", "Alta Vista", "Roscoe", "Topanga", "Darby",
    "Fruitland", "Dickens", "Garfield", "Woodbridge",
]
_all_props = set(ft_turns["Property Name"].unique())
PROPERTIES = [p for p in PROPERTY_ORDER if p in _all_props] + sorted(_all_props - set(PROPERTY_ORDER))
_PROP_RANK = {name: i for i, name in enumerate(PROPERTIES)}

def prop_sort_key(names):
    """Return sort keys for a Series/Index of property names using PROPERTY_ORDER."""
    return names.map(lambda n: _PROP_RANK.get(n, 999))


@st.cache_data
def build_turn_summary(source_df):
    return (
        source_df.groupby(["Property ID", "Property Name", "UID",
                           "Unit Label", "Move-Out Date", "Turn Type",
                           "Floor Plan", "Bedrooms", "Bathrooms", "Turn Key"])
        .agg(total_cost=("Invoice Amount", "sum"),
             line_items=("Invoice Amount", "count"),
             earliest_invoice=("Invoice Date", "min"),
             latest_invoice=("Invoice Date", "max"))
        .reset_index()
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.sidebar.markdown("### FULL TURN ANALYTICS")
st.sidebar.caption("Portfolio Renovation Intelligence Platform")
st.sidebar.markdown("---")

view = st.sidebar.radio("View", [
    "1 — Executive Summary",
    "2 — Portfolio Overview",
    "3 — Property Summary",
    "4 — Category Trends",
    "5 — Unit Search",
    "6 — Data Review LLM",
])

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**{len(ft_turns):,}** Full Turns  •  **{ft_turns['Property ID'].nunique()}** Properties  •  "
    f"**{ft_turns['UID'].nunique()}** Units  •  **{fmt(ft_turns['total_cost'].sum())}** Total Spend"
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 3: PROPERTY SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if view == "3 — Property Summary":
    banner("Property Summary", "Renovation volume, floor plan comparison, and standardized expense analysis by property")

    prop = st.selectbox("Select Property", PROPERTIES)
    p_turns = ft_turns[ft_turns["Property Name"] == prop].copy()
    p_lines = ft_lines[ft_lines["Property Name"] == prop].copy()

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Full Turns", f"{len(p_turns):,}")
    c2.metric("Total Spend", fmt(p_turns["total_cost"].sum()))
    c3.metric("Avg Cost", fmt(p_turns["total_cost"].mean()))
    c4.metric("Median Cost", fmt(p_turns["total_cost"].median()))
    dur = p_turns["Duration"].dropna()
    c5.metric("Median Duration", f"{dur.median():.0f} days" if len(dur) else "—")

    # ── YoY Count ──
    section("Year-over-Year Full Turn Count (2016 – 2025)")

    yoy = p_turns[p_turns["Year"].isin(YEARS)].groupby("Year").size().reindex(YEARS, fill_value=0)
    yoy_df = pd.DataFrame({"Year": YEARS, "Full Turns": yoy.values})

    col1, col2 = st.columns([2, 3])
    with col1:
        # Add YoY delta
        yoy_df["vs Prior Year"] = yoy_df["Full Turns"].diff()
        yoy_df["vs Prior Year"] = yoy_df["vs Prior Year"].apply(
            lambda x: f"+{x:.0f}" if pd.notna(x) and x > 0 else (f"{x:.0f}" if pd.notna(x) and x != 0 else "—")
        )
        yoy_df["Year"] = yoy_df["Year"].astype(str)
        st.dataframe(yoy_df, use_container_width=True, hide_index=True)

    with col2:
        fig = px.bar(
            yoy_df, x="Year", y="Full Turns", text="Full Turns",
            template=CHART_TEMPLATE, color_discrete_sequence=["#2563eb"],
        )
        fig.update_traces(textposition="outside", textfont_size=13)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280,
                          xaxis_title="", yaxis_title="Turns")
        st.plotly_chart(fig, use_container_width=True)

    # ── Turns by Floor Plan ──
    section("Full Turns by Floor Plan")

    fp_data = p_turns.groupby("Floor Plan").agg(
        turns=("Turn Key", "count"),
        avg_cost=("total_cost", "mean"),
        total=("total_cost", "sum"),
    ).reset_index().sort_values("turns", ascending=False)

    col1, col2 = st.columns([2, 3])
    with col1:
        fp_display = fp_data.copy()
        fp_display["avg_cost"] = fp_display["avg_cost"].apply(fmt)
        fp_display["total"] = fp_display["total"].apply(fmt)
        fp_display.columns = ["Floor Plan", "Turns", "Avg Cost", "Total Spend"]
        st.dataframe(fp_display, use_container_width=True, hide_index=True)

    with col2:
        fig_fp = px.bar(
            fp_data, x="Floor Plan", y="turns", text="turns",
            template=CHART_TEMPLATE, color_discrete_sequence=["#0ea5e9"],
        )
        fig_fp.update_traces(textposition="outside")
        fig_fp.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280,
                             xaxis_title="", yaxis_title="Turns")
        st.plotly_chart(fig_fp, use_container_width=True)

    # ══════════════════════════════════════════════════
    # EXPENSE SUMMARY BY COST TYPE
    # ══════════════════════════════════════════════════

    # ── Part 1: Combined summary (property-wide, all years) ──
    section("Avg Full Turn Cost by Floor Plan — Cost Type Breakdown")
    st.caption("Average cost per Full Turn by floor plan and cost type (property-wide, all years)")

    fp_turn_counts = p_turns.groupby("Floor Plan")["Turn Key"].nunique().reset_index(name="n_turns")
    fp_ct_spend = (
        p_lines.groupby(["Floor Plan", "Cost Type"])["Invoice Amount"]
        .sum().reset_index(name="total_spend")
    )
    fp_ct_spend = fp_ct_spend.merge(fp_turn_counts, on="Floor Plan", how="left")
    fp_ct_spend["avg_per_turn"] = fp_ct_spend.apply(
        lambda r: r["total_spend"] / r["n_turns"] if r["n_turns"] > 0 else 0, axis=1
    )

    fp_ct_pivot = fp_ct_spend.pivot_table(
        index="Floor Plan", columns="Cost Type",
        values="avg_per_turn", fill_value=0
    ).reindex(columns=COST_TYPES, fill_value=0)
    fp_ct_pivot["Total"] = fp_ct_pivot.sum(axis=1)
    fp_ct_pivot = fp_ct_pivot.sort_values("Total", ascending=False)

    fp_ct_display = fp_ct_pivot.copy()
    for col in fp_ct_display.columns:
        fp_ct_display[col] = fp_ct_display[col].apply(fmt)
    st.dataframe(fp_ct_display, use_container_width=True)

    # ── Part 2: Year-by-year Cost Type trend (property-wide) ──
    section(f"Cost Type Trend by Year — {prop}")

    trend_lines = p_lines[p_lines["Year"].isin(EXPENSE_YEARS)].copy()
    trend_turn_counts = (
        p_turns[p_turns["Year"].isin(EXPENSE_YEARS)]
        .groupby("Year")["Turn Key"].nunique()
        .reindex(EXPENSE_YEARS, fill_value=0)
    )

    ct_year_spend = (
        trend_lines.groupby(["Cost Type", "Year"])["Invoice Amount"]
        .sum().reset_index(name="total_spend")
    )
    ct_year_spend["n_turns"] = ct_year_spend["Year"].map(trend_turn_counts).fillna(0)
    ct_year_spend["avg_per_turn"] = ct_year_spend.apply(
        lambda r: r["total_spend"] / r["n_turns"] if r["n_turns"] > 0 else 0, axis=1
    )

    ct_trend_pivot = ct_year_spend.pivot_table(
        index="Cost Type", columns="Year",
        values="avg_per_turn", fill_value=0
    ).reindex(columns=EXPENSE_YEARS, fill_value=0).reindex(COST_TYPES)
    ct_trend_pivot = ct_trend_pivot.fillna(0)

    # Add Total row
    ct_trend_pivot.loc["Total"] = ct_trend_pivot.sum()

    col1, col2 = st.columns([2, 3])
    with col1:
        ct_trend_display = ct_trend_pivot.copy()
        ct_trend_display.columns = EXPENSE_YEAR_LABELS
        for col in ct_trend_display.columns:
            ct_trend_display[col] = ct_trend_display[col].apply(fmt)
        st.dataframe(ct_trend_display, use_container_width=True)

    with col2:
        fig_ct = go.Figure()
        for ct in COST_TYPES:
            row = ct_trend_pivot.loc[ct] if ct in ct_trend_pivot.index else pd.Series(0, index=EXPENSE_YEARS)
            fig_ct.add_trace(go.Bar(
                x=EXPENSE_YEAR_LABELS,
                y=[row.get(y, 0) for y in EXPENSE_YEARS],
                name=ct,
                marker_color=COST_TYPE_COLORS.get(ct, "#94a3b8"),
                hovertemplate=f"{ct}<br>%{{x}}: $%{{y:,.0f}}<extra></extra>",
            ))
        fig_ct.update_layout(
            template=CHART_TEMPLATE, barmode="stack",
            xaxis=dict(title=""), yaxis=dict(title="Avg Cost per Turn ($)"),
            legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
            margin=dict(t=10, b=50, l=10, r=10), height=340,
        )
        st.plotly_chart(fig_ct, use_container_width=True)

    # ══════════════════════════════════════════════════
    # DETAILED EXPENSE ANALYSIS BY BUDGET CATEGORY
    # ══════════════════════════════════════════════════
    section(f"Expense Analysis by Budget Category — {prop}")
    st.caption(f"Average cost per Full Turn by budget category ({EXPENSE_YEARS[0]} – {expense_year_label(EXPENSE_YEARS[-1])})")

    # Pre-compute: spend by budget category and year
    cat_year_spend = (
        trend_lines.groupby(["Budget Category", "Year"])["Invoice Amount"]
        .sum().reset_index(name="total_spend")
    )
    cat_year_spend["n_turns"] = cat_year_spend["Year"].map(trend_turn_counts).fillna(0)
    cat_year_spend["avg_per_turn"] = cat_year_spend.apply(
        lambda r: r["total_spend"] / r["n_turns"] if r["n_turns"] > 0 else 0, axis=1
    )

    prop_yoy_pairs = [(2023, 2024), (2024, 2025), (2025, 2026)]
    render_category_table("Core Labor (Avg per Turn)", CORE_LABOR, cat_year_spend,
                          yoy_pairs=prop_yoy_pairs)
    st.markdown("")
    render_category_table("Core Materials (Avg per Turn)", CORE_MATERIALS, cat_year_spend,
                          yoy_pairs=prop_yoy_pairs)
    st.markdown("")
    render_category_table("Other Categories (Avg per Turn)", OTHER_CATS, cat_year_spend,
                          yoy_pairs=prop_yoy_pairs)

    # ══════════════════════════════════════════════════
    # NARRATIVE INSIGHT
    # ══════════════════════════════════════════════════
    total_spend = p_turns["total_cost"].sum()
    avg_cost = p_turns["total_cost"].mean()
    port_avg_all = ft_turns["total_cost"].mean()
    vs_port = ((avg_cost - port_avg_all) / port_avg_all * 100) if port_avg_all > 0 else 0

    insight(
        f"<strong>{prop}</strong> has completed <strong>{len(p_turns):,}</strong> Full Turns "
        f"totaling <strong>{fmt(total_spend)}</strong>. "
        f"Average cost per turn is <strong>{fmt(avg_cost)}</strong>, which is "
        f"<strong>{pct(vs_port)}</strong> vs the portfolio average of "
        f"<strong>{fmt(port_avg_all)}</strong>."
    )

    # ══════════════════════════════════════════════════
    # RECENT TURNS BY FLOOR PLAN (drill-down)
    # ══════════════════════════════════════════════════
    st.markdown("---")
    floor_plans = sorted(p_turns["Floor Plan"].dropna().unique())
    fp_options = ["All Floor Plans"] + floor_plans
    selected_fp = st.selectbox("Filter by Floor Plan", fp_options, key="prop_fp")

    if selected_fp == "All Floor Plans":
        fp_turns = p_turns.copy()
        fp_lines = p_lines.copy()
    else:
        fp_turns = p_turns[p_turns["Floor Plan"] == selected_fp].copy()
        fp_lines = p_lines[p_lines["Floor Plan"] == selected_fp].copy()

    fp_label = selected_fp if selected_fp != "All Floor Plans" else "All Floor Plans"

    section(f"Last 5 Completed Full Turns — {fp_label}")

    last5 = fp_turns.sort_values("completion_date", ascending=False).head(5).copy()

    if len(last5) == 0:
        st.info(f"No completed Full Turns found for {fp_label}.")
    else:
        last5["#"] = range(1, len(last5) + 1)
        last5["Completion"] = last5["completion_date"].dt.strftime("%b %d, %Y").fillna("—")
        last5["Move-Out"] = last5["Move-Out Date"].dt.strftime("%b %d, %Y").fillna("—")
        last5["Cost"] = last5["total_cost"].apply(fmt)
        last5["Dur"] = last5["Duration"].apply(lambda x: f"{x:.0f}d" if pd.notna(x) else "—")

        l5_disp = last5[["#", "Unit Label", "Floor Plan", "Move-Out", "Completion",
                         "Cost", "Dur", "line_items"]].copy()
        l5_disp.columns = ["#", "Unit", "Floor Plan", "Move-Out", "Completion",
                           "Total Cost", "Duration", "Invoices"]
        st.dataframe(l5_disp, use_container_width=True, hide_index=True)

        for _, t in last5.iterrows():
            with st.expander(f"Detail — {t['Unit Label']} — {t['Move-Out']}"):
                items = fp_lines[fp_lines["Turn Key"] == t["Turn Key"]].sort_values("Invoice Date")
                d = items[["Vendor Name", "Budget Category", "Cost Type",
                           "Invoice Amount", "Invoice Date", "Line Item Notes"]].copy()
                d["Invoice Amount"] = d["Invoice Amount"].apply(lambda x: fmt(x, 2))
                d["Invoice Date"] = d["Invoice Date"].dt.strftime("%b %d, %Y").fillna("—")
                d["Line Item Notes"] = d["Line Item Notes"].fillna("")
                st.dataframe(d, use_container_width=True, hide_index=True)

    footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 2: PORTFOLIO OVERVIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif view == "2 — Portfolio Overview":
    banner("Portfolio Summary", "Average Full Turn cost per property — 2016 through 2025")

    SUMMARY_YEARS = YEARS

    # KPIs
    recent = ft_turns[ft_turns["Year"].isin(SUMMARY_YEARS)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Full Turns (2016–25)", f"{len(recent):,}")
    c2.metric("Total Spend", fmt(recent["total_cost"].sum()))
    c3.metric("Portfolio Avg Cost", fmt(recent["total_cost"].mean()))
    c4.metric("Properties Active", f"{recent['Property ID'].nunique()}")

    section("Average Full Turn Cost by Property & Year")

    # Avg cost matrix
    avg_matrix = (
        recent.groupby(["Property Name", "Year"])["total_cost"]
        .mean().unstack(fill_value=0)
        .reindex(columns=SUMMARY_YEARS, fill_value=0)
    )

    # Count matrix
    count_matrix = (
        recent.groupby(["Property Name", "Year"]).size()
        .unstack(fill_value=0)
        .reindex(columns=SUMMARY_YEARS, fill_value=0)
    )

    # All-years avg
    avg_matrix["Avg (All Years)"] = avg_matrix[SUMMARY_YEARS].replace(0, np.nan).mean(axis=1)

    # YoY change
    avg_matrix["2024 → 2025"] = avg_matrix.apply(
        lambda r: ((r[2025] - r[2024]) / r[2024] * 100) if r[2024] > 0 and r[2025] > 0 else np.nan,
        axis=1,
    )

    avg_matrix = avg_matrix.loc[sorted(avg_matrix.index, key=lambda n: _PROP_RANK.get(n, 999))]

    # Portfolio total row
    portfolio_row = {}
    for y in SUMMARY_YEARS:
        yr_data = recent[recent["Year"] == y]["total_cost"]
        portfolio_row[y] = yr_data.mean() if len(yr_data) else 0
    portfolio_row["Avg (All Years)"] = recent["total_cost"].mean()
    portfolio_row["2024 → 2025"] = (
        ((portfolio_row[2025] - portfolio_row[2024]) / portfolio_row[2024] * 100)
        if portfolio_row[2024] > 0 and portfolio_row[2025] > 0 else np.nan
    )
    avg_matrix.loc["PORTFOLIO AVG"] = portfolio_row

    tab_avg, tab_vol = st.tabs(["Average Cost per Turn", "Turn Volume"])

    with tab_avg:
        display = avg_matrix.copy()
        for y in SUMMARY_YEARS + ["Avg (All Years)"]:
            display[y] = display[y].apply(lambda x: fmt(x) if x > 0 else "—")
        display["2024 → 2025"] = display["2024 → 2025"].apply(
            lambda x: pct(x) if pd.notna(x) else "—"
        )
        st.dataframe(display, use_container_width=True, height=560)

        # Narrative
        prop_only = avg_matrix.drop("PORTFOLIO AVG")
        port_avg = avg_matrix.loc["PORTFOLIO AVG", "Avg (All Years)"]
        yoy_chg = avg_matrix.loc["PORTFOLIO AVG", "2024 → 2025"]

        if len(prop_only) >= 2:
            top_prop = prop_only.sort_values("Avg (All Years)", ascending=False).index[0]
            low_candidates = prop_only[prop_only["Avg (All Years)"] > 0].sort_values("Avg (All Years)", ascending=True)
            low_prop = low_candidates.index[0] if len(low_candidates) else top_prop
            insight(
                f"Portfolio-wide average Full Turn cost is <strong>{fmt(port_avg)}</strong> over 2016–2025. "
                f"<strong>{top_prop}</strong> has the highest average, while <strong>{low_prop}</strong> runs lowest. "
                f"Year-over-year, portfolio costs moved <strong>{pct(yoy_chg)}</strong> from 2024 to 2025."
            )
        else:
            insight(
                f"Portfolio-wide average Full Turn cost is <strong>{fmt(port_avg)}</strong> over 2016–2025. "
                f"Year-over-year, portfolio costs moved <strong>{pct(yoy_chg)}</strong> from 2024 to 2025."
            )

    with tab_vol:
        count_matrix["Total"] = count_matrix.sum(axis=1)
        count_matrix = count_matrix.loc[sorted(count_matrix.index, key=lambda n: _PROP_RANK.get(n, 999))]
        count_matrix.loc["PORTFOLIO TOTAL"] = count_matrix.sum()
        count_matrix = count_matrix.astype(int)
        st.dataframe(count_matrix, use_container_width=True, height=560)

    footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 4: CATEGORY TRENDS (Portfolio-Wide)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif view == "4 — Category Trends":
    banner("Category Cost Trends",
           f"Portfolio-wide budget category analysis — {TREND_YEARS[0]} through {TREND_YEARS[-1]}")

    # ── Data prep (portfolio-wide, TREND_YEARS) ──
    ft_trend = ft_lines[ft_lines["Year"].isin(TREND_YEARS)].copy()
    trend_turn_counts = (
        ft_turns[ft_turns["Year"].isin(TREND_YEARS)]
        .groupby("Year")["Turn Key"].nunique()
        .reindex(TREND_YEARS, fill_value=0)
    )
    n_turns_total = ft_turns[ft_turns["Year"].isin(TREND_YEARS)]["Turn Key"].nunique()
    total_spend_5yr = ft_trend["Invoice Amount"].sum()
    avg_per_turn_overall = total_spend_5yr / n_turns_total if n_turns_total > 0 else 0

    mat_spend = ft_trend[ft_trend["Budget Category"].isin(MATERIALS_CATS)]["Invoice Amount"].sum()
    lab_spend = ft_trend[ft_trend["Budget Category"].isin(LABOR_CATS)]["Invoice Amount"].sum()
    mat_per_turn = mat_spend / n_turns_total if n_turns_total > 0 else 0
    lab_per_turn = lab_spend / n_turns_total if n_turns_total > 0 else 0

    # Cost-type level aggregation
    ct_year_spend = (
        ft_trend.groupby(["Cost Type", "Year"])["Invoice Amount"]
        .sum().reset_index(name="total_spend")
    )
    ct_year_spend["n_turns"] = ct_year_spend["Year"].map(trend_turn_counts).fillna(0)
    ct_year_spend["avg_per_turn"] = ct_year_spend.apply(
        lambda r: r["total_spend"] / r["n_turns"] if r["n_turns"] > 0 else 0, axis=1
    )

    # Budget-category level aggregation
    cat_year_spend = (
        ft_trend.groupby(["Budget Category", "Year"])["Invoice Amount"]
        .sum().reset_index(name="total_spend")
    )
    cat_year_spend["n_turns"] = cat_year_spend["Year"].map(trend_turn_counts).fillna(0)
    cat_year_spend["avg_per_turn"] = cat_year_spend.apply(
        lambda r: r["total_spend"] / r["n_turns"] if r["n_turns"] > 0 else 0, axis=1
    )

    # YoY change on total avg cost per turn
    avg_2024 = ct_year_spend[ct_year_spend["Year"] == 2024]["avg_per_turn"].sum()
    avg_2025 = ct_year_spend[ct_year_spend["Year"] == 2025]["avg_per_turn"].sum()
    yoy_total = ((avg_2025 - avg_2024) / avg_2024 * 100) if avg_2024 > 0 else np.nan

    # ── KPIs ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Cost / Turn", fmt(avg_per_turn_overall),
              (pct(yoy_total) + " YoY") if pd.notna(yoy_total) else None)
    c2.metric("Materials / Turn", fmt(mat_per_turn))
    c3.metric("Labor / Turn", fmt(lab_per_turn))
    c4.metric("Total Turns (5-Yr)", f"{n_turns_total:,}")
    c5.metric("Total Spend (5-Yr)", fmt(total_spend_5yr))

    # ══════════════════════════════════════════════════
    # COST TYPE TREND BY YEAR (table + stacked bar)
    # ══════════════════════════════════════════════════
    section("Cost Type Trend by Year — Portfolio")

    TREND_YEAR_LABELS = [str(y) for y in TREND_YEARS]

    ct_trend_pivot = ct_year_spend.pivot_table(
        index="Cost Type", columns="Year",
        values="avg_per_turn", fill_value=0
    ).reindex(columns=TREND_YEARS, fill_value=0).reindex(COST_TYPES)
    ct_trend_pivot = ct_trend_pivot.fillna(0)
    ct_trend_pivot.loc["Total"] = ct_trend_pivot.sum()

    col1, col2 = st.columns([2, 3])
    with col1:
        ct_display = ct_trend_pivot.copy()
        yoy_ct = ct_trend_pivot[TREND_YEARS[-1]] / ct_trend_pivot[TREND_YEARS[-2]].replace(0, np.nan) - 1
        ct_display.columns = TREND_YEAR_LABELS
        for col in ct_display.columns:
            ct_display[col] = ct_display[col].apply(fmt)
        ct_display["'24→'25"] = yoy_ct.apply(
            lambda x: f"{x:+.0%}" if pd.notna(x) and np.isfinite(x) else "—"
        )
        st.dataframe(ct_display, use_container_width=True)

    with col2:
        fig_ct = go.Figure()
        for ct in COST_TYPES:
            row = ct_trend_pivot.loc[ct] if ct in ct_trend_pivot.index else pd.Series(0, index=TREND_YEARS)
            fig_ct.add_trace(go.Bar(
                x=TREND_YEAR_LABELS,
                y=[row.get(y, 0) for y in TREND_YEARS],
                name=ct,
                marker_color=COST_TYPE_COLORS.get(ct, "#94a3b8"),
                hovertemplate=f"{ct}<br>%{{x}}: $%{{y:,.0f}}<extra></extra>",
            ))
        fig_ct.update_layout(
            template=CHART_TEMPLATE, barmode="stack",
            xaxis=dict(title=""), yaxis=dict(title="Avg Cost per Turn ($)"),
            legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
            margin=dict(t=10, b=50, l=10, r=10), height=340,
        )
        st.plotly_chart(fig_ct, use_container_width=True)

    # ══════════════════════════════════════════════════
    # EXPENSE ANALYSIS BY BUDGET CATEGORY
    # ══════════════════════════════════════════════════
    section("Expense Analysis by Budget Category — Portfolio")
    st.caption(f"Average cost per Full Turn by budget category ({TREND_YEARS[0]}–{TREND_YEARS[-1]})")

    render_category_table("Core Labor (Avg per Turn)", CORE_LABOR, cat_year_spend,
                          years=TREND_YEARS, year_labels=TREND_YEAR_LABELS)
    st.markdown("")
    render_category_table("Core Materials (Avg per Turn)", CORE_MATERIALS, cat_year_spend,
                          years=TREND_YEARS, year_labels=TREND_YEAR_LABELS)
    st.markdown("")
    render_category_table("Other Categories (Avg per Turn)", OTHER_CATS, cat_year_spend,
                          years=TREND_YEARS, year_labels=TREND_YEAR_LABELS)

    # ══════════════════════════════════════════════════
    # CATEGORY OUTLIERS — LAST 12 MONTHS
    # ══════════════════════════════════════════════════
    section("Category Outliers — Last 12 Months")
    st.caption("Per-property category spend exceeding mean + 1.5 standard deviations on recent Full Turns.")

    ft_all = ft_lines.copy()
    prop_cat = (
        ft_all.groupby(["Property Name", "Budget Category", "Turn Key"])["Invoice Amount"]
        .sum().reset_index()
    )
    prop_cat_stats = (
        prop_cat.groupby(["Property Name", "Budget Category"])["Invoice Amount"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    prop_cat_stats["threshold"] = prop_cat_stats["mean"] + 1.5 * prop_cat_stats["std"]

    flagged = prop_cat.merge(prop_cat_stats, on=["Property Name", "Budget Category"])
    flagged = flagged[
        (flagged["Invoice Amount"] > flagged["threshold"])
        & (flagged["count"] >= 3)
    ].copy()
    flagged["Excess"] = flagged["Invoice Amount"] - flagged["mean"]

    turn_info = ft_turns[["Turn Key", "Property Name", "Unit Label", "Move-Out Date"]].drop_duplicates("Turn Key")
    flagged = flagged.merge(turn_info, on=["Turn Key", "Property Name"], how="left")

    cutoff_12m = pd.Timestamp.now() - pd.DateOffset(months=12)
    recent_outliers = flagged[flagged["Move-Out Date"] >= cutoff_12m].copy()
    recent_outliers["_order"] = prop_sort_key(recent_outliers["Property Name"])
    recent_outliers = recent_outliers.sort_values(["_order", "Excess"], ascending=[True, False]).drop(columns="_order")

    c1, c2, c3 = st.columns(3)
    c1.metric("Outlier Flags", len(recent_outliers))
    c2.metric("Total Excess Spend", fmt(recent_outliers["Excess"].sum()) if len(recent_outliers) else "$0")
    top_cat = recent_outliers.groupby("Budget Category")["Excess"].sum().idxmax() if len(recent_outliers) else "None"
    c3.metric("Top Flagged Category", top_cat)

    if len(recent_outliers) > 0:
        r_display = recent_outliers[[
            "Property Name", "Unit Label", "Move-Out Date", "Budget Category",
            "Invoice Amount", "mean", "Excess"
        ]].copy()
        r_display["Move-Out Date"] = r_display["Move-Out Date"].dt.strftime("%b %d, %Y").fillna("—")
        r_display["Invoice Amount"] = r_display["Invoice Amount"].apply(fmt)
        r_display["mean"] = r_display["mean"].apply(fmt)
        r_display["Excess"] = r_display["Excess"].apply(fmt)
        r_display.columns = ["Property", "Unit", "Move-Out", "Category",
                              "Actual", "Property Avg", "Excess"]
        st.dataframe(r_display, use_container_width=True, hide_index=True,
                     height=min(400, 60 + len(recent_outliers) * 35))
    else:
        st.success("No category outliers detected in the past 12 months.")

    # ══════════════════════════════════════════════════
    # NARRATIVE INSIGHT
    # ══════════════════════════════════════════════════
    mat_share = f"{mat_spend / total_spend_5yr * 100:.0f}%" if total_spend_5yr > 0 else "—"
    lab_share = f"{lab_spend / total_spend_5yr * 100:.0f}%" if total_spend_5yr > 0 else "—"

    cat_2024 = cat_year_spend[cat_year_spend["Year"] == 2024].set_index("Budget Category")["avg_per_turn"]
    cat_2025 = cat_year_spend[cat_year_spend["Year"] == 2025].set_index("Budget Category")["avg_per_turn"]
    cat_delta = (cat_2025 - cat_2024).dropna().sort_values()

    insight_parts = [
        f"Across the portfolio ({TREND_YEARS[0]}–{TREND_YEARS[-1]}), "
        f"Materials represent <strong>{mat_share}</strong> of tracked spend "
        f"(<strong>{fmt(mat_per_turn)}</strong>/turn) while Labor accounts for "
        f"<strong>{lab_share}</strong> (<strong>{fmt(lab_per_turn)}</strong>/turn).",
    ]
    if len(cat_delta) > 0 and cat_delta.iloc[-1] > 0:
        insight_parts.append(
            f"<strong>{cat_delta.index[-1]}</strong> saw the largest per-turn cost increase "
            f"year-over-year (<strong>+{fmt(cat_delta.iloc[-1])}</strong>)."
        )
    if len(cat_delta) > 0 and cat_delta.iloc[0] < 0:
        insight_parts.append(
            f"<strong>{cat_delta.index[0]}</strong> declined by "
            f"<strong>{fmt(abs(cat_delta.iloc[0]))}</strong>/turn."
        )

    insight(" ".join(insight_parts))

    footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 5: UNIT SEARCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif view == "5 — Unit Search":
    banner("Unit Search & History", "Full renovation history for any unit in the portfolio")

    # Use all-types turn summary for this view
    all_turns = build_turn_summary(_df_all)

    col1, col2 = st.columns(2)
    with col1:
        _all_unit_props = list(_df_all["Property Name"].unique())
        _all_unit_props_ordered = [p for p in PROPERTIES if p in _all_unit_props] + sorted(set(_all_unit_props) - set(PROPERTIES))
        prop_choice = st.selectbox("Property", _all_unit_props_ordered)
    with col2:
        prop_units = sorted(_df_all[_df_all["Property Name"] == prop_choice]["Unit Label"].unique())
        unit_choice = st.selectbox("Unit", prop_units)

    unit_ts = all_turns[
        (all_turns["Property Name"] == prop_choice) & (all_turns["Unit Label"] == unit_choice)
    ].sort_values("Move-Out Date", ascending=False)

    unit_df = _df_all[
        (_df_all["Property Name"] == prop_choice) & (_df_all["Unit Label"] == unit_choice)
    ]

    if len(unit_ts) == 0:
        st.info("No turn history found for this unit.")
    else:
        sample = unit_df.iloc[0]
        bed = f"{sample['Bedrooms']:.0f}" if pd.notna(sample['Bedrooms']) else "—"
        bath = f"{sample['Bathrooms']:.0f}" if pd.notna(sample['Bathrooms']) else "—"
        st.markdown(
            f"**{prop_choice} — Unit {unit_choice}** &nbsp;|&nbsp; "
            f"Floor Plan: `{sample['Floor Plan']}` &nbsp;|&nbsp; "
            f"Bed/Bath: `{bed} / {bath}` &nbsp;|&nbsp; "
            f"ID: `{sample['UID']}`"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Turns", len(unit_ts))
        c2.metric("Lifetime Spend", fmt(unit_ts["total_cost"].sum()))
        c3.metric("Avg Cost / Turn", fmt(unit_ts["total_cost"].mean()))

        section("Turn History")
        for _, turn in unit_ts.iterrows():
            with st.expander(
                f"**{turn['Turn Type']}** — {turn['Move-Out Date'].strftime('%b %d, %Y') if pd.notna(turn['Move-Out Date']) else '—'} — "
                f"{fmt(turn['total_cost'])} ({turn['line_items']} invoices)"
            ):
                items = unit_df[unit_df["Turn Key"] == turn["Turn Key"]].sort_values("Invoice Date")
                disp = items[["Invoice Date", "Vendor Name", "Budget Category",
                              "Cost Type", "Invoice Amount", "Line Item Notes"]].copy()
                disp["Invoice Date"] = disp["Invoice Date"].dt.strftime("%b %d, %Y").fillna("—")
                disp["Invoice Amount"] = disp["Invoice Amount"].apply(lambda x: fmt(x, 2))
                disp["Line Item Notes"] = disp["Line Item Notes"].fillna("")
                st.dataframe(disp, use_container_width=True, hide_index=True)

        # ══════════════════════════════════════════════════
        # PRE-SCOPING — MOST RECENT FULL TURN
        # ══════════════════════════════════════════════════
        unit_ft = unit_ts[unit_ts["Turn Type"] == "Full Turn"]
        if len(unit_ft) > 0:
            last_ft = unit_ft.sort_values("Move-Out Date", ascending=False).iloc[0]
            last_ft_date = last_ft["Move-Out Date"]
            last_ft_key = last_ft["Turn Key"]
            last_ft_items = unit_df[unit_df["Turn Key"] == last_ft_key].copy()

            section(f"Pre-Scoping — Most Recent Full Turn ({last_ft_date.strftime('%b %Y') if pd.notna(last_ft_date) else '—'})")

            # Category breakdown
            cat_brkdn = (
                last_ft_items.groupby("Budget Category")["Invoice Amount"]
                .sum().reset_index(name="Amount")
            )
            total_last_ft = cat_brkdn["Amount"].sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Last Full Turn Cost", fmt(total_last_ft))
            c2.metric("Move-Out", last_ft_date.strftime("%b %d, %Y") if pd.notna(last_ft_date) else "—")
            dur_last = last_ft["latest_invoice"] - last_ft_date if pd.notna(last_ft.get("latest_invoice")) and pd.notna(last_ft_date) else pd.NaT
            c3.metric("Duration", f"{dur_last.days}d" if pd.notna(dur_last) and hasattr(dur_last, "days") and dur_last.days >= 0 else "—")

            # Grouped display: Core Labor → Core Materials → Other
            cat_amounts = cat_brkdn.set_index("Budget Category")["Amount"]
            for group_label, group_cats in [
                ("Core Labor", CORE_LABOR),
                ("Core Materials", CORE_MATERIALS),
                ("Other", OTHER_CATS),
            ]:
                group_data = [(c, cat_amounts.get(c, 0)) for c in group_cats if cat_amounts.get(c, 0) > 0]
                if not group_data:
                    continue
                group_data.sort(key=lambda x: x[1], reverse=True)
                subtotal = sum(v for _, v in group_data)
                rows = [{"Budget Category": c, "Amount": v, "% of Total": f"{v / total_last_ft * 100:.0f}%" if total_last_ft > 0 else "—"} for c, v in group_data]
                rows.append({"Budget Category": f"{group_label} Subtotal", "Amount": subtotal, "% of Total": f"{subtotal / total_last_ft * 100:.0f}%" if total_last_ft > 0 else "—"})
                gdf = pd.DataFrame(rows)
                gdf["Amount"] = gdf["Amount"].apply(lambda x: fmt(x, 2))
                st.markdown(f"**{group_label}**")
                st.dataframe(gdf, use_container_width=True, hide_index=True)

            # Vendor summary
            vendor_brkdn = (
                last_ft_items.groupby("Vendor Name")["Invoice Amount"]
                .sum().reset_index(name="Amount")
                .sort_values("Amount", ascending=False)
            )
            vendor_brkdn["Amount"] = vendor_brkdn["Amount"].apply(lambda x: fmt(x, 2))
            vendor_brkdn.columns = ["Vendor", "Total"]
            with st.expander("Vendor Breakdown"):
                st.dataframe(vendor_brkdn, use_container_width=True, hide_index=True)
        else:
            section("Pre-Scoping")
            st.info("No Full Turn history for this unit. See projected costs below for budget guidance.")

        # ══════════════════════════════════════════════════
        # PROJECTED TURN COST
        # ══════════════════════════════════════════════════
        section("Projected Turn Cost — Recommended Scope & Budget")

        unit_floor_plan = unit_df.iloc[0]["Floor Plan"] if len(unit_df) > 0 else None
        has_prior_ft = len(unit_ft) > 0

        # Determine projected turn type
        if has_prior_ft:
            proj_type = "Make Ready"
            st.caption(
                f"This unit has a prior Full Turn — projecting a **Make Ready** scope. "
                f"Based on {unit_floor_plan} Make Ready data at {prop_choice} over the last 2 years."
            )
        else:
            proj_type = "Full Turn"
            st.caption(
                f"No prior Full Turn on record — projecting a **Full Turn** scope. "
                f"Based on {unit_floor_plan} Full Turn data at {prop_choice} over the last 2 years."
            )

        # Build comps: same floor plan, same property, same turn type, last 2 years
        cutoff_2yr = pd.Timestamp.now() - pd.DateOffset(years=2)
        comp_lines = _df_all[
            (_df_all["Property Name"] == prop_choice)
            & (_df_all["Floor Plan"] == unit_floor_plan)
            & (_df_all["Turn Type"] == proj_type)
            & (_df_all["Move-Out Date"] >= cutoff_2yr)
        ].copy()

        # Count comp turns
        comp_turn_keys = comp_lines["Turn Key"].nunique()

        if comp_turn_keys >= 1:
            # Category-level avg per turn
            comp_cat = (
                comp_lines.groupby("Budget Category")["Invoice Amount"]
                .sum().reset_index(name="total_spend")
            )
            comp_cat["Avg per Turn"] = comp_cat["total_spend"] / comp_turn_keys
            projected_total = comp_cat["Avg per Turn"].sum()
            proj_amounts = comp_cat.set_index("Budget Category")["Avg per Turn"]

            c1, c2, c3 = st.columns(3)
            c1.metric(f"Projected {proj_type} Cost", fmt(projected_total))
            c2.metric("Based on Comps", f"{comp_turn_keys} {proj_type}s")
            c3.metric("Floor Plan", unit_floor_plan if unit_floor_plan else "—")

            # Grouped scope of work: Core Labor → Core Materials → Other
            st.markdown(f"**Recommended Scope of Work — {proj_type}**")
            for group_label, group_cats in [
                ("Core Labor", CORE_LABOR),
                ("Core Materials", CORE_MATERIALS),
                ("Other", OTHER_CATS),
            ]:
                group_data = [(c, proj_amounts.get(c, 0)) for c in group_cats if proj_amounts.get(c, 0) > 0]
                if not group_data:
                    continue
                group_data.sort(key=lambda x: x[1], reverse=True)
                subtotal = sum(v for _, v in group_data)
                rows = [{"Budget Category": c, "Projected Cost": v, "% of Total": f"{v / projected_total * 100:.0f}%" if projected_total > 0 else "—"} for c, v in group_data]
                rows.append({"Budget Category": f"{group_label} Subtotal", "Projected Cost": subtotal, "% of Total": f"{subtotal / projected_total * 100:.0f}%" if projected_total > 0 else "—"})
                gdf = pd.DataFrame(rows)
                gdf["Projected Cost"] = gdf["Projected Cost"].apply(fmt)
                st.markdown(f"**{group_label}**")
                st.dataframe(gdf, use_container_width=True, hide_index=True)

            # Comparison to last Full Turn if available
            if has_prior_ft:
                delta = projected_total - total_last_ft
                insight(
                    f"<strong>Budget Guidance:</strong> Based on <strong>{comp_turn_keys}</strong> recent "
                    f"{proj_type}s for {unit_floor_plan} units at {prop_choice}, "
                    f"expect approximately <strong>{fmt(projected_total)}</strong>. "
                    f"The last Full Turn on this unit cost <strong>{fmt(total_last_ft)}</strong> — "
                    f"a Make Ready is typically a fraction of Full Turn scope."
                )
            else:
                insight(
                    f"<strong>Budget Guidance:</strong> Based on <strong>{comp_turn_keys}</strong> recent "
                    f"Full Turns for {unit_floor_plan} units at {prop_choice}, "
                    f"expect approximately <strong>{fmt(projected_total)}</strong>."
                )
        else:
            # Fallback: no comps for this exact combination, try property-wide
            fallback_lines = _df_all[
                (_df_all["Property Name"] == prop_choice)
                & (_df_all["Turn Type"] == proj_type)
                & (_df_all["Move-Out Date"] >= cutoff_2yr)
            ].copy()
            fallback_turns = fallback_lines["Turn Key"].nunique()
            if fallback_turns > 0:
                fallback_total = fallback_lines["Invoice Amount"].sum() / fallback_turns
                st.info(
                    f"No {proj_type} comps for {unit_floor_plan} at {prop_choice} in the last 2 years. "
                    f"Using property-wide {proj_type} average: **{fmt(fallback_total)}** "
                    f"(based on {fallback_turns} turns)."
                )
            else:
                st.info(f"No recent {proj_type} data available at {prop_choice} for projection.")

    footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 1: EXECUTIVE SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif view == "1 — Executive Summary":
    banner("Executive Intelligence", "Strategic performance overview for senior leadership — Full Turn portfolio analytics")

    # ── Compute core metrics ──
    curr_year = ft_turns[ft_turns["Year"] == 2025]
    prev_year = ft_turns[ft_turns["Year"] == 2024]

    portfolio_avg = ft_turns["total_cost"].mean()
    curr_avg = curr_year["total_cost"].mean() if len(curr_year) else 0
    prev_avg = prev_year["total_cost"].mean() if len(prev_year) else 0
    yoy_delta = ((curr_avg - prev_avg) / prev_avg * 100) if prev_avg > 0 else np.nan

    curr_vol = len(curr_year)
    prev_vol = len(prev_year)
    vol_delta = ((curr_vol - prev_vol) / prev_vol * 100) if prev_vol > 0 else np.nan

    dur_curr = curr_year["Duration"].dropna()
    dur_prev = prev_year["Duration"].dropna()
    dur_now = dur_curr.median() if len(dur_curr) else np.nan
    dur_then = dur_prev.median() if len(dur_prev) else np.nan

    # ── Top-Line KPIs ──
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("2025 Full Turns", f"{curr_vol:,}", f"{vol_delta:+.0f}% vs '24" if pd.notna(vol_delta) else "—")
    c2.metric("2025 Avg Cost", fmt(curr_avg), (pct(yoy_delta) + " vs '24") if pd.notna(yoy_delta) else "—")
    c3.metric("Portfolio Avg (All)", fmt(portfolio_avg))
    c4.metric("2025 Total Spend", fmt(curr_year["total_cost"].sum()))
    c5.metric("Median Duration", f"{dur_now:.0f}d" if pd.notna(dur_now) else "—",
              f"{dur_now - dur_then:+.0f}d vs '24" if (pd.notna(dur_now) and pd.notna(dur_then)) else "")
    c6.metric("Active Properties", f"{curr_year['Property ID'].nunique()}" if len(curr_year) else "0")

    # ━━ Section 1: Cost Trajectory ━━
    section("Cost Trajectory — Are We Getting More Efficient?")

    yearly_stats = ft_turns[ft_turns["Year"].isin(YEARS)].groupby("Year").agg(
        avg_cost=("total_cost", "mean"),
        median_cost=("total_cost", "median"),
        total_spend=("total_cost", "sum"),
        turn_count=("Turn Key", "count"),
    ).reset_index()

    col1, col2 = st.columns([3, 2])
    with col1:
        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(
            x=yearly_stats["Year"], y=yearly_stats["avg_cost"],
            name="Avg Cost", mode="lines+markers",
            line=dict(color="#2563eb", width=3), marker=dict(size=9),
            hovertemplate="Avg: $%{y:,.0f}<extra></extra>",
        ))
        fig_traj.add_trace(go.Scatter(
            x=yearly_stats["Year"], y=yearly_stats["median_cost"],
            name="Median Cost", mode="lines+markers",
            line=dict(color="#10b981", width=2, dash="dash"), marker=dict(size=7),
            hovertemplate="Median: $%{y:,.0f}<extra></extra>",
        ))
        fig_traj.update_layout(
            template=CHART_TEMPLATE,
            xaxis=dict(dtick=1, title=""), yaxis=dict(title="Cost per Full Turn ($)"),
            legend=dict(orientation="h", y=-0.12), margin=dict(t=10, b=50), height=380,
            hovermode="x unified",
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    with col2:
        # Year-over-year change table
        yearly_stats["YoY Change"] = yearly_stats["avg_cost"].pct_change() * 100
        ys_display = yearly_stats[["Year", "turn_count", "avg_cost", "median_cost", "total_spend", "YoY Change"]].copy()
        ys_display["avg_cost"] = ys_display["avg_cost"].apply(fmt)
        ys_display["median_cost"] = ys_display["median_cost"].apply(fmt)
        ys_display["total_spend"] = ys_display["total_spend"].apply(fmt)
        ys_display["YoY Change"] = ys_display["YoY Change"].apply(lambda x: pct(x) if pd.notna(x) else "—")
        ys_display["Year"] = ys_display["Year"].astype(int).astype(str)
        ys_display.columns = ["Year", "Turns", "Avg Cost", "Median", "Total Spend", "YoY Δ"]
        st.dataframe(ys_display, use_container_width=True, hide_index=True, height=380)

    # Cost efficiency narrative
    if len(yearly_stats) >= 2:
        first_yr = yearly_stats.iloc[0]
        last_yr = yearly_stats.iloc[-1]
        total_chg = ((last_yr["avg_cost"] - first_yr["avg_cost"]) / first_yr["avg_cost"]) * 100 if first_yr["avg_cost"] > 0 else 0
        insight(
            f"From <strong>{int(first_yr['Year'])}</strong> to <strong>{int(last_yr['Year'])}</strong>, "
            f"average Full Turn cost moved from <strong>{fmt(first_yr['avg_cost'])}</strong> to "
            f"<strong>{fmt(last_yr['avg_cost'])}</strong> — a cumulative shift of <strong>{pct(total_chg)}</strong>. "
            f"Total capital deployed: <strong>{fmt(ft_turns['total_cost'].sum())}</strong> across "
            f"<strong>{len(ft_turns):,}</strong> Full Turns."
        )

    # ━━ Section 2: Property Benchmarking ━━
    section("Property Benchmarking — Cost Efficiency Ranking")

    prop_bench = ft_turns[ft_turns["Year"].isin(YEARS)].groupby("Property Name").agg(
        turns=("Turn Key", "count"),
        avg_cost=("total_cost", "mean"),
        median_cost=("total_cost", "median"),
        total_spend=("total_cost", "sum"),
        avg_duration=("Duration", "median"),
    ).reset_index().sort_values("avg_cost", ascending=False)

    # Add rank (1 = highest avg cost)
    prop_bench["Rank"] = range(1, len(prop_bench) + 1)
    prop_bench["vs Portfolio"] = ((prop_bench["avg_cost"] - portfolio_avg) / portfolio_avg * 100) if portfolio_avg > 0 else 0

    col1, col2 = st.columns([3, 2])
    with col1:
        # Chart: lowest cost at top, highest at bottom (ascending for horizontal bar)
        chart_data = prop_bench.sort_values("avg_cost", ascending=True)
        fig_bench = px.bar(
            chart_data,
            y="Property Name", x="avg_cost", orientation="h",
            text=chart_data["avg_cost"].apply(fmt),
            template=CHART_TEMPLATE,
            color="avg_cost",
            color_continuous_scale=["#10b981", "#f59e0b", "#dc2626"],
        )
        fig_bench.update_traces(textposition="outside")
        fig_bench.add_vline(x=portfolio_avg, line_dash="dash", line_color="#2563eb",
                            annotation_text=f"Portfolio Avg: {fmt(portfolio_avg)}",
                            annotation_position="top right", annotation_font_size=11)
        fig_bench.update_layout(
            margin=dict(t=20, b=10, l=10, r=100), height=480,
            xaxis_title="Avg Full Turn Cost ($)", yaxis_title="",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bench, use_container_width=True)

    with col2:
        pb_display = prop_bench[["Rank", "Property Name", "turns", "avg_cost", "vs Portfolio", "avg_duration"]].copy()
        pb_display["avg_cost"] = pb_display["avg_cost"].apply(fmt)
        pb_display["vs Portfolio"] = pb_display["vs Portfolio"].apply(lambda x: pct(x))
        pb_display["avg_duration"] = pb_display["avg_duration"].apply(lambda x: f"{x:.0f}d" if pd.notna(x) else "—")
        pb_display.columns = ["#", "Property", "Turns", "Avg Cost", "vs Portfolio", "Med Duration"]
        st.dataframe(pb_display, use_container_width=True, hide_index=True, height=480)

    # Identify best and worst performers
    if len(prop_bench) >= 2:
        best = prop_bench.sort_values("avg_cost").iloc[0]
        worst = prop_bench.sort_values("avg_cost").iloc[-1]
        insight(
            f"<strong>{worst['Property Name']}</strong> has the highest average Full Turn cost at "
            f"<strong>{fmt(worst['avg_cost'])}</strong> ({pct(worst['vs Portfolio'])} above portfolio avg), "
            f"while <strong>{best['Property Name']}</strong> runs most efficiently at "
            f"<strong>{fmt(best['avg_cost'])}</strong>. Investigating what drives this gap could yield significant savings."
        )

    # ━━ Section 3: Vendor Concentration ━━
    section("Vendor Concentration — Risk & Spend Distribution")

    vendor_data = ft_lines[ft_lines["Year"].isin(YEARS)].groupby("Vendor Name").agg(
        total_spend=("Invoice Amount", "sum"),
        invoices=("Invoice Amount", "count"),
        properties=("Property Name", "nunique"),
        avg_invoice=("Invoice Amount", "mean"),
    ).reset_index().sort_values("total_spend", ascending=False)

    total_vendor_spend = vendor_data["total_spend"].sum()
    vendor_data["Share"] = (vendor_data["total_spend"] / total_vendor_spend * 100) if total_vendor_spend > 0 else 0
    vendor_data["Cumulative"] = vendor_data["Share"].cumsum()

    top_n = 10
    top_vendors = vendor_data.head(top_n)
    top_share = top_vendors["Share"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Vendors", f"{len(vendor_data):,}")
    c2.metric(f"Top {top_n} Share", f"{top_share:.1f}%")
    c3.metric("Single-Property Vendors", f"{len(vendor_data[vendor_data['properties'] == 1]):,}")

    col1, col2 = st.columns([3, 2])
    with col1:
        fig_vendor = px.bar(
            top_vendors, y="Vendor Name", x="total_spend", orientation="h",
            text=top_vendors["total_spend"].apply(fmt),
            template=CHART_TEMPLATE, color_discrete_sequence=["#6366f1"],
        )
        fig_vendor.update_traces(textposition="outside")
        fig_vendor.update_layout(
            margin=dict(t=10, b=10, l=10, r=100), height=400,
            xaxis_title="Total Spend ($)", yaxis_title="",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_vendor, use_container_width=True)

    with col2:
        v_display = top_vendors[["Vendor Name", "total_spend", "Share", "invoices", "properties"]].copy()
        v_display["total_spend"] = v_display["total_spend"].apply(fmt)
        v_display["Share"] = v_display["Share"].apply(lambda x: f"{x:.1f}%")
        v_display.columns = ["Vendor", "Total Spend", "Portfolio %", "Invoices", "Properties"]
        st.dataframe(v_display, use_container_width=True, hide_index=True, height=400)

    if top_share > 60:
        st.markdown(f'<div class="outlier-flag"><strong>Concentration Risk:</strong> Top {top_n} vendors control '
                    f'<strong>{top_share:.1f}%</strong> of spend. Consider diversifying to reduce dependency and '
                    f'improve pricing leverage.</div>', unsafe_allow_html=True)

    # ━━ Section 4: Capital Forecast ━━
    section("Capital Forecast — Projected Annual Full Turn Spend")

    # Use 3-year trailing average for projection
    hist_years = [2023, 2024, 2025]
    hist_data = ft_turns[ft_turns["Year"].isin(hist_years)].groupby("Year").agg(
        turns=("Turn Key", "count"),
        total=("total_cost", "sum"),
        avg_cost=("total_cost", "mean"),
    ).reset_index()

    if len(hist_data) >= 2:
        avg_turns = hist_data["turns"].mean()
        avg_cost_trend = hist_data["avg_cost"].mean()
        projected_spend = avg_turns * avg_cost_trend

        # Cost trend (linear)
        if len(hist_data) >= 2:
            cost_slope = np.polyfit(hist_data["Year"], hist_data["avg_cost"], 1)
            projected_cost_2026 = np.polyval(cost_slope, 2026)
            projected_cost_2026 = max(projected_cost_2026, 0)

            vol_slope = np.polyfit(hist_data["Year"], hist_data["turns"], 1)
            projected_vol_2026 = max(np.polyval(vol_slope, 2026), 0)

            forecast_spend = projected_vol_2026 * projected_cost_2026

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("3-Year Avg Turns/Year", f"{avg_turns:.0f}")
        c2.metric("3-Year Avg Cost", fmt(avg_cost_trend))
        c3.metric("2026 Proj. Avg Cost", fmt(projected_cost_2026))
        c4.metric("2026 Proj. Spend", fmt(forecast_spend))

        # Forecast chart
        all_years_data = ft_turns[ft_turns["Year"].isin(YEARS)].groupby("Year").agg(
            total=("total_cost", "sum"),
        ).reset_index()

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Bar(
            x=all_years_data["Year"], y=all_years_data["total"],
            name="Actual", marker_color="#2563eb",
            hovertemplate="Actual: $%{y:,.0f}<extra></extra>",
        ))
        fig_fc.add_trace(go.Bar(
            x=[2026], y=[forecast_spend],
            name="Projected", marker_color="#94a3b8", opacity=0.6,
            hovertemplate="Projected: $%{y:,.0f}<extra></extra>",
        ))
        fig_fc.update_layout(
            template=CHART_TEMPLATE, barmode="group",
            xaxis=dict(dtick=1, title=""), yaxis=dict(title="Annual Full Turn Spend ($)"),
            legend=dict(orientation="h", y=-0.12), margin=dict(t=10, b=50), height=380,
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        insight(
            f"Based on 2023–2025 trends, the portfolio is projected to complete approximately "
            f"<strong>{projected_vol_2026:.0f}</strong> Full Turns in 2026 at an average cost of "
            f"<strong>{fmt(projected_cost_2026)}</strong>, totaling an estimated "
            f"<strong>{fmt(forecast_spend)}</strong> in capital deployment."
        )

    # ━━ Section 5: Key Risk Flags ━━
    section("Risk Flags — Items Requiring Executive Attention")

    risk_items = []

    # 1. Properties with rising costs (2024 → 2025)
    for prop in PROPERTIES:
        p24 = ft_turns[(ft_turns["Property Name"] == prop) & (ft_turns["Year"] == 2024)]["total_cost"]
        p25 = ft_turns[(ft_turns["Property Name"] == prop) & (ft_turns["Year"] == 2025)]["total_cost"]
        if len(p24) >= 2 and len(p25) >= 2 and p24.mean() > 0:
            chg = ((p25.mean() - p24.mean()) / p24.mean()) * 100
            if chg > 20:
                risk_items.append({
                    "Risk": "Cost Escalation",
                    "Detail": f"{prop}: avg cost up {chg:.0f}% YoY ({fmt(p24.mean())} → {fmt(p25.mean())})",
                    "Severity": "High" if chg > 40 else "Medium",
                })

    # 2. Properties with slow duration
    for prop in PROPERTIES:
        p_dur = ft_turns[(ft_turns["Property Name"] == prop) & (ft_turns["Year"] == 2025)]["Duration"].dropna()
        if len(p_dur) >= 3 and p_dur.median() > 60:
            risk_items.append({
                "Risk": "Slow Velocity",
                "Detail": f"{prop}: median turn duration {p_dur.median():.0f} days (>60d threshold)",
                "Severity": "Medium",
            })

    # 3. Category inflation (per-turn spend, all 17 raw categories)
    for cat_name in MATERIALS_CATS + LABOR_CATS:
        cat24_turns = ft_lines[(ft_lines["Year"] == 2024) & (ft_lines["Budget Category"] == cat_name)].groupby("Turn Key")["Invoice Amount"].sum()
        cat25_turns = ft_lines[(ft_lines["Year"] == 2025) & (ft_lines["Budget Category"] == cat_name)].groupby("Turn Key")["Invoice Amount"].sum()
        if len(cat24_turns) >= 5 and len(cat25_turns) >= 5:
            avg24, avg25 = cat24_turns.mean(), cat25_turns.mean()
            chg = ((avg25 - avg24) / avg24) * 100 if avg24 > 0 else 0
            if chg > 25 and (avg25 - avg24) > 100:
                risk_items.append({
                    "Risk": "Category Inflation",
                    "Detail": f"{cat_name}: avg per-turn spend up {chg:.0f}% YoY ({fmt(avg24)} → {fmt(avg25)})",
                    "Severity": "High" if chg > 50 else "Medium",
                })

    # 4. High-frequency units (>4 Full Turns all-time)
    freq_units = ft_turns.groupby(["Property Name", "Unit Label"])["Turn Key"].nunique().reset_index(name="turns")
    chronic = freq_units[freq_units["turns"] >= 5]
    if len(chronic) > 0:
        risk_items.append({
            "Risk": "Chronic Vacancy",
            "Detail": f"{len(chronic)} units with 5+ Full Turns — potential workmanship or tenant screening issues",
            "Severity": "Medium",
        })

    if risk_items:
        risk_df = pd.DataFrame(risk_items)
        # Sort: High first
        severity_order = {"High": 0, "Medium": 1, "Low": 2}
        risk_df["_sort"] = risk_df["Severity"].map(severity_order)
        risk_df = risk_df.sort_values("_sort").drop("_sort", axis=1)

        for _, r in risk_df.iterrows():
            color = "#dc2626" if r["Severity"] == "High" else "#f59e0b"
            icon = "🔴" if r["Severity"] == "High" else "🟡"
            st.markdown(
                f'<div style="background: {"#fef2f2" if r["Severity"] == "High" else "#fffbeb"}; '
                f'border: 1px solid {"#fecaca" if r["Severity"] == "High" else "#fde68a"}; '
                f'border-left: 4px solid {color}; border-radius: 8px; padding: 12px 16px; margin: 6px 0; font-size: 13px;">'
                f'<strong>{icon} {r["Risk"]}</strong> — {r["Detail"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")  # spacer
        insight(
            f"<strong>{len([r for r in risk_items if r['Severity'] == 'High'])}</strong> high-severity and "
            f"<strong>{len([r for r in risk_items if r['Severity'] == 'Medium'])}</strong> medium-severity risk "
            f"flags identified. Cost escalation items should be investigated first — they represent the most "
            f"immediate impact to capital deployment efficiency."
        )
    else:
        st.success("No risk flags identified — portfolio metrics are within normal ranges.")

    footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 6: DATA REVIEW LLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif view == "6 — Data Review LLM":
    banner("Data Review LLM", "Ask questions about your Full Turn portfolio data — powered by AI")

    # ── Build data context for the LLM ──
    @st.cache_data
    def build_data_context(_ft_turns, _ft_lines, _df_all):
        """Generate a compact summary of the portfolio data for the LLM system prompt."""
        lines = []
        lines.append("=== PORTFOLIO OVERVIEW ===")
        lines.append(f"Total Full Turns: {len(_ft_turns):,}")
        lines.append(f"Properties: {_ft_turns['Property Name'].nunique()}")
        lines.append(f"Unique Units: {_ft_turns['UID'].nunique()}")
        min_date = _ft_turns['Move-Out Date'].min()
        max_date = _ft_turns['Move-Out Date'].max()
        if pd.notna(min_date) and pd.notna(max_date):
            lines.append(f"Date Range: {min_date.strftime('%b %Y')} to {max_date.strftime('%b %Y')}")
        lines.append(f"Total Spend: ${_ft_turns['total_cost'].sum():,.0f}")
        lines.append(f"Avg Cost per Turn: ${_ft_turns['total_cost'].mean():,.0f}")
        lines.append(f"Median Cost per Turn: ${_ft_turns['total_cost'].median():,.0f}")
        dur = _ft_turns["Duration"].dropna()
        if len(dur) > 0:
            lines.append(f"Avg Duration: {dur.mean():.0f} days | Median: {dur.median():.0f} days")

        lines.append("\n=== PROPERTY BREAKDOWN ===")
        prop_stats = _ft_turns.groupby("Property Name").agg(
            turns=("Turn Key", "count"),
            avg_cost=("total_cost", "mean"),
            total_spend=("total_cost", "sum"),
        )
        prop_stats = prop_stats.loc[sorted(prop_stats.index, key=lambda n: _PROP_RANK.get(n, 999))]
        for prop, row in prop_stats.iterrows():
            lines.append(f"  {prop}: {row['turns']} turns, avg ${row['avg_cost']:,.0f}, total ${row['total_spend']:,.0f}")

        lines.append("\n=== YEARLY TRENDS ===")
        yearly = _ft_turns.groupby("Year").agg(
            turns=("Turn Key", "count"),
            avg_cost=("total_cost", "mean"),
            total_spend=("total_cost", "sum"),
        )
        for yr, row in yearly.iterrows():
            lines.append(f"  {int(yr)}: {row['turns']} turns, avg ${row['avg_cost']:,.0f}, total ${row['total_spend']:,.0f}")

        lines.append("\n=== TOP BUDGET CATEGORIES (by total spend) ===")
        cat_spend = _ft_lines.groupby("Budget Category")["Invoice Amount"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
        for cat, row in cat_spend.head(17).iterrows():
            lines.append(f"  {cat}: ${row['sum']:,.0f} total, ${row['mean']:,.0f} avg invoice, {int(row['count'])} invoices")

        lines.append("\n=== FLOOR PLAN MIX ===")
        fp_stats = _ft_turns.groupby("Floor Plan").agg(
            turns=("Turn Key", "count"),
            avg_cost=("total_cost", "mean"),
        ).sort_values("turns", ascending=False)
        for fp, row in fp_stats.head(10).iterrows():
            lines.append(f"  {fp}: {row['turns']} turns, avg ${row['avg_cost']:,.0f}")

        lines.append("\n=== VENDOR SUMMARY (Top 15) ===")
        vendor_stats = _ft_lines.groupby("Vendor Name")["Invoice Amount"].agg(["sum", "count"]).sort_values("sum", ascending=False)
        for v, row in vendor_stats.head(15).iterrows():
            lines.append(f"  {v}: ${row['sum']:,.0f} total, {int(row['count'])} invoices")

        return "\n".join(lines)

    data_context = build_data_context(ft_turns, ft_lines, _df_all)

    SYSTEM_PROMPT = f"""You are a senior multifamily real estate analytics assistant embedded in a Full Turn renovation dashboard.
Your role is to answer questions about this portfolio's renovation (Full Turn) data with precision, clarity, and executive-level insight.

Key definitions:
- Full Turn: A complete unit renovation after a tenant moves out (avg $15-23K)
- Turn Key: Unique identifier for each turn event (Property + Unit + Move-Out Date)
- Duration: Days from move-out to last invoice (renovation timeline)
- Budget Categories: 17 categories split into Materials (Supplies, Appliances, Flooring Materials, Cabinets Materials, Countertops Materials, Windows) and Labor (Labor General, Flooring Labor, Electric General, Countertops Labor, Plumbing, Powerwash and Demo, Management Fee, Scrape Ceiling, Glaze, Cabinets Labor, Paint)

Here is the current portfolio data summary:

{data_context}

Guidelines:
- Always cite specific numbers from the data when answering
- Format currency as $X,XXX
- If you're unsure or the data doesn't support an answer, say so clearly
- Provide actionable insights when relevant
- Keep responses concise but thorough (3-5 sentences for simple questions, more for complex analysis)
- When comparing properties, always rank them
- Use percentage changes for YoY comparisons"""

    # ── API Key handling ──
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key to enable the Data Review LLM. Your key is never stored."
    )

    if not api_key:
        st.info("🔑 Enter your OpenAI API key above to start asking questions about your data. "
                "Your key is used only for this session and is never stored.")
        st.markdown("**Example questions you can ask:**")
        st.markdown("""
        - *Which property has the highest average Full Turn cost?*
        - *How has our total spend changed year over year?*
        - *What are the top 3 budget categories by spend?*
        - *Compare 2024 vs 2025 performance across the portfolio*
        - *Which properties are getting more expensive over time?*
        - *What's our average renovation duration?*
        - *Who are our top vendors and how much do we spend with each?*
        """)
        footer()
    else:
        client = OpenAI(api_key=api_key)

        # Initialize chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Display chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your Full Turn data..."):
            # Show user message
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Build messages for API
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            # Include last 10 messages for context
            for msg in st.session_state.chat_messages[-10:]:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your data..."):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=api_messages,
                            temperature=0.3,
                            max_tokens=1000,
                        )
                        answer = response.choices[0].message.content
                        st.markdown(answer)
                        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = str(e)
                        if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
                            st.error("❌ Invalid API key. Please check your OpenAI API key and try again.")
                        else:
                            st.error(f"❌ Error: {error_msg}")

        # Clear chat button
        if st.session_state.chat_messages:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_messages = []
                st.rerun()

        footer()
