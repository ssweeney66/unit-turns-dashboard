import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Full Turn Analytics | Portfolio Dashboard",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

YEARS = [2021, 2022, 2023, 2024, 2025]
CHART_TEMPLATE = "plotly_white"

# The 5 tracked spend categories and how they map to raw Budget Category values
CATEGORY_MAP = {
    "Paint":        ["Paint"],
    "Labor General": ["Labor General"],
    "Flooring":     ["Flooring Materials", "Flooring Labor"],
    "Countertops":  ["Countertops Labor", "Countertops Materials"],
    "Appliances":   ["Appliances"],
}
CAT_COLORS = {
    "Paint":         "#6366f1",
    "Labor General": "#0ea5e9",
    "Flooring":      "#f59e0b",
    "Countertops":   "#10b981",
    "Appliances":    "#ef4444",
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
    if pd.isna(val) or val == 0:
        return "—"
    return f"${val:,.{decimals}f}"

def pct(val):
    if pd.isna(val):
        return "—"
    return f"{val:+.1f}%"


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

    # ── Map to tracked categories ──
    reverse_map = {}
    for group, raw_cats in CATEGORY_MAP.items():
        for rc in raw_cats:
            reverse_map[rc] = group
    ft["Tracked Category"] = ft["Budget Category"].map(reverse_map).fillna("Other")

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
PROPERTIES = sorted(ft_turns["Property Name"].unique())


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
st.sidebar.caption("Filtered to Full Renovations only")
st.sidebar.markdown("---")

view = st.sidebar.radio("View", [
    "1 — Property Level",
    "2 — Portfolio Summary",
    "3 — 5-Year Category Trend",
    "4 — Recent 10 Audit",
    "5 — Unit Search",
    "6 — Anomaly Detection",
])

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**{len(ft_turns):,}** Full Turns  •  **{ft_turns['Property ID'].nunique()}** Properties  •  "
    f"**{ft_turns['UID'].nunique()}** Units  •  **{fmt(ft_turns['total_cost'].sum())}** Total Spend"
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 1: PROPERTY LEVEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if view == "1 — Property Level":
    banner("Property-Level Full Turn Analysis", "Select a property to review renovation volume, floor plan mix, and recent completions")

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

    # ── YoY Count 2021-2025 ──
    section("Year-over-Year Full Turn Count (2021 – 2025)")

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

    # ── Last 5 Completed Turns ──
    section("Last 5 Completed Full Turns")

    last5 = p_turns.sort_values("completion_date", ascending=False).head(5).copy()
    last5["#"] = range(1, len(last5) + 1)
    last5["Completion"] = last5["completion_date"].dt.strftime("%b %d, %Y").fillna("—")
    last5["Move-Out"] = last5["Move-Out Date"].dt.strftime("%b %d, %Y")
    last5["Cost"] = last5["total_cost"].apply(fmt)
    last5["Dur"] = last5["Duration"].apply(lambda x: f"{x:.0f}d" if pd.notna(x) else "—")

    disp = last5[["#", "Unit Label", "Floor Plan", "Move-Out", "Completion", "Cost", "Dur", "line_items"]]
    disp = disp.rename(columns={"Unit Label": "Unit", "Dur": "Duration", "line_items": "Invoices"})
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # Expandable detail for each
    for _, t in last5.iterrows():
        with st.expander(f"Detail — {t['Unit Label']} — {t['Move-Out']}"):
            items = p_lines[p_lines["Turn Key"] == t["Turn Key"]].sort_values("Invoice Date")
            d = items[["Vendor Name", "Budget Category", "Invoice Amount", "Invoice Date", "Line Item Notes"]].copy()
            d["Invoice Amount"] = d["Invoice Amount"].apply(lambda x: fmt(x, 2))
            d["Invoice Date"] = d["Invoice Date"].dt.strftime("%b %d, %Y").fillna("—")
            st.dataframe(d, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 2: PORTFOLIO SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif view == "2 — Portfolio Summary":
    banner("Portfolio Summary", "Average Full Turn cost per property — 2022 through 2025")

    SUMMARY_YEARS = [2022, 2023, 2024, 2025]

    # KPIs
    recent = ft_turns[ft_turns["Year"].isin(SUMMARY_YEARS)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Full Turns (2022–25)", f"{len(recent):,}")
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
    avg_matrix["4-Year Avg"] = avg_matrix[SUMMARY_YEARS].replace(0, np.nan).mean(axis=1)

    # YoY change
    avg_matrix["2024 → 2025"] = avg_matrix.apply(
        lambda r: ((r[2025] - r[2024]) / r[2024] * 100) if r[2024] > 0 and r[2025] > 0 else np.nan,
        axis=1,
    )

    avg_matrix = avg_matrix.sort_values("4-Year Avg", ascending=False)

    # Portfolio total row
    portfolio_row = {}
    for y in SUMMARY_YEARS:
        yr_data = recent[recent["Year"] == y]["total_cost"]
        portfolio_row[y] = yr_data.mean() if len(yr_data) else 0
    portfolio_row["4-Year Avg"] = recent["total_cost"].mean()
    portfolio_row["2024 → 2025"] = (
        ((portfolio_row[2025] - portfolio_row[2024]) / portfolio_row[2024] * 100)
        if portfolio_row[2024] > 0 and portfolio_row[2025] > 0 else np.nan
    )
    avg_matrix.loc["PORTFOLIO AVG"] = portfolio_row

    tab_avg, tab_vol = st.tabs(["Average Cost per Turn", "Turn Volume"])

    with tab_avg:
        display = avg_matrix.copy()
        for y in SUMMARY_YEARS + ["4-Year Avg"]:
            display[y] = display[y].apply(lambda x: fmt(x) if x > 0 else "—")
        display["2024 → 2025"] = display["2024 → 2025"].apply(
            lambda x: pct(x) if pd.notna(x) else "—"
        )
        st.dataframe(display, use_container_width=True, height=560)

        # Narrative
        top_prop = avg_matrix.drop("PORTFOLIO AVG").sort_values("4-Year Avg", ascending=False).index[0]
        low_prop = avg_matrix.drop("PORTFOLIO AVG").sort_values("4-Year Avg", ascending=True)
        low_prop = low_prop[low_prop["4-Year Avg"] > 0].index[0]
        port_avg = avg_matrix.loc["PORTFOLIO AVG", "4-Year Avg"]
        yoy_chg = avg_matrix.loc["PORTFOLIO AVG", "2024 → 2025"]

        insight(
            f"Portfolio-wide average Full Turn cost is <strong>{fmt(port_avg)}</strong> over 2022–2025. "
            f"<strong>{top_prop}</strong> has the highest average, while <strong>{low_prop}</strong> runs lowest. "
            f"Year-over-year, portfolio costs moved <strong>{pct(yoy_chg)}</strong> from 2024 to 2025."
        )

    with tab_vol:
        count_matrix["Total"] = count_matrix.sum(axis=1)
        count_matrix = count_matrix.sort_values("Total", ascending=False)
        count_matrix.loc["PORTFOLIO TOTAL"] = count_matrix.sum()
        count_matrix = count_matrix.astype(int)
        st.dataframe(count_matrix, use_container_width=True, height=560)

    # Bar chart comparison
    section("Avg Full Turn Cost by Property — Most Recent Year (2025)")

    data_25 = ft_turns[ft_turns["Year"] == 2025]
    if len(data_25) > 0:
        prop_25 = data_25.groupby("Property Name")["total_cost"].mean().reset_index()
        prop_25 = prop_25.sort_values("total_cost", ascending=True)
        fig = px.bar(
            prop_25, y="Property Name", x="total_cost", orientation="h",
            text=prop_25["total_cost"].apply(fmt),
            template=CHART_TEMPLATE, color_discrete_sequence=["#2563eb"],
            labels={"total_cost": "Avg Full Turn Cost ($)", "Property Name": ""},
        )
        fig.update_traces(textposition="outside")
        portfolio_avg_25 = data_25["total_cost"].mean()
        fig.add_vline(x=portfolio_avg_25, line_dash="dash", line_color="#dc2626",
                       annotation_text=f"Portfolio Avg: {fmt(portfolio_avg_25)}",
                       annotation_position="top right")
        fig.update_layout(margin=dict(t=30, b=10, l=10, r=80), height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 2025 data available.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 3: 5-YEAR LINE-ITEM TREND
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif view == "3 — 5-Year Category Trend":
    banner("5-Year Category Cost Trend", "Tracking Paint, Labor General, Flooring, Countertops, and Appliances per Full Turn")

    ft_5yr = ft_lines[ft_lines["Year"].isin(YEARS)].copy()
    ft_5yr_tracked = ft_5yr[ft_5yr["Tracked Category"] != "Other"].copy()

    # Avg cost per tracked category per turn per year
    cat_year_spend = ft_5yr_tracked.groupby(["Tracked Category", "Year"])["Invoice Amount"].sum().reset_index()
    turns_year = ft_turns[ft_turns["Year"].isin(YEARS)].groupby("Year").size().reset_index(name="turns")
    cat_year_spend = cat_year_spend.merge(turns_year, on="Year")
    cat_year_spend["Avg Per Turn"] = cat_year_spend["Invoice Amount"] / cat_year_spend["turns"]

    # KPIs per category (latest year)
    latest = cat_year_spend[cat_year_spend["Year"] == 2025]
    cols = st.columns(5)
    for i, cat in enumerate(CATEGORY_MAP.keys()):
        val = latest[latest["Tracked Category"] == cat]["Avg Per Turn"]
        cols[i].metric(cat, fmt(val.iloc[0]) if len(val) else "—")

    section("Avg Cost per Full Turn — 5 Tracked Categories")

    # Multi-line chart
    fig = go.Figure()
    for cat, color in CAT_COLORS.items():
        subset = cat_year_spend[cat_year_spend["Tracked Category"] == cat].sort_values("Year")
        fig.add_trace(go.Scatter(
            x=subset["Year"], y=subset["Avg Per Turn"],
            name=cat, mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=8),
            hovertemplate=f"{cat}<br>%{{x}}: %{{y:$,.0f}}<extra></extra>",
        ))

    fig.update_layout(
        template=CHART_TEMPLATE,
        xaxis=dict(dtick=1, title=""),
        yaxis=dict(title="Avg Cost per Full Turn ($)", gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="top", y=-0.1, font=dict(size=12)),
        margin=dict(t=20, b=60), height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    pivot = cat_year_spend.pivot_table(
        index="Tracked Category", columns="Year", values="Avg Per Turn", fill_value=0
    ).reindex(columns=YEARS, fill_value=0)
    pivot = pivot.reindex(CATEGORY_MAP.keys())
    pivot_display = pivot.copy()
    for c in pivot_display.columns:
        pivot_display[c] = pivot_display[c].apply(lambda x: fmt(x) if x > 0 else "—")
    st.dataframe(pivot_display, use_container_width=True)

    # ── Outlier Detection: 1.5 SD ──
    section("Outlier Detection — Category Spend Exceeding Property Mean by >1.5 Std Dev")
    st.caption("Flags individual Full Turns where a tracked category's spend exceeds that property's historical mean + 1.5σ for that category.")

    # Compute per-property, per-category stats
    prop_cat = (
        ft_5yr_tracked.groupby(["Property Name", "Tracked Category", "Turn Key"])["Invoice Amount"]
        .sum().reset_index()
    )
    prop_cat_stats = (
        prop_cat.groupby(["Property Name", "Tracked Category"])["Invoice Amount"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    prop_cat_stats["threshold"] = prop_cat_stats["mean"] + 1.5 * prop_cat_stats["std"]

    # Flag outliers
    flagged = prop_cat.merge(prop_cat_stats, on=["Property Name", "Tracked Category"])
    flagged = flagged[
        (flagged["Invoice Amount"] > flagged["threshold"])
        & (flagged["count"] >= 3)  # need min data to compute meaningful SD
    ].copy()
    flagged["Excess"] = flagged["Invoice Amount"] - flagged["mean"]
    flagged["SDs Over"] = (flagged["Invoice Amount"] - flagged["mean"]) / flagged["std"]
    flagged = flagged.sort_values("SDs Over", ascending=False)

    # Enrich with turn info
    turn_info = ft_turns[["Turn Key", "Property Name", "Unit Label", "Move-Out Date"]].drop_duplicates("Turn Key")
    flagged = flagged.merge(turn_info, on=["Turn Key", "Property Name"], how="left")

    st.metric("Outlier Line Items Flagged", len(flagged))

    if len(flagged) > 0:
        flag_display = flagged[[
            "Property Name", "Unit Label", "Move-Out Date", "Tracked Category",
            "Invoice Amount", "mean", "threshold", "SDs Over"
        ]].copy()
        flag_display["Move-Out Date"] = flag_display["Move-Out Date"].dt.strftime("%b %d, %Y")
        flag_display["Invoice Amount"] = flag_display["Invoice Amount"].apply(fmt)
        flag_display["mean"] = flag_display["mean"].apply(fmt)
        flag_display["threshold"] = flag_display["threshold"].apply(fmt)
        flag_display["SDs Over"] = flag_display["SDs Over"].apply(lambda x: f"{x:.1f}σ")
        flag_display.columns = ["Property", "Unit", "Move-Out", "Category",
                                 "Actual Spend", "Property Avg", "Threshold (μ+1.5σ)", "Std Devs Over"]
        st.dataframe(flag_display, use_container_width=True, hide_index=True, height=400)

        # Which categories flag most
        cat_flags = flagged.groupby("Tracked Category").size().reset_index(name="Flags").sort_values("Flags", ascending=False)
        fig_flags = px.bar(
            cat_flags, x="Tracked Category", y="Flags", text="Flags",
            template=CHART_TEMPLATE, color="Tracked Category",
            color_discrete_map=CAT_COLORS,
        )
        fig_flags.update_traces(textposition="outside")
        fig_flags.update_layout(margin=dict(t=10, b=10), height=300, showlegend=False,
                                xaxis_title="", yaxis_title="Outlier Count")
        st.plotly_chart(fig_flags, use_container_width=True)
    else:
        st.success("No outliers detected across tracked categories.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 4: RECENT 10 AUDIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif view == "4 — Recent 10 Audit":
    banner("Recent 10 Full Turn Audit",
           "Detailed cost breakdown of the 10 most recent Full Turns — benchmarked against the portfolio 5-year average")

    recent_10 = ft_turns.sort_values("completion_date", ascending=False).head(10).copy()

    # Portfolio-wide 5-year avg per category per turn
    ft_5yr = ft_lines[ft_lines["Year"].isin(YEARS)].copy()
    total_turns_5yr = ft_turns[ft_turns["Year"].isin(YEARS)]["Turn Key"].nunique()

    portfolio_cat_avg = (
        ft_5yr.groupby("Budget Category")["Invoice Amount"].sum() / total_turns_5yr
    ).reset_index()
    portfolio_cat_avg.columns = ["Budget Category", "Portfolio 5yr Avg"]

    # Summary table
    section("Overview — 10 Most Recent Full Turns")

    r10_display = recent_10.copy()
    r10_display["#"] = range(1, 11)
    r10_display["Completion"] = r10_display["completion_date"].dt.strftime("%b %d, %Y").fillna("—")
    r10_display["Move-Out"] = r10_display["Move-Out Date"].dt.strftime("%b %d, %Y")
    r10_display["Cost"] = r10_display["total_cost"].apply(fmt)
    r10_display["Dur"] = r10_display["Duration"].apply(lambda x: f"{x:.0f}d" if pd.notna(x) else "—")

    # Portfolio avg for context
    port_avg = ft_turns[ft_turns["Year"].isin(YEARS)]["total_cost"].mean()
    r10_display["vs Portfolio Avg"] = r10_display["total_cost"].apply(
        lambda x: pct((x - port_avg) / port_avg * 100)
    )

    overview = r10_display[["#", "Property Name", "Unit Label", "Floor Plan",
                             "Move-Out", "Completion", "Cost", "vs Portfolio Avg",
                             "Dur", "line_items"]].copy()
    overview.columns = ["#", "Property", "Unit", "Floor Plan", "Move-Out",
                         "Completion", "Total Cost", "vs Portfolio Avg",
                         "Duration", "Invoices"]
    st.dataframe(overview, use_container_width=True, hide_index=True)

    insight(
        f"Portfolio 5-year average Full Turn cost: <strong>{fmt(port_avg)}</strong>. "
        f"Values in the 'vs Portfolio Avg' column show how each recent turn compares. "
        f"Expand any turn below to see line-item detail benchmarked against portfolio norms."
    )

    # ── Detailed Breakdown per Turn ──
    section("Line-Item Breakdown with Portfolio Benchmarks")

    for idx, (_, turn) in enumerate(recent_10.iterrows()):
        label = (
            f"**{idx+1}. {turn['Property Name']}** — Unit {turn['Unit Label']} — "
            f"{fmt(turn['total_cost'])} — {turn['Move-Out Date'].strftime('%b %d, %Y')}"
        )
        with st.expander(label, expanded=(idx == 0)):
            # Get line items
            items = ft_lines[ft_lines["Turn Key"] == turn["Turn Key"]].copy()

            # Aggregate by category
            cat_agg = (
                items.groupby("Budget Category")["Invoice Amount"]
                .sum().reset_index()
                .sort_values("Invoice Amount", ascending=False)
            )
            cat_agg = cat_agg.merge(portfolio_cat_avg, on="Budget Category", how="left")
            cat_agg["Portfolio 5yr Avg"] = cat_agg["Portfolio 5yr Avg"].fillna(0)
            cat_agg["Variance"] = cat_agg["Invoice Amount"] - cat_agg["Portfolio 5yr Avg"]
            cat_agg["Var %"] = cat_agg.apply(
                lambda r: ((r["Variance"]) / r["Portfolio 5yr Avg"] * 100)
                if r["Portfolio 5yr Avg"] > 0 else np.nan, axis=1
            )

            # Display
            cat_display = cat_agg.copy()
            cat_display["Invoice Amount"] = cat_display["Invoice Amount"].apply(fmt)
            cat_display["Portfolio 5yr Avg"] = cat_display["Portfolio 5yr Avg"].apply(fmt)
            cat_display["Variance"] = cat_display["Variance"].apply(lambda x: fmt(x) if x >= 0 else f"-{fmt(abs(x))}")
            cat_display["Var %"] = cat_display["Var %"].apply(lambda x: pct(x) if pd.notna(x) else "—")
            cat_display.columns = ["Category", "This Turn", "Portfolio Avg", "Variance ($)", "Variance (%)"]
            st.dataframe(cat_display, use_container_width=True, hide_index=True)

            # Horizontal bar comparing this turn vs portfolio
            col1, col2 = st.columns([3, 2])
            with col1:
                top_cats = cat_agg.head(8)
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(
                    y=top_cats["Budget Category"], x=top_cats["Invoice Amount"],
                    name="This Turn", orientation="h",
                    marker_color="#2563eb", opacity=0.9,
                ))
                fig_cmp.add_trace(go.Bar(
                    y=top_cats["Budget Category"], x=top_cats["Portfolio 5yr Avg"],
                    name="Portfolio 5yr Avg", orientation="h",
                    marker_color="#94a3b8", opacity=0.6,
                ))
                fig_cmp.update_layout(
                    template=CHART_TEMPLATE, barmode="group",
                    legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
                    margin=dict(t=10, b=50, l=10, r=10), height=280,
                    xaxis_title="Cost ($)", yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

            with col2:
                st.markdown("**Turn Summary**")
                st.markdown(f"- **Total Cost:** {fmt(turn['total_cost'])}")
                st.markdown(f"- **Floor Plan:** {turn['Floor Plan']}")
                st.markdown(f"- **Duration:** {turn['Duration']:.0f} days" if pd.notna(turn['Duration']) else "- **Duration:** —")
                st.markdown(f"- **Invoices:** {turn['line_items']}")
                variance_total = turn["total_cost"] - port_avg
                st.markdown(f"- **vs Portfolio:** {fmt(variance_total)} ({pct(variance_total/port_avg*100)})")

            # Raw invoice list
            with st.expander("View All Invoices"):
                raw = items[["Vendor Name", "Budget Category", "Invoice Amount",
                             "Invoice Date", "Cost Type", "Line Item Notes"]].copy()
                raw["Invoice Amount"] = raw["Invoice Amount"].apply(lambda x: fmt(x, 2))
                raw["Invoice Date"] = raw["Invoice Date"].dt.strftime("%b %d, %Y").fillna("—")
                raw = raw.sort_values("Invoice Date")
                st.dataframe(raw, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 5: UNIT SEARCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif view == "5 — Unit Search":
    banner("Unit Search & History", "Full renovation history for any unit in the portfolio")

    # Use all-types turn summary for this view
    all_turns = build_turn_summary(_df_all)

    col1, col2 = st.columns(2)
    with col1:
        prop_choice = st.selectbox("Property", sorted(_df_all["Property Name"].unique()))
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
        st.markdown(
            f"**{prop_choice} — Unit {unit_choice}** &nbsp;|&nbsp; "
            f"Floor Plan: `{sample['Floor Plan']}` &nbsp;|&nbsp; "
            f"Bed/Bath: `{sample['Bedrooms']:.0f} / {sample['Bathrooms']:.0f}` &nbsp;|&nbsp; "
            f"ID: `{sample['UID']}`"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Turns", len(unit_ts))
        c2.metric("Lifetime Spend", fmt(unit_ts["total_cost"].sum()))
        c3.metric("Avg Cost / Turn", fmt(unit_ts["total_cost"].mean()))

        section("Turn History")
        for _, turn in unit_ts.iterrows():
            with st.expander(
                f"**{turn['Turn Type']}** — {turn['Move-Out Date'].strftime('%b %d, %Y')} — "
                f"{fmt(turn['total_cost'])} ({turn['line_items']} invoices)"
            ):
                items = unit_df[unit_df["Turn Key"] == turn["Turn Key"]].sort_values("Invoice Date")
                disp = items[["Invoice Date", "Vendor Name", "Budget Category",
                              "Cost Type", "Invoice Amount", "Line Item Notes"]].copy()
                disp["Invoice Date"] = disp["Invoice Date"].dt.strftime("%b %d, %Y").fillna("—")
                disp["Invoice Amount"] = disp["Invoice Amount"].apply(lambda x: fmt(x, 2))
                st.dataframe(disp, use_container_width=True, hide_index=True)

        section("Spending History")
        turn_colors = {"Full Turn": "#dc2626", "Partial Turn": "#f59e0b", "Make Ready": "#10b981"}
        fig = px.bar(
            unit_ts, x="Move-Out Date", y="total_cost", color="Turn Type",
            color_discrete_map=turn_colors, template=CHART_TEMPLATE,
            labels={"total_cost": "Turn Cost ($)", "Move-Out Date": ""},
        )
        fig.update_layout(margin=dict(t=10, b=10), height=350)
        st.plotly_chart(fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 6: ANOMALY DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif view == "6 — Anomaly Detection":
    banner("Anomaly Detection", "Identify unusual patterns across all turn types requiring investigation")

    all_turns = build_turn_summary(_df_all)

    tab1, tab2, tab3 = st.tabs(["High-Frequency Units", "Cost Outliers (IQR)", "Data Quality"])

    with tab1:
        st.caption("Units with unusually high turn frequency — may indicate chronic vacancy or workmanship issues.")
        threshold = st.slider("Minimum turn count to flag", 3, 10, 4)

        unit_freq = (
            all_turns.groupby(["Property Name", "Unit Display"])
            .agg(turn_count=("Turn Key", "nunique"), total_spend=("total_cost", "sum"),
                 types=("Turn Type", lambda x: ", ".join(sorted(x.unique()))))
            .reset_index().sort_values("turn_count", ascending=False)
        )
        flagged = unit_freq[unit_freq["turn_count"] >= threshold]
        st.metric("Flagged Units", len(flagged))

        if len(flagged) > 0:
            disp = flagged.copy()
            disp["total_spend"] = disp["total_spend"].apply(fmt)
            disp.columns = ["Property", "Unit", "Turns", "Total Spend", "Turn Types"]
            st.dataframe(disp, use_container_width=True, hide_index=True)

            fig = px.bar(
                flagged.head(20), x="Unit Label", y="turn_count",
                color="Property Name", text="turn_count",
                template=CHART_TEMPLATE,
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_tickangle=-45, margin=dict(t=10, b=60), height=380,
                              xaxis_title="", yaxis_title="Turns",
                              legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.caption("Turns where total cost exceeds Q3 + 1.5 × IQR for their type.")

        outlier_rows = []
        for tt in all_turns["Turn Type"].unique():
            subset = all_turns[all_turns["Turn Type"] == tt]["total_cost"]
            if len(subset) < 4:
                continue
            q1, q3 = subset.quantile(0.25), subset.quantile(0.75)
            upper = q3 + 1.5 * (q3 - q1)
            for _, row in all_turns[(all_turns["Turn Type"] == tt) & (all_turns["total_cost"] > upper)].iterrows():
                outlier_rows.append({
                    "Property": row["Property Name"], "Unit": row["Unit Label"],
                    "Move-Out": row["Move-Out Date"].strftime("%b %d, %Y"),
                    "Type": row["Turn Type"],
                    "Cost": row["total_cost"], "Threshold": upper,
                })

        if outlier_rows:
            odf = pd.DataFrame(outlier_rows).sort_values("Cost", ascending=False)
            st.metric("Statistical Outliers", len(odf))
            odf_disp = odf.copy()
            odf_disp["Cost"] = odf_disp["Cost"].apply(fmt)
            odf_disp["Threshold"] = odf_disp["Threshold"].apply(fmt)
            st.dataframe(odf_disp, use_container_width=True, hide_index=True, height=400)
        else:
            st.success("No statistical outliers detected.")

    with tab3:
        st.caption("Data integrity checks across all records.")
        negatives = _df_all[_df_all["Invoice Amount"] < 0]
        missing_unit = _df_all[_df_all["Unit Number"].str.strip() == ""]
        missing_inv = _df_all[_df_all["Invoice Number"].isna() | (_df_all["Invoice Number"].astype(str).str.strip() == "")]

        c1, c2, c3 = st.columns(3)
        c1.metric("Negative Invoices", len(negatives))
        c2.metric("Missing Unit #", len(missing_unit))
        c3.metric("Missing Invoice #", len(missing_inv))

        if len(negatives) > 0:
            st.markdown("**Negative Invoice Records:**")
            neg_d = negatives[["Property Name", "Unit Label", "Vendor Name",
                               "Invoice Amount", "Budget Category", "Line Item Notes"]].head(20).copy()
            neg_d["Invoice Amount"] = neg_d["Invoice Amount"].apply(lambda x: fmt(x, 2))
            st.dataframe(neg_d, use_container_width=True, hide_index=True)
