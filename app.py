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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
TREND_YEARS = [2021, 2022, 2023, 2024, 2025]
CHART_TEMPLATE = "plotly_white"

# ── Materials vs Labor category classification ──
MATERIALS_CATS = [
    "Supplies", "Appliances", "Flooring Materials", "Paint",
    "Cabinets Materials", "Countertops Materials", "Windows",
]
LABOR_CATS = [
    "Labor General", "Flooring Labor", "Electric General",
    "Countertops Labor", "Plumbing", "Powerwash and Demo",
    "Management Fee", "Scrape Ceiling", "Glaze", "Cabinets Labor",
]

MAT_COLORS = {
    "Supplies":              "#2563eb",
    "Appliances":            "#ef4444",
    "Flooring Materials":    "#f59e0b",
    "Paint":                 "#6366f1",
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

    /* Divider */
    .section-divider { height: 1px; background: #e2e8f0; margin: 32px 0; }
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

def divider():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

def footer():
    st.markdown(
        '<div class="dashboard-footer">'
        'CONFIDENTIAL — Full Turn Analytics Dashboard &nbsp;|&nbsp; '
        f'Data as of Feb 2026 &nbsp;|&nbsp; '
        'Prepared for Executive Review'
        '</div>',
        unsafe_allow_html=True,
    )


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
st.sidebar.caption("Portfolio Renovation Intelligence Platform")
st.sidebar.markdown("---")

view = st.sidebar.radio("View", [
    "1 — Executive Summary",
    "2 — Portfolio Overview",
    "3 — Category Trends",
    "4 — Property Drilldown",
    "5 — Recent Turns Audit",
    "6 — Unit Search",
    "7 — Anomaly Detection",
])

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**{len(ft_turns):,}** Full Turns  •  **{ft_turns['Property ID'].nunique()}** Properties  •  "
    f"**{ft_turns['UID'].nunique()}** Units  •  **{fmt(ft_turns['total_cost'].sum())}** Total Spend"
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 1: PROPERTY LEVEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if view == "4 — Property Drilldown":
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

    # Property narrative
    total_spend = p_turns["total_cost"].sum()
    avg_cost = p_turns["total_cost"].mean()
    port_avg_all = ft_turns["total_cost"].mean()
    vs_port = ((avg_cost - port_avg_all) / port_avg_all * 100) if port_avg_all > 0 else 0
    insight(
        f"<strong>{prop}</strong> has completed <strong>{len(p_turns):,}</strong> Full Turns "
        f"totaling <strong>{fmt(total_spend)}</strong>. "
        f"Average cost per turn is <strong>{fmt(avg_cost)}</strong>, which is "
        f"<strong>{pct(vs_port)}</strong> vs the portfolio average of <strong>{fmt(port_avg_all)}</strong>."
    )
    footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 2: PORTFOLIO SUMMARY
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

    avg_matrix = avg_matrix.sort_values("Avg (All Years)", ascending=False)

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
        top_prop = avg_matrix.drop("PORTFOLIO AVG").sort_values("Avg (All Years)", ascending=False).index[0]
        low_prop = avg_matrix.drop("PORTFOLIO AVG").sort_values("Avg (All Years)", ascending=True)
        low_prop = low_prop[low_prop["Avg (All Years)"] > 0].index[0]
        port_avg = avg_matrix.loc["PORTFOLIO AVG", "Avg (All Years)"]
        yoy_chg = avg_matrix.loc["PORTFOLIO AVG", "2024 → 2025"]

        insight(
            f"Portfolio-wide average Full Turn cost is <strong>{fmt(port_avg)}</strong> over 2016–2025. "
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

    footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 3: CATEGORY TRENDS (Last 5 Years)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif view == "3 — Category Trends":
    banner("Category Cost Trends", f"5-year trend ({TREND_YEARS[0]}–{TREND_YEARS[-1]}) — Materials and Labor analyzed separately per Full Turn")

    ft_trend = ft_lines[ft_lines["Year"].isin(TREND_YEARS)].copy()
    ft_all = ft_lines.copy()  # Keep all years for outlier detection
    turns_year = ft_turns[ft_turns["Year"].isin(TREND_YEARS)].groupby("Year").size().reset_index(name="turns")

    # Compute avg cost per category per turn per year (all raw categories)
    cat_year_all = ft_trend.groupby(["Budget Category", "Year"])["Invoice Amount"].sum().reset_index()
    cat_year_all = cat_year_all.merge(turns_year, on="Year")
    cat_year_all["Avg Per Turn"] = cat_year_all["Invoice Amount"] / cat_year_all["turns"]

    # Materials vs Labor totals (5-year window)
    mat_5yr = ft_trend[ft_trend["Budget Category"].isin(MATERIALS_CATS)]
    lab_5yr = ft_trend[ft_trend["Budget Category"].isin(LABOR_CATS)]
    mat_total = mat_5yr["Invoice Amount"].sum()
    lab_total = lab_5yr["Invoice Amount"].sum()
    total_5yr = mat_total + lab_total

    n_turns_total = ft_turns[ft_turns["Year"].isin(TREND_YEARS)]["Turn Key"].nunique()
    mat_per_turn = mat_total / n_turns_total if n_turns_total > 0 else 0
    lab_per_turn = lab_total / n_turns_total if n_turns_total > 0 else 0

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Materials / Turn", fmt(mat_per_turn))
    c2.metric("Labor / Turn", fmt(lab_per_turn))
    c3.metric("Materials Share", f"{mat_total/total_5yr*100:.0f}%" if total_5yr > 0 else "—")
    c4.metric("Labor Share", f"{lab_total/total_5yr*100:.0f}%" if total_5yr > 0 else "—")
    c5.metric("Budget Categories", "17")

    # ══════════════════════════════════════════════════
    # MATERIALS CHART
    # ══════════════════════════════════════════════════
    section("Materials — Avg Cost per Full Turn by Category")

    mat_data = cat_year_all[cat_year_all["Budget Category"].isin(MATERIALS_CATS)]
    fig_mat = go.Figure()
    for cat in MATERIALS_CATS:
        subset = mat_data[mat_data["Budget Category"] == cat].sort_values("Year")
        if len(subset) > 0:
            fig_mat.add_trace(go.Scatter(
                x=subset["Year"], y=subset["Avg Per Turn"],
                name=cat, mode="lines+markers",
                line=dict(color=MAT_COLORS.get(cat, "#94a3b8"), width=2.5),
                marker=dict(size=7),
                hovertemplate=f"{cat}<br>%{{x}}: $%{{y:,.0f}}<extra></extra>",
            ))
    fig_mat.update_layout(
        template=CHART_TEMPLATE,
        xaxis=dict(dtick=1, title=""), yaxis=dict(title="Avg Cost per Full Turn ($)", gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="top", y=-0.12, font=dict(size=11)),
        margin=dict(t=10, b=80), height=420, hovermode="x unified",
    )
    st.plotly_chart(fig_mat, use_container_width=True)

    # Materials data table with YoY % change
    mat_pivot = cat_year_all[cat_year_all["Budget Category"].isin(MATERIALS_CATS)].pivot_table(
        index="Budget Category", columns="Year", values="Avg Per Turn", fill_value=0
    ).reindex(columns=TREND_YEARS, fill_value=0).reindex(MATERIALS_CATS)
    mat_yoy = mat_pivot[TREND_YEARS[-1]] / mat_pivot[TREND_YEARS[-2]].replace(0, np.nan) - 1
    mat_pivot_d = mat_pivot.copy()
    for c in mat_pivot_d.columns:
        mat_pivot_d[c] = mat_pivot_d[c].apply(lambda x: fmt(x) if x > 0 else "—")
    mat_pivot_d["YoY Δ"] = mat_yoy.apply(lambda x: f"{x:+.0%}" if pd.notna(x) and np.isfinite(x) else "—")
    st.dataframe(mat_pivot_d, use_container_width=True)

    # ══════════════════════════════════════════════════
    # LABOR CHART
    # ══════════════════════════════════════════════════
    section("Labor & Services — Avg Cost per Full Turn by Category")

    lab_data = cat_year_all[cat_year_all["Budget Category"].isin(LABOR_CATS)]
    fig_lab = go.Figure()
    for cat in LABOR_CATS:
        subset = lab_data[lab_data["Budget Category"] == cat].sort_values("Year")
        if len(subset) > 0:
            fig_lab.add_trace(go.Scatter(
                x=subset["Year"], y=subset["Avg Per Turn"],
                name=cat, mode="lines+markers",
                line=dict(color=LAB_COLORS.get(cat, "#94a3b8"), width=2.5),
                marker=dict(size=7),
                hovertemplate=f"{cat}<br>%{{x}}: $%{{y:,.0f}}<extra></extra>",
            ))
    fig_lab.update_layout(
        template=CHART_TEMPLATE,
        xaxis=dict(dtick=1, title=""), yaxis=dict(title="Avg Cost per Full Turn ($)", gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="top", y=-0.12, font=dict(size=11)),
        margin=dict(t=10, b=80), height=420, hovermode="x unified",
    )
    st.plotly_chart(fig_lab, use_container_width=True)

    # Labor data table with YoY % change
    lab_pivot = cat_year_all[cat_year_all["Budget Category"].isin(LABOR_CATS)].pivot_table(
        index="Budget Category", columns="Year", values="Avg Per Turn", fill_value=0
    ).reindex(columns=TREND_YEARS, fill_value=0).reindex(LABOR_CATS)
    lab_yoy = lab_pivot[TREND_YEARS[-1]] / lab_pivot[TREND_YEARS[-2]].replace(0, np.nan) - 1
    lab_pivot_d = lab_pivot.copy()
    for c in lab_pivot_d.columns:
        lab_pivot_d[c] = lab_pivot_d[c].apply(lambda x: fmt(x) if x > 0 else "—")
    lab_pivot_d["YoY Δ"] = lab_yoy.apply(lambda x: f"{x:+.0%}" if pd.notna(x) and np.isfinite(x) else "—")
    st.dataframe(lab_pivot_d, use_container_width=True)

    # ══════════════════════════════════════════════════
    # COMBINED STACKED VIEW
    # ══════════════════════════════════════════════════
    section("Materials vs Labor — Total Spend Split by Year")

    mat_by_yr = mat_5yr.groupby("Year")["Invoice Amount"].sum().reindex(TREND_YEARS, fill_value=0)
    lab_by_yr = lab_5yr.groupby("Year")["Invoice Amount"].sum().reindex(TREND_YEARS, fill_value=0)

    fig_stack = go.Figure()
    fig_stack.add_trace(go.Bar(
        x=[str(y) for y in TREND_YEARS], y=mat_by_yr.values,
        name="Materials", marker_color="#2563eb",
        hovertemplate="Materials: $%{y:,.0f}<extra></extra>",
    ))
    fig_stack.add_trace(go.Bar(
        x=[str(y) for y in TREND_YEARS], y=lab_by_yr.values,
        name="Labor & Services", marker_color="#f59e0b",
        hovertemplate="Labor: $%{y:,.0f}<extra></extra>",
    ))
    fig_stack.update_layout(
        template=CHART_TEMPLATE, barmode="stack",
        xaxis=dict(title=""), yaxis=dict(title="Total Spend ($)"),
        legend=dict(orientation="h", y=-0.12, font=dict(size=12)),
        margin=dict(t=10, b=50), height=380,
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # Narrative
    mat_share = f"{mat_total/total_5yr*100:.0f}%" if total_5yr > 0 else "—"
    lab_share = f"{lab_total/total_5yr*100:.0f}%" if total_5yr > 0 else "—"
    insight(
        f"Over the last 5 years ({TREND_YEARS[0]}–{TREND_YEARS[-1]}), Materials represent "
        f"<strong>{mat_share}</strong> of tracked spend "
        f"(<strong>{fmt(mat_per_turn)}</strong>/turn) while Labor & Services account for "
        f"<strong>{lab_share}</strong> (<strong>{fmt(lab_per_turn)}</strong>/turn). "
        f"<strong>Supplies</strong> is the largest single materials category, and "
        f"<strong>Labor General</strong> leads the services side."
    )

    # ══════════════════════════════════════════════════
    # OUTLIER DETECTION (uses ALL raw categories now)
    # ══════════════════════════════════════════════════

    # Compute per-property, per-raw-category stats (using ALL years for statistical depth)
    prop_cat = (
        ft_all.groupby(["Property Name", "Budget Category", "Turn Key"])["Invoice Amount"]
        .sum().reset_index()
    )
    prop_cat_stats = (
        prop_cat.groupby(["Property Name", "Budget Category"])["Invoice Amount"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    prop_cat_stats["threshold"] = prop_cat_stats["mean"] + 1.5 * prop_cat_stats["std"]

    # Flag outliers
    flagged = prop_cat.merge(prop_cat_stats, on=["Property Name", "Budget Category"])
    flagged = flagged[
        (flagged["Invoice Amount"] > flagged["threshold"])
        & (flagged["count"] >= 3)
    ].copy()
    flagged["Excess"] = flagged["Invoice Amount"] - flagged["mean"]
    flagged["Excess %"] = (flagged["Excess"] / flagged["mean"] * 100)
    flagged["SDs Over"] = (flagged["Invoice Amount"] - flagged["mean"]) / flagged["std"]

    # Enrich with turn info
    turn_info = ft_turns[["Turn Key", "Property Name", "Unit Label", "Move-Out Date"]].drop_duplicates("Turn Key")
    flagged = flagged.merge(turn_info, on=["Turn Key", "Property Name"], how="left")

    # ── Section A: Last 12 Months Outliers ──
    section("Outliers — Last 12 Months")
    st.caption("Category spend exceeding property mean + 1.5σ on Full Turns completed in the past 12 months.")

    cutoff_12m = pd.Timestamp.now() - pd.DateOffset(months=12)
    recent_outliers = flagged[flagged["Move-Out Date"] >= cutoff_12m].sort_values("Excess", ascending=False).copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Outliers (Last 12 Mo)", len(recent_outliers))
    c2.metric("Total Excess Spend", fmt(recent_outliers["Excess"].sum()) if len(recent_outliers) else "—")
    top_cat_recent = recent_outliers.groupby("Budget Category")["Excess"].sum().idxmax() if len(recent_outliers) else "—"
    c3.metric("Top Flagged Category", top_cat_recent)

    if len(recent_outliers) > 0:
        r_display = recent_outliers[[
            "Property Name", "Unit Label", "Move-Out Date", "Budget Category",
            "Invoice Amount", "mean", "Excess", "Excess %", "SDs Over"
        ]].copy()
        r_display["Move-Out Date"] = r_display["Move-Out Date"].dt.strftime("%b %d, %Y")
        r_display["Invoice Amount"] = r_display["Invoice Amount"].apply(fmt)
        r_display["mean"] = r_display["mean"].apply(fmt)
        r_display["Excess"] = r_display["Excess"].apply(fmt)
        r_display["Excess %"] = r_display["Excess %"].apply(lambda x: f"+{x:.0f}%" if pd.notna(x) and np.isfinite(x) else "—")
        r_display["SDs Over"] = r_display["SDs Over"].apply(lambda x: f"{x:.1f}σ" if pd.notna(x) and np.isfinite(x) else "—")
        r_display.columns = ["Property", "Unit", "Move-Out", "Category",
                              "Actual Spend", "Property Avg", "Excess ($)", "Excess (%)", "Std Devs Over"]
        st.dataframe(r_display, use_container_width=True, hide_index=True, height=min(400, 60 + len(recent_outliers) * 35))

        worst = recent_outliers.iloc[0]
        insight(
            f"Highest recent outlier: <strong>{worst['Budget Category']}</strong> at "
            f"<strong>{worst['Property Name']}</strong> (Unit {worst['Unit Label']}) — "
            f"spent <strong>{fmt(worst['Invoice Amount'])}</strong> vs property avg of "
            f"<strong>{fmt(worst['mean'])}</strong>, exceeding by <strong>{fmt(worst['Excess'])}</strong>."
        )
    else:
        st.success("No category outliers detected in the past 12 months.")

    # ── Section B: All-Time Historical Outliers ──
    section("Historical Outliers — All Time (Highest Excess First)")
    st.caption("All flagged category overspends across the full dataset, ranked by dollar excess over property average.")

    historical_outliers = flagged.sort_values("Excess", ascending=False).copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Historical Outliers", len(historical_outliers))
    c2.metric("Cumulative Excess", fmt(historical_outliers["Excess"].sum()) if len(historical_outliers) else "—")
    top_prop = historical_outliers.groupby("Property Name")["Excess"].sum().idxmax() if len(historical_outliers) else "—"
    c3.metric("Most Flagged Property", top_prop)

    if len(historical_outliers) > 0:
        h_display = historical_outliers[[
            "Property Name", "Unit Label", "Move-Out Date", "Budget Category",
            "Invoice Amount", "mean", "Excess", "Excess %", "SDs Over"
        ]].copy()
        h_display["Move-Out Date"] = h_display["Move-Out Date"].dt.strftime("%b %d, %Y")
        h_display["Invoice Amount"] = h_display["Invoice Amount"].apply(fmt)
        h_display["mean"] = h_display["mean"].apply(fmt)
        h_display["Excess"] = h_display["Excess"].apply(fmt)
        h_display["Excess %"] = h_display["Excess %"].apply(lambda x: f"+{x:.0f}%" if pd.notna(x) and np.isfinite(x) else "—")
        h_display["SDs Over"] = h_display["SDs Over"].apply(lambda x: f"{x:.1f}σ" if pd.notna(x) and np.isfinite(x) else "—")
        h_display.columns = ["Property", "Unit", "Move-Out", "Category",
                              "Actual Spend", "Property Avg", "Excess ($)", "Excess (%)", "Std Devs Over"]
        st.dataframe(h_display, use_container_width=True, hide_index=True, height=500)

        # Which categories flag most
        all_cat_colors = {**MAT_COLORS, **LAB_COLORS}
        cat_flags = historical_outliers.groupby("Budget Category").agg(
            Flags=("Turn Key", "count"),
            Total_Excess=("Excess", "sum"),
        ).reset_index().sort_values("Total_Excess", ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            fig_flags = px.bar(
                cat_flags, x="Budget Category", y="Flags", text="Flags",
                template=CHART_TEMPLATE, color="Budget Category",
                color_discrete_map=all_cat_colors,
            )
            fig_flags.update_traces(textposition="outside")
            fig_flags.update_layout(margin=dict(t=10, b=60), height=340, showlegend=False,
                                    xaxis_title="", yaxis_title="Outlier Count", xaxis_tickangle=-45,
                                    title=dict(text="Outlier Frequency by Category", font=dict(size=13)))
            st.plotly_chart(fig_flags, use_container_width=True)

        with col2:
            fig_excess = px.bar(
                cat_flags, x="Budget Category", y="Total_Excess",
                text=cat_flags["Total_Excess"].apply(fmt),
                template=CHART_TEMPLATE, color="Budget Category",
                color_discrete_map=all_cat_colors,
            )
            fig_excess.update_traces(textposition="outside")
            fig_excess.update_layout(margin=dict(t=10, b=60), height=340, showlegend=False,
                                     xaxis_title="", yaxis_title="Total Excess ($)", xaxis_tickangle=-45,
                                     title=dict(text="Cumulative Excess by Category", font=dict(size=13)))
            st.plotly_chart(fig_excess, use_container_width=True)

        # Property breakdown
        prop_flags = historical_outliers.groupby("Property Name").agg(
            Flags=("Turn Key", "count"),
            Excess=("Excess", "sum"),
        ).reset_index().sort_values("Excess", ascending=False)
        fig_prop = px.bar(
            prop_flags, x="Excess", y="Property Name", orientation="h",
            text=prop_flags["Excess"].apply(fmt),
            template=CHART_TEMPLATE, color_discrete_sequence=["#dc2626"],
        )
        fig_prop.update_traces(textposition="outside")
        fig_prop.update_layout(margin=dict(t=10, b=10, r=80), height=400, showlegend=False,
                                xaxis_title="Total Excess Spend ($)", yaxis_title="",
                                yaxis=dict(autorange="reversed"),
                                title=dict(text="Excess Spend by Property", font=dict(size=13)))
        st.plotly_chart(fig_prop, use_container_width=True)
    else:
        st.success("No outliers detected across tracked categories.")

    footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 4: RECENT 10 AUDIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif view == "5 — Recent Turns Audit":
    banner("Recent 10 Full Turn Audit",
           "Detailed cost breakdown of the 10 most recent Full Turns — benchmarked against the portfolio average")

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
    r10_display["#"] = range(1, len(recent_10) + 1)
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
        f"Portfolio average Full Turn cost: <strong>{fmt(port_avg)}</strong>. "
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
                vs_pct = pct(variance_total / port_avg * 100) if port_avg > 0 else "—"
                st.markdown(f"- **vs Portfolio:** {fmt(variance_total)} ({vs_pct})")

            # Raw invoice list
            with st.expander("View All Invoices"):
                raw = items[["Vendor Name", "Budget Category", "Invoice Amount",
                             "Invoice Date", "Cost Type", "Line Item Notes"]].copy()
                raw["Invoice Amount"] = raw["Invoice Amount"].apply(lambda x: fmt(x, 2))
                raw["Invoice Date"] = raw["Invoice Date"].dt.strftime("%b %d, %Y").fillna("—")
                raw = raw.sort_values("Invoice Date")
                st.dataframe(raw, use_container_width=True, hide_index=True)

    footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 5: UNIT SEARCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif view == "6 — Unit Search":
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

    footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 6: ANOMALY DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif view == "7 — Anomaly Detection":
    banner("Anomaly Detection", "Identify unusual patterns across all turn types requiring investigation")

    all_turns = build_turn_summary(_df_all)

    tab1, tab2, tab3 = st.tabs(["High-Frequency Units", "Cost Outliers (IQR)", "Data Quality"])

    with tab1:
        st.caption("Units with unusually high turn frequency — may indicate chronic vacancy or workmanship issues.")
        threshold = st.slider("Minimum turn count to flag", 3, 10, 4)

        unit_freq = (
            all_turns.groupby(["Property Name", "Unit Label"])
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

    footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW 7: CEO DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif view == "1 — Executive Summary":
    banner("Executive Intelligence", "Strategic performance overview for senior leadership — Full Turn portfolio analytics")

    # ── Compute core metrics ──
    recent_3yr = ft_turns[ft_turns["Year"].isin([2023, 2024, 2025])].copy()
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
    c2.metric("2025 Avg Cost", fmt(curr_avg), pct(yoy_delta) + " vs '24" if pd.notna(yoy_delta) else "—")
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

    # Add rank
    prop_bench["Rank"] = range(1, len(prop_bench) + 1)
    prop_bench["vs Portfolio"] = ((prop_bench["avg_cost"] - portfolio_avg) / portfolio_avg * 100) if portfolio_avg > 0 else 0

    col1, col2 = st.columns([3, 2])
    with col1:
        fig_bench = px.bar(
            prop_bench.sort_values("avg_cost", ascending=True),
            y="Property Name", x="avg_cost", orientation="h",
            text=prop_bench.sort_values("avg_cost", ascending=True)["avg_cost"].apply(fmt),
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

    vendor_data["Share"] = vendor_data["total_spend"] / vendor_data["total_spend"].sum() * 100
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
        if len(p24) >= 2 and len(p25) >= 2:
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
            if chg > 25:
                risk_items.append({
                    "Risk": "Category Inflation",
                    "Detail": f"{cat_name}: avg per-turn spend up {chg:.0f}% YoY ({fmt(avg24)} → {fmt(avg25)})",
                    "Severity": "High" if chg > 50 else "Medium",
                })

    # 4. High-frequency units (>4 turns all-time)
    all_turns_count = build_turn_summary(_df_all)
    freq_units = all_turns_count.groupby(["Property Name", "Unit Label"]).size().reset_index(name="turns")
    chronic = freq_units[freq_units["turns"] >= 5]
    if len(chronic) > 0:
        risk_items.append({
            "Risk": "Chronic Vacancy",
            "Detail": f"{len(chronic)} units with 5+ turns — potential workmanship or tenant screening issues",
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
