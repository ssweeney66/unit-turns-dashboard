import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Renovation Analytics | Portfolio Dashboard",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Executive-Grade Styling ───────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    h1 { font-weight: 700 !important; letter-spacing: -0.5px; }
    h2, h3 { font-weight: 600 !important; color: #1a1a2e; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * { color: #e8e8e8 !important; }
    [data-testid="stSidebar"] h1 { color: #ffffff !important; font-size: 20px !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 14px; }

    /* ── KPI Cards ── */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border-left: 4px solid #2563eb;
    }
    div[data-testid="stMetric"] label { font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.5px; color: #6b7280 !important; font-weight: 500 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 700 !important; color: #111827 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 12px !important; }

    /* ── Tables ── */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .stDataFrame table { font-size: 13px !important; }
    .stDataFrame th { background: #f8fafc !important; font-weight: 600 !important;
        text-transform: uppercase; font-size: 11px !important; letter-spacing: 0.5px; color: #475569 !important; }

    /* ── Section Dividers ── */
    .section-header {
        background: linear-gradient(90deg, #f8fafc, #ffffff);
        border-left: 4px solid #2563eb;
        padding: 12px 18px;
        margin: 30px 0 18px 0;
        border-radius: 0 8px 8px 0;
    }
    .section-header h3 { margin: 0 !important; font-size: 16px !important; color: #1e293b !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] { font-weight: 500; font-size: 14px; }

    /* ── Narrative ── */
    .exec-narrative {
        background: #f0f4ff;
        border: 1px solid #dbeafe;
        border-radius: 10px;
        padding: 20px 24px;
        margin: 10px 0 20px 0;
        font-size: 14px;
        line-height: 1.7;
        color: #1e293b;
    }
    .exec-narrative strong { color: #1e40af; }

    /* ── Page Title Bar ── */
    .page-header {
        background: linear-gradient(135deg, #1a1a2e, #2563eb);
        color: white;
        padding: 28px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .page-header h1 { color: white !important; margin: 0 !important; font-size: 28px !important; }
    .page-header p { color: #bfdbfe; margin: 6px 0 0 0; font-size: 14px; }

    /* ── Outlier highlight ── */
    .outlier-alert {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Color Palette ─────────────────────────────────────────
COLORS = {
    "Full Turn": "#dc2626",
    "Partial Turn": "#f59e0b",
    "Make Ready": "#10b981",
    "primary": "#2563eb",
    "secondary": "#64748b",
    "accent": "#7c3aed",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#dc2626",
}

CHART_TEMPLATE = "plotly_white"
PROP_COLORS = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel2


# ── Data Loading ───────────────────────────────────────────
@st.cache_data
def load_data():
    data_path = Path(__file__).parent / "Unit Turns - AI Clean - 2.26.2026.xlsx"
    df = pd.read_excel(data_path)

    df["Building Code"] = df["Building Code"].fillna("").astype(str).str.strip()
    df["Unit Number"] = df["Unit Number"].fillna("").astype(str).str.strip()
    df["Turn Type"] = df["Turn Type"].fillna("").astype(str).str.strip()
    df["Property ID"] = df["Property ID"].fillna("").astype(str).str.strip()
    df["Invoice Amount"] = pd.to_numeric(df["Invoice Amount"], errors="coerce").fillna(0)
    df["Move-Out Date"] = pd.to_datetime(df["Move-Out Date"], errors="coerce")
    df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], errors="coerce")

    # Cap bad dates
    max_valid = pd.Timestamp(datetime.now().year + 1, 12, 31)
    df.loc[df["Invoice Date"] > max_valid, "Invoice Date"] = pd.NaT

    df = df[df["Property ID"] != ""].copy()

    # Unique Unit: Property ID + Building Code + Unit Number
    def make_unit_key(r):
        if r["Building Code"] != "":
            return f"{r['Property ID']}|{r['Building Code']}|{r['Unit Number']}"
        return f"{r['Property ID']}|{r['Unit Number']}"

    def make_unit_display(r):
        if r["Building Code"] != "":
            return f"{r['Building Code']} | {r['Unit Number']}"
        return r["Unit Number"]

    df["Unique Unit ID"] = df.apply(make_unit_key, axis=1)
    df["Unit Display"] = df.apply(make_unit_display, axis=1)
    df["Turn Key"] = df["Unique Unit ID"] + "|" + df["Move-Out Date"].astype(str) + "|" + df["Turn Type"]

    return df


df = load_data()


@st.cache_data
def build_turn_summary(_df):
    return (
        _df.groupby(["Property ID", "Property Name", "Unique Unit ID",
                      "Unit Display", "Move-Out Date", "Turn Type",
                      "Floor Plan", "Bedrooms", "Bathrooms", "Turn Key"])
        .agg(total_cost=("Invoice Amount", "sum"),
             line_items=("Invoice Amount", "count"),
             earliest_invoice=("Invoice Date", "min"),
             latest_invoice=("Invoice Date", "max"))
        .reset_index()
    )


@st.cache_data
def build_ft_data(_df):
    ft = _df[_df["Turn Type"] == "Full Turn"].copy()
    ts = (
        ft.groupby(["Property ID", "Property Name", "Unique Unit ID",
                     "Unit Display", "Move-Out Date", "Turn Key",
                     "Floor Plan", "Bedrooms", "Bathrooms"])
        .agg(total_cost=("Invoice Amount", "sum"),
             line_items=("Invoice Amount", "count"),
             completion_date=("Invoice Date", "max"))
        .reset_index()
    )
    ts["Year"] = ts["Move-Out Date"].dt.year
    ts["duration_days"] = (ts["completion_date"] - ts["Move-Out Date"]).dt.days
    ts.loc[ts["duration_days"] < 0, "duration_days"] = pd.NA
    return ft, ts


turn_summary = build_turn_summary(df)
ft_lines, ft_turns = build_ft_data(df)

# Portfolio unit count
portfolio_units = df.drop_duplicates("Unique Unit ID")["Unique Unit ID"].nunique()


# ── Helper: section header ────────────────────────────────
def section(number, title):
    st.markdown(
        f'<div class="section-header"><h3>{number}. {title}</h3></div>',
        unsafe_allow_html=True,
    )


def page_header(title, subtitle):
    st.markdown(
        f'<div class="page-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def fmt_currency(val, decimals=0):
    if pd.isna(val):
        return "—"
    return f"${val:,.{decimals}f}"


def fmt_pct(val, decimals=1):
    if pd.isna(val):
        return "—"
    return f"{val:.{decimals}f}%"


# ── Sidebar ───────────────────────────────────────────────
st.sidebar.markdown("### PORTFOLIO ANALYTICS")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Full Renovation Performance",
     "Portfolio Overview", "Property Drilldown", "Unit Search",
     "Anomaly Detection", "Vendor Analysis"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")

all_properties = sorted(df["Property Name"].unique())
selected_properties = st.sidebar.multiselect(
    "Properties", all_properties, default=all_properties
)

if page != "Full Renovation Performance":
    all_turn_types = sorted(df["Turn Type"].unique())
    selected_turn_types = st.sidebar.multiselect(
        "Turn Types", all_turn_types, default=all_turn_types
    )
else:
    selected_turn_types = ["Full Turn"]

date_min = df["Move-Out Date"].min().date()
date_max = df["Move-Out Date"].max().date()
date_range = st.sidebar.date_input(
    "Move-Out Date Range", value=(date_min, date_max),
    min_value=date_min, max_value=date_max,
)

# Apply filters
if len(date_range) == 2:
    mask = (
        df["Property Name"].isin(selected_properties)
        & df["Turn Type"].isin(selected_turn_types)
        & (df["Move-Out Date"].dt.date >= date_range[0])
        & (df["Move-Out Date"].dt.date <= date_range[1])
    )
    mask_ts = (
        turn_summary["Property Name"].isin(selected_properties)
        & turn_summary["Turn Type"].isin(selected_turn_types)
        & (turn_summary["Move-Out Date"].dt.date >= date_range[0])
        & (turn_summary["Move-Out Date"].dt.date <= date_range[1])
    )
else:
    mask = df["Property Name"].isin(selected_properties) & df["Turn Type"].isin(selected_turn_types)
    mask_ts = turn_summary["Property Name"].isin(selected_properties) & turn_summary["Turn Type"].isin(selected_turn_types)

fdf = df[mask].copy()
fts = turn_summary[mask_ts].copy()

st.sidebar.markdown("---")
st.sidebar.caption(f"Data: {len(df):,} line items | {df['Property ID'].nunique()} properties | {portfolio_units} units")


# ══════════════════════════════════════════════════════════
# PAGE: FULL RENOVATION PERFORMANCE
# ══════════════════════════════════════════════════════════
if page == "Full Renovation Performance":

    page_header(
        "Full Renovation Performance",
        f"Comprehensive analysis of Full Turn renovations across {len(selected_properties)} properties"
    )

    # Filter FT data
    ft_f = ft_turns[ft_turns["Property Name"].isin(selected_properties)].copy()
    ft_l = ft_lines[ft_lines["Property Name"].isin(selected_properties)].copy()
    if len(date_range) == 2:
        ft_f = ft_f[(ft_f["Move-Out Date"].dt.date >= date_range[0]) & (ft_f["Move-Out Date"].dt.date <= date_range[1])]
        ft_l = ft_l[ft_l["Turn Key"].isin(ft_f["Turn Key"])]

    YEARS = [2021, 2022, 2023, 2024, 2025]
    ft_recent = ft_f[ft_f["Year"].isin(YEARS)]

    # ── Executive Summary Narrative ──────────────────────
    total_turns = len(ft_f)
    total_spend = ft_f["total_cost"].sum()
    avg_cost = ft_f["total_cost"].mean() if total_turns else 0
    median_cost = ft_f["total_cost"].median() if total_turns else 0
    median_dur = ft_f["duration_days"].dropna().median() if total_turns else 0

    # YoY change
    turns_24 = len(ft_f[ft_f["Year"] == 2024])
    turns_25 = len(ft_f[ft_f["Year"] == 2025])
    avg_24 = ft_f[ft_f["Year"] == 2024]["total_cost"].mean() if turns_24 else 0
    avg_25 = ft_f[ft_f["Year"] == 2025]["total_cost"].mean() if turns_25 else 0
    cost_change = ((avg_25 - avg_24) / avg_24 * 100) if avg_24 else 0

    # Penetration rate
    ft_units = ft_f["Unique Unit ID"].nunique()
    penetration = ft_units / portfolio_units * 100 if portfolio_units else 0

    st.markdown(f"""
    <div class="exec-narrative">
        The portfolio has executed <strong>{total_turns:,} Full Turn renovations</strong> totaling
        <strong>{fmt_currency(total_spend)}</strong> in capital deployment. In 2025, <strong>{turns_25} turns</strong>
        were completed at an average cost of <strong>{fmt_currency(avg_25)}</strong> per unit
        ({'↑' if cost_change > 0 else '↓'} <strong>{abs(cost_change):.1f}%</strong> vs. 2024).
        Median turn duration is <strong>{median_dur:.0f} days</strong> from move-out to final invoice.
        <strong>{fmt_pct(penetration)}</strong> of portfolio units ({ft_units} of {portfolio_units}) have undergone
        at least one full renovation.
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Full Turns", f"{total_turns:,}")
    c2.metric("Capital Deployed", fmt_currency(total_spend))
    c3.metric("Avg Cost / Turn", fmt_currency(avg_cost))
    c4.metric("Median Cost", fmt_currency(median_cost))
    c5.metric("Median Duration", f"{median_dur:.0f} days" if not pd.isna(median_dur) else "—")
    c6.metric("Penetration Rate", fmt_pct(penetration))

    # ── MODULE 1: YoY Volume Matrix ──────────────────────
    section("1", "Year-over-Year Volume Matrix")

    yoy_data = ft_f[ft_f["Year"].isin(YEARS)].copy()
    yoy_matrix = (
        yoy_data.groupby(["Property Name", "Year"]).size()
        .unstack(fill_value=0).reindex(columns=YEARS, fill_value=0)
    )
    yoy_matrix["Total"] = yoy_matrix.sum(axis=1)
    yoy_matrix = yoy_matrix.sort_values("Total", ascending=False)
    yoy_matrix.loc["PORTFOLIO TOTAL"] = yoy_matrix.sum()
    yoy_matrix = yoy_matrix.astype(int)

    # Cost per turn matrix
    yoy_cost = (
        yoy_data.groupby(["Property Name", "Year"])["total_cost"].mean()
        .unstack(fill_value=0).reindex(columns=YEARS, fill_value=0)
    )

    tab_vol, tab_cost = st.tabs(["Turn Volume", "Avg Cost per Turn"])

    with tab_vol:
        def style_volume(val):
            if not isinstance(val, (int, float)):
                return ""
            if val == 0:
                return "color: #d1d5db;"
            elif val >= 10:
                return "background: #fee2e2; font-weight: 600; color: #991b1b;"
            elif val >= 5:
                return "background: #fef3c7; font-weight: 500; color: #92400e;"
            return "color: #1f2937;"

        st.dataframe(yoy_matrix.style.map(style_volume), use_container_width=True)

    with tab_cost:
        cost_display = yoy_cost.copy()
        for col in cost_display.columns:
            cost_display[col] = cost_display[col].apply(lambda x: fmt_currency(x) if x > 0 else "—")
        st.dataframe(cost_display, use_container_width=True)

    # Stacked bar
    yoy_chart = yoy_data.groupby(["Property Name", "Year"]).size().reset_index(name="Turns")
    fig_yoy = px.bar(
        yoy_chart, x="Year", y="Turns", color="Property Name",
        barmode="stack", template=CHART_TEMPLATE,
        color_discrete_sequence=PROP_COLORS,
        labels={"Turns": "Full Turns Completed"},
    )
    fig_yoy.update_layout(
        xaxis=dict(dtick=1, title=""),
        yaxis=dict(title="Full Turns"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, font=dict(size=11)),
        margin=dict(t=20, b=80),
        height=420,
    )
    st.plotly_chart(fig_yoy, use_container_width=True)

    # ── MODULE 2: Recent 10 Full Turns ───────────────────
    section("2", "Most Recently Completed Full Turns")

    recent = ft_f.sort_values("completion_date", ascending=False).head(10).copy()
    recent["Turn Duration"] = recent["duration_days"].apply(lambda x: f"{x:.0f} days" if pd.notna(x) else "—")
    recent["Completion"] = recent["completion_date"].dt.strftime("%b %d, %Y").fillna("—")
    recent["Move-Out"] = recent["Move-Out Date"].dt.strftime("%b %d, %Y")
    recent["Cost"] = recent["total_cost"].apply(fmt_currency)
    recent["#"] = range(1, len(recent) + 1)

    display_recent = recent[["#", "Property Name", "Unit Display", "Cost",
                              "Completion", "Move-Out", "Turn Duration", "line_items"]].copy()
    display_recent.columns = ["#", "Property", "Unit", "Total Cost",
                               "Completion Date", "Move-Out Date", "Duration", "Invoices"]
    st.dataframe(display_recent, use_container_width=True, hide_index=True)

    # ── MODULE 3: Categorical Spend YoY ──────────────────
    section("3", "Categorical Spend Analysis")
    st.caption("Average spend per budget category per Full Turn — tracks how renovation cost composition evolves year over year.")

    cat_lines = ft_l[ft_l["Move-Out Date"].dt.year.isin(YEARS)].copy()
    cat_lines["Year"] = cat_lines["Move-Out Date"].dt.year

    cat_year_spend = cat_lines.groupby(["Budget Category", "Year"])["Invoice Amount"].sum().reset_index()
    turns_per_year = yoy_data.groupby("Year").size().reset_index(name="turn_count")
    cat_year_spend = cat_year_spend.merge(turns_per_year, on="Year", how="left")
    cat_year_spend["Avg Per Turn"] = cat_year_spend["Invoice Amount"] / cat_year_spend["turn_count"]

    cat_pivot = (
        cat_year_spend.pivot_table(index="Budget Category", columns="Year", values="Avg Per Turn", fill_value=0)
        .reindex(columns=YEARS, fill_value=0)
    )
    cat_pivot["5-Year Avg"] = cat_pivot[YEARS].mean(axis=1)
    cat_pivot = cat_pivot.sort_values("5-Year Avg", ascending=False)

    # Add YoY % change column
    cat_pivot["2024 → 2025"] = cat_pivot.apply(
        lambda r: ((r[2025] - r[2024]) / r[2024] * 100) if r[2024] > 0 and r[2025] > 0 else None, axis=1
    )

    tab_table, tab_chart = st.tabs(["Data Table", "Trend Visualization"])

    with tab_table:
        cat_display = cat_pivot.copy()
        for col in YEARS + ["5-Year Avg"]:
            cat_display[col] = cat_display[col].apply(lambda x: fmt_currency(x) if x > 0 else "—")
        cat_display["2024 → 2025"] = cat_display["2024 → 2025"].apply(
            lambda x: f"+{x:.1f}%" if pd.notna(x) and x > 0 else (f"{x:.1f}%" if pd.notna(x) else "—")
        )
        st.dataframe(cat_display, use_container_width=True)

    with tab_chart:
        top_cats = cat_pivot.index[:8].tolist()
        trend_data = cat_year_spend[cat_year_spend["Budget Category"].isin(top_cats)]

        fig_cat = px.line(
            trend_data, x="Year", y="Avg Per Turn", color="Budget Category",
            markers=True, template=CHART_TEMPLATE,
            labels={"Avg Per Turn": "Average Cost per Full Turn ($)", "Year": ""},
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig_cat.update_layout(
            xaxis=dict(dtick=1),
            legend=dict(orientation="h", yanchor="top", y=-0.15, font=dict(size=11)),
            margin=dict(t=20, b=80),
            height=450,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

        # Stacked area: total composition
        st.markdown("**Cost Composition Over Time** — Stacked view of top 8 categories")
        fig_area = px.area(
            trend_data, x="Year", y="Avg Per Turn", color="Budget Category",
            template=CHART_TEMPLATE,
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={"Avg Per Turn": "Avg Cost per Turn ($)"},
        )
        fig_area.update_layout(
            xaxis=dict(dtick=1, title=""),
            legend=dict(orientation="h", yanchor="top", y=-0.15, font=dict(size=11)),
            margin=dict(t=20, b=80),
            height=420,
        )
        st.plotly_chart(fig_area, use_container_width=True)

    # ── MODULE 4: Outlier Detection ──────────────────────
    section("4", "Outlier Detection — Turns Exceeding Property Average by >15%")

    prop_avg = ft_f.groupby("Property Name")["total_cost"].mean().reset_index().rename(columns={"total_cost": "property_avg"})
    outlier_df = ft_f.merge(prop_avg, on="Property Name")
    outlier_df["threshold"] = outlier_df["property_avg"] * 1.15
    outlier_df["pct_over"] = (outlier_df["total_cost"] - outlier_df["property_avg"]) / outlier_df["property_avg"] * 100
    outliers = outlier_df[outlier_df["total_cost"] > outlier_df["threshold"]].sort_values("pct_over", ascending=False)
    non_outliers = outlier_df[outlier_df["total_cost"] <= outlier_df["threshold"]]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Outlier Turns", f"{len(outliers)} of {len(ft_f)}")
    col2.metric("Outlier Rate", fmt_pct(len(outliers) / len(ft_f) * 100 if len(ft_f) else 0))
    col3.metric("Avg Overage", fmt_currency((outliers["total_cost"] - outliers["property_avg"]).mean()) if len(outliers) else "—")
    col4.metric("Total Excess Spend", fmt_currency((outliers["total_cost"] - outliers["property_avg"]).sum()) if len(outliers) else "$0")

    if len(outliers) > 0:
        tab_list, tab_scatter, tab_by_prop = st.tabs(["Outlier List", "Scatter Analysis", "By Property"])

        with tab_list:
            out_display = outliers[[
                "Property Name", "Unit Display", "Move-Out Date",
                "total_cost", "property_avg", "pct_over", "line_items"
            ]].copy()
            out_display["Move-Out Date"] = out_display["Move-Out Date"].dt.strftime("%b %d, %Y")
            out_display["total_cost"] = out_display["total_cost"].apply(fmt_currency)
            out_display["property_avg"] = out_display["property_avg"].apply(fmt_currency)
            out_display["pct_over"] = out_display["pct_over"].apply(lambda x: f"+{x:.1f}%")
            out_display.columns = ["Property", "Unit", "Move-Out", "Actual Cost",
                                    "Property Avg", "% Over Avg", "Invoices"]
            st.dataframe(out_display, use_container_width=True, hide_index=True, height=450)

        with tab_scatter:
            fig_out = go.Figure()

            # Normal turns
            fig_out.add_trace(go.Scatter(
                x=non_outliers["property_avg"], y=non_outliers["total_cost"],
                mode="markers", name="Within Threshold",
                marker=dict(color="#94a3b8", size=7, opacity=0.4),
                hovertemplate="Property Avg: %{x:$,.0f}<br>Actual: %{y:$,.0f}<extra></extra>",
            ))

            # Outliers
            fig_out.add_trace(go.Scatter(
                x=outliers["property_avg"], y=outliers["total_cost"],
                mode="markers", name="Outlier (>15% Over Avg)",
                marker=dict(color="#dc2626", size=10, opacity=0.8, line=dict(width=1, color="white")),
                text=outliers["Property Name"] + " — " + outliers["Unit Display"],
                hovertemplate="%{text}<br>Property Avg: %{x:$,.0f}<br>Actual: %{y:$,.0f}<extra></extra>",
            ))

            max_val = max(outlier_df["total_cost"].max(), outlier_df["property_avg"].max()) * 1.1
            fig_out.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val], mode="lines", name="Expected (1:1)",
                line=dict(dash="dash", color="#94a3b8", width=1),
            ))
            fig_out.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val * 1.15], mode="lines", name="+15% Threshold",
                line=dict(dash="dot", color="#dc2626", width=1.5),
            ))

            fig_out.update_layout(
                template=CHART_TEMPLATE,
                xaxis_title="Property Historical Avg Full Turn Cost ($)",
                yaxis_title="Actual Full Turn Cost ($)",
                legend=dict(orientation="h", yanchor="top", y=-0.12, font=dict(size=11)),
                margin=dict(t=20, b=80),
                height=500,
            )
            st.plotly_chart(fig_out, use_container_width=True)

        with tab_by_prop:
            out_by_prop = (
                outliers.groupby("Property Name")
                .agg(outlier_count=("total_cost", "count"),
                     avg_overage=("pct_over", "mean"),
                     total_excess=("total_cost", lambda x: (x - outliers.loc[x.index, "property_avg"]).sum()))
                .reset_index()
                .sort_values("outlier_count", ascending=False)
            )
            out_by_prop["avg_overage"] = out_by_prop["avg_overage"].apply(lambda x: f"+{x:.1f}%")
            out_by_prop["total_excess"] = out_by_prop["total_excess"].apply(fmt_currency)
            out_by_prop.columns = ["Property", "Outlier Count", "Avg % Over", "Total Excess Spend"]
            st.dataframe(out_by_prop, use_container_width=True, hide_index=True)
    else:
        st.success("No outliers detected with current filters.")


# ══════════════════════════════════════════════════════════
# PAGE: PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════
elif page == "Portfolio Overview":
    page_header("Portfolio Overview", "All turn types across the full portfolio")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Turns", f"{fts['Turn Key'].nunique():,}")
    c2.metric("Total Spend", fmt_currency(fts["total_cost"].sum()))
    c3.metric("Avg Cost / Turn", fmt_currency(fts["total_cost"].mean()) if len(fts) else "—")
    c4.metric("Unique Units", f"{fts['Unique Unit ID'].nunique():,}")

    section("A", "Breakdown by Turn Type")
    col1, col2 = st.columns(2)

    with col1:
        type_summary = (
            fts.groupby("Turn Type")
            .agg(count=("Turn Key", "nunique"), total=("total_cost", "sum"),
                 avg=("total_cost", "mean"), median=("total_cost", "median"))
            .reset_index().sort_values("total", ascending=False)
        )
        type_summary.columns = ["Turn Type", "Count", "Total Spend", "Avg Cost", "Median Cost"]
        for col_name in ["Total Spend", "Avg Cost", "Median Cost"]:
            type_summary[col_name] = type_summary[col_name].apply(fmt_currency)
        st.dataframe(type_summary, use_container_width=True, hide_index=True)

    with col2:
        fig_pie = px.pie(
            fts, values="total_cost", names="Turn Type", color="Turn Type",
            color_discrete_map=COLORS, hole=0.45, template=CHART_TEMPLATE,
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=12)
        fig_pie.update_layout(margin=dict(t=20, b=20), height=320, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    section("B", "Monthly Spend Trend")
    fts_time = fts.copy()
    fts_time["Month"] = fts_time["Move-Out Date"].dt.to_period("M").astype(str)
    monthly = fts_time.groupby(["Month", "Turn Type"]).agg(spend=("total_cost", "sum")).reset_index()
    fig_trend = px.bar(
        monthly, x="Month", y="spend", color="Turn Type",
        color_discrete_map=COLORS, template=CHART_TEMPLATE,
        labels={"spend": "Total Spend ($)", "Month": ""},
    )
    fig_trend.update_layout(xaxis_tickangle=-45, bargap=0.1, margin=dict(t=20, b=60),
                            legend=dict(orientation="h", y=-0.2), height=400)
    st.plotly_chart(fig_trend, use_container_width=True)

    section("C", "Property Comparison")
    prop_summary = fts.groupby(["Property Name", "Turn Type"]).agg(count=("Turn Key", "nunique")).reset_index()
    fig_prop = px.bar(
        prop_summary, x="Property Name", y="count", color="Turn Type",
        color_discrete_map=COLORS, barmode="stack", template=CHART_TEMPLATE,
        labels={"count": "Turns"},
    )
    fig_prop.update_layout(xaxis_tickangle=-45, margin=dict(t=20, b=60),
                           legend=dict(orientation="h", y=-0.2), height=400)
    st.plotly_chart(fig_prop, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE: PROPERTY DRILLDOWN
# ══════════════════════════════════════════════════════════
elif page == "Property Drilldown":
    page_header("Property Drilldown", "Deep-dive into individual property performance")

    prop_names = sorted(fts["Property Name"].unique())
    if not prop_names:
        st.warning("No data with current filters.")
    else:
        prop_choice = st.selectbox("Select Property", prop_names)
        prop_ts = fts[fts["Property Name"] == prop_choice].copy()
        prop_df = fdf[fdf["Property Name"] == prop_choice].copy()

        if len(prop_ts) == 0:
            st.warning("No turns for this property with current filters.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Turns", f"{len(prop_ts):,}")
            c2.metric("Total Spend", fmt_currency(prop_ts["total_cost"].sum()))
            c3.metric("Avg Cost / Turn", fmt_currency(prop_ts["total_cost"].mean()))
            c4.metric("Unique Units", f"{prop_ts['Unique Unit ID'].nunique():,}")

            col1, col2 = st.columns(2)
            with col1:
                section("", "Turn Type Breakdown")
                fig = px.pie(
                    prop_ts, values="total_cost", names="Turn Type", color="Turn Type",
                    color_discrete_map=COLORS, hole=0.45, template=CHART_TEMPLATE,
                )
                fig.update_traces(textinfo="percent+label")
                fig.update_layout(margin=dict(t=10, b=10), height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                section("", "Top Budget Categories")
                cat_spend = (
                    prop_df.groupby("Budget Category")["Invoice Amount"]
                    .sum().reset_index().sort_values("Invoice Amount", ascending=True).tail(10)
                )
                fig_cat = px.bar(
                    cat_spend, x="Invoice Amount", y="Budget Category",
                    orientation="h", template=CHART_TEMPLATE,
                    color_discrete_sequence=[COLORS["primary"]],
                    labels={"Invoice Amount": "Total ($)"},
                )
                fig_cat.update_layout(margin=dict(t=10, b=10), height=300)
                st.plotly_chart(fig_cat, use_container_width=True)

            section("", "Turn Timeline")
            timeline = prop_ts.sort_values("Move-Out Date")
            fig_tl = px.scatter(
                timeline, x="Move-Out Date", y="Unit Display", color="Turn Type",
                size="total_cost", color_discrete_map=COLORS, template=CHART_TEMPLATE,
                labels={"total_cost": "Cost ($)"}, hover_data=["total_cost", "line_items"],
            )
            fig_tl.update_layout(
                height=max(400, len(timeline["Unit Display"].unique()) * 22),
                margin=dict(t=10, b=10),
                yaxis_title="",
            )
            st.plotly_chart(fig_tl, use_container_width=True)

            section("", "All Turns")
            dtable = prop_ts[["Unit Display", "Move-Out Date", "Turn Type", "total_cost", "line_items", "Floor Plan"]].copy()
            dtable["Move-Out Date"] = dtable["Move-Out Date"].dt.strftime("%b %d, %Y")
            dtable["total_cost"] = dtable["total_cost"].apply(fmt_currency)
            dtable.columns = ["Unit", "Move-Out", "Type", "Total Cost", "Invoices", "Floor Plan"]
            dtable = dtable.sort_values("Move-Out", ascending=False)
            st.dataframe(dtable, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# PAGE: UNIT SEARCH
# ══════════════════════════════════════════════════════════
elif page == "Unit Search":
    page_header("Unit Search & History", "Look up the complete renovation history for any unit")

    col1, col2 = st.columns(2)
    with col1:
        prop_choice = st.selectbox("Property", sorted(df["Property Name"].unique()))
    with col2:
        prop_units = sorted(df[df["Property Name"] == prop_choice]["Unit Display"].unique())
        unit_choice = st.selectbox("Unit", prop_units)

    unit_ts = turn_summary[
        (turn_summary["Property Name"] == prop_choice) & (turn_summary["Unit Display"] == unit_choice)
    ].sort_values("Move-Out Date", ascending=False)

    unit_df = df[(df["Property Name"] == prop_choice) & (df["Unit Display"] == unit_choice)]

    if len(unit_ts) == 0:
        st.info("No turn history found for this unit.")
    else:
        sample = unit_df.iloc[0]
        st.markdown(
            f"**{prop_choice} — Unit {unit_choice}** &nbsp;|&nbsp; "
            f"Floor Plan: `{sample['Floor Plan']}` &nbsp;|&nbsp; "
            f"Bed/Bath: `{sample['Bedrooms']:.0f} / {sample['Bathrooms']:.0f}` &nbsp;|&nbsp; "
            f"ID: `{sample['Unique Unit ID']}`"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Turns", len(unit_ts))
        c2.metric("Lifetime Spend", fmt_currency(unit_ts["total_cost"].sum()))
        c3.metric("Avg Cost / Turn", fmt_currency(unit_ts["total_cost"].mean()))

        section("", "Turn History")
        for _, turn in unit_ts.iterrows():
            label = (
                f"**{turn['Turn Type']}** — {turn['Move-Out Date'].strftime('%b %d, %Y')}"
                f" — {fmt_currency(turn['total_cost'])} ({turn['line_items']} invoices)"
            )
            with st.expander(label):
                items = unit_df[unit_df["Turn Key"] == turn["Turn Key"]].sort_values("Invoice Date")
                disp = items[["Invoice Date", "Vendor Name", "Budget Category",
                              "Cost Type", "Invoice Amount", "Line Item Notes"]].copy()
                disp["Invoice Date"] = disp["Invoice Date"].dt.strftime("%b %d, %Y").fillna("—")
                disp["Invoice Amount"] = disp["Invoice Amount"].apply(lambda x: fmt_currency(x, 2))
                st.dataframe(disp, use_container_width=True, hide_index=True)

        section("", "Spending History")
        fig = px.bar(
            unit_ts, x="Move-Out Date", y="total_cost", color="Turn Type",
            color_discrete_map=COLORS, template=CHART_TEMPLATE,
            labels={"total_cost": "Turn Cost ($)", "Move-Out Date": ""},
        )
        fig.update_layout(margin=dict(t=10, b=10), height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE: ANOMALY DETECTION
# ══════════════════════════════════════════════════════════
elif page == "Anomaly Detection":
    page_header("Anomaly Detection", "Identify unusual patterns requiring investigation")

    tab1, tab2, tab3 = st.tabs(["High-Frequency Units", "Statistical Cost Outliers", "Data Quality"])

    with tab1:
        st.caption("Units with unusually high turn frequency may indicate chronic vacancy or workmanship issues.")
        threshold = st.slider("Minimum turn count to flag", 3, 10, 4)

        unit_freq = (
            fts.groupby(["Property Name", "Unit Display"])
            .agg(turn_count=("Turn Key", "nunique"), total_spend=("total_cost", "sum"),
                 types=("Turn Type", lambda x: ", ".join(sorted(x.unique()))))
            .reset_index().sort_values("turn_count", ascending=False)
        )
        flagged = unit_freq[unit_freq["turn_count"] >= threshold]
        st.metric("Flagged Units", len(flagged))

        if len(flagged) > 0:
            disp = flagged.copy()
            disp["total_spend"] = disp["total_spend"].apply(fmt_currency)
            disp.columns = ["Property", "Unit", "Turns", "Total Spend", "Turn Types"]
            st.dataframe(disp, use_container_width=True, hide_index=True)

    with tab2:
        st.caption("Turns where total cost exceeds Q3 + 1.5 x IQR for their type (statistical outlier method).")
        outlier_rows = []
        for tt in fts["Turn Type"].unique():
            subset = fts[fts["Turn Type"] == tt]["total_cost"]
            if len(subset) < 4:
                continue
            q1, q3 = subset.quantile(0.25), subset.quantile(0.75)
            upper = q3 + 1.5 * (q3 - q1)
            for _, row in fts[(fts["Turn Type"] == tt) & (fts["total_cost"] > upper)].iterrows():
                outlier_rows.append({
                    "Property": row["Property Name"], "Unit": row["Unit Display"],
                    "Move-Out": row["Move-Out Date"].strftime("%b %d, %Y"),
                    "Type": row["Turn Type"],
                    "Cost": row["total_cost"], "Threshold": upper,
                })

        if outlier_rows:
            odf = pd.DataFrame(outlier_rows).sort_values("Cost", ascending=False)
            st.metric("Statistical Outliers", len(odf))
            odf["Cost"] = odf["Cost"].apply(fmt_currency)
            odf["Threshold"] = odf["Threshold"].apply(fmt_currency)
            st.dataframe(odf, use_container_width=True, hide_index=True)
        else:
            st.success("No statistical outliers detected.")

    with tab3:
        st.caption("Data integrity checks on the filtered dataset.")
        negatives = fdf[fdf["Invoice Amount"] < 0]
        missing_unit = fdf[fdf["Unit Number"].str.strip() == ""]
        missing_inv = fdf[fdf["Invoice Number"].isna() | (fdf["Invoice Number"].astype(str).str.strip() == "")]

        c1, c2, c3 = st.columns(3)
        c1.metric("Negative Invoices", len(negatives))
        c2.metric("Missing Unit #", len(missing_unit))
        c3.metric("Missing Invoice #", len(missing_inv))

        if len(negatives) > 0:
            st.markdown("**Negative Invoice Records:**")
            neg_d = negatives[["Property Name", "Unit Display", "Vendor Name",
                               "Invoice Amount", "Budget Category", "Line Item Notes"]].head(20).copy()
            neg_d["Invoice Amount"] = neg_d["Invoice Amount"].apply(lambda x: fmt_currency(x, 2))
            st.dataframe(neg_d, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# PAGE: VENDOR ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "Vendor Analysis":
    page_header("Vendor Analysis", "Spend concentration, benchmarking, and vendor drilldown")

    vendor_summary = (
        fdf.groupby("Vendor Name")
        .agg(invoices=("Invoice Amount", "count"), total_spend=("Invoice Amount", "sum"),
             avg_invoice=("Invoice Amount", "mean"), properties=("Property Name", "nunique"))
        .reset_index().sort_values("total_spend", ascending=False)
    )

    section("", "Top 15 Vendors by Spend")
    top15 = vendor_summary.head(15)
    fig_v = px.bar(
        top15, x="Vendor Name", y="total_spend",
        text=top15["total_spend"].apply(fmt_currency),
        template=CHART_TEMPLATE,
        color_discrete_sequence=[COLORS["primary"]],
        labels={"total_spend": "Total Spend ($)"},
    )
    fig_v.update_layout(xaxis_tickangle=-45, margin=dict(t=10, b=60), height=400)
    fig_v.update_traces(textposition="outside")
    st.plotly_chart(fig_v, use_container_width=True)

    section("", "All Vendors")
    vd = vendor_summary.copy()
    vd["total_spend"] = vd["total_spend"].apply(fmt_currency)
    vd["avg_invoice"] = vd["avg_invoice"].apply(fmt_currency)
    vd.columns = ["Vendor", "Invoices", "Total Spend", "Avg Invoice", "Properties"]
    st.dataframe(vd, use_container_width=True, hide_index=True)

    section("", "Vendor Drilldown")
    vendor_choice = st.selectbox("Select Vendor", vendor_summary["Vendor Name"].tolist())
    vdf = fdf[fdf["Vendor Name"] == vendor_choice]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**Invoices:** {len(vdf):,}")
        st.markdown(f"**Total Spend:** {fmt_currency(vdf['Invoice Amount'].sum())}")
        st.markdown(f"**Properties:** {vdf['Property Name'].nunique()}")
        st.markdown(f"**Avg Invoice:** {fmt_currency(vdf['Invoice Amount'].mean())}")
    with col2:
        cat_b = vdf.groupby("Budget Category")["Invoice Amount"].sum().reset_index().sort_values("Invoice Amount", ascending=False)
        fig_vc = px.pie(cat_b, values="Invoice Amount", names="Budget Category", hole=0.45, template=CHART_TEMPLATE)
        fig_vc.update_layout(margin=dict(t=10, b=10), height=300)
        st.plotly_chart(fig_vc, use_container_width=True)

    st.markdown("**Recent Invoices**")
    vi = vdf[["Property Name", "Unit Display", "Invoice Date", "Invoice Number",
              "Invoice Amount", "Budget Category", "Line Item Notes"]].copy()
    vi["Invoice Date"] = vi["Invoice Date"].dt.strftime("%b %d, %Y").fillna("—")
    vi["Invoice Amount"] = vi["Invoice Amount"].apply(lambda x: fmt_currency(x, 2))
    vi = vi.sort_values("Invoice Date", ascending=False)
    st.dataframe(vi.head(50), use_container_width=True, hide_index=True)
