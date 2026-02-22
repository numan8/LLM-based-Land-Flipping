# app.py — Cash Sales Velocity Dashboard + AI Stats (LLM Q&A + Safe Plotting)
# Put ai_stats_clean_for_velocity.csv next to this file.

import os
import json
import re
from typing import Any, Dict, Optional, Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from openai import OpenAI


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Cash Sales Velocity Dashboard", layout="wide")
st.title("Cash Sales Velocity Dashboard")
st.caption(
    "Goal: Find pricing/markup sweet spots that maximize probability of selling in ≤30 or ≤60 days, "
    "and increase compounding cycles."
)

# ----------------------------
# Config knobs (client-safety)
# ----------------------------
MARKUP_CAP_DEFAULT = 10.0       # cap outliers (prevents 1495x)
MIN_ROWS_FOR_BINS = 20          # require enough rows to show "Best bin"
MIN_ROWS_FOR_CURVE = 40         # require enough rows for stable curve
MIN_ROWS_SOFT_WARNING = 40      # show caution when sample is small

DEFAULT_RELIABILITY_PROB = 0.70  # "reliably sells fast" threshold for highest markup guidance
DEFAULT_MIN_N_THRESHOLD = 40     # minimum rows to make strong threshold claims


# ----------------------------
# OpenAI config (REQUIRED)
# ----------------------------
def get_openai_client_and_model() -> Tuple[OpenAI, str]:
    api_key = None
    model = None

    # Prefer Streamlit secrets, then env vars
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    if "OPENAI_MODEL" in st.secrets:
        model = st.secrets["OPENAI_MODEL"]

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    model = model or os.getenv("OPENAI_MODEL") or "gpt-5.2"

    if not api_key:
        st.error(
            "OPENAI_API_KEY is missing. Add it to `.streamlit/secrets.toml` or set it as an environment variable."
        )
        st.stop()

    return OpenAI(api_key=api_key), model


# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Required columns that must exist in some form
    required = [
        "Acres",
        "County, State",
        "Property Location or City",
        "PURCHASE DATE",
        "SALE DATE - start",
        "Cash Sales Price - amount",
        "days_to_sale",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Dates
    df["PURCHASE DATE"] = pd.to_datetime(df["PURCHASE DATE"], errors="coerce")
    df["SALE DATE - start"] = pd.to_datetime(df["SALE DATE - start"], errors="coerce")

    # Numeric conversions (some optional)
    for c in ["Acres", "Cash Sales Price - amount", "days_to_sale"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional cost components
    for col in ["Purchase Price", "Title Escrow Costs", "Taxes we Paid", "Total Purchase Price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Total cost: prefer Total Purchase Price, else compute from components if present
    if "Total Purchase Price" not in df.columns or df["Total Purchase Price"].isna().all():
        base = df["Purchase Price"] if "Purchase Price" in df.columns else np.nan
        escrow = df["Title Escrow Costs"] if "Title Escrow Costs" in df.columns else 0.0
        taxes = df["Taxes we Paid"] if "Taxes we Paid" in df.columns else 0.0
        df["Total Purchase Price"] = pd.to_numeric(base, errors="coerce") + pd.to_numeric(escrow, errors="coerce") + pd.to_numeric(taxes, errors="coerce")

    # Markup multiple: prefer existing, else compute sale / total cost
    if "markup_multiple" in df.columns:
        df["markup_multiple"] = pd.to_numeric(df["markup_multiple"], errors="coerce")
    else:
        df["markup_multiple"] = np.where(
            df["Total Purchase Price"] > 0,
            df["Cash Sales Price - amount"] / df["Total Purchase Price"],
            np.nan
        )

    # Derived metrics
    df["profit_$"] = df["Cash Sales Price - amount"] - df["Total Purchase Price"]
    df["profit_pct_cost"] = np.where(df["Total Purchase Price"] > 0, df["profit_$"] / df["Total Purchase Price"], np.nan)

    # Commission assumptions (client instructions)
    # - Acq Agent: 4% of PROFIT
    # - Sales Agent: 4% of PROFIT
    # - Affiliate/Listing Agent: 10% of SALE PRICE
    df["commission_profit_based_$"] = 0.08 * df["profit_$"].clip(lower=0)
    df["commission_sale_based_$"] = 0.10 * df["Cash Sales Price - amount"].clip(lower=0)

    df["net_profit_$"] = df["profit_$"] - df["commission_profit_based_$"] - df["commission_sale_based_$"]
    df["net_profit_pct_cost"] = np.where(df["Total Purchase Price"] > 0, df["net_profit_$"] / df["Total Purchase Price"], np.nan)

    # Buckets
    df["speed_bucket"] = pd.cut(
        df["days_to_sale"],
        bins=[-1, 30, 60, 90, 180, 3650],
        labels=["≤30", "31–60", "61–90", "91–180", "181+"]
    )

    df["acres_bucket"] = pd.cut(
        df["Acres"],
        bins=[-0.01, 0.25, 0.5, 1, 2, 5, 10, 1e9],
        labels=["≤0.25", "0.26–0.5", "0.51–1", "1.01–2", "2.01–5", "5.01–10", "10+"]
    )

    return df


DATA_PATH = "ai_stats_clean_for_velocity.csv"
df = load_data(DATA_PATH)

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filters")

min_date = df["PURCHASE DATE"].min()
max_date = df["PURCHASE DATE"].max()

date_range = st.sidebar.date_input(
    "Purchase date range",
    value=(min_date.date() if pd.notna(min_date) else None,
           max_date.date() if pd.notna(max_date) else None),
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date, end_date = min_date, max_date

county_options = sorted([c for c in df["County, State"].dropna().unique()])
selected_counties = st.sidebar.multiselect("County, State", county_options, default=[])

city_options = sorted([c for c in df["Property Location or City"].dropna().unique()])
selected_cities = st.sidebar.multiselect("City", city_options, default=[])

# days slider
dmin = int(np.nanmin(df["days_to_sale"])) if df["days_to_sale"].notna().any() else 0
dmax = int(np.nanmax(df["days_to_sale"])) if df["days_to_sale"].notna().any() else 365
days_range = st.sidebar.slider("Days to sale range", min_value=dmin, max_value=dmax, value=(dmin, min(dmax, 365)))

# markup slider with cap
st.sidebar.subheader("Markup controls")
markup_cap = st.sidebar.number_input(
    "Markup cap (max multiple shown/used)",
    min_value=1.0,
    max_value=100.0,
    value=float(MARKUP_CAP_DEFAULT),
    step=0.5,
    help="Caps extreme outliers so the dashboard stays realistic."
)

mm_min_raw = float(np.nanmin(df["markup_multiple"])) if df["markup_multiple"].notna().any() else 0.5
mm_min = max(0.0, mm_min_raw)
mm_max = min(float(np.nanmax(df["markup_multiple"])), float(markup_cap)) if df["markup_multiple"].notna().any() else float(markup_cap)
mm_max = max(mm_max, mm_min + 0.01)

markup_range = st.sidebar.slider(
    "Markup multiple range (Sale / Total Cost)",
    min_value=float(mm_min),
    max_value=float(mm_max),
    value=(float(mm_min), float(mm_max))
)

# Reliability threshold controls (for "highest markup that still sells fast")
st.sidebar.subheader("Velocity-first thresholds")
reliability_prob = st.sidebar.slider(
    "Reliability threshold P(sell fast) ≥",
    min_value=0.50,
    max_value=0.90,
    value=float(DEFAULT_RELIABILITY_PROB),
    step=0.05,
    help="Used to estimate the HIGHEST markup multiple that still sells within the target window with this probability."
)
min_n_threshold = st.sidebar.number_input(
    "Minimum deals required for threshold claims",
    min_value=10,
    max_value=500,
    value=int(DEFAULT_MIN_N_THRESHOLD),
    step=5,
    help="If the filtered sample is smaller than this, the app will warn and treat thresholds as directional."
)

# Apply filters
f = df.copy()
f = f[(f["PURCHASE DATE"] >= start_date) & (f["PURCHASE DATE"] <= end_date)]
if selected_counties:
    f = f[f["County, State"].isin(selected_counties)]
if selected_cities:
    f = f[f["Property Location or City"].isin(selected_cities)]
f = f[(f["days_to_sale"] >= days_range[0]) & (f["days_to_sale"] <= days_range[1])]
f = f[(f["markup_multiple"] >= markup_range[0]) & (f["markup_multiple"] <= markup_range[1])]
f = f[f["markup_multiple"] <= float(markup_cap)]  # hard cap safety

# ----------------------------
# Helper: sweet-spot table
# ----------------------------
def sweet_spot_table(data: pd.DataFrame, target_days: int = 30, n_bins: int = 8) -> pd.DataFrame:
    d = data.dropna(subset=["markup_multiple", "days_to_sale"]).copy()
    if len(d) < 10:
        return pd.DataFrame()

    # If too few rows, reduce bins automatically
    n_bins_eff = int(min(n_bins, max(3, len(d) // 10)))
    try:
        d["mm_bin"] = pd.qcut(d["markup_multiple"], q=n_bins_eff, duplicates="drop")
    except ValueError:
        d["mm_bin"] = pd.cut(d["markup_multiple"], bins=n_bins_eff)

    g = d.groupby("mm_bin", observed=True).agg(
        n=("markup_multiple", "size"),
        median_markup=("markup_multiple", "median"),
        median_days=("days_to_sale", "median"),
        p_sell_within_target=("days_to_sale", lambda x: float(np.mean(x <= target_days))),
        median_net_profit_pct=("net_profit_pct_cost", "median"),
        median_net_profit_dollars=("net_profit_$", "median"),
    ).reset_index()

    g["mm_bin"] = g["mm_bin"].astype(str)
    g = g.sort_values("median_markup")

    # Velocity-first score: probability dominates
    g["score"] = g["p_sell_within_target"] * 0.85 + g["median_net_profit_pct"].fillna(0) * 0.15

    g["p_sell_within_target_pct"] = (g["p_sell_within_target"] * 100).round(1)
    g["median_net_profit_pct_disp"] = (g["median_net_profit_pct"] * 100).round(1)
    g["median_net_profit_dollars"] = g["median_net_profit_dollars"].round(0)

    return g


def best_bin_message(t: pd.DataFrame, label: str, min_n: int) -> None:
    if t.empty:
        st.info("Not enough rows in current filter to compute bins.")
        return

    t_ok = t[t["n"] >= max(3, min_n // 8)].copy()
    if len(t_ok) == 0 or t["n"].sum() < min_n:
        st.warning(
            f"Not enough sample size to confidently declare a “best” {label} sweet spot. "
            f"Try widening filters or aggregating to county/state. "
            f"(Filtered deals: {int(t['n'].sum())})"
        )
        return

    best = t_ok.sort_values(["score", "p_sell_within_target", "median_net_profit_pct"], ascending=False).head(1)

    st.success(
        f"Best {label} bin (min sample enforced): **{best['mm_bin'].iloc[0]}** | "
        f"Median markup **{best['median_markup'].iloc[0]:.2f}x** | "
        f"Sell{label} **{best['p_sell_within_target_pct'].iloc[0]:.1f}%** | "
        f"Median net margin **{best['median_net_profit_pct_disp'].iloc[0]:.1f}%**"
    )


def max_markup_for_probability(
    data: pd.DataFrame,
    target_days: int,
    min_prob: float = 0.70,
    min_n: int = 40,
) -> Dict[str, Any]:
    """
    Estimates the HIGHEST markup multiple bin that still meets P(sell ≤ target_days) ≥ min_prob.
    Uses quantile bins for stability and enforces minimum sample size.
    """
    d = data.dropna(subset=["markup_multiple", "days_to_sale"]).copy()
    if len(d) < min_n:
        return {"ok": False, "reason": f"Not enough rows (n={len(d)})", "n": int(len(d)), "min_n": int(min_n), "target_days": int(target_days), "min_prob": float(min_prob)}

    n_bins = min(10, max(5, len(d) // 15))
    try:
        d["mm_bin"] = pd.qcut(d["markup_multiple"], q=n_bins, duplicates="drop")
    except ValueError:
        d["mm_bin"] = pd.cut(d["markup_multiple"], bins=n_bins)

    g = d.groupby("mm_bin", observed=True).agg(
        n=("markup_multiple", "size"),
        lo=("markup_multiple", "min"),
        hi=("markup_multiple", "max"),
        median_markup=("markup_multiple", "median"),
        p=("days_to_sale", lambda x: float(np.mean(x <= target_days))),
        median_days=("days_to_sale", "median"),
        median_net_profit_pct=("net_profit_pct_cost", "median"),
        median_net_profit_dollars=("net_profit_$", "median"),
    ).reset_index()

    g = g[g["n"] >= max(5, min_n // 10)].sort_values("hi")
    if g.empty:
        return {"ok": False, "reason": "No bins with enough sample size.", "target_days": int(target_days), "min_prob": float(min_prob)}

    ok_bins = g[g["p"] >= min_prob]
    if ok_bins.empty:
        best = g.sort_values(["p", "hi"], ascending=[False, True]).iloc[0].to_dict()
        best.update({
            "ok": False,
            "reason": f"No bin meets P≥{min_prob:.0%}; returning best available.",
            "target_days": int(target_days),
            "min_prob": float(min_prob),
        })
        return _coerce_jsonable(best)

    best = ok_bins.sort_values(["hi", "p"], ascending=[False, False]).iloc[0].to_dict()
    best.update({"ok": True, "target_days": int(target_days), "min_prob": float(min_prob)})
    return _coerce_jsonable(best)


def _coerce_jsonable(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in obj.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (pd.Interval,)):
            out[k] = str(v)
        elif isinstance(v, (pd.Timestamp,)):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


# ----------------------------
# KPI row (velocity-first)
# ----------------------------
col1, col2, col3, col4, col5, col6 = st.columns(6)

n = len(f)
pct_30 = (f["days_to_sale"] <= 30).mean() * 100 if n else 0
pct_60 = (f["days_to_sale"] <= 60).mean() * 100 if n else 0

median_days = float(np.nanmedian(f["days_to_sale"])) if n else np.nan
median_mm = float(np.nanmedian(f["markup_multiple"])) if n else np.nan
cycles_est = (365 / median_days) if (n and pd.notna(median_days) and median_days > 0) else np.nan

col1.metric("Deals (filtered)", f"{n:,}")
col2.metric("Sell ≤30 days", f"{pct_30:.1f}%")
col3.metric("Sell ≤60 days", f"{pct_60:.1f}%")
col4.metric("Median days to sale", f"{median_days:.0f}" if pd.notna(median_days) else "—")
col5.metric("Median markup multiple", f"{median_mm:.2f}" if pd.notna(median_mm) else "—")
col6.metric("Est. cycles/year (median)", f"{cycles_est:.1f}" if pd.notna(cycles_est) else "—")

if n and n < MIN_ROWS_SOFT_WARNING:
    st.warning(
        f"Small sample size in current filters (n={n}). Sweet spots/curves may be unstable. "
        "For client decisions, prefer county/state or larger segments."
    )

st.divider()

# ----------------------------
# LLM + safe plotting helpers
# ----------------------------
SUPPORTED_CHARTS = {"scatter", "line", "bar", "histogram", "box"}

def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None


def render_plot_from_spec(df_: pd.DataFrame, spec: Dict[str, Any]):
    chart_type = (spec.get("type") or "").lower().strip()
    if chart_type not in SUPPORTED_CHARTS:
        st.info(f"Unsupported chart type: {chart_type!r}. Supported: {sorted(SUPPORTED_CHARTS)}")
        return

    x = spec.get("x")
    y = spec.get("y")
    color = spec.get("color")
    facet_row = spec.get("facet_row")
    facet_col = spec.get("facet_col")
    nbins = spec.get("nbins", 40)
    title = spec.get("title") or "AI Chart"
    agg = (spec.get("agg") or "").lower().strip()

    cols = set(df_.columns)
    for col in [x, y, color, facet_row, facet_col]:
        if col is None:
            continue
        if col not in cols:
            st.warning(f"AI requested a column that does not exist: {col!r}")
            return

    d = df_.copy()

    if chart_type == "scatter":
        if not x or not y:
            st.warning("Scatter plot requires x and y.")
            return
        fig = px.scatter(d, x=x, y=y, color=color, facet_row=facet_row, facet_col=facet_col, title=title)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "line":
        if not x or not y:
            st.warning("Line plot requires x and y.")
            return
        fig = px.line(d.sort_values(x), x=x, y=y, color=color, facet_row=facet_row, facet_col=facet_col, title=title)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "bar":
        if x and y and not agg:
            fig = px.bar(d, x=x, y=y, color=color, facet_row=facet_row, facet_col=facet_col, title=title)
            st.plotly_chart(fig, use_container_width=True)
        else:
            if not x:
                st.warning("Aggregated bar requires x.")
                return
            if agg not in {"mean", "median", "sum", "count"}:
                agg = "mean"
            if agg == "count":
                g = d.groupby(x, observed=True).size().reset_index(name="count")
                fig = px.bar(g, x=x, y="count", title=title)
                st.plotly_chart(fig, use_container_width=True)
            else:
                if not y:
                    st.warning("Aggregated bar requires y for mean/median/sum.")
                    return
                fn = {"mean": np.mean, "median": np.median, "sum": np.sum}[agg]
                g = d.dropna(subset=[x, y]).groupby(x, observed=True)[y].apply(fn).reset_index(name=f"{agg}_{y}")
                fig = px.bar(g, x=x, y=f"{agg}_{y}", title=title)
                st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "histogram":
        if not x:
            st.warning("Histogram requires x.")
            return
        fig = px.histogram(d.dropna(subset=[x]), x=x, nbins=int(nbins), color=color, title=title)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "box":
        if not x or not y:
            st.warning("Box plot requires x and y.")
            return
        fig = px.box(d.dropna(subset=[x, y]), x=x, y=y, color=color, title=title)
        st.plotly_chart(fig, use_container_width=True)


def ask_llm_about_stats(user_question: str, df_filtered: pd.DataFrame) -> Tuple[str, Optional[Dict[str, Any]]]:
    client, model = get_openai_client_and_model()

    # Core KPIs for velocity + compounding
    kpis = {
        "n": int(len(df_filtered)),
        "pct_sell_30": float((df_filtered["days_to_sale"] <= 30).mean()) if len(df_filtered) else None,
        "pct_sell_60": float((df_filtered["days_to_sale"] <= 60).mean()) if len(df_filtered) else None,
        "median_days": float(np.nanmedian(df_filtered["days_to_sale"])) if len(df_filtered) else None,
        "median_markup_multiple": float(np.nanmedian(df_filtered["markup_multiple"])) if len(df_filtered) else None,
        "median_net_profit_pct_cost": float(np.nanmedian(df_filtered["net_profit_pct_cost"])) if len(df_filtered) else None,
        "median_net_profit_$": float(np.nanmedian(df_filtered["net_profit_$"])) if len(df_filtered) else None,
    }
    kpis["cycles_per_year_median"] = (365 / kpis["median_days"]) if kpis["median_days"] and kpis["median_days"] > 0 else None

    # Sweet spot tables
    t30 = sweet_spot_table(df_filtered, target_days=30, n_bins=8)
    t60 = sweet_spot_table(df_filtered, target_days=60, n_bins=8)

    # Highest markup that still sells fast with reliability threshold
    max30 = max_markup_for_probability(df_filtered, target_days=30, min_prob=float(reliability_prob), min_n=int(min_n_threshold))
    max60 = max_markup_for_probability(df_filtered, target_days=60, min_prob=float(reliability_prob), min_n=int(min_n_threshold))

    system = (
        "You are an AI Stats analyst for a real-estate cash sales velocity dashboard.\n"
        "The client’s #1 objective is SPEED: sell within 30 days preferred, 60 days acceptable. "
        "Profit is secondary but must remain positive after commissions.\n\n"
        "Commission assumptions are already baked into net profit fields:\n"
        "- 4% of PROFIT to Acquisitions Agent\n"
        "- 4% of PROFIT to Sales Agent\n"
        "- 10% of SALE PRICE to Affiliate/Listing Agent\n\n"
        "Always frame insights in terms of:\n"
        "- Sweet spot markup multiple ranges for ≤30 and ≤60 day cash sales\n"
        "- The HIGHEST markup multiple that still reliably sells fast (use provided thresholds)\n"
        "- Compounding cycles/year (365 / median days-to-sale) and why velocity beats margin\n"
        "- Sample size warnings when n is small or thresholds are not met\n\n"
        "If the user asks for a chart, return a safe plot specification.\n"
        "You MUST output JSON ONLY and NOTHING ELSE.\n\n"
        "Output format:\n"
        "{\"answer\": \"...\", \"plot\": null}\n"
        "OR\n"
        "{\"answer\": \"...\", \"plot\": {\"type\":\"scatter|line|bar|histogram|box\","
        "\"title\":\"...\",\"x\":\"col\",\"y\":\"col_or_null\",\"color\":\"col_or_null\","
        "\"facet_row\":null,\"facet_col\":null,\"agg\":\"mean|median|sum|count\", \"nbins\": 10}}\n\n"
        "Rules:\n"
        "- Use only columns that exist.\n"
        "- Prefer simple, client-friendly explanations.\n"
        "- Never output code. Never output markdown."
    )

    payload = {
        "question": user_question,
        "kpis": kpis,
        "reliability_threshold_prob": float(reliability_prob),
        "min_n_threshold": int(min_n_threshold),
        "highest_markup_threshold_30": max30,
        "highest_markup_threshold_60": max60,
        "sweet_spot_30_table": t30.replace({np.nan: None}).to_dict(orient="records") if not t30.empty else [],
        "sweet_spot_60_table": t60.replace({np.nan: None}).to_dict(orient="records") if not t60.empty else [],
        "columns_available": list(df_filtered.columns),
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)}
        ],
    )

    text = getattr(resp, "output_text", None) or ""
    parsed = extract_json_block(text)

    if not parsed or "answer" not in parsed:
        return (text.strip() or "I couldn't parse the model response. Please re-ask more directly."), None

    answer = str(parsed.get("answer", "")).strip()
    plot = parsed.get("plot", None)
    if isinstance(plot, dict):
        return answer, plot
    return answer, None


# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Sweet Spot (≤30/≤60)", "Velocity & Pricing", "Geo / Property Segments", "Data Table", "Ask AI (LLM)"]
)

# ----------------------------
# Tab 1: Sweet Spot
# ----------------------------
with tab1:
    st.subheader("Markup sweet spot: probability of selling fast")

    left, right = st.columns(2)

    with left:
        st.markdown("### ≤30 days sweet spot")
        t30 = sweet_spot_table(f, target_days=30, n_bins=8)
        if not t30.empty:
            st.dataframe(
                t30[["mm_bin", "n", "median_markup", "median_days",
                     "p_sell_within_target_pct", "median_net_profit_pct_disp",
                     "median_net_profit_dollars", "score"]],
                use_container_width=True
            )
        best_bin_message(t30, "≤30", MIN_ROWS_FOR_BINS)

    with right:
        st.markdown("### ≤60 days sweet spot")
        t60 = sweet_spot_table(f, target_days=60, n_bins=8)
        if not t60.empty:
            st.dataframe(
                t60[["mm_bin", "n", "median_markup", "median_days",
                     "p_sell_within_target_pct", "median_net_profit_pct_disp",
                     "median_net_profit_dollars", "score"]],
                use_container_width=True
            )
        best_bin_message(t60, "≤60", MIN_ROWS_FOR_BINS)

    st.markdown("### Highest markup that still sells fast (reliability threshold)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**≤30 days target (P ≥ {reliability_prob:.0%})**")
        m30 = max_markup_for_probability(f, target_days=30, min_prob=float(reliability_prob), min_n=int(min_n_threshold))
        st.json(m30)
    with c2:
        st.markdown(f"**≤60 days target (P ≥ {reliability_prob:.0%})**")
        m60 = max_markup_for_probability(f, target_days=60, min_prob=float(reliability_prob), min_n=int(min_n_threshold))
        st.json(m60)

    st.markdown("### Probability curve (Sell within X days vs markup multiple)")
    target_days = st.slider("Choose target days", 10, 120, 30, step=5)

    d = f.dropna(subset=["markup_multiple", "days_to_sale"]).copy()
    if len(d) < 10:
        st.info("Not enough rows to plot probability curve.")
    else:
        if len(d) < MIN_ROWS_FOR_CURVE:
            st.warning(
                f"Curve stability warning: only {len(d)} rows in this segment. "
                "Interpret the curve as directional; for client decisions use larger segments."
            )

        d = d.sort_values("markup_multiple")
        window = max(5, int(len(d) * 0.12))
        d["p_sell"] = (d["days_to_sale"] <= target_days).astype(int)
        d["p_sell_smooth"] = d["p_sell"].rolling(window=window, min_periods=max(3, window // 2)).mean()

        fig = px.line(
            d,
            x="markup_multiple",
            y="p_sell_smooth",
            labels={
                "markup_multiple": "Markup multiple (Sale / Total Cost)",
                "p_sell_smooth": f"Smoothed P(sell ≤ {target_days} days)"
            },
            title=f"Probability of Fast Sale vs Markup (rolling window={window})"
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Tab 2: Velocity & Pricing
# ----------------------------
with tab2:
    st.subheader("Velocity drivers and pricing behavior")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            f.dropna(subset=["days_to_sale"]),
            x="days_to_sale",
            nbins=40,
            title="Distribution of Days to Sale"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(
            f.dropna(subset=["markup_multiple"]),
            x="markup_multiple",
            nbins=40,
            title="Distribution of Markup Multiple (Sale / Total Cost)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Markup vs Days to Sale (scatter)")
    d = f.dropna(subset=["markup_multiple", "days_to_sale"]).copy()
    if len(d) >= 3:
        fig = px.scatter(
            d,
            x="markup_multiple",
            y="days_to_sale",
            color="speed_bucket",
            hover_data=[
                "County, State",
                "Property Location or City",
                "Acres",
                "Total Purchase Price",
                "Cash Sales Price - amount",
                "net_profit_$",
                "net_profit_pct_cost",
            ],
            title="Markup Multiple vs Days to Sale (colored by speed bucket)"
        )
        st.plotly_chart(fig, use_container_width=True)

        corr = d[["markup_multiple", "days_to_sale"]].corr().iloc[0, 1]
        st.info(f"Correlation (markup_multiple vs days_to_sale): **{corr:.2f}** (directional only, not causation).")
    else:
        st.info("Not enough rows to build scatter plot.")

    st.markdown("### Profit vs Velocity tradeoff (net)")
    d2 = f.dropna(subset=["net_profit_pct_cost", "days_to_sale"]).copy()
    if len(d2) >= 3:
        fig = px.scatter(
            d2,
            x="net_profit_pct_cost",
            y="days_to_sale",
            color="speed_bucket",
            hover_data=["County, State", "Property Location or City", "Acres", "net_profit_$"],
            labels={"net_profit_pct_cost": "Net Profit % of Cost"},
            title="Net Profit % vs Days to Sale"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough rows to plot profit vs velocity.")

# ----------------------------
# Tab 3: Geo / Segments
# ----------------------------
with tab3:
    st.subheader("Which counties / cities / acreage ranges sell faster?")

    c1, c2 = st.columns(2)

    with c1:
        g = f.dropna(subset=["County, State", "days_to_sale"]).groupby("County, State", observed=True).agg(
            n=("days_to_sale", "size"),
            median_days=("days_to_sale", "median"),
            pct_30=("days_to_sale", lambda x: (x <= 30).mean() * 100),
            pct_60=("days_to_sale", lambda x: (x <= 60).mean() * 100),
            median_markup=("markup_multiple", "median"),
        ).reset_index()

        g = g[g["n"] >= 5].sort_values("median_days")

        fig = px.bar(
            g.head(20),
            x="median_days",
            y="County, State",
            orientation="h",
            hover_data=["n", "pct_30", "pct_60", "median_markup"],
            title="Top 20 Counties by Fastest Median Days (min 5 deals)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        a = f.dropna(subset=["acres_bucket", "days_to_sale"]).groupby("acres_bucket", observed=True).agg(
            n=("days_to_sale", "size"),
            median_days=("days_to_sale", "median"),
            pct_30=("days_to_sale", lambda x: (x <= 30).mean() * 100),
            pct_60=("days_to_sale", lambda x: (x <= 60).mean() * 100),
            median_markup=("markup_multiple", "median"),
        ).reset_index().sort_values("median_days")

        fig = px.bar(
            a,
            x="acres_bucket",
            y="median_days",
            hover_data=["n", "pct_30", "pct_60", "median_markup"],
            title="Median Days by Acres Bucket"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### City leaderboard (fastest median days)")
    city = f.dropna(subset=["Property Location or City", "days_to_sale"]).groupby("Property Location or City", observed=True).agg(
        n=("days_to_sale", "size"),
        median_days=("days_to_sale", "median"),
        pct_30=("days_to_sale", lambda x: (x <= 30).mean() * 100),
        pct_60=("days_to_sale", lambda x: (x <= 60).mean() * 100),
        median_markup=("markup_multiple", "median"),
    ).reset_index()

    city = city[city["n"] >= 5].sort_values("median_days")
    st.dataframe(city.head(30), use_container_width=True)

# ----------------------------
# Tab 4: Data table + export
# ----------------------------
with tab4:
    st.subheader("Filtered dataset")

    show_cols = [
        "County, State",
        "Property Location or City",
        "Acres",
        "Total Purchase Price",
        "Cash Sales Price - amount",
        "markup_multiple",
        "days_to_sale",
        "profit_$",
        "net_profit_$",
        "net_profit_pct_cost",
        "PURCHASE DATE",
        "SALE DATE - start",
    ]

    # include optional raw components if present
    for extra in ["Purchase Price", "Title Escrow Costs", "Taxes we Paid"]:
        if extra in f.columns:
            show_cols.insert(show_cols.index("Total Purchase Price"), extra)

    st.dataframe(
        f[show_cols].sort_values("SALE DATE - start", ascending=False),
        use_container_width=True
    )

    csv = f[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        data=csv,
        file_name="filtered_cash_sales_velocity.csv",
        mime="text/csv"
    )

# ----------------------------
# Tab 5: Ask AI (LLM)
# ----------------------------
with tab5:
    st.subheader("Ask AI about velocity + sweet spots (filtered data)")
    st.caption(
        "Ask questions like:\n"
        "• “What is the highest markup multiple we can use and still sell within 30 days reliably?”\n"
        "• “What’s the sweet spot markup range for ≤60 days?”\n"
        "• “Plot markup_multiple vs days_to_sale colored by speed_bucket.”\n"
        "Charts are generated safely from a plot spec (no code execution)."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Ask me about the current filtered deals. If you want a chart, say “plot …” or “chart …”."}
        ]

    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    prompt = st.chat_input("Ask a question (add 'plot' if you want a chart)")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing AI Stats..."):
                answer, plot_spec = ask_llm_about_stats(prompt, f)

            st.write(answer)
            if plot_spec:
                st.markdown("**AI chart**")
                render_plot_from_spec(f, plot_spec)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Chat cleared. Ask a new question about the current filters."}
        ]
        st.rerun()

st.caption(
    "Notes: (1) Markup multiples are capped to reduce outlier distortion. "
    "(2) “Best bin” and “highest markup” enforce minimum sample sizes to avoid misleading conclusions. "
    "(3) This dashboard is descriptive; a predictive model can be added next."
)
