
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SKLViz Revenue Dashboard", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def format_inr(x: float) -> str:
    if pd.isna(x) or x is None:
        return "₹0"
    return f"₹{float(x):,.0f}"

def format_int(x: float) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "0"

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def clean_month(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\s+2025|\s+25|\s+2024|\s+2023", "", regex=True)
    return s.str.title()

def auto_parse_dates(d: pd.DataFrame, min_parse_rate: float = 0.60) -> list:
    parsed_cols = []
    for c in d.columns:
        if d[c].dtype == "object":
            name_hint = any(k in c.lower() for k in ["date", "time", "created", "updated", "modified", "dt"])
            if not name_hint:
                continue
            dt = pd.to_datetime(d[c], errors="coerce")
            rate = dt.notna().mean()
            if rate >= min_parse_rate:
                d[c] = dt
                parsed_cols.append(c)
    return parsed_cols

def pick_best_date_col(d: pd.DataFrame):
    dt_cols = d.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    if not dt_cols:
        return None
    priority = [
        "payment_received_date", "payment date", "payment_received",
        "enrollment_date", "enrolled_date",
        "converted_date", "won_date",
        "lead_created_date", "created date", "create date",
        "lead_date", "date"
    ]
    dt_lower_map = {c.lower(): c for c in dt_cols}
    for p in priority:
        for lc, orig in dt_lower_map.items():
            if p in lc:
                return orig
    return dt_cols[0]

def detect_enrollment_mode(d: pd.DataFrame):
    cols = {c.lower(): c for c in d.columns}

    for candidate in ["total_enrollments", "enrollments", "enrollment_count", "enrollment"]:
        if candidate in cols:
            col = cols[candidate]
            def f(x):
                return pd.to_numeric(x[col], errors="coerce").fillna(0)
            return ("sum:" + col, f, f"Enrollments = SUM({col}).")

    if "converted" in cols:
        col = cols["converted"]
        def f(x):
            v = (
                x[col].astype(str).str.strip().str.lower()
                .map({"1":1,"true":1,"yes":1,"y":1,"converted":1})
                .fillna(0).astype(int)
            )
            return v
        return ("count_where:" + col, f, "Enrollments = COUNT(Converted = 1).")

    for lc, orig in cols.items():
        if "payment" in lc and "date" in lc:
            col = orig
            def f(x):
                return x[col].notna().astype(int)
            return ("count_nonnull:" + col, f, f"Enrollments = COUNT(non-null {col}).")

    def f(x):
        return pd.Series(np.ones(len(x), dtype=int), index=x.index)
    return ("row_count", f, "Enrollments = row count (fallback).")

def add_period_col(t: pd.DataFrame, date_col: str, granularity: str) -> pd.Series:
    if granularity == "Day":
        return t[date_col].dt.date.astype(str)
    if granularity == "Week":
        return t[date_col].dt.to_period("W-MON").apply(lambda p: str(p.start_time.date()))
    if granularity == "Month":
        return t[date_col].dt.to_period("M").astype(str)
    return t[date_col].dt.date.astype(str)

def aggregate_sum_trend(d: pd.DataFrame, date_col: str, values: pd.Series, granularity: str) -> pd.DataFrame:
    t = d.dropna(subset=[date_col]).copy()
    if t.empty:
        return pd.DataFrame()
    t[date_col] = pd.to_datetime(t[date_col], errors="coerce")
    t = t.dropna(subset=[date_col])
    if t.empty:
        return pd.DataFrame()

    v = values.loc[t.index].copy()
    t["period"] = add_period_col(t, date_col, granularity)

    out = (
        pd.DataFrame({"period": t["period"], "value": v})
        .groupby("period")["value"].sum()
        .reset_index()
        .sort_values("period")
        .set_index("period")
    )
    return out

def aggregate_aov_trend(d: pd.DataFrame, date_col: str, revenue: pd.Series, enroll: pd.Series, granularity: str) -> pd.DataFrame:
    t = d.dropna(subset=[date_col]).copy()
    if t.empty:
        return pd.DataFrame()
    t[date_col] = pd.to_datetime(t[date_col], errors="coerce")
    t = t.dropna(subset=[date_col])
    if t.empty:
        return pd.DataFrame()

    r = revenue.loc[t.index].copy()
    e = enroll.loc[t.index].copy()
    t["period"] = add_period_col(t, date_col, granularity)

    agg = (
        pd.DataFrame({"period": t["period"], "revenue": r, "enroll": e})
        .groupby("period")[["revenue", "enroll"]].sum()
        .reset_index()
        .sort_values("period")
    )
    agg["AOV"] = np.where(agg["enroll"] > 0, agg["revenue"] / agg["enroll"], 0.0)
    return agg.set_index("period")[["AOV"]]

def rep_metric_table(df_base: pd.DataFrame, rep_col: str, revenue_series: pd.Series, enroll_series: pd.Series) -> pd.DataFrame:
    t = df_base.copy()
    t["_rev"] = revenue_series
    t["_enr"] = enroll_series
    g = t.groupby(rep_col)[["_rev","_enr"]].sum().reset_index()
    g["AOV"] = np.where(g["_enr"] > 0, g["_rev"] / g["_enr"], 0.0)
    g = g.rename(columns={rep_col:"Sales_Rep","_rev":"Revenue","_enr":"Enrollments"})
    return g

def pick_reps_by_bucket(rep_df: pd.DataFrame, metric: str, bucket: str) -> list:
    rep_df = rep_df.copy()
    rep_df = rep_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
    rep_df = rep_df.sort_values(metric, ascending=False).reset_index(drop=True)

    n = len(rep_df)
    if n == 0:
        return []

    if bucket == "All":
        return rep_df["Sales_Rep"].astype(str).tolist()

    if bucket == "Top 20%":
        k = max(1, int(np.ceil(0.20 * n)))
        return rep_df.head(k)["Sales_Rep"].astype(str).tolist()

    q = int(np.ceil(n / 4))
    top = rep_df.iloc[0:q]
    q2  = rep_df.iloc[q:2*q]
    q3  = rep_df.iloc[2*q:3*q]
    bot = rep_df.iloc[3*q:n]

    if bucket == "Top quartile (Q1)":
        return top["Sales_Rep"].astype(str).tolist()
    if bucket == "2nd quartile (Q2)":
        return q2["Sales_Rep"].astype(str).tolist()
    if bucket == "3rd quartile (Q3)":
        return q3["Sales_Rep"].astype(str).tolist()
    if bucket == "Bottom quartile (Q4)":
        return bot["Sales_Rep"].astype(str).tolist()

    return rep_df["Sales_Rep"].astype(str).tolist()

# ---------------------------
# App
# ---------------------------
st.title("SKLViz – Sales Review Dashboard")

df = load_data("sv_data.csv")

required_cols = {"Month", "Total_Value_INR"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

df["Month"] = clean_month(df["Month"])
df["Total_Value_INR"] = pd.to_numeric(df["Total_Value_INR"], errors="coerce").fillna(0)

_ = auto_parse_dates(df)
_, enroll_series_fn, enroll_note = detect_enrollment_mode(df)

# Sidebar
with st.sidebar:
    st.header("Controls")
    with st.expander("Filters", expanded=True):
        view_metric = st.radio("View Metric", ["Total Revenue", "Total Enrollments", "AOV"])
        month_choice = st.radio("Select Period", ["July", "August", "September", "Consolidated (Q3 / 30-day)"])
        show_raw = st.checkbox("Show filtered raw data preview", value=False)

# Mode pill (top-right)
top_left, top_right = st.columns([3, 1.2])
with top_right:
    mode = st.radio("Mode", options=["Overall", "Sales Rep"], horizontal=True)

# Period filter
if month_choice.startswith("Consolidated"):
    df_base = df[df["Month"].isin(["July", "August", "September"])].copy()
    selected_period_label = "Q3 (Jul–Sep) Consolidated"
else:
    df_base = df[df["Month"] == month_choice].copy()
    selected_period_label = month_choice

# Base series (for rep ranking)
base_revenue = df_base["Total_Value_INR"].copy()
base_enroll  = enroll_series_fn(df_base)

df_f = df_base.copy()
rep_note = ""
bucket_selected_table = None

if mode == "Sales Rep":
    if "Sales_Rep" not in df_base.columns:
        st.warning("Sales Rep mode selected, but column 'Sales_Rep' is not present in your CSV.")
    else:
        rep_df = rep_metric_table(df_base, "Sales_Rep", base_revenue, base_enroll)

        rank_metric = "Revenue" if view_metric == "Total Revenue" else ("Enrollments" if view_metric == "Total Enrollments" else "AOV")

        with top_left:
            st.caption("Sales Rep mode is ON — selection uses your left filters.")
            sel_type = st.radio("Sales Rep Selection", ["Selected reps", "Bucket (Top/Quartiles)"], horizontal=True)

        if sel_type == "Selected reps":
            reps = rep_df["Sales_Rep"].astype(str).dropna().sort_values().unique().tolist()
            chosen = st.multiselect("Pick Sales Rep(s)", options=reps, default=[])
            if chosen:
                df_f = df_base[df_base["Sales_Rep"].astype(str).isin(chosen)].copy()
                rep_note = f"Filtered to selected reps: {', '.join(chosen[:5])}" + (" ..." if len(chosen) > 5 else "")
                bucket_selected_table = rep_df[rep_df["Sales_Rep"].astype(str).isin(chosen)].copy()
            else:
                rep_note = "No reps selected — showing all reps (Sales Rep mode)."
                bucket_selected_table = rep_df.copy()
        else:
            bucket = st.selectbox(
                f"Pick bucket based on {rank_metric}",
                options=["All", "Top 20%", "Top quartile (Q1)", "2nd quartile (Q2)", "3rd quartile (Q3)", "Bottom quartile (Q4)"],
                index=1
            )
            reps_in_bucket = pick_reps_by_bucket(rep_df, rank_metric, bucket)
            if reps_in_bucket:
                df_f = df_base[df_base["Sales_Rep"].astype(str).isin(reps_in_bucket)].copy()
                rep_note = f"Bucket: {bucket} (based on {rank_metric}) | Reps: {len(reps_in_bucket)}"
                bucket_selected_table = rep_df[rep_df["Sales_Rep"].astype(str).isin(reps_in_bucket)].copy().sort_values(rank_metric, ascending=False)
            else:
                rep_note = f"Bucket: {bucket} returned 0 reps. Showing all reps."
                df_f = df_base.copy()
                bucket_selected_table = rep_df.copy()

# Compute on filtered df_f
revenue_series = df_f["Total_Value_INR"].copy()
enroll_series  = enroll_series_fn(df_f)

total_revenue = float(revenue_series.sum())
total_enrollments = float(enroll_series.sum())
aov_total = (total_revenue / total_enrollments) if total_enrollments > 0 else 0.0

if view_metric == "Total Revenue":
    primary_value_fmt = format_inr(total_revenue)
    primary_note = "Revenue = SUM(Total_Value_INR)."
elif view_metric == "Total Enrollments":
    primary_value_fmt = format_int(total_enrollments)
    primary_note = enroll_note
else:
    primary_value_fmt = format_inr(aov_total)
    primary_note = "AOV = Total Revenue / Total Enrollments."

tabs = st.tabs(["Review"])

with tabs[0]:
    if mode == "Sales Rep":
        st.info(rep_note)

        if bucket_selected_table is not None:
            st.subheader("Sales Rep List (for selected reps / bucket)")
            show = bucket_selected_table.copy()
            show["Revenue"] = show["Revenue"].apply(format_inr)
            show["Enrollments"] = show["Enrollments"].astype(int)
            show["AOV"] = show["AOV"].apply(format_inr)
            st.dataframe(show[["Sales_Rep","Revenue","Enrollments","AOV"]], use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected Period", selected_period_label)
    c2.metric(view_metric, primary_value_fmt)
    c3.metric("Total Enrollments", format_int(total_enrollments))
    c4.metric("AOV", format_inr(aov_total))
    st.caption(primary_note)

    st.divider()

    st.subheader(f"{view_metric} Trend")
    date_col = pick_best_date_col(df_f)

    if not date_col:
        st.info("No usable date/time column found for trend plotting.")
    else:
        granularity = st.selectbox("Trend Granularity", ["Day", "Week", "Month"])

        if view_metric == "AOV":
            trend_tbl = aggregate_aov_trend(df_f, date_col, revenue_series, enroll_series, granularity)
            if not trend_tbl.empty:
                st.line_chart(trend_tbl.rename(columns={"AOV": "AOV (INR)"}))
        else:
            metric_series = revenue_series if view_metric == "Total Revenue" else enroll_series
            label = "Revenue (INR)" if view_metric == "Total Revenue" else "Enrollments"
            trend_tbl = aggregate_sum_trend(df_f, date_col, metric_series, granularity)
            if not trend_tbl.empty:
                st.line_chart(trend_tbl.rename(columns={"value": label}))

    st.divider()

    if show_raw:
        st.subheader("Filtered Data Preview")
        st.dataframe(df_f.head(200), use_container_width=True)

    st.download_button(
        "Download filtered rows as CSV",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="filtered_sv_data.csv",
        mime="text/csv",
    )
