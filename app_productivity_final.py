
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

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

def group_metric_table(df_base: pd.DataFrame, group_col: str, revenue_series: pd.Series, enroll_series: pd.Series) -> pd.DataFrame:
    t = df_base.copy()
    t["_rev"] = revenue_series
    t["_enr"] = enroll_series
    g = t.groupby(group_col)[["_rev","_enr"]].sum().reset_index()
    g["AOV"] = np.where(g["_enr"] > 0, g["_rev"] / g["_enr"], 0.0)
    g = g.rename(columns={group_col:"Group","_rev":"Revenue","_enr":"Enrollments"})
    return g

def manager_metric_table_with_team(df_base: pd.DataFrame, manager_col: str, revenue_series: pd.Series, enroll_series: pd.Series) -> pd.DataFrame:
    g = group_metric_table(df_base, manager_col, revenue_series, enroll_series)

    if "Sales_Rep" in df_base.columns:
        team = (
            df_base.groupby(manager_col)["Sales_Rep"]
            .nunique(dropna=True)
            .reset_index()
            .rename(columns={"Sales_Rep":"Team_Size"})
        )
        team = team.rename(columns={manager_col: "Group"})
        g = g.merge(team, on="Group", how="left")
    else:
        g["Team_Size"] = np.nan

    g["Revenue_per_Rep"] = np.where(g["Team_Size"] > 0, g["Revenue"] / g["Team_Size"], np.nan)
    g["Enrollments_per_Rep"] = np.where(g["Team_Size"] > 0, g["Enrollments"] / g["Team_Size"], np.nan)
    return g

def pick_groups_by_bucket(group_df: pd.DataFrame, metric: str, bucket: str) -> list:
    group_df = group_df.copy()
    group_df = group_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
    group_df = group_df.sort_values(metric, ascending=False).reset_index(drop=True)

    n = len(group_df)
    if n == 0:
        return []

    if bucket == "All":
        return group_df["Group"].astype(str).tolist()

    if bucket == "Top 20%":
        k = max(1, int(np.ceil(0.20 * n)))
        return group_df.head(k)["Group"].astype(str).tolist()

    q = int(np.ceil(n / 4))
    top = group_df.iloc[0:q]
    q2  = group_df.iloc[q:2*q]
    q3  = group_df.iloc[2*q:3*q]
    bot = group_df.iloc[3*q:n]

    if bucket == "Top quartile (Q1)":
        return top["Group"].astype(str).tolist()
    if bucket == "2nd quartile (Q2)":
        return q2["Group"].astype(str).tolist()
    if bucket == "3rd quartile (Q3)":
        return q3["Group"].astype(str).tolist()
    if bucket == "Bottom quartile (Q4)":
        return bot["Group"].astype(str).tolist()

    return group_df["Group"].astype(str).tolist()

def build_sales_rep_clusters(rep_table: pd.DataFrame) -> pd.DataFrame:
    '''
    4 'ideal clusters' using a simple, explainable quadrant split:
    - High vs Low Revenue (median split)
    - High vs Low Enrollments (median split)

    Cluster labels:
    1) Champions: High Revenue + High Enrollments
    2) Value-Heavy: High Revenue + Low Enrollments
    3) Volume-Heavy: Low Revenue + High Enrollments
    4) Needs Focus: Low Revenue + Low Enrollments
    '''
    t = rep_table.copy()
    t = t.replace([np.inf, -np.inf], np.nan).dropna(subset=["Revenue","Enrollments"]).copy()
    if t.empty:
        return t

    rev_med = float(t["Revenue"].median())
    enr_med = float(t["Enrollments"].median())

    t["rev_band"] = np.where(t["Revenue"] >= rev_med, "High Revenue", "Low Revenue")
    t["enr_band"] = np.where(t["Enrollments"] >= enr_med, "High Enrollments", "Low Enrollments")

    def label(row):
        if row["rev_band"] == "High Revenue" and row["enr_band"] == "High Enrollments":
            return "Champions (High Rev, High Enr)"
        if row["rev_band"] == "High Revenue" and row["enr_band"] == "Low Enrollments":
            return "Value-Heavy (High Rev, Low Enr)"
        if row["rev_band"] == "Low Revenue" and row["enr_band"] == "High Enrollments":
            return "Volume-Heavy (Low Rev, High Enr)"
        return "Needs Focus (Low Rev, Low Enr)"

    t["Cluster"] = t.apply(label, axis=1)
    t["Revenue_median"] = rev_med
    t["Enrollments_median"] = enr_med
    return t



def detect_inquiry_id_col(d: pd.DataFrame):
    cols = {c.lower(): c for c in d.columns}
    candidates = [
        "lead_id", "leadid", "lead id",
        "prospect_id", "prospectid", "prospect id",
        "inquiry_id", "inquiryid", "inquiry id",
        "enquiry_id", "enquiryid", "enquiry id",
        "opportunity_id", "opportunity id",
        "leadsquared", "application_id", "application id"
    ]
    for cand in candidates:
        for lc, orig in cols.items():
            if cand in lc:
                return orig
    return None

def safe_qcut(series: pd.Series, q: int = 4):
    s = series.copy()
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0)
    try:
        return pd.qcut(s.rank(method="first"), q, labels=[f"Q{i}" for i in range(1, q+1)])
    except Exception:
        return pd.cut(s.rank(method="first"), bins=q, labels=[f"Q{i}" for i in range(1, q+1)])

def build_sales_rep_productivity(df_filtered: pd.DataFrame, master_df: pd.DataFrame, activity_date_col: str):
    if "Sales_Rep" not in df_filtered.columns:
        return None, "Column 'Sales_Rep' not found in sv_data.csv."

    if activity_date_col is None or activity_date_col not in df_filtered.columns:
        return None, "No usable activity date column found for active-days calculation."

    t = df_filtered.copy()
    t["Sales_Rep"] = t["Sales_Rep"].astype(str)

    dt = pd.to_datetime(t[activity_date_col], errors="coerce")
    t = t.loc[dt.notna()].copy()
    t["_activity_day"] = dt.loc[dt.notna()].dt.date.astype(str)

    inquiry_id_col = detect_inquiry_id_col(t)

    grp = t.groupby("Sales_Rep")
    if inquiry_id_col:
        prod = pd.DataFrame({
            "Sales_Rep": grp.size().index.astype(str),
            "Active_Days": grp["_activity_day"].nunique().values,
            "Total_Inquiries": grp[inquiry_id_col].nunique(dropna=True).values,
        })
    else:
        prod = pd.DataFrame({
            "Sales_Rep": grp.size().index.astype(str),
            "Active_Days": grp["_activity_day"].nunique().values,
            "Total_Inquiries": grp.size().values,
        })

    m = master_df.copy()
    m["Sales_Rep"] = m["Sales_Rep"].astype(str)
    m["Date_of_Joining"] = pd.to_datetime(m["Date_of_Joining"], errors="coerce")
    m["Tenure_Months"] = pd.to_numeric(m["Tenure_Months"], errors="coerce")

    out = prod.merge(m, on="Sales_Rep", how="left")

    out["Inquiries_per_Active_Day"] = np.where(out["Active_Days"] > 0, out["Total_Inquiries"] / out["Active_Days"], 0.0)
    out["Inquiries_per_Tenure_Month"] = np.where(out["Tenure_Months"] > 0, out["Total_Inquiries"] / out["Tenure_Months"], np.nan)

    out["Productivity_Index"] = (0.6 * out["Inquiries_per_Active_Day"]) + (0.4 * out["Inquiries_per_Tenure_Month"].fillna(0))

    qlabels = safe_qcut(out["Productivity_Index"], q=4).astype(str)
    out["Productivity_Quartile"] = qlabels
    out["Productivity_Bucket"] = out["Productivity_Quartile"].map({
        "Q4": "Top quartile",
        "Q3": "2nd quartile",
        "Q2": "3rd quartile",
        "Q1": "Bottom quartile",
    }).fillna("NA")

    out = out.sort_values("Productivity_Index", ascending=False).reset_index(drop=True)

    note = f"Inquiries counted by {'unique ' + inquiry_id_col if inquiry_id_col else 'row count (no inquiry id column found)'}; Active days from '{activity_date_col}'."
    return out, note

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

# MASTER MAPPING (Reporting_Manager ↔ Sales_Rep ↔ DOJ ↔ Tenure)
MASTER_TSV = """Reporting_Manager    Sales_Rep    Date_of_Joining    Tenure_Months
TL_Amit    Agent_A1    2024-01-15    23
TL_Amit    Agent_A2    2024-03-01    21
TL_Amit    Agent_A3    2024-02-01    22
TL_Amit    Agent_A4    2024-02-15    22
TL_Amit    Agent_A5    2024-04-01    20
TL_Priya    Agent_P1    2023-06-01    31
TL_Priya    Agent_P2    2023-09-01    28
TL_Priya    Agent_P3    2023-11-01    26
TL_Priya    Agent_P4    2024-01-01    24
TL_Priya    Agent_P5    2024-02-01    23
TL_Priya    Agent_P6    2024-03-15    21
TL_Priya    Agent_P7    2024-05-01    19
TL_Priya    Agent_P8    2024-06-01    18
TL_Priya    Agent_P9    2024-09-01    15
TL_Rahul    Agent_R1    2024-01-01    24
TL_Rahul    Agent_R2    2024-02-01    23
TL_Rahul    Agent_R3    2024-03-01    22
TL_Rahul    Agent_R4    2024-07-01    17
TL_Rahul    Agent_R5    2024-04-01    20
TL_Rahul    Agent_R6    2024-05-01    19
TL_Rahul    Agent_R7    2024-02-15    22
TL_Rahul    Agent_R8    2024-08-01    16
TL_Sneha    Agent_S1    2024-04-01    20
TL_Sneha    Agent_S2    2024-06-01    18
TL_Sneha    Agent_S3    2024-07-01    17
TL_Sneha    Agent_S4    2024-05-01    19
TL_Sneha    Agent_S5    2024-06-15    18
TL_Sneha    Agent_S6    2024-03-01    21
TL_Sneha    Agent_S7    2024-09-01    15
TL_Sneha    Agent_S8    2024-07-15    17
"""
master_map = pd.read_csv(StringIO(MASTER_TSV), sep=r"\t")
master_map["Date_of_Joining"] = pd.to_datetime(master_map["Date_of_Joining"], errors="coerce")
master_map["Tenure_Months"] = pd.to_numeric(master_map["Tenure_Months"], errors="coerce")


# Sidebar (left)
with st.sidebar:
    st.header("Controls")
    with st.expander("Filters", expanded=True):
        view_metric = st.radio("View Metric", ["Total Revenue", "Total Enrollments", "AOV"])
        month_choice = st.radio("Select Period", ["July", "August", "September", "Consolidated (Q3 / 30-day)"])
        show_raw = st.checkbox("Show filtered raw data preview", value=False)

# Mode pill (top-right)
top_left, top_right = st.columns([3, 1.9])
with top_right:
    mode = st.radio("Mode", options=["Overall", "Sales Rep", "Reporting Manager"], horizontal=True)

# Period filter
if month_choice.startswith("Consolidated"):
    df_base = df[df["Month"].isin(["July", "August", "September"])].copy()
    selected_period_label = "Q3 (Jul–Sep) Consolidated"
else:
    df_base = df[df["Month"] == month_choice].copy()
    selected_period_label = month_choice

# Base series (for ranking)
base_revenue = df_base["Total_Value_INR"].copy()
base_enroll  = enroll_series_fn(df_base)

df_f = df_base.copy()
mode_note = ""
selected_table = None
rank_metric = "Revenue"

def metric_name_for_rank(vm: str) -> str:
    if vm == "Total Revenue":
        return "Revenue"
    if vm == "Total Enrollments":
        return "Enrollments"
    return "AOV"

# Mode filters
if mode in ["Sales Rep", "Reporting Manager"]:
    group_col = "Sales_Rep" if mode == "Sales Rep" else "Reporting_Manager"
    if group_col not in df_base.columns:
        st.warning(f"{mode} mode selected, but column '{group_col}' is not present in your CSV.")
    else:
        rank_metric = metric_name_for_rank(view_metric)

        if mode == "Reporting Manager":
            group_df = manager_metric_table_with_team(df_base, group_col, base_revenue, base_enroll)
        else:
            group_df = group_metric_table(df_base, group_col, base_revenue, base_enroll)

        with top_left:
            st.caption(f"{mode} mode is ON — selection uses your left filters.")
            sel_type = st.radio(f"{mode} Selection", ["Selected", "Bucket (Top/Quartiles)"], horizontal=True)

        if sel_type == "Selected":
            options = group_df["Group"].astype(str).dropna().sort_values().unique().tolist()
            chosen = st.multiselect(f"Pick {mode}(s)", options=options, default=[])

            if chosen:
                df_f = df_base[df_base[group_col].astype(str).isin(chosen)].copy()
                mode_note = f"Filtered to selected {mode}(s): {', '.join(chosen[:5])}" + (" ..." if len(chosen) > 5 else "")
                selected_table = group_df[group_df["Group"].astype(str).isin(chosen)].copy()
            else:
                mode_note = f"No {mode} selected — showing all."
                selected_table = group_df.copy()
        else:
            bucket = st.selectbox(
                f"Pick bucket based on {rank_metric}",
                options=["All", "Top 20%", "Top quartile (Q1)", "2nd quartile (Q2)", "3rd quartile (Q3)", "Bottom quartile (Q4)"],
                index=1
            )
            groups_in_bucket = pick_groups_by_bucket(group_df, rank_metric, bucket)
            if groups_in_bucket:
                df_f = df_base[df_base[group_col].astype(str).isin(groups_in_bucket)].copy()
                mode_note = f"Bucket: {bucket} (based on {rank_metric}) | {mode}s: {len(groups_in_bucket)}"
                selected_table = group_df[group_df["Group"].astype(str).isin(groups_in_bucket)].copy().sort_values(rank_metric, ascending=False)
            else:
                mode_note = f"Bucket: {bucket} returned 0. Showing all."
                df_f = df_base.copy()
                selected_table = group_df.copy()

# Compute on df_f
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
    if mode != "Overall":
        st.info(mode_note)

        if selected_table is not None:
            label = "Sales Rep" if mode == "Sales Rep" else "Reporting Manager"
            st.subheader(f"{label} List (selected / bucket)")

            show = selected_table.copy()
            show["Revenue_fmt"] = show["Revenue"].apply(format_inr)
            show["Enrollments_fmt"] = show["Enrollments"].astype(int)
            show["AOV_fmt"] = show["AOV"].apply(format_inr)

            cols = ["Group", "Revenue_fmt", "Enrollments_fmt", "AOV_fmt"]
            rename_map = {"Group": label, "Revenue_fmt": "Revenue", "Enrollments_fmt": "Enrollments", "AOV_fmt": "AOV"}

            if mode == "Reporting Manager":
                show["Team_Size_fmt"] = show["Team_Size"].fillna(0).astype(int)
                show["Revenue_per_Rep_fmt"] = show["Revenue_per_Rep"].apply(lambda v: format_inr(v) if pd.notna(v) else "NA")
                show["Enrollments_per_Rep_fmt"] = show["Enrollments_per_Rep"].apply(lambda v: f"{float(v):.2f}" if pd.notna(v) else "NA")
                cols = ["Group", "Team_Size_fmt", "Revenue_fmt", "Enrollments_fmt", "AOV_fmt", "Revenue_per_Rep_fmt", "Enrollments_per_Rep_fmt"]
                rename_map.update({
                    "Team_Size_fmt": "# Sales Reps",
                    "Revenue_per_Rep_fmt": "Revenue / Rep",
                    "Enrollments_per_Rep_fmt": "Enrollments / Rep"
                })

            st.dataframe(show[cols].rename(columns=rename_map), use_container_width=True)

            # Sales Rep clustering (only in Sales Rep mode)
            if mode == "Sales Rep":
                st.subheader("Sales Rep Clusters (Revenue × Enrollments)")
                st.caption("4 clusters using median Revenue and median Enrollments. Colors show cluster membership. Dashed lines are medians.")

                rep_tbl = selected_table.rename(columns={"Group":"Sales_Rep"}).copy()
                clustered = build_sales_rep_clusters(rep_tbl)

                if clustered.empty:
                    st.info("Not enough Sales Rep data to create clusters.")
                else:
                    rev_med = float(clustered["Revenue_median"].iloc[0])
                    enr_med = float(clustered["Enrollments_median"].iloc[0])

                    st.vega_lite_chart(
                        clustered,
                        {
                            "width": "container",
                            "height": 420,
                            "layer": [
                                {
                                    "mark": {"type": "point", "filled": True, "size": 90},
                                    "encoding": {
                                        "x": {"field": "Enrollments", "type": "quantitative", "axis": {"title": "Enrollments"}},
                                        "y": {"field": "Revenue", "type": "quantitative", "axis": {"title": "Revenue (INR)"}},
                                        "color": {"field": "Cluster", "type": "nominal", "legend": {"title": "Cluster"}},
                                        "tooltip": [
                                            {"field": "Sales_Rep", "type": "nominal", "title": "Sales Rep"},
                                            {"field": "Cluster", "type": "nominal"},
                                            {"field": "Revenue", "type": "quantitative"},
                                            {"field": "Enrollments", "type": "quantitative"},
                                            {"field": "AOV", "type": "quantitative"}
                                        ]
                                    }
                                },
                                {"mark": {"type": "rule", "strokeDash": [4, 4], "opacity": 0.6}, "encoding": {"x": {"datum": enr_med}}},
                                {"mark": {"type": "rule", "strokeDash": [4, 4], "opacity": 0.6}, "encoding": {"y": {"datum": rev_med}}},
                            ]
                        },
                        use_container_width=True
                    )

                    out = clustered[["Sales_Rep","Cluster","Revenue","Enrollments","AOV"]].copy()
                    out["Revenue"] = out["Revenue"].apply(format_inr)
                    out["Enrollments"] = out["Enrollments"].astype(int)
                    out["AOV"] = out["AOV"].apply(format_inr)
                    st.dataframe(out.sort_values(["Cluster","Sales_Rep"]), use_container_width=True)

# ---------------------------
                # Sales Rep Productivity (DOJ/Tenure × Active Days × Inquiries)
                # ---------------------------
                st.subheader("Sales Rep Productivity (DOJ/Tenure × Active Days × Inquiries)")
                st.caption("Activity = unique days with records; Inquiries = unique Lead/Inquiry ID if present, else row count.")

                activity_date_col = pick_best_date_col(df_f)
                prod_df, prod_note = build_sales_rep_productivity(df_f, master_map, activity_date_col)

                if prod_df is None or prod_df.empty:
                    st.info(prod_note if prod_note else "Not enough data to compute productivity.")
                else:
                    st.caption(prod_note)

                    q_choice = st.selectbox(
                        "Productivity Quartile Filter",
                        options=["All", "Top quartile", "2nd quartile", "3rd quartile", "Bottom quartile"],
                        index=0
                    )
                    view_df = prod_df.copy()
                    if q_choice != "All":
                        view_df = view_df[view_df["Productivity_Bucket"] == q_choice].copy()

                    showp = view_df.copy()
                    showp["Date_of_Joining"] = showp["Date_of_Joining"].dt.date.astype(str)
                    showp["Tenure_Months"] = showp["Tenure_Months"].fillna(0).astype(int)
                    showp["Active_Days"] = showp["Active_Days"].fillna(0).astype(int)
                    showp["Total_Inquiries"] = showp["Total_Inquiries"].fillna(0).astype(int)
                    showp["Inquiries_per_Active_Day"] = showp["Inquiries_per_Active_Day"].round(2)
                    showp["Inquiries_per_Tenure_Month"] = showp["Inquiries_per_Tenure_Month"].round(2)
                    showp["Productivity_Index"] = showp["Productivity_Index"].round(2)

                    st.dataframe(
                        showp[[
                            "Sales_Rep","Reporting_Manager","Date_of_Joining","Tenure_Months",
                            "Active_Days","Total_Inquiries","Inquiries_per_Active_Day","Inquiries_per_Tenure_Month",
                            "Productivity_Index","Productivity_Bucket"
                        ]],
                        use_container_width=True,
                        height=420
                    )

                    chartp = view_df.copy()
                    chartp["Tenure_Months"] = pd.to_numeric(chartp["Tenure_Months"], errors="coerce").fillna(0)
                    st.vega_lite_chart(
                        chartp,
                        {
                            "width": "container",
                            "height": 420,
                            "mark": {"type": "circle", "opacity": 0.85},
                            "encoding": {
                                "x": {"field": "Tenure_Months", "type": "quantitative", "axis": {"title": "Tenure (Months)"}},
                                "y": {"field": "Inquiries_per_Active_Day", "type": "quantitative", "axis": {"title": "Inquiries / Active Day"}},
                                "size": {"field": "Total_Inquiries", "type": "quantitative", "legend": {"title": "Total Inquiries"}},
                                "color": {"field": "Productivity_Bucket", "type": "nominal", "legend": {"title": "Quartile"}},
                                "tooltip": [
                                    {"field": "Sales_Rep", "type": "nominal"},
                                    {"field": "Reporting_Manager", "type": "nominal"},
                                    {"field": "Date_of_Joining", "type": "temporal"},
                                    {"field": "Tenure_Months", "type": "quantitative"},
                                    {"field": "Active_Days", "type": "quantitative"},
                                    {"field": "Total_Inquiries", "type": "quantitative"},
                                    {"field": "Inquiries_per_Active_Day", "type": "quantitative"},
                                    {"field": "Inquiries_per_Tenure_Month", "type": "quantitative"},
                                    {"field": "Productivity_Index", "type": "quantitative"},
                                    {"field": "Productivity_Bucket", "type": "nominal"},
                                ]
                            }
                        },
                        use_container_width=True
                    )

                    st.download_button(
                        "Download productivity table as CSV",
                        data=view_df.to_csv(index=False).encode("utf-8"),
                        file_name="sales_rep_productivity.csv",
                        mime="text/csv",
                    )


            st.subheader(f"{label} Comparison: Revenue (bar) + AOV & Enrollments (lines)")
            chart_df = selected_table.copy()
            chart_df = chart_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Revenue","AOV","Enrollments"])
            chart_df["Group"] = chart_df["Group"].astype(str)

            max_groups = 30
            if len(chart_df) > max_groups:
                chart_df = chart_df.sort_values(rank_metric, ascending=False).head(max_groups)

            sort_order = chart_df.sort_values(rank_metric, ascending=False)["Group"].tolist()

            tooltips = [
                {"field": "Group", "type": "nominal", "title": label},
                {"field": "Revenue", "type": "quantitative"},
                {"field": "Enrollments", "type": "quantitative"},
                {"field": "AOV", "type": "quantitative"},
            ]
            if mode == "Reporting Manager":
                if "Team_Size" in chart_df.columns:
                    tooltips.append({"field": "Team_Size", "type": "quantitative", "title": "# Sales Reps"})
                if "Revenue_per_Rep" in chart_df.columns:
                    tooltips.append({"field": "Revenue_per_Rep", "type": "quantitative", "title": "Revenue / Rep"})
                if "Enrollments_per_Rep" in chart_df.columns:
                    tooltips.append({"field": "Enrollments_per_Rep", "type": "quantitative", "title": "Enrollments / Rep"})

            st.vega_lite_chart(
                chart_df,
                {
                    "width": "container",
                    "height": 360,
                    "layer": [
                        {
                            "mark": {"type": "bar", "opacity": 0.6},
                            "encoding": {
                                "x": {"field": "Group", "type": "nominal", "sort": sort_order, "axis": {"labelAngle": -45}},
                                "y": {"field": "Revenue", "type": "quantitative", "axis": {"title": "Revenue (INR)"}},
                                "tooltip": tooltips
                            }
                        },
                        {
                            "mark": {"type": "line", "point": True},
                            "encoding": {
                                "x": {"field": "Group", "type": "nominal", "sort": sort_order},
                                "y": {"field": "AOV", "type": "quantitative", "axis": {"title": "AOV (INR)"}},
                                "tooltip": tooltips
                            }
                        },
                        {
                            "mark": {"type": "line", "point": True, "strokeDash": [4, 2]},
                            "encoding": {
                                "x": {"field": "Group", "type": "nominal", "sort": sort_order},
                                "y": {"field": "Enrollments", "type": "quantitative", "axis": {"title": "Enrollments"}},
                                "tooltip": tooltips
                            }
                        }
                    ],
                    "resolve": {"scale": {"y": "independent"}}
                },
                use_container_width=True
            )

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