
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

def load_master_map(master_text: str) -> pd.DataFrame:
    """Robust loader for embedded master mapping."""
    text = (master_text or '').strip()
    if not text:
        return pd.DataFrame()
    # Try common separators
    for sep in ['\t', ',', ';']:
        try:
            dfm = pd.read_csv(StringIO(text), sep=sep, engine='python')
            dfm.columns = [str(c).replace('\ufeff','').strip() for c in dfm.columns]
            if 'Sales_Rep' in dfm.columns and len(dfm.columns) >= 3:
                return dfm
        except Exception:
            pass
    # Whitespace fallback
    try:
        dfm = pd.read_csv(StringIO(text), sep=r'\s+', engine='python')
        dfm.columns = [str(c).replace('\ufeff','').strip() for c in dfm.columns]
        if 'Sales_Rep' in dfm.columns and len(dfm.columns) >= 3:
            return dfm
    except Exception:
        pass
    # Manual tab split last resort
    raw = [ln for ln in text.splitlines() if ln.strip()]
    header = [h.strip() for h in raw[0].split('\t')]
    rows = [[c.strip() for c in ln.split('\t')] for ln in raw[1:]]
    dfm = pd.DataFrame(rows, columns=header)
    dfm.columns = [str(c).replace('\ufeff','').strip() for c in dfm.columns]
    return dfm

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
master_map = load_master_map(MASTER_TSV)
master_map.columns = [str(c).replace('\ufeff','').strip() for c in master_map.columns]
rename_map = {}
for c in list(master_map.columns):
    lc = str(c).lower().strip()
    if lc in ['reporting_manager','reporting manager','tl','team_lead','team lead']:
        rename_map[c] = 'Reporting_Manager'
    elif lc in ['sales_rep','sales rep','agent','agent_name','salesrep']:
        rename_map[c] = 'Sales_Rep'
    elif lc in ['date_of_joining','date of joining','doj','joining_date','joining date']:
        rename_map[c] = 'Date_of_Joining'
    elif lc in ['tenure_months','tenure months','tenure','tenure_month','tenure month']:
        rename_map[c] = 'Tenure_Months'
master_map = master_map.rename(columns=rename_map)
if 'Date_of_Joining' in master_map.columns:
    master_map['Date_of_Joining'] = pd.to_datetime(master_map['Date_of_Joining'], errors='coerce')
if 'Tenure_Months' in master_map.columns:
    master_map['Tenure_Months'] = pd.to_numeric(master_map['Tenure_Months'], errors='coerce')
else:
    master_map['Tenure_Months'] = np.nan


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

tabs = st.tabs(["Review", "Input Metrics", "UTM Campaign", "Bottom of Funnel"])

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

                st.markdown(
                    """
**Productivity Index Formula (Weights = 60% / 40%)**  
- **Inquiries per Active Day** = Total Inquiries ÷ Active Days  
- **Inquiries per Tenure Month** = Total Inquiries ÷ Tenure Months  

**Productivity Index** = **0.6 × (Inquiries per Active Day)** + **0.4 × (Inquiries per Tenure Month)**
                    """
                )

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
                    
                    # Bubble chart: Productivity vs Enrollments, bubble size = Revenue
                    # (Bigger bubble => higher revenue; higher position => higher enrollments)
                    st.subheader("Bubble View: Productivity vs Enrollments (Bubble Size = Revenue)")

                    # Attach revenue & enrollments to productivity table (from rep aggregation table)
                    rep_metrics = selected_table.rename(columns={"Group": "Sales_Rep"}).copy()
                    bubble_df = view_df.merge(rep_metrics[["Sales_Rep", "Revenue", "Enrollments", "AOV"]], on="Sales_Rep", how="left")

                    # Keep only valid rows
                    bubble_df = bubble_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Productivity_Index", "Enrollments"])
                    bubble_df["Revenue"] = pd.to_numeric(bubble_df["Revenue"], errors="coerce").fillna(0)
                    bubble_df["Enrollments"] = pd.to_numeric(bubble_df["Enrollments"], errors="coerce").fillna(0)

                    st.vega_lite_chart(
                        bubble_df,
                        {
                            "width": "container",
                            "height": 460,
                            "mark": {"type": "circle", "opacity": 0.80},
                            "encoding": {
                                "x": {
                                    "field": "Productivity_Index",
                                    "type": "quantitative",
                                    "axis": {"title": "Productivity Index (0.6*Daily + 0.4*Tenure-normalized)"}
                                },
                                "y": {
                                    "field": "Enrollments",
                                    "type": "quantitative",
                                    "axis": {"title": "Enrollments (higher = bubble higher)"}
                                },
                                "size": {
                                    "field": "Revenue",
                                    "type": "quantitative",
                                    "legend": {"title": "Revenue (bubble size)"}
                                },
                                "color": {
                                    "field": "Productivity_Bucket",
                                    "type": "nominal",
                                    "legend": {"title": "Productivity Quartile"}
                                },
                                "tooltip": [
                                    {"field": "Sales_Rep", "type": "nominal", "title": "Sales Rep"},
                                    {"field": "Reporting_Manager", "type": "nominal", "title": "Team Lead"},
                                    {"field": "Productivity_Index", "type": "quantitative", "title": "Productivity Index"},
                                    {"field": "Total_Inquiries", "type": "quantitative", "title": "Total Inquiries"},
                                    {"field": "Active_Days", "type": "quantitative", "title": "Active Days"},
                                    {"field": "Revenue", "type": "quantitative", "title": "Revenue (INR)"},
                                    {"field": "Enrollments", "type": "quantitative", "title": "Enrollments"},
                                    {"field": "AOV", "type": "quantitative", "title": "AOV (INR)"},
                                    {"field": "Productivity_Bucket", "type": "nominal", "title": "Quartile"},
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

with tabs[1]:
    st.subheader("Input Metrics – Speed to Lead & First Contact")

    required_cols = ["Hrs_To_First_Dial", "First_Contact_Time", "Contact_Made"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        st.warning(f"These columns are missing in your CSV, so this tab is partial: {missing_cols}")
    else:
        # Prepare data
        im = df.copy()

        # Hrs_To_First_Dial
        im["Hrs_To_First_Dial"] = pd.to_numeric(im["Hrs_To_First_Dial"], errors="coerce")
        im = im.dropna(subset=["Hrs_To_First_Dial"]).copy()

        # First_Contact_Time (parse datetime)
        im["First_Contact_Time"] = pd.to_datetime(im["First_Contact_Time"], errors="coerce")
        # If First_Contact_Time is missing but Date exists, fallback to Date
        if im["First_Contact_Time"].isna().all() and "Date" in im.columns:
            im["First_Contact_Time"] = pd.to_datetime(im["Date"], errors="coerce")

        # Contact_Made normalization to boolean
        cm = im["Contact_Made"].astype(str).str.strip().str.lower()
        im["_contact_made"] = cm.isin(["1", "true", "yes", "y", "contacted", "made", "connected"])

        # Apply same period filter as Review tab for comparability
        if month_choice.startswith("Consolidated"):
            im = im[clean_month(im["Month"]).isin(["July", "August", "September"])].copy() if "Month" in im.columns else im
            period_label = "Q3 (Jul–Sep) Consolidated"
        else:
            im = im[clean_month(im["Month"]) == month_choice].copy() if "Month" in im.columns else im
            period_label = month_choice

        st.caption(f"Period filter applied: **{period_label}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows in view", f"{len(im):,}")
        c2.metric("Median hrs to first dial", f"{im['Hrs_To_First_Dial'].median():.2f}")
        c3.metric("P75 hrs to first dial", f"{im['Hrs_To_First_Dial'].quantile(0.75):.2f}")
        c4.metric("Contact Made rate", f"{(im['_contact_made'].mean()*100):.0f}%")

        st.divider()

        # 1) Distribution of Hrs_To_First_Dial
        st.subheader("Distribution: Hrs_To_First_Dial (Speed-to-Lead)")
        st.caption("Shows latency from lead creation to first dial. Lower is better.")

        hist_df = im[["Hrs_To_First_Dial"]].dropna().copy()
        st.vega_lite_chart(
            hist_df,
            {
                "width": "container",
                "height": 340,
                "mark": "bar",
                "encoding": {
                    "x": {
                        "field": "Hrs_To_First_Dial",
                        "type": "quantitative",
                        "bin": {"maxbins": 30},
                        "axis": {"title": "Hours to First Dial"}
                    },
                    "y": {"aggregate": "count", "type": "quantitative", "axis": {"title": "Count of leads"}},
                    "tooltip": [{"aggregate": "count", "type": "quantitative", "title": "Count"}]
                }
            },
            use_container_width=True
        )

        st.divider()

        # 2) When do we call? Hour-of-day distribution for First_Contact_Time
        st.subheader("When calls happen: First_Contact_Time by Hour-of-Day")
        if im["First_Contact_Time"].notna().any():
            im["_contact_hour"] = im["First_Contact_Time"].dt.hour

            hour_df = im.dropna(subset=["_contact_hour"]).copy()
            st.vega_lite_chart(
                hour_df,
                {
                    "width": "container",
                    "height": 320,
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "_contact_hour", "type": "ordinal", "axis": {"title": "Hour of Day (0-23)"}},
                        "y": {"aggregate": "count", "type": "quantitative", "axis": {"title": "Count of first contacts"}},
                        "tooltip": [{"field": "_contact_hour", "type": "ordinal", "title": "Hour"}, {"aggregate": "count", "type": "quantitative", "title": "Count"}]
                    }
                },
                use_container_width=True
            )
        else:
            st.info("First_Contact_Time could not be parsed; please ensure it is a valid datetime column.")

        st.divider()

        # 3) Likelihood of Contact Made vs Hrs_To_First_Dial (conversion/engagement likelihood proxy)
        st.subheader("Likelihood of Contact Made vs Response Time")
        st.caption("This shows: if we call faster, do we get higher Contact_Made probability?")

        # bins for hours
        bins = [0, 0.25, 0.5, 1, 2, 4, 8, 24, np.inf]
        labels = ["0-15m", "15-30m", "30-60m", "1-2h", "2-4h", "4-8h", "8-24h", "24h+"]
        im["_rt_bin"] = pd.cut(im["Hrs_To_First_Dial"], bins=bins, labels=labels, include_lowest=True, right=False)

        rate_tbl = (
            im.groupby("_rt_bin")["_contact_made"]
            .agg(Contact_Made_Rate="mean", Leads="size")
            .reset_index()
        )
        rate_tbl["Contact_Made_Rate"] = rate_tbl["Contact_Made_Rate"] * 100

        st.vega_lite_chart(
            rate_tbl,
            {
                "width": "container",
                "height": 340,
                "mark": {"type": "line", "point": True},
                "encoding": {
                    "x": {"field": "_rt_bin", "type": "ordinal", "axis": {"title": "Response time bucket"}},
                    "y": {"field": "Contact_Made_Rate", "type": "quantitative", "axis": {"title": "Contact Made Rate (%)"}},
                    "tooltip": [
                        {"field": "_rt_bin", "type": "ordinal", "title": "Bucket"},
                        {"field": "Contact_Made_Rate", "type": "quantitative", "title": "Rate (%)"},
                        {"field": "Leads", "type": "quantitative", "title": "Leads"},
                    ]
                }
            },
            use_container_width=True
        )

        st.dataframe(rate_tbl.rename(columns={"_rt_bin": "Response_Time_Bucket"}), use_container_width=True)

        st.divider()

        # 4) Sales Rep defaulters (slowest responders)
        st.subheader("Sales Rep Defaulters: Slowest Response Time")
        if "Sales_Rep" in im.columns:
            rep_tbl = (
                im.groupby("Sales_Rep")["Hrs_To_First_Dial"]
                .agg(Leads="size", Avg_Hours="mean", Median_Hours="median", P75_Hours=lambda s: s.quantile(0.75))
                .reset_index()
            )
            rep_tbl["Avg_Hours"] = rep_tbl["Avg_Hours"].round(2)
            rep_tbl["Median_Hours"] = rep_tbl["Median_Hours"].round(2)
            rep_tbl["P75_Hours"] = rep_tbl["P75_Hours"].round(2)

            # show top slowest by median, only those with meaningful lead count
            min_leads = st.slider("Min leads per rep (filter noise)", min_value=1, max_value=int(max(1, rep_tbl["Leads"].max())), value=min(20, int(max(1, rep_tbl["Leads"].max()))))
            rep_tbl_f = rep_tbl[rep_tbl["Leads"] >= min_leads].copy()
            rep_tbl_f = rep_tbl_f.sort_values("Median_Hours", ascending=False)

            st.dataframe(rep_tbl_f.head(30), use_container_width=True)

            st.vega_lite_chart(
                rep_tbl_f.head(20),
                {
                    "width": "container",
                    "height": 380,
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "Sales_Rep", "type": "nominal", "sort": "-y", "axis": {"labelAngle": -45, "title": "Sales Rep"}},
                        "y": {"field": "Median_Hours", "type": "quantitative", "axis": {"title": "Median Hrs_To_First_Dial"}},
                        "tooltip": [
                            {"field": "Sales_Rep", "type": "nominal"},
                            {"field": "Leads", "type": "quantitative"},
                            {"field": "Avg_Hours", "type": "quantitative", "title": "Avg Hours"},
                            {"field": "Median_Hours", "type": "quantitative", "title": "Median Hours"},
                            {"field": "P75_Hours", "type": "quantitative", "title": "P75 Hours"},
                        ]
                    }
                },
                use_container_width=True
            )
        else:
            st.warning("Sales_Rep column not found in your CSV, so defaulter ranking cannot be computed.")

        st.download_button(
            "Download Input Metrics (filtered) as CSV",
            data=im.to_csv(index=False).encode("utf-8"),
            file_name="input_metrics_filtered.csv",
            mime="text/csv",
        )



with tabs[2]:
    st.subheader("UTM Campaign Performance")

    if "Utm_Campaign" not in df.columns:
        st.warning("Column 'Utm_Campaign' not found in your CSV.")
    else:
        utm_df = df.copy()
        if "Month" in utm_df.columns:
            utm_df["Month"] = clean_month(utm_df["Month"])

        if month_choice.startswith("Consolidated"):
            if "Month" in utm_df.columns:
                utm_df = utm_df[utm_df["Month"].isin(["July", "August", "September"])].copy()
            period_label = "Q3 (Jul–Sep) Consolidated"
        else:
            if "Month" in utm_df.columns:
                utm_df = utm_df[utm_df["Month"] == month_choice].copy()
            period_label = month_choice

        st.caption(f"Period filter applied: **{period_label}**")

        # Determine inquiry identifier (prefer LeadID/InquiryID; fallback to row-count)
        inquiry_id_col = None
        for cand in ["LeadID", "LeadId", "Lead_ID", "leadid", "lead_id", "InquiryID", "Inquiry_Id", "ProspectID", "ProspectId"]:
            if cand in utm_df.columns:
                inquiry_id_col = cand
                break

        if inquiry_id_col:
            lead_counts = utm_df.groupby("Utm_Campaign")[inquiry_id_col].nunique(dropna=True).reset_index(name="Leads")
        else:
            lead_counts = utm_df.groupby("Utm_Campaign").size().reset_index(name="Leads")

        lead_counts = lead_counts.sort_values("Leads", ascending=False)
        total_leads = float(lead_counts["Leads"].sum())
        lead_counts["Percent"] = np.where(total_leads > 0, (lead_counts["Leads"] / total_leads) * 100, 0.0)
        lead_counts["Percent_int"] = lead_counts["Percent"].round(0).astype(int)

        st.subheader("Campaign Mix (Share of Leads)")
        st.caption("Pie = distribution of leads by UTM campaign (based on unique LeadID if available).")

        top_n = st.slider("Show top N campaigns in pie (rest grouped as 'Others')", min_value=5, max_value=30, value=12)
        pie_df = lead_counts.copy()
        if len(pie_df) > top_n:
            top = pie_df.head(top_n).copy()
            others = pd.DataFrame({
                "Utm_Campaign": ["Others"],
                "Leads": [pie_df.iloc[top_n:]["Leads"].sum()],
            })
            others["Percent"] = np.where(total_leads > 0, (others["Leads"] / total_leads) * 100, 0.0)
            others["Percent_int"] = others["Percent"].round(0).astype(int)
            pie_df = pd.concat([top, others], ignore_index=True)

        st.vega_lite_chart(
            pie_df,
            {
                "width": "container",
                "height": 360,
                "mark": {"type": "arc", "innerRadius": 40},
                "encoding": {
                    "theta": {"field": "Leads", "type": "quantitative"},
                    "color": {"field": "Utm_Campaign", "type": "nominal", "legend": {"title": "Campaign"}},
                    "tooltip": [
                        {"field": "Utm_Campaign", "type": "nominal", "title": "Campaign"},
                        {"field": "Leads", "type": "quantitative", "title": "Leads"},
                        {"field": "Percent", "type": "quantitative", "title": "Percent (%)"}
                    ]
                }
            },
            use_container_width=True
        )

        tbl = lead_counts[["Utm_Campaign", "Leads", "Percent_int"]].rename(columns={"Percent_int": "Percent (%)"})
        st.dataframe(tbl, use_container_width=True, height=420)

        st.divider()

        st.subheader("Which Campaign Converts With Which Sales Rep?")
        st.caption("Conversion proxy = Enrollments computed from your dataset logic (same as Review).")

        if "Sales_Rep" not in utm_df.columns:
            st.warning("Column 'Sales_Rep' not found in your CSV, so Campaign × Sales Rep view can't be generated.")
        else:
            enr_series = enroll_series_fn(utm_df)

            tmp = utm_df.copy()
            tmp["_enroll"] = pd.to_numeric(enr_series, errors="coerce").fillna(0)

            if inquiry_id_col:
                grp_leads = tmp.groupby(["Utm_Campaign", "Sales_Rep"])[inquiry_id_col].nunique(dropna=True)
            else:
                grp_leads = tmp.groupby(["Utm_Campaign", "Sales_Rep"]).size()

            grp_enr = tmp.groupby(["Utm_Campaign", "Sales_Rep"])["_enroll"].sum()

            mat = (
                pd.DataFrame({"Leads": grp_leads, "Enrollments": grp_enr})
                .reset_index()
            )
            mat["Enrollments"] = pd.to_numeric(mat["Enrollments"], errors="coerce").fillna(0)
            mat["Conv_Rate"] = np.where(mat["Leads"] > 0, (mat["Enrollments"] / mat["Leads"]) * 100, 0.0)

            top_campaigns = st.slider("Top campaigns to show (by leads)", min_value=5, max_value=50, value=15)
            top_reps = st.slider("Top sales reps to show (by leads)", min_value=5, max_value=50, value=15)

            camp_rank = lead_counts.head(top_campaigns)["Utm_Campaign"].astype(str).tolist()
            rep_rank = (
                tmp.groupby("Sales_Rep")
                .size()
                .sort_values(ascending=False)
                .head(top_reps)
                .index.astype(str)
                .tolist()
            )

            mat_f = mat[mat["Utm_Campaign"].astype(str).isin(camp_rank) & mat["Sales_Rep"].astype(str).isin(rep_rank)].copy()

            st.vega_lite_chart(
                mat_f,
                {
                    "width": "container",
                    "height": 520,
                    "mark": {"type": "rect"},
                    "encoding": {
                        "x": {"field": "Sales_Rep", "type": "nominal", "axis": {"labelAngle": -45, "title": "Sales Rep"}},
                        "y": {"field": "Utm_Campaign", "type": "nominal", "axis": {"title": "UTM Campaign"}},
                        "color": {"field": "Conv_Rate", "type": "quantitative", "legend": {"title": "Conversion % (Enroll/Leads)"}},
                        "tooltip": [
                            {"field": "Utm_Campaign", "type": "nominal", "title": "Campaign"},
                            {"field": "Sales_Rep", "type": "nominal", "title": "Sales Rep"},
                            {"field": "Leads", "type": "quantitative", "title": "Leads"},
                            {"field": "Enrollments", "type": "quantitative", "title": "Enrollments"},
                            {"field": "Conv_Rate", "type": "quantitative", "title": "Conversion %"},
                        ]
                    }
                },
                use_container_width=True
            )

            st.subheader("Top Campaign × Sales Rep pairs (by Conversion %)")
            min_leads_pair = st.slider("Min leads in pair (filter noise)", min_value=1, max_value=int(max(1, mat["Leads"].max())), value=min(20, int(max(1, mat["Leads"].max()))))
            mat_tbl = mat[mat["Leads"] >= min_leads_pair].copy()
            mat_tbl["Conv_Rate"] = mat_tbl["Conv_Rate"].round(1)
            mat_tbl = mat_tbl.sort_values(["Conv_Rate", "Enrollments", "Leads"], ascending=False).head(50)

            st.dataframe(mat_tbl, use_container_width=True, height=520)

            st.download_button(
                "Download campaign × rep conversion table as CSV",
                data=mat.to_csv(index=False).encode("utf-8"),
                file_name="utm_campaign_salesrep_conversion.csv",
                mime="text/csv",
            )



with tabs[3]:
    st.subheader("Bottom of Funnel – Conversion Speed")

    required_cols = ["Form_Submitted_At", "Conversion_Date"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.warning(f"Missing required columns for Bottom of Funnel: {missing_cols}")
    else:
        bof = df.copy()

        # Apply same period filter as Review tab
        if "Month" in bof.columns:
            bof["Month"] = clean_month(bof["Month"])
        if month_choice.startswith("Consolidated"):
            if "Month" in bof.columns:
                bof = bof[bof["Month"].isin(["July", "August", "September"])].copy()
            period_label = "Q3 (Jul–Sep) Consolidated"
        else:
            if "Month" in bof.columns:
                bof = bof[bof["Month"] == month_choice].copy()
            period_label = month_choice

        st.caption(f"Period filter applied: **{period_label}**")

        # Parse only YYYY-MM-DD part (first 10 chars)
        bof["_form_date"] = pd.to_datetime(bof["Form_Submitted_At"].astype(str).str.slice(0, 10), errors="coerce")
        bof["_conv_date"] = pd.to_datetime(bof["Conversion_Date"].astype(str).str.slice(0, 10), errors="coerce")

        # Define converted rows
        bof["_converted"] = bof["_conv_date"].notna() & bof["_form_date"].notna()

        conv = bof[bof["_converted"]].copy()
        if conv.empty:
            st.info("No converted rows found (Conversion_Date missing or unparsable).")
        else:
            conv["Conversion_Speed_Days"] = (conv["_conv_date"] - conv["_form_date"]).dt.days
            conv = conv[conv["Conversion_Speed_Days"].notna()].copy()
            # Remove negative lags (data issues)
            conv = conv[conv["Conversion_Speed_Days"] >= 0].copy()

            if conv.empty:
                st.info("Converted rows exist, but conversion speed could not be computed (negative/invalid dates).")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Converted rows", f"{len(conv):,}")
                c2.metric("Avg conversion speed (days)", f"{conv['Conversion_Speed_Days'].mean():.1f}")
                c3.metric("Median speed (days)", f"{conv['Conversion_Speed_Days'].median():.0f}")
                c4.metric("P75 speed (days)", f"{conv['Conversion_Speed_Days'].quantile(0.75):.0f}")

                st.divider()

                # View selector
                view = st.radio(
                    "View",
                    options=["Overall Distribution", "Sales Rep-wise", "Campaign-wise", "Speed Buckets"],
                    horizontal=True
                )

                # Optional filters
                min_conversions = st.slider("Min conversions for ranking views", min_value=1, max_value=int(max(1, conv.shape[0])), value=min(20, int(max(1, conv.shape[0]))))

                # 1) Overall distribution
                if view == "Overall Distribution":
                    st.subheader("Distribution of Conversion Speed (Days)")
                    st.caption("Days from Form_Submitted_At → Conversion_Date (converted rows only).")

                    hist_df = conv[["Conversion_Speed_Days"]].copy()
                    st.vega_lite_chart(
                        hist_df,
                        {
                            "width": "container",
                            "height": 360,
                            "mark": "bar",
                            "encoding": {
                                "x": {"field": "Conversion_Speed_Days", "type": "quantitative", "bin": {"maxbins": 30},
                                      "axis": {"title": "Conversion Speed (days)"}},
                                "y": {"aggregate": "count", "type": "quantitative", "axis": {"title": "Converted leads"}},
                                "tooltip": [{"aggregate": "count", "type": "quantitative", "title": "Converted leads"}]
                            }
                        },
                        use_container_width=True
                    )

                    # Quick SLA buckets
                    bins = [0, 1, 3, 7, 14, 30, 60, 90, np.inf]
                    labels = ["0d", "1-2d", "3-6d", "7-13d", "14-29d", "30-59d", "60-89d", "90d+"]
                    conv["_speed_bucket"] = pd.cut(conv["Conversion_Speed_Days"], bins=bins, labels=labels, include_lowest=True, right=False)
                    bucket_tbl = conv.groupby("_speed_bucket")["Conversion_Speed_Days"].agg(Conversions="size").reset_index()
                    bucket_tbl["Percent"] = (bucket_tbl["Conversions"] / bucket_tbl["Conversions"].sum()) * 100
                    bucket_tbl["Percent"] = bucket_tbl["Percent"].round(0).astype(int)
                    st.subheader("Speed Bucket Share")
                    st.dataframe(bucket_tbl.rename(columns={"_speed_bucket": "Speed Bucket", "Percent": "Percent (%)"}), use_container_width=True)

                # 2) Sales Rep-wise
                elif view == "Sales Rep-wise":
                    st.subheader("Sales Rep-wise Conversion Speed")
                    if "Sales_Rep" not in conv.columns:
                        st.warning("Sales_Rep column not found in your CSV.")
                    else:
                        rep_tbl = (
                            conv.groupby("Sales_Rep")["Conversion_Speed_Days"]
                            .agg(Conversions="size", Avg_Days="mean", Median_Days="median", P75_Days=lambda s: s.quantile(0.75))
                            .reset_index()
                        )
                        rep_tbl = rep_tbl[rep_tbl["Conversions"] >= min_conversions].copy()
                        rep_tbl["Avg_Days"] = rep_tbl["Avg_Days"].round(1)
                        rep_tbl["Median_Days"] = rep_tbl["Median_Days"].round(0).astype(int)
                        rep_tbl["P75_Days"] = rep_tbl["P75_Days"].round(0).astype(int)

                        # Best = fastest (lowest avg days)
                        rep_tbl = rep_tbl.sort_values("Avg_Days", ascending=True)

                        st.dataframe(rep_tbl, use_container_width=True, height=520)

                        st.vega_lite_chart(
                            rep_tbl.head(25),
                            {
                                "width": "container",
                                "height": 420,
                                "mark": "bar",
                                "encoding": {
                                    "x": {"field": "Sales_Rep", "type": "nominal", "sort": "y",
                                          "axis": {"labelAngle": -45, "title": "Sales Rep"}},
                                    "y": {"field": "Avg_Days", "type": "quantitative", "axis": {"title": "Avg Conversion Speed (days)"}},
                                    "tooltip": [
                                        {"field": "Sales_Rep", "type": "nominal"},
                                        {"field": "Conversions", "type": "quantitative"},
                                        {"field": "Avg_Days", "type": "quantitative"},
                                        {"field": "Median_Days", "type": "quantitative"},
                                        {"field": "P75_Days", "type": "quantitative"},
                                    ]
                                }
                            },
                            use_container_width=True
                        )

                # 3) Campaign-wise
                elif view == "Campaign-wise":
                    st.subheader("Campaign-wise Conversion Speed")
                    if "Utm_Campaign" not in conv.columns:
                        st.warning("Utm_Campaign column not found in your CSV.")
                    else:
                        camp_tbl = (
                            conv.groupby("Utm_Campaign")["Conversion_Speed_Days"]
                            .agg(Conversions="size", Avg_Days="mean", Median_Days="median", P75_Days=lambda s: s.quantile(0.75))
                            .reset_index()
                        )
                        camp_tbl = camp_tbl[camp_tbl["Conversions"] >= min_conversions].copy()
                        camp_tbl["Avg_Days"] = camp_tbl["Avg_Days"].round(1)
                        camp_tbl["Median_Days"] = camp_tbl["Median_Days"].round(0).astype(int)
                        camp_tbl["P75_Days"] = camp_tbl["P75_Days"].round(0).astype(int)

                        camp_tbl = camp_tbl.sort_values("Avg_Days", ascending=True)

                        st.dataframe(camp_tbl, use_container_width=True, height=520)

                        st.vega_lite_chart(
                            camp_tbl.head(25),
                            {
                                "width": "container",
                                "height": 420,
                                "mark": "bar",
                                "encoding": {
                                    "x": {"field": "Utm_Campaign", "type": "nominal", "sort": "y",
                                          "axis": {"labelAngle": -45, "title": "UTM Campaign"}},
                                    "y": {"field": "Avg_Days", "type": "quantitative", "axis": {"title": "Avg Conversion Speed (days)"}},
                                    "tooltip": [
                                        {"field": "Utm_Campaign", "type": "nominal", "title": "Campaign"},
                                        {"field": "Conversions", "type": "quantitative"},
                                        {"field": "Avg_Days", "type": "quantitative"},
                                        {"field": "Median_Days", "type": "quantitative"},
                                        {"field": "P75_Days", "type": "quantitative"},
                                    ]
                                }
                            },
                            use_container_width=True
                        )

                # 4) Speed buckets (what speed is “right”)
                else:
                    st.subheader("What conversion speed is ‘best’?")
                    st.caption("This shows how conversions are distributed across speed buckets. You can use it to set follow-up SLAs.")

                    bins = [0, 1, 3, 7, 14, 30, 60, 90, np.inf]
                    labels = ["0d", "1-2d", "3-6d", "7-13d", "14-29d", "30-59d", "60-89d", "90d+"]
                    conv["_speed_bucket"] = pd.cut(conv["Conversion_Speed_Days"], bins=bins, labels=labels, include_lowest=True, right=False)

                    speed_tbl = (
                        conv.groupby("_speed_bucket")["Conversion_Speed_Days"]
                        .agg(Conversions="size", Avg_Days="mean")
                        .reset_index()
                    )
                    speed_tbl["Avg_Days"] = speed_tbl["Avg_Days"].round(1)
                    speed_tbl["Percent"] = (speed_tbl["Conversions"] / speed_tbl["Conversions"].sum()) * 100
                    speed_tbl["Percent"] = speed_tbl["Percent"].round(0).astype(int)

                    st.vega_lite_chart(
                        speed_tbl,
                        {
                            "width": "container",
                            "height": 360,
                            "mark": {"type": "bar"},
                            "encoding": {
                                "x": {"field": "_speed_bucket", "type": "ordinal", "axis": {"title": "Speed bucket"}},
                                "y": {"field": "Conversions", "type": "quantitative", "axis": {"title": "Conversions"}},
                                "tooltip": [
                                    {"field": "_speed_bucket", "type": "ordinal", "title": "Bucket"},
                                    {"field": "Conversions", "type": "quantitative"},
                                    {"field": "Percent", "type": "quantitative", "title": "Percent (%)"},
                                    {"field": "Avg_Days", "type": "quantitative", "title": "Avg Days"},
                                ]
                            }
                        },
                        use_container_width=True
                    )

                    st.dataframe(speed_tbl.rename(columns={"_speed_bucket": "Speed Bucket", "Percent": "Percent (%)"}), use_container_width=True)

                st.divider()

                st.download_button(
                    "Download converted rows with speed as CSV",
                    data=conv.to_csv(index=False).encode("utf-8"),
                    file_name="bottom_of_funnel_conversion_speed.csv",
                    mime="text/csv",
                )
