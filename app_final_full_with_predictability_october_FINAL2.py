
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

tabs = st.tabs(["Review", "Input Metrics", "UTM Campaign", "Bottom of Funnel", "Add-on Products", "Webinars", "Input", "Marketing Performance", "Business Predictability (Oct)"])

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


with tabs[4]:
    st.subheader("Add-on Products – Attach Rate, Revenue Lift & Ownership")

    required_cols = ["Program_Fee_INR", "Addon_Purchased", "Addon_Type", "Addon_Billed_Separately", "Addon_Revenue_INR", "Total_Value_INR"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.warning(f"Missing required columns for Add-on analysis: {missing_cols}")
    else:
        ad = df.copy()

        # Apply same period filter as Review tab (so numbers are comparable)
        if "Month" in ad.columns:
            ad["Month"] = clean_month(ad["Month"])
        if month_choice.startswith("Consolidated"):
            if "Month" in ad.columns:
                ad = ad[ad["Month"].isin(["July", "August", "September"])].copy()
            period_label = "Q3 (Jul–Sep) Consolidated"
        else:
            if "Month" in ad.columns:
                ad = ad[ad["Month"] == month_choice].copy()
            period_label = month_choice

        st.caption(f"Period filter applied: **{period_label}**")

        # Normalize numeric
        ad["Program_Fee_INR"] = pd.to_numeric(ad["Program_Fee_INR"], errors="coerce")
        ad["Addon_Revenue_INR"] = pd.to_numeric(ad["Addon_Revenue_INR"], errors="coerce")
        ad["Total_Value_INR"] = pd.to_numeric(ad["Total_Value_INR"], errors="coerce")

        # Normalize booleans
        ap = ad["Addon_Purchased"].astype(str).str.strip().str.lower()
        ad["_addon_purchased"] = ap.isin(["1", "true", "yes", "y", "purchased", "bought"])

        bs = ad["Addon_Billed_Separately"].astype(str).str.strip().str.lower()
        ad["_addon_billed_separately"] = bs.isin(["1", "true", "yes", "y", "separate", "separately"])

        ad["Addon_Revenue_INR"] = ad["Addon_Revenue_INR"].fillna(0)
        ad["Program_Fee_INR"] = ad["Program_Fee_INR"].fillna(0)
        ad["Total_Value_INR"] = ad["Total_Value_INR"].fillna(0)

        total_orders = len(ad)
        addon_orders = int(ad["_addon_purchased"].sum())
        attach_rate = (addon_orders / total_orders) * 100 if total_orders > 0 else 0.0
        addon_revenue = float(ad.loc[ad["_addon_purchased"], "Addon_Revenue_INR"].sum())
        total_revenue = float(ad["Total_Value_INR"].sum())
        addon_share = (addon_revenue / total_revenue) * 100 if total_revenue > 0 else 0.0

        with_addon = ad.loc[ad["_addon_purchased"], "Total_Value_INR"]
        without_addon = ad.loc[~ad["_addon_purchased"], "Total_Value_INR"]
        avg_with = float(with_addon.mean()) if len(with_addon) else 0.0
        avg_without = float(without_addon.mean()) if len(without_addon) else 0.0
        lift = ((avg_with - avg_without) / avg_without) * 100 if avg_without > 0 else np.nan

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total orders (rows)", f"{total_orders:,}")
        c2.metric("Add-on attach rate", f"{attach_rate:.0f}%")
        c3.metric("Add-on revenue", format_inr(addon_revenue))
        c4.metric("Add-on revenue share", f"{addon_share:.0f}%")

        c5, c6 = st.columns(2)
        c5.metric("Avg order value (with add-on)", format_inr(avg_with))
        c6.metric("Avg order value (without add-on)", format_inr(avg_without))

        if pd.notna(lift):
            st.caption(f"Revenue lift proxy (Avg Total_Value_INR with add-on vs without): **{lift:.0f}%** (interpret cautiously; depends on lead mix).")
        else:
            st.caption("Revenue lift proxy not computed (avg without add-on is 0 or missing).")

        st.divider()

        st.subheader("1) Add-on Purchased – Mix")
        mix = pd.DataFrame({
            "Addon_Purchased": ["Yes", "No"],
            "Orders": [int(ad["_addon_purchased"].sum()), int((~ad["_addon_purchased"]).sum())]
        })
        mix["Percent"] = np.where(mix["Orders"].sum() > 0, mix["Orders"] / mix["Orders"].sum() * 100, 0.0).round(0).astype(int)

        st.vega_lite_chart(
            mix,
            {
                "width": "container",
                "height": 320,
                "mark": {"type": "arc", "innerRadius": 40},
                "encoding": {
                    "theta": {"field": "Orders", "type": "quantitative"},
                    "color": {"field": "Addon_Purchased", "type": "nominal", "legend": {"title": "Add-on Purchased"}},
                    "tooltip": [
                        {"field": "Addon_Purchased", "type": "nominal"},
                        {"field": "Orders", "type": "quantitative"},
                        {"field": "Percent", "type": "quantitative", "title": "Percent (%)"},
                    ]
                }
            },
            use_container_width=True
        )
        st.dataframe(mix.rename(columns={"Percent": "Percent (%)"}), use_container_width=True)

        st.divider()

        st.subheader("2) Which Campaign drives Add-ons?")
        if "Utm_Campaign" not in ad.columns:
            st.warning("Utm_Campaign not found, so campaign-wise add-on view is skipped.")
        else:
            ad["Utm_Campaign"] = ad["Utm_Campaign"].astype(str).fillna("Unknown").str.strip()
            camp = (
                ad.groupby("Utm_Campaign")
                .agg(
                    Orders=("Utm_Campaign", "size"),
                    Addon_Orders=("_addon_purchased", "sum"),
                    Addon_Revenue=("Addon_Revenue_INR", "sum"),
                    Total_Revenue=("Total_Value_INR", "sum"),
                )
                .reset_index()
            )
            camp["Attach_Rate_%"] = np.where(camp["Orders"] > 0, camp["Addon_Orders"] / camp["Orders"] * 100, 0.0)
            camp["Attach_Rate_%"] = camp["Attach_Rate_%"].round(1)
            camp["Addon_Share_%"] = np.where(camp["Total_Revenue"] > 0, camp["Addon_Revenue"] / camp["Total_Revenue"] * 100, 0.0)
            camp["Addon_Share_%"] = camp["Addon_Share_%"].round(1)

            camp = camp.sort_values("Addon_Revenue", ascending=False)

            st.vega_lite_chart(
                camp.head(25),
                {
                    "width": "container",
                    "height": 420,
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "Utm_Campaign", "type": "nominal", "sort": "-y", "axis": {"labelAngle": -45, "title": "Campaign"}},
                        "y": {"field": "Attach_Rate_%", "type": "quantitative", "axis": {"title": "Add-on Attach Rate (%)"}},
                        "tooltip": [
                            {"field": "Utm_Campaign", "type": "nominal"},
                            {"field": "Orders", "type": "quantitative"},
                            {"field": "Addon_Orders", "type": "quantitative"},
                            {"field": "Attach_Rate_%", "type": "quantitative", "title": "Attach Rate (%)"},
                            {"field": "Addon_Revenue", "type": "quantitative", "title": "Add-on Revenue (INR)"},
                            {"field": "Addon_Share_%", "type": "quantitative", "title": "Add-on Share (%)"},
                        ]
                    }
                },
                use_container_width=True
            )

            camp_tbl = camp.copy()
            camp_tbl["Addon_Revenue"] = camp_tbl["Addon_Revenue"].apply(format_inr)
            camp_tbl["Total_Revenue"] = camp_tbl["Total_Revenue"].apply(format_inr)
            st.dataframe(camp_tbl[["Utm_Campaign", "Orders", "Addon_Orders", "Attach_Rate_%", "Addon_Revenue", "Addon_Share_%"]], use_container_width=True, height=520)

        st.divider()

        st.subheader("3) Which Sales Rep sells most Add-ons?")
        if "Sales_Rep" not in ad.columns:
            st.warning("Sales_Rep not found, so sales-rep add-on ownership view is skipped.")
        else:
            rep = (
                ad.groupby("Sales_Rep")
                .agg(
                    Orders=("Sales_Rep", "size"),
                    Addon_Orders=("_addon_purchased", "sum"),
                    Addon_Revenue=("Addon_Revenue_INR", "sum"),
                    Total_Revenue=("Total_Value_INR", "sum"),
                )
                .reset_index()
            )
            rep["Attach_Rate_%"] = np.where(rep["Orders"] > 0, rep["Addon_Orders"] / rep["Orders"] * 100, 0.0)
            rep["Attach_Rate_%"] = rep["Attach_Rate_%"].round(1)
            rep["Addon_Share_%"] = np.where(rep["Total_Revenue"] > 0, rep["Addon_Revenue"] / rep["Total_Revenue"] * 100, 0.0)
            rep["Addon_Share_%"] = rep["Addon_Share_%"].round(1)

            rep = rep.sort_values(["Addon_Orders", "Addon_Revenue"], ascending=False)

            st.vega_lite_chart(
                rep.head(25),
                {
                    "width": "container",
                    "height": 420,
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "Sales_Rep", "type": "nominal", "sort": "-y", "axis": {"labelAngle": -45, "title": "Sales Rep"}},
                        "y": {"field": "Addon_Orders", "type": "quantitative", "axis": {"title": "Add-on Orders (count)"}},
                        "tooltip": [
                            {"field": "Sales_Rep", "type": "nominal"},
                            {"field": "Orders", "type": "quantitative"},
                            {"field": "Addon_Orders", "type": "quantitative"},
                            {"field": "Attach_Rate_%", "type": "quantitative", "title": "Attach Rate (%)"},
                            {"field": "Addon_Revenue", "type": "quantitative", "title": "Add-on Revenue (INR)"},
                            {"field": "Addon_Share_%", "type": "quantitative", "title": "Add-on Share (%)"},
                        ]
                    }
                },
                use_container_width=True
            )

            rep_tbl = rep.copy()
            rep_tbl["Addon_Revenue"] = rep_tbl["Addon_Revenue"].apply(format_inr)
            rep_tbl["Total_Revenue"] = rep_tbl["Total_Revenue"].apply(format_inr)
            st.dataframe(rep_tbl[["Sales_Rep", "Orders", "Addon_Orders", "Attach_Rate_%", "Addon_Revenue", "Addon_Share_%"]], use_container_width=True, height=520)

        st.divider()

        st.subheader("4) Are Add-ons billed separately?")
        if ad["_addon_purchased"].sum() == 0:
            st.info("No add-on purchases found in this filtered view.")
        else:
            ap_only = ad[ad["_addon_purchased"]].copy()
            split = (
                ap_only.groupby("_addon_billed_separately")
                .agg(Orders=("_addon_billed_separately", "size"), Addon_Revenue=("Addon_Revenue_INR", "sum"))
                .reset_index()
            )
            split["_addon_billed_separately"] = split["_addon_billed_separately"].map({True: "Yes", False: "No"})
            split["Percent"] = (split["Orders"] / split["Orders"].sum() * 100).round(0).astype(int)

            st.vega_lite_chart(
                split,
                {
                    "width": "container",
                    "height": 320,
                    "mark": {"type": "arc", "innerRadius": 40},
                    "encoding": {
                        "theta": {"field": "Orders", "type": "quantitative"},
                        "color": {"field": "_addon_billed_separately", "type": "nominal", "legend": {"title": "Billed Separately"}},
                        "tooltip": [
                            {"field": "_addon_billed_separately", "type": "nominal", "title": "Billed Separately"},
                            {"field": "Orders", "type": "quantitative"},
                            {"field": "Percent", "type": "quantitative", "title": "Percent (%)"},
                            {"field": "Addon_Revenue", "type": "quantitative", "title": "Add-on Revenue (INR)"},
                        ]
                    }
                },
                use_container_width=True
            )
            split_tbl = split.copy()
            split_tbl["Addon_Revenue"] = split_tbl["Addon_Revenue"].apply(format_inr)
            st.dataframe(split_tbl.rename(columns={"_addon_billed_separately": "Billed Separately", "Percent": "Percent (%)"}), use_container_width=True)

            if "Addon_Type" in ap_only.columns:
                ap_only["Addon_Type"] = ap_only["Addon_Type"].astype(str).fillna("Unknown").str.strip()
                type_tbl = (
                    ap_only.groupby(["Addon_Type", "_addon_billed_separately"])
                    .agg(Orders=("Addon_Type", "size"), Addon_Revenue=("Addon_Revenue_INR", "sum"))
                    .reset_index()
                )
                type_tbl["_addon_billed_separately"] = type_tbl["_addon_billed_separately"].map({True: "Yes", False: "No"})
                st.subheader("Add-on Type × Billed Separately")
                st.vega_lite_chart(
                    type_tbl,
                    {
                        "width": "container",
                        "height": 420,
                        "mark": "bar",
                        "encoding": {
                            "x": {"field": "Addon_Type", "type": "nominal", "axis": {"labelAngle": -45, "title": "Add-on Type"}},
                            "y": {"field": "Orders", "type": "quantitative", "axis": {"title": "Orders"}},
                            "color": {"field": "_addon_billed_separately", "type": "nominal", "legend": {"title": "Billed Separately"}},
                            "tooltip": [
                                {"field": "Addon_Type", "type": "nominal"},
                                {"field": "_addon_billed_separately", "type": "nominal", "title": "Billed Separately"},
                                {"field": "Orders", "type": "quantitative"},
                                {"field": "Addon_Revenue", "type": "quantitative", "title": "Add-on Revenue (INR)"},
                            ]
                        }
                    },
                    use_container_width=True
                )

                type_tbl2 = type_tbl.copy()
                type_tbl2["Addon_Revenue"] = type_tbl2["Addon_Revenue"].apply(format_inr)
                st.dataframe(type_tbl2.rename(columns={"_addon_billed_separately": "Billed Separately"}), use_container_width=True, height=520)

        st.divider()

        st.download_button(
            "Download add-on analysis rows (filtered) as CSV",
            data=ad.to_csv(index=False).encode("utf-8"),
            file_name="addon_products_filtered.csv",
            mime="text/csv",
        )



with tabs[5]:
    st.subheader("Webinars – Registration → Attendance → Conversion")

    st.caption("Upload webinar session summary and (optional) webinar-attendee level data to analyze webinar funnel and sales rep ownership.")

    c_u1, c_u2 = st.columns(2)
    with c_u1:
        session_file = st.file_uploader("Upload Webinar Session Summary CSV", type=["csv"], key="webinar_sessions")
        st.caption("Expected columns: Session_ID, Session_Date, Day, Registered, Attended, Converted")
    with c_u2:
        attendee_file = st.file_uploader("Upload Webinar Attendee/Lead CSV (optional)", type=["csv"], key="webinar_attendees")
        st.caption("Expected columns (best-effort): Session_ID, Session_Date, Session_Day, Webinar_Attended, Hot_Lead_Flag, Converted, Sales_Rep (optional)")

    if session_file is None:
        st.info("Upload the webinar session summary CSV to begin.")
    else:
        try:
            sess = pd.read_csv(session_file)
        except Exception:
            st.error("Could not read the session CSV. Please ensure it's a valid CSV.")
            st.stop()

        # Normalize / parse
        for col in ["Session_ID", "Day"]:
            if col in sess.columns:
                sess[col] = sess[col].astype(str).str.strip()

        if "Session_Date" in sess.columns:
            sess["Session_Date"] = pd.to_datetime(sess["Session_Date"].astype(str).str.slice(0,10), errors="coerce")
        else:
            st.warning("Session_Date missing — charts by date will be limited.")

        for col in ["Registered", "Attended", "Converted"]:
            if col in sess.columns:
                sess[col] = pd.to_numeric(sess[col], errors="coerce").fillna(0)
            else:
                st.error(f"Missing required column in session file: {col}")
                st.stop()

        # Derived rates
        sess["Attend_Rate_%"] = np.where(sess["Registered"] > 0, (sess["Attended"] / sess["Registered"]) * 100, 0.0)
        sess["Conv_from_Attended_%"] = np.where(sess["Attended"] > 0, (sess["Converted"] / sess["Attended"]) * 100, 0.0)
        sess["Conv_from_Registered_%"] = np.where(sess["Registered"] > 0, (sess["Converted"] / sess["Registered"]) * 100, 0.0)

        # Sort by date if available
        if "Session_Date" in sess.columns:
            sess = sess.sort_values("Session_Date")

        # Summary metrics
        tot_reg = float(sess["Registered"].sum())
        tot_att = float(sess["Attended"].sum())
        tot_conv = float(sess["Converted"].sum())
        attend_rate = (tot_att / tot_reg) * 100 if tot_reg > 0 else 0.0
        conv_rate_att = (tot_conv / tot_att) * 100 if tot_att > 0 else 0.0
        conv_rate_reg = (tot_conv / tot_reg) * 100 if tot_reg > 0 else 0.0

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Sessions", f"{len(sess):,}")
        k2.metric("Registered", f"{tot_reg:,.0f}")
        k3.metric("Attended", f"{tot_att:,.0f}")
        k4.metric("Converted", f"{tot_conv:,.0f}")
        k5.metric("Conv% (Reg→Conv)", f"{conv_rate_reg:.1f}%")

        st.caption(f"Attendance Rate = {attend_rate:.1f}% | Conversion Rate (Att→Conv) = {conv_rate_att:.1f}%")

        st.divider()

        # 1) Funnel trend over time
        st.subheader("1) Webinar Funnel Trend (by session date)")
        if "Session_Date" in sess.columns and sess["Session_Date"].notna().any():
            trend = sess[["Session_Date", "Registered", "Attended", "Converted"]].copy()
            trend["Session_Date"] = trend["Session_Date"].dt.date.astype(str)

            trend_melt = trend.melt(id_vars=["Session_Date"], var_name="Stage", value_name="Count")
            st.vega_lite_chart(
                trend_melt,
                {
                    "width": "container",
                    "height": 380,
                    "mark": {"type": "line", "point": True},
                    "encoding": {
                        "x": {"field": "Session_Date", "type": "ordinal", "axis": {"title": "Session Date", "labelAngle": -45}},
                        "y": {"field": "Count", "type": "quantitative", "axis": {"title": "Count"}},
                        "color": {"field": "Stage", "type": "nominal", "legend": {"title": "Stage"}},
                        "tooltip": [{"field": "Session_Date", "type": "ordinal"}, {"field": "Stage", "type": "nominal"}, {"field": "Count", "type": "quantitative"}]
                    }
                },
                use_container_width=True
            )
        else:
            st.info("Session_Date is missing/unparseable; showing session table only.")

        st.divider()

        # 2) Day-wise performance
        st.subheader("2) Day-wise Webinar Performance")
        if "Day" in sess.columns:
            day_tbl = (
                sess.groupby("Day")[["Registered", "Attended", "Converted"]]
                .sum()
                .reset_index()
            )
            day_tbl["Attend_Rate_%"] = np.where(day_tbl["Registered"] > 0, (day_tbl["Attended"] / day_tbl["Registered"]) * 100, 0.0)
            day_tbl["Conv_from_Attended_%"] = np.where(day_tbl["Attended"] > 0, (day_tbl["Converted"] / day_tbl["Attended"]) * 100, 0.0)
            day_tbl["Conv_from_Registered_%"] = np.where(day_tbl["Registered"] > 0, (day_tbl["Converted"] / day_tbl["Registered"]) * 100, 0.0)

            st.vega_lite_chart(
                day_tbl,
                {
                    "width": "container",
                    "height": 380,
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "Day", "type": "nominal", "sort": "-y", "axis": {"title": "Day"}},
                        "y": {"field": "Conv_from_Registered_%", "type": "quantitative", "axis": {"title": "Conversion % (Reg→Conv)"}},
                        "tooltip": [
                            {"field": "Day", "type": "nominal"},
                            {"field": "Registered", "type": "quantitative"},
                            {"field": "Attended", "type": "quantitative"},
                            {"field": "Converted", "type": "quantitative"},
                            {"field": "Attend_Rate_%", "type": "quantitative", "title": "Attend %"},
                            {"field": "Conv_from_Attended_%", "type": "quantitative", "title": "Conv% (Att→Conv)"},
                            {"field": "Conv_from_Registered_%", "type": "quantitative", "title": "Conv% (Reg→Conv)"},
                        ]
                    }
                },
                use_container_width=True
            )
            st.dataframe(
                day_tbl.round({"Attend_Rate_%":1, "Conv_from_Attended_%":1, "Conv_from_Registered_%":1}),
                use_container_width=True,
                height=260
            )
        else:
            st.info("Day column missing in the session file.")

        st.divider()

        # 3) Session leaderboard (best sessions)
        st.subheader("3) Best Sessions (by conversions and conversion rate)")
        sess_tbl = sess.copy()
        # show top by converted
        top_by_conv = sess_tbl.sort_values("Converted", ascending=False).head(10)
        st.caption("Top 10 sessions by conversions")
        st.dataframe(
            top_by_conv[[c for c in ["Session_ID","Session_Date","Day","Registered","Attended","Converted","Attend_Rate_%","Conv_from_Registered_%","Conv_from_Attended_%"] if c in top_by_conv.columns]].round(2),
            use_container_width=True
        )

        st.divider()

        # 4) Sales Rep ownership (optional attendee file)
        st.subheader("4) Sales Rep Ownership (optional)")
        if attendee_file is None:
            st.info("Upload the attendee/lead file to compute Sales Rep-wise webinar conversions (and Hot Lead flag impact).")
        else:
            try:
                att = pd.read_csv(attendee_file)
            except Exception:
                st.error("Could not read attendee CSV. Please ensure it's a valid CSV.")
                st.stop()

            # Normalize booleans
            def to_bool(s):
                v = s.astype(str).str.strip().str.lower()
                return v.isin(["1","true","yes","y","hot","contacted","made","connected","attended"])

            for col in ["Webinar_Attended", "Hot_Lead_Flag", "Converted"]:
                if col in att.columns:
                    att["_b_" + col] = to_bool(att[col])
                else:
                    st.warning(f"Column '{col}' not found in attendee file; related insights may be skipped.")

            # Basic session join (optional)
            if "Session_ID" in att.columns and "Session_ID" in sess.columns:
                # keep as string
                att["Session_ID"] = att["Session_ID"].astype(str).str.strip()
                sess_id_list = set(sess["Session_ID"].astype(str).str.strip())
                att = att[att["Session_ID"].isin(sess_id_list)].copy()

            # Sales rep wise
            if "Sales_Rep" not in att.columns:
                st.warning("Sales_Rep not found in attendee file. Upload a file with Sales_Rep to get rep-wise insights.")
            else:
                att["Sales_Rep"] = att["Sales_Rep"].astype(str).str.strip()

                # Use Converted boolean if present else try fallback from df enrollment logic
                if "_b_Converted" in att.columns:
                    conv_flag = att["_b_Converted"].astype(int)
                elif "Converted" in att.columns:
                    conv_flag = pd.to_numeric(att["Converted"], errors="coerce").fillna(0).astype(int)
                else:
                    conv_flag = pd.Series(np.zeros(len(att), dtype=int))

                att["_conv"] = conv_flag

                rep_tbl = (
                    att.groupby("Sales_Rep")
                    .agg(
                        Leads=("Sales_Rep", "size"),
                        Conversions=("_conv", "sum"),
                        Hot_Leads=("_b_Hot_Lead_Flag", "sum") if "_b_Hot_Lead_Flag" in att.columns else ("Sales_Rep","size")
                    )
                    .reset_index()
                )
                rep_tbl["Conv_%"] = np.where(rep_tbl["Leads"] > 0, rep_tbl["Conversions"] / rep_tbl["Leads"] * 100, 0.0)
                rep_tbl["Conv_%"] = rep_tbl["Conv_%"].round(1)
                rep_tbl = rep_tbl.sort_values(["Conv_%","Conversions"], ascending=False)

                st.dataframe(rep_tbl, use_container_width=True, height=520)

                st.vega_lite_chart(
                    rep_tbl.head(25),
                    {
                        "width": "container",
                        "height": 420,
                        "mark": "bar",
                        "encoding": {
                            "x": {"field": "Sales_Rep", "type": "nominal", "sort": "-y", "axis": {"labelAngle": -45}},
                            "y": {"field": "Conv_%", "type": "quantitative", "axis": {"title": "Conversion %"}},
                            "tooltip": [
                                {"field": "Sales_Rep", "type": "nominal"},
                                {"field": "Leads", "type": "quantitative"},
                                {"field": "Conversions", "type": "quantitative"},
                                {"field": "Conv_%", "type": "quantitative", "title": "Conv %"},
                            ]
                        }
                    },
                    use_container_width=True
                )

            # Hot lead impact (if present)
            if "_b_Hot_Lead_Flag" in att.columns and "_conv" in att.columns:
                st.subheader("Hot Lead Flag impact on conversion")
                hot_tbl = (
                    att.groupby("_b_Hot_Lead_Flag")["_conv"]
                    .agg(Leads="size", Conversions="sum")
                    .reset_index()
                )
                hot_tbl["_b_Hot_Lead_Flag"] = hot_tbl["_b_Hot_Lead_Flag"].map({True:"Hot", False:"Not Hot"})
                hot_tbl["Conv_%"] = np.where(hot_tbl["Leads"]>0, hot_tbl["Conversions"]/hot_tbl["Leads"]*100, 0.0).round(1)
                st.dataframe(hot_tbl.rename(columns={"_b_Hot_Lead_Flag":"Hot Lead"}), use_container_width=True)

        st.download_button(
            "Download cleaned session table as CSV",
            data=sess.to_csv(index=False).encode("utf-8"),
            file_name="webinar_sessions_cleaned.csv",
            mime="text/csv",
        )



with tabs[6]:
    st.subheader("Input – Total Dials (Call Effort → Conversion Predictability)")

    if "Total_Dials" not in df.columns:
        st.warning("Column 'Total_Dials' not found in your CSV.")
    else:
        inp = df.copy()

        # Period filter (same as Review)
        if "Month" in inp.columns:
            inp["Month"] = clean_month(inp["Month"])

        if month_choice.startswith("Consolidated"):
            if "Month" in inp.columns:
                inp = inp[inp["Month"].isin(["July", "August", "September"])].copy()
            period_label = "Q3 (Jul–Sep) Consolidated"
        else:
            if "Month" in inp.columns:
                inp = inp[inp["Month"] == month_choice].copy()
            period_label = month_choice

        st.caption(f"Period filter applied: **{period_label}**")

        # Numeric
        inp["Total_Dials"] = pd.to_numeric(inp["Total_Dials"], errors="coerce")
        inp = inp.dropna(subset=["Total_Dials"]).copy()
        inp["Total_Dials"] = inp["Total_Dials"].clip(lower=0)

        # Conversion flag (reuse app's enrollment logic) → treat as 0/1
        conv_flag = enroll_series_fn(inp)
        conv_flag = pd.to_numeric(conv_flag, errors="coerce").fillna(0)
        inp["_converted"] = (conv_flag > 0).astype(int)

        # Basic stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows in view", f"{len(inp):,}")
        c2.metric("Avg Total Dials", f"{inp['Total_Dials'].mean():.1f}")
        c3.metric("Median Total Dials", f"{inp['Total_Dials'].median():.0f}")
        c4.metric("Conversion rate (proxy)", f"{(inp['_converted'].mean()*100):.0f}%")

        st.divider()

        # 1) Sales Rep-wise trend of Total Dials (over time)
        st.subheader("1) Sales Rep-wise Total Dials Trend")
        if "Sales_Rep" not in inp.columns:
            st.info("Sales_Rep column not available, so rep-wise trend cannot be shown.")
        else:
            date_col = pick_best_date_col(inp)
            if not date_col:
                st.info("No usable date column found for time trend.")
            else:
                gran = st.selectbox("Trend granularity", ["Day", "Week", "Month"], key="dials_gran")
                inp2 = inp.copy()
                inp2[date_col] = pd.to_datetime(inp2[date_col], errors="coerce")
                inp2 = inp2.dropna(subset=[date_col]).copy()
                inp2["period"] = add_period_col(inp2, date_col, gran)

                rep_counts = inp2.groupby("Sales_Rep").size().sort_values(ascending=False)
                max_n = int(max(3, min(25, len(rep_counts))))
                top_reps = st.slider("Show top N reps (by rows)", 3, max_n, value=min(8, max_n), key="dials_top_reps")
                reps_to_show = rep_counts.head(top_reps).index.astype(str).tolist()

                trend = (
                    inp2[inp2["Sales_Rep"].astype(str).isin(reps_to_show)]
                    .groupby(["period", "Sales_Rep"])["Total_Dials"]
                    .sum()
                    .reset_index()
                )

                st.vega_lite_chart(
                    trend,
                    {
                        "width": "container",
                        "height": 420,
                        "mark": {"type": "line", "point": True},
                        "encoding": {
                            "x": {"field": "period", "type": "ordinal", "axis": {"title": f"{gran}"}},
                            "y": {"field": "Total_Dials", "type": "quantitative", "axis": {"title": "Total Dials (sum)"}},
                            "color": {"field": "Sales_Rep", "type": "nominal", "legend": {"title": "Sales Rep"}},
                            "tooltip": [
                                {"field": "period", "type": "ordinal"},
                                {"field": "Sales_Rep", "type": "nominal"},
                                {"field": "Total_Dials", "type": "quantitative", "title": "Dials"},
                            ]
                        }
                    },
                    use_container_width=True
                )

        st.divider()

        # 2) Sweet spot: conversion likelihood vs Total_Dials
        st.subheader("2) Sweet Spot: Conversion likelihood vs Total Dials")
        st.caption("Conversion proxy = your enrollments logic. This shows whether more dials improves conversion probability.")

        bins = [0, 1, 3, 5, 8, 12, 20, 50, np.inf]
        labels = ["0", "1-2", "3-4", "5-7", "8-11", "12-19", "20-49", "50+"]
        inp["_dial_bin"] = pd.cut(inp["Total_Dials"], bins=bins, labels=labels, include_lowest=True, right=False)

        tbl = (
            inp.groupby("_dial_bin")["_converted"]
            .agg(Leads="size", Conversions="sum", Conv_Rate="mean")
            .reset_index()
        )
        tbl["Conv_Rate_%"] = (tbl["Conv_Rate"] * 100).round(1)
        tbl = tbl.drop(columns=["Conv_Rate"])

        min_leads = st.slider("Min leads per dial bucket", 1, int(max(1, tbl["Leads"].max())), value=min(50, int(max(1, tbl["Leads"].max()))), key="dial_min_leads")
        tbl_f = tbl[tbl["Leads"] >= min_leads].copy()

        sweet = None
        if not tbl_f.empty:
            sweet = tbl_f.sort_values(["Conv_Rate_%", "Conversions", "Leads"], ascending=False).head(1).iloc[0]

        st.vega_lite_chart(
            tbl,
            {
                "width": "container",
                "height": 360,
                "mark": {"type": "line", "point": True},
                "encoding": {
                    "x": {"field": "_dial_bin", "type": "ordinal", "axis": {"title": "Total Dials bucket"}},
                    "y": {"field": "Conv_Rate_%", "type": "quantitative", "axis": {"title": "Conversion rate (%)"}},
                    "tooltip": [
                        {"field": "_dial_bin", "type": "ordinal", "title": "Bucket"},
                        {"field": "Leads", "type": "quantitative"},
                        {"field": "Conversions", "type": "quantitative"},
                        {"field": "Conv_Rate_%", "type": "quantitative", "title": "Conv %"},
                    ]
                }
            },
            use_container_width=True
        )

        st.dataframe(tbl.rename(columns={"_dial_bin": "Total_Dials Bucket"}), use_container_width=True, height=320)

        if sweet is not None:
            st.success(
                f"Sweet spot (meeting min-leads threshold): Bucket **{sweet['_dial_bin']}** with **{sweet['Conv_Rate_%']}%** conversion "
                f"({int(sweet['Conversions'])}/{int(sweet['Leads'])})."
            )
        else:
            st.info("No bucket meets the min-leads threshold to declare a sweet spot.")

        st.divider()

        # 3) Sales Rep defaulters (low dials + low conversion)
        st.subheader("3) Sales Rep Defaulters (Low Dials + Low Conversion)")
        if "Sales_Rep" not in inp.columns:
            st.info("Sales_Rep column not available.")
        else:
            rep = (
                inp.groupby("Sales_Rep")
                .agg(Leads=("Sales_Rep","size"), Avg_Dials=("Total_Dials","mean"), Median_Dials=("Total_Dials","median"), Conversions=("_converted","sum"))
                .reset_index()
            )
            rep["Conv_%"] = np.where(rep["Leads"]>0, rep["Conversions"]/rep["Leads"]*100, 0.0)
            rep["Avg_Dials"] = rep["Avg_Dials"].round(1)
            rep["Median_Dials"] = rep["Median_Dials"].round(0).astype(int)
            rep["Conv_%"] = rep["Conv_%"].round(1)

            min_rep_leads = st.slider("Min leads per rep", 1, int(max(1, rep["Leads"].max())), value=min(50, int(max(1, rep["Leads"].max()))), key="min_rep_leads_dials")
            rep_f = rep[rep["Leads"] >= min_rep_leads].copy()
            rep_f = rep_f.sort_values(["Conv_%","Avg_Dials"], ascending=[True, True])

            st.dataframe(rep_f.head(30), use_container_width=True, height=520)

            st.vega_lite_chart(
                rep_f.head(25),
                {
                    "width": "container",
                    "height": 420,
                    "mark": {"type": "circle", "opacity": 0.85},
                    "encoding": {
                        "x": {"field": "Avg_Dials", "type": "quantitative", "axis": {"title": "Avg Total Dials"}},
                        "y": {"field": "Conv_%", "type": "quantitative", "axis": {"title": "Conversion %"}},
                        "size": {"field": "Leads", "type": "quantitative", "legend": {"title": "Leads (rows)"}},
                        "tooltip": [
                            {"field": "Sales_Rep", "type": "nominal"},
                            {"field": "Leads", "type": "quantitative"},
                            {"field": "Avg_Dials", "type": "quantitative"},
                            {"field": "Median_Dials", "type": "quantitative"},
                            {"field": "Conversions", "type": "quantitative"},
                            {"field": "Conv_%", "type": "quantitative", "title": "Conv %"},
                        ]
                    }
                },
                use_container_width=True
            )

        st.divider()

        

        # 4) Correlation: Do more dials relate to higher conversion probability?
        st.divider()
        st.subheader("4) Correlation: Total Dials vs Conversion (Does effort matter?)")
        st.caption("Conversion is treated as 0/1 using the same enrollments logic. Correlation is directional only (not causation).")

        # Pearson (point-biserial) and Spearman correlations
        try:
            pearson = float(inp["Total_Dials"].corr(inp["_converted"], method="pearson"))
        except Exception:
            pearson = np.nan
        try:
            spearman = float(inp["Total_Dials"].corr(inp["_converted"], method="spearman"))
        except Exception:
            spearman = np.nan

        c1, c2 = st.columns(2)
        c1.metric("Pearson corr (Dials vs Converted)", "NA" if pd.isna(pearson) else f"{pearson:.3f}")
        c2.metric("Spearman corr (rank-based)", "NA" if pd.isna(spearman) else f"{spearman:.3f}")

        # Binned conversion curve (more granular than earlier buckets)
        max_d = int(np.nanmax(inp["Total_Dials"].values)) if len(inp) else 0
        max_d = min(max_d, 200)  # keep chart readable
        tmpc = inp.copy()
        tmpc["_d_clip"] = tmpc["Total_Dials"].clip(upper=max_d)
        tmpc["_d_bin"] = pd.cut(tmpc["_d_clip"], bins=min(30, max(5, int(max_d/2)+1)), include_lowest=True)

        corr_tbl = (
            tmpc.groupby("_d_bin")["_converted"]
            .agg(Leads="size", Conversions="sum", ConvRate="mean")
            .reset_index()
        )
        corr_tbl["ConvRate_%"] = (corr_tbl["ConvRate"] * 100).round(1)
        corr_tbl = corr_tbl.drop(columns=["ConvRate"])

        st.vega_lite_chart(
            corr_tbl,
            {
                "width": "container",
                "height": 340,
                "mark": {"type": "line", "point": True},
                "encoding": {
                    "x": {"field": "_d_bin", "type": "ordinal", "axis": {"title": "Total Dials (binned)"}},
                    "y": {"field": "ConvRate_%", "type": "quantitative", "axis": {"title": "Conversion rate (%)"}},
                    "tooltip": [
                        {"field": "_d_bin", "type": "ordinal", "title": "Bin"},
                        {"field": "Leads", "type": "quantitative"},
                        {"field": "Conversions", "type": "quantitative"},
                        {"field": "ConvRate_%", "type": "quantitative", "title": "Conv %"},
                    ]
                }
            },
            use_container_width=True
        )

        st.dataframe(corr_tbl.rename(columns={"_d_bin": "Total_Dials Bin"}), use_container_width=True, height=280)

        # 5) For high-revenue reps: Avg daily revenue vs Avg daily dials
        st.divider()
        st.subheader("5) High-Revenue Reps: Average Daily Revenue vs Daily Dials")
        st.caption("This helps see if top revenue reps are also consistently putting in dials (normalized by active days).")

        if "Sales_Rep" not in inp.columns:
            st.info("Sales_Rep not available, so rep-wise daily view cannot be computed.")
        else:
            # Need revenue and a date column for active days
            if "Total_Value_INR" not in inp.columns:
                st.warning("Total_Value_INR not found in CSV, so revenue comparison cannot be computed here.")
            else:
                date_col2 = pick_best_date_col(inp)
                if not date_col2:
                    st.warning("No usable date column found to compute active days.")
                else:
                    tmp = inp.copy()
                    tmp["Total_Value_INR"] = pd.to_numeric(tmp["Total_Value_INR"], errors="coerce").fillna(0)
                    tmp[date_col2] = pd.to_datetime(tmp[date_col2], errors="coerce")
                    tmp = tmp.dropna(subset=[date_col2]).copy()
                    tmp["_day"] = tmp[date_col2].dt.date.astype(str)

                    rep_daily = (
                        tmp.groupby("Sales_Rep")
                        .agg(
                            Active_Days=("_day", "nunique"),
                            Total_Dials=("Total_Dials", "sum"),
                            Total_Revenue=("Total_Value_INR", "sum"),
                            Conversions=("_converted", "sum"),
                            Leads=("Sales_Rep", "size"),
                        )
                        .reset_index()
                    )
                    rep_daily["Avg_Dials_per_Day"] = np.where(rep_daily["Active_Days"] > 0, rep_daily["Total_Dials"] / rep_daily["Active_Days"], 0.0)
                    rep_daily["Avg_Revenue_per_Day"] = np.where(rep_daily["Active_Days"] > 0, rep_daily["Total_Revenue"] / rep_daily["Active_Days"], 0.0)
                    rep_daily["Conv_%"] = np.where(rep_daily["Leads"] > 0, rep_daily["Conversions"] / rep_daily["Leads"] * 100, 0.0)

                    rep_daily["Avg_Dials_per_Day"] = rep_daily["Avg_Dials_per_Day"].round(1)
                    rep_daily["Avg_Revenue_per_Day"] = rep_daily["Avg_Revenue_per_Day"].round(0)
                    rep_daily["Conv_%"] = rep_daily["Conv_%"].round(1)

                    min_rep_leads2 = st.slider("Min leads per rep (daily view)", 1, int(max(1, rep_daily["Leads"].max())), value=min(50, int(max(1, rep_daily["Leads"].max()))), key="min_rep_leads_daily")
                    rep_daily_f = rep_daily[rep_daily["Leads"] >= min_rep_leads2].copy()

                    # Focus on top revenue reps
                    top_k = st.slider("Show top K reps by total revenue", 5, min(50, max(5, len(rep_daily_f))), value=min(15, max(5, len(rep_daily_f))), key="top_k_rev")
                    rep_top = rep_daily_f.sort_values("Total_Revenue", ascending=False).head(top_k).copy()

                    # Table
                    tbl_show = rep_top.copy()
                    tbl_show["Total_Revenue"] = tbl_show["Total_Revenue"].apply(format_inr)
                    tbl_show["Avg_Revenue_per_Day"] = tbl_show["Avg_Revenue_per_Day"].apply(lambda v: format_inr(v))
                    st.dataframe(
                        tbl_show[[
                            "Sales_Rep","Active_Days","Leads","Conversions","Conv_%","Total_Revenue",
                            "Avg_Dials_per_Day","Avg_Revenue_per_Day"
                        ]].rename(columns={
                            "Conv_%":"Conv (%)",
                            "Avg_Dials_per_Day":"Avg Dials/Day",
                            "Avg_Revenue_per_Day":"Avg Revenue/Day"
                        }),
                        use_container_width=True,
                        height=420
                    )

                    # Scatter: x=dials/day, y=revenue/day, size=total revenue, color=conv%
                    st.vega_lite_chart(
                        rep_top,
                        {
                            "width": "container",
                            "height": 460,
                            "mark": {"type": "circle", "opacity": 0.85},
                            "encoding": {
                                "x": {"field": "Avg_Dials_per_Day", "type": "quantitative", "axis": {"title": "Avg Dials per Active Day"}},
                                "y": {"field": "Avg_Revenue_per_Day", "type": "quantitative", "axis": {"title": "Avg Revenue per Active Day (INR)"}},
                                "size": {"field": "Total_Revenue", "type": "quantitative", "legend": {"title": "Total Revenue (size)"}},
                                "color": {"field": "Conv_%", "type": "quantitative", "legend": {"title": "Conversion % (color)"}},
                                "tooltip": [
                                    {"field": "Sales_Rep", "type": "nominal"},
                                    {"field": "Active_Days", "type": "quantitative"},
                                    {"field": "Total_Dials", "type": "quantitative", "title": "Total Dials"},
                                    {"field": "Total_Revenue", "type": "quantitative", "title": "Total Revenue (INR)"},
                                    {"field": "Avg_Dials_per_Day", "type": "quantitative", "title": "Avg Dials/Day"},
                                    {"field": "Avg_Revenue_per_Day", "type": "quantitative", "title": "Avg Revenue/Day"},
                                    {"field": "Conv_%", "type": "quantitative", "title": "Conv %"},
                                ]
                            }
                        },
                        use_container_width=True
                    )
st.download_button(
            "Download Total Dials (filtered) as CSV",
            data=inp.to_csv(index=False).encode("utf-8"),
            file_name="total_dials_filtered.csv",
            mime="text/csv",
        )



with tabs[7]:
    st.subheader("Marketing Performance – CPL Efficiency & ROI Proxy")

    required_cols = ["Utm_Campaign", "CPL_INR"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.warning(f"Missing required columns for Marketing Performance: {missing_cols}")
    else:
        mp = df.copy()

        # Apply same period filter as Review tab
        if "Month" in mp.columns:
            mp["Month"] = clean_month(mp["Month"])

        if month_choice.startswith("Consolidated"):
            if "Month" in mp.columns:
                mp = mp[mp["Month"].isin(["July", "August", "September"])].copy()
            period_label = "Q3 (Jul–Sep) Consolidated"
        else:
            if "Month" in mp.columns:
                mp = mp[mp["Month"] == month_choice].copy()
            period_label = month_choice

        st.caption(f"Period filter applied: **{period_label}**")

        mp["Utm_Campaign"] = mp["Utm_Campaign"].astype(str).fillna("Unknown").str.strip()
        mp["CPL_INR"] = pd.to_numeric(mp["CPL_INR"], errors="coerce")

        # Identify lead/inquiry id for lead counting
        inquiry_id_col = None
        for cand in ["LeadID", "LeadId", "Lead_ID", "leadid", "lead_id", "InquiryID", "Inquiry_Id", "ProspectID", "ProspectId"]:
            if cand in mp.columns:
                inquiry_id_col = cand
                break

        # Revenue / Enrollments (reuse app logic)
        if "Total_Value_INR" in mp.columns:
            mp["Total_Value_INR"] = pd.to_numeric(mp["Total_Value_INR"], errors="coerce").fillna(0)
        else:
            mp["Total_Value_INR"] = 0.0

        enr = enroll_series_fn(mp)
        mp["_enroll"] = pd.to_numeric(enr, errors="coerce").fillna(0)

        # Lead counts by campaign
        if inquiry_id_col:
            lead_counts = mp.groupby("Utm_Campaign")[inquiry_id_col].nunique(dropna=True)
            mp["_lead_unit"] = mp[inquiry_id_col]
        else:
            lead_counts = mp.groupby("Utm_Campaign").size()
            mp["_lead_unit"] = 1

        # Campaign-level CPL: take median CPL across rows (more robust than mean)
        cpl_med = mp.groupby("Utm_Campaign")["CPL_INR"].median()

        # Spend proxy = CPL * Leads
        camp = pd.DataFrame({
            "Leads": lead_counts,
            "CPL_INR_Median": cpl_med
        }).reset_index().rename(columns={"index": "Utm_Campaign"})

        # Revenue and enrollments per campaign
        rev = mp.groupby("Utm_Campaign")["Total_Value_INR"].sum()
        enr2 = mp.groupby("Utm_Campaign")["_enroll"].sum()

        camp = camp.merge(rev.reset_index().rename(columns={"Total_Value_INR": "Revenue_INR"}), on="Utm_Campaign", how="left")
        camp = camp.merge(enr2.reset_index().rename(columns={"_enroll": "Enrollments"}), on="Utm_Campaign", how="left")

        camp["Revenue_INR"] = pd.to_numeric(camp["Revenue_INR"], errors="coerce").fillna(0)
        camp["Enrollments"] = pd.to_numeric(camp["Enrollments"], errors="coerce").fillna(0)
        camp["CPL_INR_Median"] = pd.to_numeric(camp["CPL_INR_Median"], errors="coerce").fillna(0)

        camp["Spend_INR_Proxy"] = camp["Leads"] * camp["CPL_INR_Median"]
        camp["Revenue_per_Lead"] = np.where(camp["Leads"] > 0, camp["Revenue_INR"] / camp["Leads"], 0.0)
        camp["Enroll_per_Lead_%"] = np.where(camp["Leads"] > 0, (camp["Enrollments"] / camp["Leads"]) * 100, 0.0)
        camp["CAC_Proxy_INR"] = np.where(camp["Enrollments"] > 0, camp["Spend_INR_Proxy"] / camp["Enrollments"], np.nan)

        # ROI proxies
        camp["ROAS_Proxy"] = np.where(camp["Spend_INR_Proxy"] > 0, camp["Revenue_INR"] / camp["Spend_INR_Proxy"], np.nan)  # revenue per rupee spent
        camp["ROI_Proxy_%"] = np.where(camp["Spend_INR_Proxy"] > 0, ((camp["Revenue_INR"] - camp["Spend_INR_Proxy"]) / camp["Spend_INR_Proxy"]) * 100, np.nan)

        # Summary metrics
        tot_rev = float(camp["Revenue_INR"].sum())
        tot_spend = float(camp["Spend_INR_Proxy"].sum())
        tot_enr = float(camp["Enrollments"].sum())
        roas_all = (tot_rev / tot_spend) if tot_spend > 0 else np.nan
        cpe_all = (tot_spend / tot_enr) if tot_enr > 0 else np.nan

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Revenue (INR)", format_inr(tot_rev))
        k2.metric("Total Spend (proxy)", format_inr(tot_spend))
        k3.metric("ROAS (proxy)", "NA" if pd.isna(roas_all) else f"{roas_all:.2f}×")
        k4.metric("Cost per Enrollment (proxy)", "NA" if pd.isna(cpe_all) else format_inr(cpe_all))

        st.caption("Spend is a proxy: Spend = Median(CPL) × Leads (unique LeadID if available). ROAS/ROI are directional, not audited finance.")

        st.divider()

        # 1) Campaign leaderboard by ROAS
        st.subheader("1) Campaign Efficiency Leaderboard (ROAS / Cost per Enrollment)")
        min_leads = st.slider("Min leads per campaign (filter noise)", 1, int(max(1, camp["Leads"].max())), value=min(50, int(max(1, camp["Leads"].max()))), key="mp_min_leads")
        camp_f = camp[camp["Leads"] >= min_leads].copy()

        show_sort = st.radio("Sort campaigns by", ["ROAS (proxy) – higher is better", "Cost per Enrollment (proxy) – lower is better", "Revenue – higher is better"], horizontal=True)
        if show_sort.startswith("ROAS"):
            camp_f = camp_f.sort_values("ROAS_Proxy", ascending=False)
        elif show_sort.startswith("Cost"):
            camp_f = camp_f.sort_values("CAC_Proxy_INR", ascending=True)
        else:
            camp_f = camp_f.sort_values("Revenue_INR", ascending=False)

        tbl = camp_f.copy()
        tbl["Revenue_INR"] = tbl["Revenue_INR"].apply(format_inr)
        tbl["Spend_INR_Proxy"] = tbl["Spend_INR_Proxy"].apply(format_inr)
        tbl["CAC_Proxy_INR"] = tbl["CAC_Proxy_INR"].apply(lambda v: "NA" if pd.isna(v) else format_inr(v))
        tbl["ROAS_Proxy"] = tbl["ROAS_Proxy"].round(2)
        tbl["ROI_Proxy_%"] = tbl["ROI_Proxy_%"].round(0)
        tbl["Revenue_per_Lead"] = tbl["Revenue_per_Lead"].round(0)
        tbl["Enroll_per_Lead_%"] = tbl["Enroll_per_Lead_%"].round(2)

        st.dataframe(
            tbl[[
                "Utm_Campaign","Leads","CPL_INR_Median","Revenue_INR","Spend_INR_Proxy",
                "ROAS_Proxy","ROI_Proxy_%","Enrollments","Enroll_per_Lead_%","CAC_Proxy_INR","Revenue_per_Lead"
            ]].rename(columns={
                "CPL_INR_Median":"Median CPL (INR)",
                "ROAS_Proxy":"ROAS×",
                "ROI_Proxy_%":"ROI%",
                "Enroll_per_Lead_%":"Enroll/Lead (%)",
                "CAC_Proxy_INR":"Cost/Enroll (INR)",
                "Revenue_per_Lead":"Rev/Lead (INR)"
            }),
            use_container_width=True,
            height=520
        )

        st.divider()

        # 2) Scatter: CPL vs Revenue per Lead (bubble = Leads, color = ROAS)
        st.subheader("2) CPL vs Revenue/Lead (Bubble = Lead Volume)")
        plot_df = camp_f.replace([np.inf, -np.inf], np.nan).dropna(subset=["CPL_INR_Median","Revenue_per_Lead"])

        st.vega_lite_chart(
            plot_df,
            {
                "width": "container",
                "height": 460,
                "mark": {"type": "circle", "opacity": 0.85},
                "encoding": {
                    "x": {"field": "CPL_INR_Median", "type": "quantitative", "axis": {"title": "Median CPL (INR)"} },
                    "y": {"field": "Revenue_per_Lead", "type": "quantitative", "axis": {"title": "Revenue per Lead (INR)"} },
                    "size": {"field": "Leads", "type": "quantitative", "legend": {"title": "Leads (size)"} },
                    "color": {"field": "ROAS_Proxy", "type": "quantitative", "legend": {"title": "ROAS× (color)"} },
                    "tooltip": [
                        {"field": "Utm_Campaign", "type": "nominal", "title": "Campaign"},
                        {"field": "Leads", "type": "quantitative"},
                        {"field": "CPL_INR_Median", "type": "quantitative", "title": "Median CPL"},
                        {"field": "Revenue_INR", "type": "quantitative", "title": "Revenue (INR)"},
                        {"field": "Spend_INR_Proxy", "type": "quantitative", "title": "Spend proxy (INR)"},
                        {"field": "ROAS_Proxy", "type": "quantitative", "title": "ROAS×"},
                        {"field": "CAC_Proxy_INR", "type": "quantitative", "title": "Cost/Enroll (INR)"},
                    ]
                }
            },
            use_container_width=True
        )

        st.divider()

        # 3) Average revenue by campaign (trend not required): show bar
        st.subheader("3) Revenue & Enrollment mix by Campaign")
        
# Safe Top-N selector (bulletproof using selectbox instead of slider)
_max_n = int(max(1, len(camp_f)))
options = list(range(1, _max_n + 1))
default_n = min(15, _max_n)
top_n = st.selectbox(
    "Show top N campaigns by revenue",
    options,
    index=options.index(default_n) if default_n in options else len(options) - 1,
    key="mp_top_n"
)
        melt = top[["Utm_Campaign","Revenue_INR","Enrollments","Leads"]].melt(id_vars=["Utm_Campaign"], var_name="Metric", value_name="Value")
        st.vega_lite_chart(
            melt,
            {
                "width": "container",
                "height": 420,
                "mark": "bar",
                "encoding": {
                    "x": {"field": "Utm_Campaign", "type": "nominal", "axis": {"labelAngle": -45, "title": "Campaign"}},
                    "y": {"field": "Value", "type": "quantitative", "axis": {"title": "Value"}},
                    "color": {"field": "Metric", "type": "nominal", "legend": {"title": "Metric"}},
                    "tooltip": [
                        {"field": "Utm_Campaign", "type": "nominal"},
                        {"field": "Metric", "type": "nominal"},
                        {"field": "Value", "type": "quantitative"},
                    ]
                }
            },
            use_container_width=True
        )

        st.divider()

        # 4) Suggested cuts: ROI buckets
        st.subheader("4) ROI Buckets (quick cut)")
        bucket = pd.cut(
            camp_f["ROAS_Proxy"],
            bins=[-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf],
            labels=["<0.5×", "0.5–1.0×", "1.0–1.5×", "1.5–2.0×", "2.0×+"],
        )
        btbl = camp_f.assign(ROAS_Bucket=bucket).groupby("ROAS_Bucket").agg(
            Campaigns=("Utm_Campaign","nunique"),
            Leads=("Leads","sum"),
            Revenue=("Revenue_INR","sum"),
            Spend=("Spend_INR_Proxy","sum"),
            Enrollments=("Enrollments","sum")
        ).reset_index()
        btbl["ROAS_Proxy"] = np.where(btbl["Spend"]>0, btbl["Revenue"]/btbl["Spend"], np.nan)
        btbl["ROAS_Proxy"] = btbl["ROAS_Proxy"].round(2)
        btbl["Revenue"] = btbl["Revenue"].apply(format_inr)
        btbl["Spend"] = btbl["Spend"].apply(format_inr)

        st.dataframe(btbl.rename(columns={"ROAS_Proxy":"ROAS×"}), use_container_width=True)

        st.download_button(
            "Download marketing performance table (campaign-level) as CSV",
            data=camp.to_csv(index=False).encode("utf-8"),
            file_name="marketing_performance_campaign.csv",
            mime="text/csv",
        )


with tabs[8]:
    st.subheader("Business Predictability in October (Regression)")
    st.caption(
        "Goal: predict October performance using (1) projected October leads (M0) and (2) leads from the preceding 3 months (M-1..M-3). "
        "This is a planning / predictability view. Accuracy is indicative (depends on data coverage)."
    )

    try:
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    except Exception:
        st.error("scikit-learn is required for this tab. Add 'scikit-learn' to requirements.txt and redeploy.")
        st.stop()

    needed = ["Month", "Utm_Campaign"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns for predictability: {missing}")
        st.stop()

    lead_id_col = None
    for cand in ["LeadID", "LeadId", "Lead_ID", "leadid", "lead_id", "InquiryID", "Inquiry_Id", "ProspectID", "ProspectId"]:
        if cand in df.columns:
            lead_id_col = cand
            break

    base = df.copy()
    base["Month"] = clean_month(base["Month"])

    hist_months = ["July", "August", "September"]
    hist = base[base["Month"].isin(hist_months)].copy()
    if hist.empty:
        st.info("No July–September rows found. This tab expects at least Jul–Sep history to forecast October.")
        st.stop()

    hist["Utm_Campaign"] = hist["Utm_Campaign"].astype(str).fillna("Unknown").str.strip()

    if "Sales_Rep" in hist.columns:
        hist["Sales_Rep"] = hist["Sales_Rep"].astype(str).fillna("Unknown").str.strip()
    else:
        hist["Sales_Rep"] = "All"

    hist["Total_Value_INR"] = pd.to_numeric(hist.get("Total_Value_INR", 0), errors="coerce").fillna(0)
    hist["_enroll"] = pd.to_numeric(enroll_series_fn(hist), errors="coerce").fillna(0)

    if lead_id_col:
        hist["_lead_unit"] = hist[lead_id_col]
    else:
        hist["_lead_unit"] = 1

    grp_cols = ["Month", "Sales_Rep", "Utm_Campaign"]
    agg = (
        hist.groupby(grp_cols)
        .agg(
            Leads=("_lead_unit", "nunique" if lead_id_col else "sum"),
            Enrollments=("_enroll", "sum"),
            Revenue=("Total_Value_INR", "sum"),
        )
        .reset_index()
    )

    month_order = {"July": 1, "August": 2, "September": 3, "October": 4}
    agg["m_idx"] = agg["Month"].map(month_order).astype(int)

    agg = agg.sort_values(["Sales_Rep", "Utm_Campaign", "m_idx"])
    for lag in [1, 2, 3]:
        agg[f"Leads_m{lag}"] = agg.groupby(["Sales_Rep", "Utm_Campaign"])["Leads"].shift(lag).fillna(0)

    agg["Leads_prev3"] = agg[["Leads_m1", "Leads_m2", "Leads_m3"]].sum(axis=1)
    agg["Leads_M0"] = agg["Leads"]

    features = ["Leads_M0", "Leads_prev3"]
    target_choice = st.radio("Predict target", ["Revenue", "Enrollments"], horizontal=True)
    y_col = "Revenue" if target_choice == "Revenue" else "Enrollments"

    model_name = st.selectbox(
        "Regression model",
        ["Linear Regression", "Ridge", "Random Forest", "Gradient Boosting"],
        index=0
    )

    def make_model(name: str):
        if name == "Linear Regression":
            return LinearRegression()
        if name == "Ridge":
            return Ridge(alpha=1.0, random_state=42)
        if name == "Random Forest":
            return RandomForestRegressor(n_estimators=300, random_state=42)
        return GradientBoostingRegressor(random_state=42)

    model = make_model(model_name)

    X = agg[features].astype(float).values
    y = agg[y_col].astype(float).values
    groups = agg["m_idx"].values

    uniq_m = np.unique(groups)
    if len(uniq_m) < 2:
        yhat = model.fit(X, y).predict(X)
        mae = mean_absolute_error(y, yhat)
        r2 = r2_score(y, yhat) if len(np.unique(y)) > 1 else np.nan
    else:
        cv = GroupKFold(n_splits=len(uniq_m))
        yhat = cross_val_predict(model, X, y, cv=cv, groups=groups)
        mae = mean_absolute_error(y, yhat)
        r2 = r2_score(y, yhat) if len(np.unique(y)) > 1 else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", model_name)
    c2.metric("MAE (CV)", f"{mae:,.2f}")
    c3.metric("R² (CV)", "NA" if pd.isna(r2) else f"{r2:.3f}")

    st.caption("MAE/R² computed on Jul–Sep at campaign×rep granularity using month-grouped CV. Treat as indicative.")

    st.divider()

    seg_hist = agg[agg["m_idx"].isin([1,2,3])].copy()

    def forecast_m0(group):
        g = group.sort_values("m_idx")
        xs = g["m_idx"].values
        ys = g["Leads"].values
        if len(g) >= 2:
            x_mean = xs.mean()
            y_mean = ys.mean()
            denom = ((xs - x_mean)**2).sum()
            if denom == 0:
                return float(ys[-1])
            slope = ((xs - x_mean) * (ys - y_mean)).sum() / denom
            pred = y_mean + slope * (4 - x_mean)
            return float(max(0, pred))
        return float(ys[-1]) if len(g) else 0.0

    m0_pred = (
        seg_hist.groupby(["Sales_Rep","Utm_Campaign"], as_index=False)
        .apply(lambda g: pd.Series({"Leads_M0": forecast_m0(g)}))
        .reset_index(drop=True)
    )

    piv = seg_hist.pivot_table(index=["Sales_Rep","Utm_Campaign"], columns="m_idx", values="Leads", aggfunc="sum").fillna(0)
    piv.columns = [f"m{int(c)}" for c in piv.columns]
    for c in ["m1","m2","m3"]:
        if c not in piv.columns:
            piv[c] = 0.0
    piv = piv.reset_index()
    # Prev3 for Oct = Sep+Aug+Jul leads
    piv["Leads_prev3"] = piv.get("m3",0) + piv.get("m2",0) + piv.get("m1",0)

    oct_X = piv.merge(m0_pred, on=["Sales_Rep","Utm_Campaign"], how="outer").fillna(0)

    model.fit(X, y)
    oct_pred = model.predict(oct_X[features].astype(float).values)
    oct_X["Predicted_" + y_col] = np.maximum(0, oct_pred)

    overall_pred = float(oct_X["Predicted_" + y_col].sum())
    st.subheader("October Forecast (Overall)")
    if y_col == "Revenue":
        st.metric("Predicted October Revenue (INR)", format_inr(overall_pred))
    else:
        st.metric("Predicted October Enrollments", f"{overall_pred:,.0f}")

    st.divider()

    st.subheader("1) October forecast by Sales Rep × Campaign")
    show_tbl = oct_X.copy()
    show_tbl["Leads_M0"] = show_tbl["Leads_M0"].round(0).astype(int)
    show_tbl["Leads_prev3"] = show_tbl["Leads_prev3"].round(0).astype(int)

    if y_col == "Revenue":
        show_tbl["Predicted_" + y_col] = show_tbl["Predicted_" + y_col].round(0)
        show_tbl["Predicted_fmt"] = show_tbl["Predicted_" + y_col].apply(format_inr)
        st.dataframe(show_tbl[["Sales_Rep","Utm_Campaign","Leads_M0","Leads_prev3","Predicted_fmt"]].rename(columns={"Leads_M0":"Oct Leads (forecast)","Leads_prev3":"Prev 3M Leads","Predicted_fmt":"Predicted Revenue"}), use_container_width=True, height=520)
    else:
        show_tbl["Predicted_" + y_col] = show_tbl["Predicted_" + y_col].round(0).astype(int)
        st.dataframe(show_tbl[["Sales_Rep","Utm_Campaign","Leads_M0","Leads_prev3","Predicted_" + y_col]].rename(columns={"Leads_M0":"Oct Leads (forecast)","Leads_prev3":"Prev 3M Leads","Predicted_" + y_col:"Predicted Enrollments"}), use_container_width=True, height=520)

    st.divider()

    st.subheader("2) Sales Rep-wise predicted contribution (October)")
    rep_pred = oct_X.groupby("Sales_Rep")["Predicted_" + y_col].sum().reset_index().sort_values("Predicted_" + y_col, ascending=False)

    st.vega_lite_chart(
        rep_pred.head(25),
        {
            "width": "container",
            "height": 420,
            "mark": "bar",
            "encoding": {
                "x": {"field": "Sales_Rep", "type": "nominal", "sort": "-y", "axis": {"labelAngle": -45, "title": "Sales Rep"}},
                "y": {"field": "Predicted_" + y_col, "type": "quantitative", "axis": {"title": f"Predicted {y_col}"}},
                "tooltip": [
                    {"field": "Sales_Rep", "type": "nominal"},
                    {"field": "Predicted_" + y_col, "type": "quantitative", "title": f"Predicted {y_col}"},
                ]
            }
        },
        use_container_width=True
    )
    st.dataframe(rep_pred, use_container_width=True, height=360)

    st.divider()

    st.subheader("3) Campaign-wise October forecast (Leads & Predicted outcome)")
    camp_pred = oct_X.groupby("Utm_Campaign").agg(
        Oct_Leads=("Leads_M0","sum"),
        Prev3_Leads=("Leads_prev3","sum"),
        Predicted=("Predicted_" + y_col,"sum")
    ).reset_index().sort_values("Oct_Leads", ascending=False)

    st.vega_lite_chart(
        camp_pred.head(25),
        {
            "width": "container",
            "height": 420,
            "mark": "bar",
            "encoding": {
                "x": {"field": "Utm_Campaign", "type": "nominal", "sort": "-y", "axis": {"labelAngle": -45, "title": "Campaign"}},
                "y": {"field": "Oct_Leads", "type": "quantitative", "axis": {"title": "Forecast October Leads"}},
                "tooltip": [
                    {"field": "Utm_Campaign", "type": "nominal"},
                    {"field": "Oct_Leads", "type": "quantitative"},
                    {"field": "Predicted", "type": "quantitative", "title": f"Predicted {y_col}"},
                ]
            }
        },
        use_container_width=True
    )
    st.dataframe(camp_pred, use_container_width=True, height=420)

    st.divider()

    st.subheader("4) Historical relationship: Leads vs Outcome (sanity)")
    rel = agg.copy()
    rel["Leads_total"] = rel["Leads_M0"] + rel["Leads_prev3"]
    st.vega_lite_chart(
        rel,
        {
            "width": "container",
            "height": 420,
            "mark": {"type": "circle", "opacity": 0.75},
            "encoding": {
                "x": {"field": "Leads_total", "type": "quantitative", "axis": {"title": "Leads (M0 + Prev3)"}},
                "y": {"field": y_col, "type": "quantitative", "axis": {"title": y_col}},
                "color": {"field": "Month", "type": "nominal", "legend": {"title": "Month"}},
                "tooltip": [
                    {"field": "Month", "type": "nominal"},
                    {"field": "Sales_Rep", "type": "nominal"},
                    {"field": "Utm_Campaign", "type": "nominal"},
                    {"field": "Leads_M0", "type": "quantitative", "title": "Leads M0"},
                    {"field": "Leads_prev3", "type": "quantitative", "title": "Prev3 Leads"},
                    {"field": y_col, "type": "quantitative", "title": y_col},
                ]
            }
        },
        use_container_width=True
    )

    st.download_button(
        "Download October forecast table as CSV",
        data=oct_X.to_csv(index=False).encode("utf-8"),
        file_name="october_predictability_forecast.csv",
        mime="text/csv",
    )