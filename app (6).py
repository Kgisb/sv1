
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SKLViz Revenue Dashboard", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def format_inr(x: float) -> str:
    if pd.isna(x):
        return "₹0"
    return f"₹{x:,.0f}"

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def clean_month(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\s+2025|\s+25|\s+2024|\s+2023", "", regex=True)
    return s.str.title()

def auto_parse_dates(d: pd.DataFrame, min_parse_rate: float = 0.60) -> list:
    '''
    Parse columns into datetime if:
    - dtype is object AND
    - column name hints date/time AND
    - at least min_parse_rate values can be parsed
    '''
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
    '''
    Choose a datetime column for trend plotting, preferring common CRM fields.
    '''
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
    return dt_cols[0]  # fallback

# ---------------------------
# App Title
# ---------------------------
st.title("SKLViz – Revenue Dashboard (Total_Value_INR)")

# ---------------------------
# Load data (expects sv_data.csv in repo folder)
# ---------------------------
df = load_data("sv_data.csv")

# Validate required columns
required_cols = {"Month", "Total_Value_INR"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

# Clean / coerce
df["Month"] = clean_month(df["Month"])
df["Total_Value_INR"] = pd.to_numeric(df["Total_Value_INR"], errors="coerce").fillna(0)

# Convert Converted if present
has_converted = "Converted" in df.columns
if has_converted:
    df["Converted"] = (
        df["Converted"]
        .astype(str).str.strip().str.lower()
        .map({"1": 1, "true": 1, "yes": 1, "y": 1, "converted": 1})
        .fillna(0)
        .astype(int)
    )

# Parse datetime columns (if any)
_ = auto_parse_dates(df)

# ---------------------------
# Sidebar "drawer" (collapsible)
# ---------------------------
with st.sidebar:
    st.header("Controls")
    with st.expander("Filters (collapse to slide back)", expanded=True):
        month_choice = st.radio(
            "Select Period",
            options=["July", "August", "September", "Consolidated (Q3 / 30-day)"],
            index=0,
        )

        only_converted = False
        if has_converted:
            only_converted = st.checkbox("Only Converted = 1", value=True)

        show_raw = st.checkbox("Show filtered raw data preview", value=False)

# Apply filters
if month_choice.startswith("Consolidated"):
    df_f = df[df["Month"].isin(["July", "August", "September"])].copy()
    selected_period_label = "Q3 (Jul–Sep) Consolidated"
else:
    df_f = df[df["Month"] == month_choice].copy()
    selected_period_label = month_choice

if has_converted and only_converted:
    df_f = df_f[df_f["Converted"] == 1].copy()

# ---------------------------
# Tabs (start with Review; you can add more later)
# ---------------------------
tabs = st.tabs(["Review"])

with tabs[0]:
    # KPIs
    total_revenue = float(df_f["Total_Value_INR"].sum())
    total_records = int(len(df_f))

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected Period", selected_period_label)
    c2.metric("Total Revenue (INR)", format_inr(total_revenue))
    c3.metric("Records Count", f"{total_records:,}")

    st.divider()

    # Revenue by Month
    st.subheader("Revenue by Month")
    month_tbl = (
        df_f.groupby("Month", dropna=False)["Total_Value_INR"]
        .sum()
        .reset_index()
        .sort_values("Total_Value_INR", ascending=False)
    )
    st.dataframe(month_tbl, use_container_width=True)

    st.divider()

    # Line chart: Revenue trend by best date column (if available)
    st.subheader("Revenue Trend (Line Chart)")
    date_col = pick_best_date_col(df_f)

    if date_col is None:
        st.info("No usable date/time column found for trend plotting. If your CSV has a date column, ensure its name contains 'date', 'time', 'created', etc.")
    else:
        trend = df_f.dropna(subset=[date_col]).copy()
        if not trend.empty:
            trend["day"] = trend[date_col].dt.date
            trend_tbl = trend.groupby("day")["Total_Value_INR"].sum().reset_index().sort_values("day")
            trend_tbl = trend_tbl.set_index("day")
            st.line_chart(trend_tbl)  # Streamlit native line chart
            st.caption(f"Trend based on: {date_col}")
        else:
            st.info("Date column exists, but there are no non-null values after filtering to plot a trend.")

    st.divider()

    if show_raw:
        st.subheader("Filtered Data Preview (Top 200 rows)")
        st.dataframe(df_f.head(200), use_container_width=True)

    # Download filtered data
    st.subheader("Download Filtered Data")
    csv_bytes = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered rows as CSV",
        data=csv_bytes,
        file_name="filtered_sv_data.csv",
        mime="text/csv",
    )
