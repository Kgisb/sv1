
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SKLViz Revenue Dashboard", layout="wide")

def format_inr(x):
    if pd.isna(x):
        return "₹0"
    return f"₹{x:,.0f}"

@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_csv(path, low_memory=False)

def clean_month(s):
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\s+2025|\s+25|\s+2024|\s+2023", "", regex=True)
    return s.str.title()

st.title("Revenue Dashboard – Total_Value_INR")

df = load_data("sv_data.csv")

required_cols = {"Month", "Total_Value_INR"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df["Month"] = clean_month(df["Month"])
df["Total_Value_INR"] = pd.to_numeric(df["Total_Value_INR"], errors="coerce").fillna(0)

has_converted = "Converted" in df.columns
if has_converted:
    df["Converted"] = (
        df["Converted"]
        .astype(str)
        .str.lower()
        .map({"1":1,"true":1,"yes":1,"y":1,"converted":1})
        .fillna(0)
        .astype(int)
    )

with st.sidebar:
    month_choice = st.radio(
        "Select Period",
        ["July", "August", "September", "Consolidated (Q3 / 30-day)"]
    )
    only_converted = False
    if has_converted:
        only_converted = st.checkbox("Only Converted = 1", value=True)

if month_choice.startswith("Consolidated"):
    df_f = df[df["Month"].isin(["July","August","September"])].copy()
else:
    df_f = df[df["Month"] == month_choice].copy()

if has_converted and only_converted:
    df_f = df_f[df_f["Converted"] == 1]

total_revenue = df_f["Total_Value_INR"].sum()

c1, c2 = st.columns(2)
c1.metric("Selected Period", month_choice)
c2.metric("Total Revenue", format_inr(total_revenue))

st.divider()

st.subheader("Revenue by Month")
rev_month = df_f.groupby("Month")["Total_Value_INR"].sum().reset_index()
st.dataframe(rev_month, use_container_width=True)

st.subheader("Download Filtered Data")
st.download_button(
    "Download CSV",
    df_f.to_csv(index=False).encode("utf-8"),
    "filtered_sv_data.csv",
    "text/csv"
)
