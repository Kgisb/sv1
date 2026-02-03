
SKLViz Revenue Streamlit App

FILES:
- app.py              -> Streamlit application (tabs + collapsible sidebar)
- sv_data.csv         -> Place your data file here
- requirements.txt    -> Python dependencies

HOW TO RUN:
1) pip install -r requirements.txt
2) streamlit run app.py

FEATURES (Review tab):
- Collapsible sidebar "drawer" (Filters expander)
- Month selector: July / August / September / Consolidated (Q3 / 30-day)
- Total Revenue calculation using Total_Value_INR
- Optional Converted filter (if column exists)
- Revenue by Month table
- Revenue Trend line chart (auto-picks a datetime column if present)
- Download filtered dataset

NOTES:
- For the line chart, the app auto-detects datetime columns whose name contains: date/time/created/updated.
  If your CSV has a date column with a different name, rename it to include "date" (e.g., Lead_Created_Date).
