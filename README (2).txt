
SKLViz Sales Review Streamlit App

FILES:
- app.py              -> Streamlit application (tabs + collapsible sidebar)
- sv_data.csv         -> Place your data file here
- requirements.txt    -> Python dependencies

HOW TO RUN:
1) pip install -r requirements.txt
2) streamlit run app.py

FEATURES (Review tab):
- Collapsible sidebar "drawer" (Filters expander)
- Metric selector: Total Revenue OR Total Enrollments
- Month selector: July / August / September / Consolidated (Q3 / 30-day)
- Trend line chart with granularity selector (Day / Week / Month)
- When granularity = Day: Weekday pie chart (% split for Mon..Sun)
- Download filtered dataset

ENROLLMENT LOGIC (auto-detected):
1) If a column like Total_Enrollments / Enrollments / Enrollment_Count exists -> SUM(column)
2) Else if Converted exists -> COUNT(Converted=1)
3) Else if Payment Received Date exists -> COUNT(non-null date)
4) Else -> row count fallback

NOTES:
- For charts, the app auto-detects datetime columns whose name contains: date/time/created/updated.
  If your CSV has a date column with a different name, rename it to include "date"
  (e.g., Lead_Created_Date).
