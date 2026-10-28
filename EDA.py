from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# FINAL THESIS EDA
# Predictive Analysis of Traffic Accident Risk in Karachi
# ============================================================

SCRIPT_FOLDER = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_FOLDER / "karachi_traffic_feature_engineered_final_v2.xlsx"
OUTPUT_FOLDER = SCRIPT_FOLDER / "EDA_Output"
OUTPUT_FOLDER.mkdir(exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
df = pd.read_excel(INPUT_FILE, sheet_name="Data")
df["Date-Time"] = pd.to_datetime(df["Date-Time"], errors="coerce")

risk_order = ["Low", "Medium", "High"]
congestion_order = ["Low", "Moderate", "High", "Severe"]
time_order = [
    "Late Night",
    "Early Morning",
    "Morning Rush",
    "Daytime",
    "Evening Build-up",
    "Evening Rush",
    "Late Evening",
]

# ------------------------------------------------------------
# 2. DATA QUALITY / OVERVIEW
# ------------------------------------------------------------
summary_lines = []
summary_lines.append("FINAL EDA SUMMARY")
summary_lines.append("=" * 70)
summary_lines.append(f"Rows: {len(df)}")
summary_lines.append(f"Columns: {df.shape[1]}")
summary_lines.append(f"Duplicate rows: {df.duplicated().sum()}")
summary_lines.append(f"Total missing values: {int(df.isna().sum().sum())}")
summary_lines.append("")

summary_lines.append("ACCIDENT RISK DISTRIBUTION")
summary_lines.append("-" * 70)
risk_counts = df["Accident Risk"].value_counts().reindex(risk_order, fill_value=0)
for label, count in risk_counts.items():
    summary_lines.append(
        f"{label}: {count} ({count / len(df) * 100:.1f}%)"
    )
summary_lines.append("")

# ------------------------------------------------------------
# 3. NUMERIC SUMMARY BY TARGET
# ------------------------------------------------------------
numeric_features = [
    "Vehicle Count",
    "Ride Activity",
    "Bus Activity",
    "Traffic Load per Lane",
    "Traffic Load Ratio",
    "Rainfall",
    "Visibility",
    "Estimated Average Speed",
]

summary_lines.append("NUMERIC MEANS BY ACCIDENT RISK")
summary_lines.append("-" * 70)
for feature in numeric_features:
    summary_lines.append(feature)
    means = df.groupby("Accident Risk")[feature].mean().reindex(risk_order)
    for label, value in means.items():
        summary_lines.append(f"  {label}: {value:.2f}")
summary_lines.append("")

# ------------------------------------------------------------
# 4. HIGH-RISK SHARE BY CATEGORICAL VARIABLES
# ------------------------------------------------------------
categorical_features = [
    "Area",
    "Road Type",
    "Road Condition",
    "Time Period",
    "Congestion Level",
    "ML Weather",
    "Traffic Signal",
    "Number of Lanes",
]

summary_lines.append("HIGH-RISK SHARE BY IMPORTANT CATEGORICAL FEATURES")
summary_lines.append("-" * 70)
for feature in categorical_features:
    table = pd.crosstab(df[feature], df["Accident Risk"])
    if "High" not in table.columns:
        table["High"] = 0
    table["Total"] = table.sum(axis=1)
    table["High %"] = table["High"] / table["Total"] * 100
    summary_lines.append(feature)
    for idx, row in table.sort_values("High %", ascending=False).iterrows():
        summary_lines.append(
            f"  {idx}: n={int(row['Total'])}, High={row['High %']:.1f}%"
        )
summary_lines.append("")

# ------------------------------------------------------------
# 5. CORRELATION / REDUNDANCY CHECK
# ------------------------------------------------------------
correlation_features = [
    "Vehicle Count",
    "Ride Activity",
    "Bus Activity",
    "Pickup Hour",
    "Is Weekend",
    "Hour Sin",
    "Hour Cos",
    "Rush Hour",
    "Construction",
    "Number of Lanes",
    "Near Intersection",
    "Traffic Signal",
    "Traffic Load per Lane",
    "Reference Traffic Load",
    "Traffic Load Ratio",
    "Rainfall",
    "Visibility",
    "Is Rainy",
    "Is Foggy",
    "Estimated Average Speed",
]

corr = df[correlation_features].corr(numeric_only=True)
summary_lines.append("HIGH NUMERIC CORRELATIONS (ABS >= 0.75)")
summary_lines.append("-" * 70)
found_pair = False
for i, left in enumerate(corr.columns):
    for j in range(i + 1, len(corr.columns)):
        right = corr.columns[j]
        value = corr.iloc[i, j]
        if pd.notna(value) and abs(value) >= 0.75:
            found_pair = True
            summary_lines.append(f"{left} <-> {right}: {value:.3f}")
if not found_pair:
    summary_lines.append("No pairs found at |r| >= 0.75")
summary_lines.append("")

# ------------------------------------------------------------
# 6. IQR OUTLIER COUNTS
# ------------------------------------------------------------
outlier_features = [
    "Vehicle Count",
    "Ride Activity",
    "Bus Activity",
    "Traffic Load per Lane",
    "Traffic Load Ratio",
    "Rainfall",
    "Visibility",
    "Estimated Average Speed",
]

summary_lines.append("IQR OUTLIER COUNTS")
summary_lines.append("-" * 70)
for feature in outlier_features:
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((df[feature] < lower) | (df[feature] > upper)).sum()
    summary_lines.append(
        f"{feature}: {int(outliers)} outliers (bounds {lower:.2f} to {upper:.2f})"
    )
summary_lines.append("")

# Save text summary.
summary_file = OUTPUT_FOLDER / "eda_summary.txt"
summary_file.write_text("\n".join(summary_lines), encoding="utf-8")

print("\n".join(summary_lines))

# ------------------------------------------------------------
# CHART HELPERS
# ------------------------------------------------------------
def finish_plot(filename, title=None, xlabel=None, ylabel=None, rotation=0):
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if rotation:
        plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / filename, dpi=200, bbox_inches="tight")
    plt.close()


def risk_percentage_table(feature, order=None):
    tab = pd.crosstab(df[feature], df["Accident Risk"], normalize="index") * 100
    tab = tab.reindex(columns=risk_order, fill_value=0)
    if order is not None:
        existing = [x for x in order if x in tab.index]
        remaining = [x for x in tab.index if x not in existing]
        tab = tab.reindex(existing + remaining)
    return tab

# ------------------------------------------------------------
# 7. FIGURES
# ------------------------------------------------------------

# 01 Target distribution
plt.figure(figsize=(7, 5))
risk_counts.plot(kind="bar")
finish_plot(
    "01_accident_risk_distribution.png",
    "Accident Risk Distribution",
    "Accident Risk",
    "Number of Records",
)

# 02 Road type vs risk
plt.figure(figsize=(10, 6))
risk_percentage_table("Road Type").plot(kind="bar", stacked=True, ax=plt.gca())
finish_plot(
    "02_risk_by_road_type.png",
    "Accident Risk Percentage by Road Type",
    "Road Type",
    "Percentage of Records",
    35,
)

# 03 Road condition vs risk
plt.figure(figsize=(9, 6))
risk_percentage_table("Road Condition").plot(kind="bar", stacked=True, ax=plt.gca())
finish_plot(
    "03_risk_by_road_condition.png",
    "Accident Risk Percentage by Road Condition",
    "Road Condition",
    "Percentage of Records",
    30,
)

# 04 Time period vs risk
plt.figure(figsize=(11, 6))
risk_percentage_table("Time Period", time_order).plot(
    kind="bar", stacked=True, ax=plt.gca()
)
finish_plot(
    "04_risk_by_time_period.png",
    "Accident Risk Percentage by Time Period",
    "Time Period",
    "Percentage of Records",
    35,
)

# 05 Congestion vs risk
plt.figure(figsize=(8, 6))
risk_percentage_table("Congestion Level", congestion_order).plot(
    kind="bar", stacked=True, ax=plt.gca()
)
finish_plot(
    "05_risk_by_congestion_level.png",
    "Accident Risk Percentage by Congestion Level",
    "Congestion Level",
    "Percentage of Records",
)

# 06 Vehicle Count by target
plt.figure(figsize=(8, 6))
data = [df.loc[df["Accident Risk"] == label, "Vehicle Count"].dropna() for label in risk_order]
plt.boxplot(data, labels=risk_order, showfliers=False)
finish_plot(
    "06_vehicle_count_by_risk.png",
    "Vehicle Count by Accident Risk",
    "Accident Risk",
    "Vehicle Count",
)

# 07 Traffic Load Ratio by target
plt.figure(figsize=(8, 6))
data = [df.loc[df["Accident Risk"] == label, "Traffic Load Ratio"].dropna() for label in risk_order]
plt.boxplot(data, labels=risk_order, showfliers=False)
finish_plot(
    "07_traffic_load_ratio_by_risk.png",
    "Traffic Load Ratio by Accident Risk",
    "Accident Risk",
    "Traffic Load Ratio",
)

# 08 Estimated speed by target
plt.figure(figsize=(8, 6))
data = [df.loc[df["Accident Risk"] == label, "Estimated Average Speed"].dropna() for label in risk_order]
plt.boxplot(data, labels=risk_order, showfliers=False)
finish_plot(
    "08_estimated_speed_by_risk.png",
    "Estimated Average Speed by Accident Risk",
    "Accident Risk",
    "Estimated Average Speed",
)

# 09 Hourly high-risk percentage
hour_table = pd.crosstab(df["Pickup Hour"], df["Accident Risk"])
if "High" not in hour_table.columns:
    hour_table["High"] = 0
hour_high = hour_table["High"] / hour_table.sum(axis=1) * 100
plt.figure(figsize=(10, 5))
plt.plot(hour_high.index, hour_high.values, marker="o")
plt.xticks(range(24))
plt.grid(True, alpha=0.3)
finish_plot(
    "09_hourly_high_risk_percentage.png",
    "High Accident-Risk Percentage by Hour",
    "Hour of Day",
    "High-Risk Records (%)",
)

# 10 Hourly mean traffic
hourly_traffic = df.groupby("Pickup Hour")["Vehicle Count"].mean()
plt.figure(figsize=(10, 5))
plt.plot(hourly_traffic.index, hourly_traffic.values, marker="o")
plt.xticks(range(24))
plt.grid(True, alpha=0.3)
finish_plot(
    "10_hourly_average_vehicle_count.png",
    "Average Vehicle Count by Hour",
    "Hour of Day",
    "Average Vehicle Count",
)

# 11 Numeric correlation matrix
plt.figure(figsize=(14, 11))
image = plt.imshow(corr.values, aspect="auto")
plt.colorbar(image, label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Numeric Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "11_numeric_correlation_matrix.png", dpi=200, bbox_inches="tight")
plt.close()

# 12 High-risk percentage by area
area_table = pd.crosstab(df["Area"], df["Accident Risk"])
if "High" not in area_table.columns:
    area_table["High"] = 0
area_high = (area_table["High"] / area_table.sum(axis=1) * 100).sort_values(ascending=False)
plt.figure(figsize=(11, 6))
area_high.plot(kind="bar")
finish_plot(
    "12_high_risk_percentage_by_area.png",
    "High Accident-Risk Percentage by Area",
    "Area",
    "High-Risk Records (%)",
    45,
)

# 13 Historical ML weather vs risk
plt.figure(figsize=(8, 6))
risk_percentage_table("ML Weather").plot(kind="bar", stacked=True, ax=plt.gca())
finish_plot(
    "13_risk_by_ml_weather.png",
    "Accident Risk Percentage by Historical Weather",
    "ML Weather",
    "Percentage of Records",
)

print("\nEDA completed successfully.")
print("Summary:", summary_file)
print("Figures folder:", OUTPUT_FOLDER)