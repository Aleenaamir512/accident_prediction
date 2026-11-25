# ============================================================
# HOUR-WISE HIGH-RISK SHAP ANALYSIS
# Karachi Traffic-Risk Thesis
#
# Purpose:
#   Translate Hour Sin + Hour Cos SHAP effects into actual
#   clock hours (0-23) using the saved Date-Time values.
#
# This script DOES NOT retrain or modify the final model.
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(r"D:\accident_prediction")

OUTPUT_DIR = (
    BASE_DIR
    / "Final_Random_Forest_Output"
)

SHAP_FILE = (
    OUTPUT_DIR
    / "shap_values_original_features_per_observation.csv"
)


# ============================================================
# 2. CHECK INPUT FILE
# ============================================================

print("=" * 75)
print("HOUR-WISE HIGH-RISK SHAP ANALYSIS")
print("=" * 75)

if not SHAP_FILE.exists():

    raise FileNotFoundError(
        "\nSaved SHAP file not found:\n"
        f"{SHAP_FILE}\n\n"
        "Run final_random_forest_interpretation.py first."
    )


# ============================================================
# 3. LOAD SAVED SHAP VALUES
# ============================================================

print(
    "\nLoading saved SHAP values..."
)

df = pd.read_csv(
    SHAP_FILE
)

print(
    "Rows loaded:",
    len(df),
)


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Date-Time",
    "Explained Class",
    "Hour Sin",
    "Hour Cos",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    raise ValueError(
        "\nRequired columns are missing:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing
        )
    )


# ============================================================
# 5. KEEP ONLY SHAP VALUES FOR HIGH-RISK OUTPUT
# ============================================================

high_df = (
    df[
        df["Explained Class"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "high"
    ]
    .copy()
)

if high_df.empty:

    raise ValueError(
        "\nNo rows with "
        "Explained Class = High "
        "were found."
    )


print(
    "High-risk SHAP observations:",
    len(high_df),
)


# ============================================================
# 6. CONVERT DATE-TIME TO ACTUAL CLOCK HOUR
# ============================================================

high_df["Date-Time"] = pd.to_datetime(
    high_df["Date-Time"],
    errors="raise",
)

high_df["Hour"] = (
    high_df["Date-Time"]
    .dt.hour
)


# Friendly hour labels

def hour_label(hour):

    timestamp = pd.Timestamp(
        year=2000,
        month=1,
        day=1,
        hour=int(hour),
    )

    return timestamp.strftime(
        "%I %p"
    ).lstrip("0")


high_df["Hour Label"] = (
    high_df["Hour"]
    .apply(hour_label)
)


# ============================================================
# 7. COMBINE HOUR SIN + HOUR COS SHAP CONTRIBUTIONS
# ============================================================

# Hour Sin and Hour Cos are two mathematical components
# of one underlying feature: time of day.
#
# Their signed SHAP contributions can therefore be added
# for each observation to obtain the total contribution
# of time-of-day toward the High-risk model output.

high_df[
    "Time SHAP Contribution"
] = (
    high_df["Hour Sin"]
    + high_df["Hour Cos"]
)


# Absolute value tells us magnitude only,
# regardless of whether the contribution is
# toward or away from High risk.

high_df[
    "Absolute Time SHAP"
] = np.abs(
    high_df[
        "Time SHAP Contribution"
    ]
)


# ============================================================
# 8. HOURLY SUMMARY
# ============================================================

hourly = (
    high_df
    .groupby(
        "Hour",
        as_index=False,
    )
    .agg(
        Observations=(
            "Time SHAP Contribution",
            "count",
        ),

        Mean_High_Risk_SHAP=(
            "Time SHAP Contribution",
            "mean",
        ),

        Median_High_Risk_SHAP=(
            "Time SHAP Contribution",
            "median",
        ),

        Mean_Absolute_SHAP=(
            "Absolute Time SHAP",
            "mean",
        ),

        SD_High_Risk_SHAP=(
            "Time SHAP Contribution",
            "std",
        ),
    )
)


# ============================================================
# 9. CALCULATE HOW OFTEN TIME PUSHED TOWARD HIGH RISK
# ============================================================

positive_share = (
    high_df
    .assign(
        Positive=(
            high_df[
                "Time SHAP Contribution"
            ]
            > 0
        )
        .groupby(
            "Hour"
        )[
            "Positive"
        ]
        .mean()
        .reset_index(
            name="Positive_SHAP_Share"
        )
)


hourly = hourly.merge(
    positive_share,
    on="Hour",
    how="left",
)


hourly[
    "Positive_SHAP_Percent"
] = (
    hourly[
        "Positive_SHAP_Share"
    ]
    * 100
)


# Add readable hour labels

hourly[
    "Hour Label"
] = (
    hourly["Hour"]
    .apply(hour_label)
)


# Ensure chronological order

hourly = (
    hourly
    .sort_values(
        "Hour"
    )
    .reset_index(
        drop=True
    )
))


# ============================================================
# 10. INTERPRET DIRECTION
# ============================================================

def interpret_direction(value):

    if value > 0.005:
        return "Pushes toward High risk"

    elif value < -0.005:
        return "Pushes away from High risk"

    else:
        return "Near-neutral effect"


hourly[
    "Interpretation"
] = (
    hourly[
        "Mean_High_Risk_SHAP"
    ]
    .apply(
        interpret_direction
    )
)


# ============================================================
# 11. SAVE HOURLY RESULTS
# ============================================================

hourly_output = (
    OUTPUT_DIR
    / "hour_wise_high_risk_shap.csv"
)

hourly.to_csv(
    hourly_output,
    index=False,
)


# Also save observation-level file
# containing actual hour + combined time SHAP.

observation_output = (
    OUTPUT_DIR
    / "high_risk_time_shap_per_observation.csv"
)

high_df[
    [
        "Date-Time",
        "Hour",
        "Hour Label",
        "Actual Risk",
        "Hour Sin",
        "Hour Cos",
        "Time SHAP Contribution",
        "Absolute Time SHAP",
    ]
].to_csv(
    observation_output,
    index=False,
)


# ============================================================
# 12. MAIN HOURLY SHAP PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    hourly["Hour"],
    hourly[
        "Mean_High_Risk_SHAP"
    ],
)

plt.axhline(
    y=0,
    linewidth=1,
)

plt.xticks(
    range(24),
    rotation=45,
)

plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Mean SHAP Contribution Toward High Risk"
)

plt.title(
    "Time-of-Day Effect on High-Risk Predictions\n"
    "Final Random Forest Model"
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "hour_wise_high_risk_shap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 13. MEAN ABSOLUTE SHAP PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    hourly["Hour"],
    hourly[
        "Mean_Absolute_SHAP"
    ],
)

plt.xticks(
    range(24),
    rotation=45,
)

plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Mean Absolute Time SHAP"
)

plt.title(
    "Magnitude of Time-of-Day Influence by Hour\n"
    "Final Random Forest Model"
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "hour_wise_absolute_time_shap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 14. POSITIVE HIGH-RISK CONTRIBUTION PERCENTAGE
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    hourly["Hour"],
    hourly[
        "Positive_SHAP_Percent"
    ],
)

plt.axhline(
    y=50,
    linestyle="--",
    linewidth=1,
)

plt.xticks(
    range(24),
    rotation=45,
)

plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "% Observations with Positive High-Risk SHAP"
)

plt.title(
    "Percentage of Time Effects Pushing Toward High Risk"
)

plt.ylim(
    0,
    100,
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "hour_wise_positive_high_risk_percentage.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 15. TIME-PERIOD ANALYSIS
# ============================================================

def assign_time_period(hour):

    if 0 <= hour <= 4:
        return "00:00-04:59 Overnight"

    elif 5 <= hour <= 6:
        return "05:00-06:59 Early Morning"

    elif 7 <= hour <= 9:
        return "07:00-09:59 Morning Rush"

    elif 10 <= hour <= 15:
        return "10:00-15:59 Daytime"

    elif hour == 16:
        return "16:00-16:59 Pre-Evening"

    elif 17 <= hour <= 20:
        return "17:00-20:59 Evening Peak"

    else:
        return "21:00-23:59 Late Evening"


high_df[
    "Time Period"
] = (
    high_df["Hour"]
    .apply(
        assign_time_period
    )
)


period_order = [
    "00:00-04:59 Overnight",
    "05:00-06:59 Early Morning",
    "07:00-09:59 Morning Rush",
    "10:00-15:59 Daytime",
    "16:00-16:59 Pre-Evening",
    "17:00-20:59 Evening Peak",
    "21:00-23:59 Late Evening",
]


period_summary = (
    high_df
    .groupby(
        "Time Period",
        as_index=False,
    )
    .agg(
        Observations=(
            "Time SHAP Contribution",
            "count",
        ),

        Mean_High_Risk_SHAP=(
            "Time SHAP Contribution",
            "mean",
        ),

        Median_High_Risk_SHAP=(
            "Time SHAP Contribution",
            "median",
        ),

        Mean_Absolute_SHAP=(
            "Absolute Time SHAP",
            "mean",
        ),
    )
)


period_summary[
    "Time Period"
] = pd.Categorical(
    period_summary[
        "Time Period"
    ],
    categories=period_order,
    ordered=True,
)


period_summary = (
    period_summary
    .sort_values(
        "Time Period"
    )
    .reset_index(
        drop=True
    )
)


period_summary.to_csv(
    OUTPUT_DIR
    / "time_period_high_risk_shap.csv",
    index=False,
)


# ============================================================
# 16. TIME-PERIOD PLOT
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.bar(
    period_summary[
        "Time Period"
    ].astype(str),

    period_summary[
        "Mean_High_Risk_SHAP"
    ],
)

plt.axhline(
    0,
    linewidth=1,
)

plt.xlabel(
    "Time Period"
)

plt.ylabel(
    "Mean SHAP Contribution Toward High Risk"
)

plt.title(
    "High-Risk SHAP Contribution by Time Period"
)

plt.xticks(
    rotation=35,
    ha="right",
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "time_period_high_risk_shap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 17. FIND HOURS MOST ASSOCIATED WITH HIGH-RISK OUTPUT
# ============================================================

toward_high = (
    hourly
    .sort_values(
        "Mean_High_Risk_SHAP",
        ascending=False,
    )
)


away_from_high = (
    hourly
    .sort_values(
        "Mean_High_Risk_SHAP",
        ascending=True,
    )
)


# ============================================================
# 18. PRINT HOURLY TABLE
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "HOURLY HIGH-RISK SHAP RESULTS"
)

print(
    "=" * 75
)


display_columns = [
    "Hour",
    "Hour Label",
    "Mean_High_Risk_SHAP",
    "Mean_Absolute_SHAP",
    "Positive_SHAP_Percent",
]


print(
    hourly[
        display_columns
    ]
    .round(
        {
            "Mean_High_Risk_SHAP": 4,
            "Mean_Absolute_SHAP": 4,
            "Positive_SHAP_Percent": 1,
        }
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 19. PRINT HOURS PUSHING MOST TOWARD HIGH RISK
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "HOURS PUSHING MOST TOWARD HIGH RISK"
)

print(
    "=" * 75
)


for rank, (_, row) in enumerate(
    toward_high.head(5).iterrows(),
    start=1,
):

    print(
        f"{rank}. "
        f"{row['Hour Label']} "
        f"(Hour {int(row['Hour']):02d}) "
        f"| Mean SHAP = "
        f"{row['Mean_High_Risk_SHAP']:.4f} "
        f"| Positive in "
        f"{row['Positive_SHAP_Percent']:.1f}% "
        f"of observations"
    )


# ============================================================
# 20. PRINT HOURS PUSHING MOST AWAY FROM HIGH RISK
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "HOURS PUSHING MOST AWAY FROM HIGH RISK"
)

print(
    "=" * 75
)


for rank, (_, row) in enumerate(
    away_from_high.head(5).iterrows(),
    start=1,
):

    print(
        f"{rank}. "
        f"{row['Hour Label']} "
        f"(Hour {int(row['Hour']):02d}) "
        f"| Mean SHAP = "
        f"{row['Mean_High_Risk_SHAP']:.4f}"
    )


# ============================================================
# 21. PRINT TIME-PERIOD RESULTS
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "TIME-PERIOD HIGH-RISK SHAP RESULTS"
)

print(
    "=" * 75
)


print(
    period_summary[
        [
            "Time Period",
            "Observations",
            "Mean_High_Risk_SHAP",
            "Mean_Absolute_SHAP",
        ]
    ]
    .round(
        {
            "Mean_High_Risk_SHAP": 4,
            "Mean_Absolute_SHAP": 4,
        }
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 22. STRONGEST POSITIVE TIME PERIOD
# ============================================================

best_period = (
    period_summary
    .sort_values(
        "Mean_High_Risk_SHAP",
        ascending=False,
    )
    .iloc[0]
)


print(
    "\nStrongest time period pushing "
    "toward High-risk prediction:"
)

print(
    f"{best_period['Time Period']} "
    f"(Mean SHAP = "
    f"{best_period['Mean_High_Risk_SHAP']:.4f})"
)


# ============================================================
# 23. INTERPRETATION NOTES
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "HOW TO INTERPRET THESE RESULTS"
)

print(
    "=" * 75
)

print(
    """
Mean High-Risk SHAP > 0:
    Time-of-day pushed the Random Forest prediction
    toward the High-risk class.

Mean High-Risk SHAP < 0:
    Time-of-day pushed the prediction away from
    the High-risk class.

Mean Absolute SHAP:
    Measures the strength of the time effect,
    regardless of direction.

Positive SHAP Percent:
    Percentage of observations during that hour
    where the time-of-day variables contributed
    toward a High-risk prediction.

IMPORTANT:
These results explain the behaviour of the fitted
Random Forest model. They do not prove that a specific
hour causes real-world traffic accidents.
"""
)


# ============================================================
# 24. OUTPUT FILES
# ============================================================

print(
    "=" * 75
)

print(
    "FILES SAVED"
)

print(
    "=" * 75
)

print(
    f"""
{OUTPUT_DIR}

hour_wise_high_risk_shap.csv

high_risk_time_shap_per_observation.csv

hour_wise_high_risk_shap.png

hour_wise_absolute_time_shap.png

hour_wise_positive_high_risk_percentage.png

time_period_high_risk_shap.csv

time_period_high_risk_shap.png
"""
)


print(
    "\nHour-wise analysis completed successfully."
))