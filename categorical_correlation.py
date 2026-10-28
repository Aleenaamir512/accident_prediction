
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
df = pd.read_excel(
    r"D:\accident_prediction\karachi_traffic_feature_engineered_final_v2.xlsx",
    sheet_name="Data"
)

# Sort by time
df["Date-Time"] = pd.to_datetime(df["Date-Time"])
df = df.sort_values("Date-Time").reset_index(drop=True)

# Use training data for feature investigation
train_end = int(len(df) * 0.55)
train_df = df.iloc[:train_end].copy()

# Categorical features
categorical_features = [
    "Area",
    "Road Type",
    "Road Condition",
    "Weather",
    "Time of Day",
    "Day Name",
    "Month",
    "Time Period",
    "Congestion Level",
    "Historical Weather",
    "ML Weather"
]

def cramers_v(x, y):

    table = pd.crosstab(x, y)

    if min(table.shape) < 2:
        return np.nan

    chi2, p, dof, expected = chi2_contingency(
        table
    )

    n = table.to_numpy().sum()

    v = np.sqrt(
        chi2 / (n * (min(table.shape) - 1))
    )

    return v

# Calculate X-Y association
results = []

for feature in categorical_features:

    score = cramers_v(
        train_df[feature],
        train_df["Accident Risk"]
    )

    results.append({
        "Feature": feature,
        "Cramers V": score
    })

# Results table
result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    "Cramers V",
    ascending=False
)

print("\nCATEGORICAL X-Y ASSOCIATION")
print(result_df.to_string(index=False))
