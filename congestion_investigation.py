
import pandas as pd

df = pd.read_excel(
    r"D:\accident_prediction\karachi_traffic_feature_engineered_final_v2.xlsx",
    sheet_name="Data"
)

df["Date-Time"] = pd.to_datetime(df["Date-Time"])
df = df.sort_values("Date-Time").reset_index(drop=True)

# Same training data as feature experiments
train_end = int(len(df) * 0.44)
train = df.iloc[:train_end].copy()

# 1. Congestion Level vs Accident Risk
print("\nCONGESTION LEVEL VS ACCIDENT RISK")

print(pd.crosstab(
    train["Congestion Level"],
    train["Accident Risk"]
))

# 2. Row-wise percentage distribution
print("\nRISK PERCENTAGE BY CONGESTION LEVEL")

print(pd.crosstab(
    train["Congestion Level"],
    train["Accident Risk"],
    normalize="index"
).round(3) * 100)

# 3. Correlation between two traffic features
print("\nVEHICLE COUNT VS TRAFFIC LOAD RATIO")

print(
    train["Vehicle Count"].corr(
        train["Traffic Load Ratio"],
        method="spearman"
    )
)

# 4. Inspect how Congestion Level was assigned
print("\nTRAFFIC LOAD RATIO BY CONGESTION LEVEL")

print(
    train.groupby("Congestion Level")[
        "Traffic Load Ratio"
    ].agg(["count", "min", "median", "max"])
)
