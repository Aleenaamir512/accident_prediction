
import pandas as pd

# Load dataset
df = pd.read_excel(
    r"D:\accident_prediction\karachi_traffic_feature_engineered_final_v2.xlsx",
    sheet_name="Data"
)

# Convert target into ordered numbers
risk_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["Risk_Numeric"] = df["Accident Risk"].map(risk_mapping)

# Select numerical columns
numeric_features = df.select_dtypes(
    include="number"
).columns.tolist()

numeric_features.remove("Risk_Numeric")

# Calculate Spearman correlation
results = []

for feature in numeric_features:
    correlation = df[feature].corr(
        df["Risk_Numeric"],
        method="spearman"
    )

    results.append({
        "Feature": feature,
        "Correlation": correlation
    })

# Create results table
corr_df = pd.DataFrame(results)

# Sort by absolute correlation strength
corr_df["Absolute Correlation"] = (
    corr_df["Correlation"].abs()
)

corr_df = corr_df.sort_values(
    "Absolute Correlation",
    ascending=False
)

print("\nX-Y CORRELATION RESULTS")
print(corr_df.to_string(index=False))
