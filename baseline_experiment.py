
import pandas as pd

# Step 1: Load dataset
df = pd.read_excel(
    r"D:\accident_prediction\karachi_traffic_feature_engineered_final_v2.xlsx",
    sheet_name="Data"
)

# Step 2: Check rows and columns
print("Dataset shape:", df.shape)

# Step 3: Check all feature names
print("\nAvailable columns:")
print(df.columns.tolist())

# Step 4: Check target distribution
print("\nAccident Risk distribution:")
print(df["Accident Risk"].value_counts())

# Step 5: Check missing values
print("\nMissing values:")
print(df.isnull().sum())
