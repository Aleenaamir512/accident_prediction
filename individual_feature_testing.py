import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load dataset
df = pd.read_excel(
    r"D:\accident_prediction\karachi_traffic_feature_engineered_final_v2.xlsx",
    sheet_name="Data"
)

df["Date-Time"] = pd.to_datetime(df["Date-Time"])
df = df.sort_values("Date-Time").reset_index(drop=True)

# 2. Chronological training and validation
# Use only the first 55% of the dataset
train_end = int(len(df) * 0.44)
valid_end = int(len(df) * 0.55)

train = df.iloc[:train_end]
valid = df.iloc[train_end:valid_end]

y_train = train["Accident Risk"]
y_valid = valid["Accident Risk"]

# 3. Original feature order (unchanged)
features_order = [
    "Vehicle Count",
    "Road Type",
    "Road Condition",
    "Area",
    "Number of Lanes",
    "Traffic Signal",
    "ML Weather",
    "Hour Sin",
    "Hour Cos",
    "Is Weekend",
    "Rush Hour",
    "Bus Activity",
    "Ride Activity"
]

# Keep the first 9 features fixed; test each extra feature separately.
base_features = features_order[:9]

extra_features = [
    "Traffic Load Ratio",
    "Congestion Level",
    "Rainfall",
    "Visibility",
    "Construction",
    "Near Intersection",
    "Estimated Average Speed"
]


# First 9 features
base_features = features_order[:9]

# Remove Vehicle Count from base
without_vehicle = [
    f for f in base_features
    if f != "Vehicle Count"
]

# Controlled experiments
experiments = [
    ("Base 9 Features", base_features),

    ("Base + Congestion",
     base_features + ["Congestion Level"]),

    ("Base Without Vehicle",
     without_vehicle),

    ("Replace Vehicle with Congestion",
     without_vehicle + ["Congestion Level"])
]

results = []


# 4. Evaluate the base set and one extra feature at a time
for i, (experiment_name, features) in enumerate(experiments, start=1):
    X_train = train[features]
    X_valid = valid[features]

    categorical = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             categorical)
        ],
        remainder="passthrough"
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)

    precision = precision_score(
        y_valid, predictions,
        average="macro", zero_division=0
    )

    recall = recall_score(
        y_valid, predictions,
        average="macro", zero_division=0
    )

    f1 = f1_score(
        y_valid, predictions,
        average="macro", zero_division=0
    )

    accuracy = accuracy_score(y_valid, predictions)

    results.append({
        "Step": i,
        "Feature Added": experiment_name,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
        "Accuracy": accuracy
    })

# 5. Compare every extra feature against the SAME base model
results_df = pd.DataFrame(results)
base_precision = results_df.loc[0, "Macro Precision"]
results_df["Precision Change vs Base"] = (
    results_df["Macro Precision"] - base_precision
)

print("\nINDIVIDUAL FEATURE EXPERIMENT")
print(results_df.round(4).to_string(index=False))

# 6. Save separately so the previous incremental results are preserved
results_df.to_csv(
    r"D:\accident_prediction\individual_feature_results.csv",
    index=False
)

print("\nResults saved successfully!")
