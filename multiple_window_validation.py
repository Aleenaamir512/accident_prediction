from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

BASE = Path(__file__).resolve().parent
INPUT_FILE = BASE / "karachi_traffic_feature_engineered_final_v2.xlsx"
OUTPUT_FILE = BASE / "multiple_window_results_fixed.csv"

# 1. Load, order chronologically.
df = pd.read_excel(INPUT_FILE, sheet_name="Data")
df["Date-Time"] = pd.to_datetime(df["Date-Time"], errors="raise")
df = df.sort_values("Date-Time", kind="mergesort").reset_index(drop=True)


base_features = [
    "Vehicle Count",
    "Road Type",
    "Road Condition",
    "Area",
    "Number of Lanes",
    "Traffic Signal",
    "ML Weather",
    "Hour Sin",
    "Hour Cos",
]
experiments = {
    "Base 9": base_features,
    "Base + Congestion": base_features + ["Congestion Level"],
    "Base + Rush Hour": base_features + ["Rush Hour"],
    "Base + Bus Activity": base_features + ["Bus Activity"],
    "Base + Ride Activity": base_features + ["Ride Activity"],
    "Base + Is Weekend": base_features + ["Is Weekend"],
}

windows = [(450, 575), (575, 700), (700, 825)]
results = []

for window_no, (train_end, valid_end) in enumerate(windows, start=1):
    train = df.iloc[:train_end]
    valid = df.iloc[train_end:valid_end]
    print(f"\nRunning Window {window_no} ({len(train)} train, {len(valid)} validation)")

    for name, features in experiments.items():
        X_train = train[features]
        X_valid = valid[features]
        y_train = train["Accident Risk"]
        y_valid = valid["Accident Risk"]

        categorical = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
            ],
            remainder="passthrough",
        )
        rf = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", rf),
        ])
        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)

        
        results.append({
            "Window": window_no,
            "Experiment": name,
            "Macro Precision": precision_score(
                y_valid, predictions, average="macro", zero_division=0
            ),
            "Macro Recall": recall_score(
                y_valid, predictions, average="macro", zero_division=0
            ),
            "Macro F1": f1_score(
                y_valid, predictions, average="macro", zero_division=0
            ),
            "Accuracy": accuracy_score(y_valid, predictions),
        })
        print(f"  Saved: {name}")

# 6 experiments * 3 windows = 18 rows expected.
if not results:
    raise RuntimeError("No experiment results were created. Check the loops above.")

results_df = pd.DataFrame(results)
print("\nALL WINDOW RESULTS")
print(results_df.round(4).to_string(index=False))

average_results = (
    results_df.groupby("Experiment", as_index=False)[
        ["Macro Precision", "Macro Recall", "Macro F1", "Accuracy"]
    ].mean()
)
print("\nAVERAGE PERFORMANCE")
print(average_results.round(4).to_string(index=False))

precision_table = results_df.pivot(
    index="Window", columns="Experiment", values="Macro Precision"
)
print("\nPRECISION DIFFERENCE VS BASE 9")
for name in experiments:
    if name != "Base 9":
        print(f"\n{name}")
        print((precision_table[name] - precision_table["Base 9"]).round(4).to_string())

results_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(results_df)} results to: {OUTPUT_FILE}")
