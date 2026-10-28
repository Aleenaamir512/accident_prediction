
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


train_end = int(len(df) * 0.44)
valid_end = int(len(df) * 0.55)

train = df.iloc[:train_end]
valid = df.iloc[train_end:valid_end]

y_train = train["Accident Risk"]
y_valid = valid["Accident Risk"]

# 3. Features to add one at a time
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

results = []

# 4. Add features one by one
for i in range(1, len(features_order) + 1):

    features = features_order[:i]

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
        "Feature Added": features_order[i - 1],
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
        "Accuracy": accuracy
    })

# 5. Compare experiments
results_df = pd.DataFrame(results)

results_df["Precision Change"] = (
    results_df["Macro Precision"].diff()
)

print("\nINCREMENTAL FEATURE EXPERIMENT")
print(results_df.round(4).to_string(index=False))

# 6. Save results
results_df.to_csv(
    r"D:\accident_prediction\incremental_feature_results.csv",
    index=False
)

print("\nResults saved successfully!")
