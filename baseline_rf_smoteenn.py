
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. Load dataset
df = pd.read_excel(
    r"D:\accident_prediction\karachi_traffic_feature_engineered_final_v2.xlsx",
    sheet_name="Data"
)

# 2. Sort data by time
df["Date-Time"] = pd.to_datetime(df["Date-Time"])
df = df.sort_values("Date-Time").reset_index(drop=True)

# 3. Choose input features
features = [
    "Area",
    "Road Type",
    "Road Condition",
    "Vehicle Count",
    "Number of Lanes",
    "Traffic Signal",
    "Hour Sin",
    "Hour Cos",
    "Is Weekend",
    "ML Weather"
]

X = df[features]
y = df["Accident Risk"]

# 4. Chronological train/test split
# First test window from your existing experiment
n = len(df)
train_end = int(n * 0.55)
test_end = int(n * 0.70)

X_train = X.iloc[:train_end]
X_test = X.iloc[train_end:test_end]

y_train = y.iloc[:train_end]
y_test = y.iloc[train_end:test_end]

# 5. Encode categorical variables
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

# 6. Create basic Random Forest

# Same Random Forest settings as basic baseline
rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

# Handle categorical and numerical features
smote = SMOTENC(
    categorical_features=list(
        dict.fromkeys(categorical + ["Is Weekend"])
    ),
    random_state=42
)

# SMOTENC + ENN + Random Forest
model = ImbPipeline([
    ("smotenc", smote),
    ("preprocessor", preprocessor),
    ("enn", EditedNearestNeighbours()),
    ("classifier", rf)
])

# 7. Train model
model.fit(X_train, y_train)

# 8. Predict
y_pred = model.predict(X_test)

# 9. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Macro Precision:",
      precision_score(y_test, y_pred,
                      average="macro", zero_division=0))

print("Macro Recall:",
      recall_score(y_test, y_pred,
                   average="macro", zero_division=0))

print("Macro F1:",
      f1_score(y_test, y_pred,
               average="macro", zero_division=0))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

from sklearn.metrics import confusion_matrix

labels = ["Low", "Medium", "High"]

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Low", "Actual Medium", "Actual High"],
    columns=["Pred Low", "Pred Medium", "Pred High"]
)

print("\nCONFUSION MATRIX")
print(cm_df)

# Compare actual and predicted classes
print("\nACTUAL RISK COUNTS")
print(y_test.value_counts().reindex(labels, fill_value=0))

print("\nPREDICTED RISK COUNTS")
print(pd.Series(y_pred).value_counts().reindex(
    labels, fill_value=0
))


# Check training records before and after resampling

print("\nBEFORE RESAMPLING:")
print(y_train.value_counts())

# After SMOTENC
X_smote, y_smote = model.named_steps[
    "smotenc"
].fit_resample(X_train, y_train)

print("\nAFTER SMOTENC:")
print(pd.Series(y_smote).value_counts())

# Transform features using fitted preprocessor
X_encoded = model.named_steps[
    "preprocessor"
].transform(X_smote)

# After ENN
X_enn, y_enn = model.named_steps[
    "enn"
].fit_resample(X_encoded, y_smote)

print("\nAFTER ENN:")
print(pd.Series(y_enn).value_counts())
