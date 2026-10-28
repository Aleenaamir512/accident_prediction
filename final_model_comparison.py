from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTENC
    from imblearn.combine import SMOTEENN
    from imblearn.under_sampling import EditedNearestNeighbours
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTENC = None
    SMOTEENN = None
    EditedNearestNeighbours = None
    ImbPipeline = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "karachi_traffic_feature_engineered_final_v2.xlsx"
if not INPUT_FILE.exists():
    INPUT_FILE = HERE / "karachi_traffic_feature_engineered_final_v2(1).xlsx"
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Put karachi_traffic_feature_engineered_final_v2.xlsx beside this script."
    )
OUTPUT = HERE / "Final_Model_Comparison_With_SMOTEENN"
OUTPUT.mkdir(exist_ok=True)

TARGET = "Accident Risk"
CLASSES = ["Low", "Medium", "High"]
TO_NUM = {s: i for i, s in enumerate(CLASSES)}
FEATURES = [
    "Vehicle Count", "Road Type", "Road Condition", "Area",
    "Number of Lanes", "Traffic Signal", "ML Weather", "Hour Sin", "Hour Cos",
]
CATEGORICAL = ["Road Type", "Road Condition", "Area", "ML Weather"]
NUMERICAL = [f for f in FEATURES if f not in CATEGORICAL]
SEED = 42

# 3 time-ordered development evaluations, then the final later 300 rows.
DEVELOPMENT_WINDOWS = [(600, 800), (800, 1000), (1000, 1200)]
FINAL_TRAIN_END = 1200


def ohe():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def plain_preprocessor(scale_numeric=False):
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler() if scale_numeric else "passthrough"),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe()),
    ])
    return ColumnTransformer([
        ("categorical", categorical_pipe, CATEGORICAL),
        ("numeric", numeric_pipe, NUMERICAL),
    ], sparse_threshold=0)


def smotenc_pipeline(with_enn=False):
    if SMOTENC is None:
        return None
    
    n = len(NUMERICAL)
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERICAL),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), CATEGORICAL),
    ], sparse_threshold=0)
    post = ColumnTransformer([
        ("numeric", "passthrough", list(range(n))),
        ("categorical", ohe(), list(range(n, n + len(CATEGORICAL)))),
    ], sparse_threshold=0)
    steps = [
        ("prepare", prep),
        ("smotenc", SMOTENC(categorical_features=list(range(n, n+len(CATEGORICAL))),
                             k_neighbors=5, random_state=SEED)),
        ("onehot_after_sampling", post),
    ]
    if with_enn:
        # Clean AFTER categorical-aware sampling and one-hot encoding.
        
        steps.append(("enn", EditedNearestNeighbours(
            n_neighbors=3, sampling_strategy="all")))
    steps.append(("classifier", RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3,
        class_weight=None, random_state=SEED, n_jobs=-1,
    )))
    return ImbPipeline(steps)


def smoteenn_pipeline():
    if SMOTEENN is None:
        return None

    return ImbPipeline([
        ("preprocess", plain_preprocessor(scale_numeric=True)),
        ("smoteenn", SMOTEENN(random_state=SEED)),
        ("classifier", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3,
            class_weight=None, random_state=SEED, n_jobs=-1,
        )),
    ])


def model_factories():
    factories = {
        "Always Medium (baseline)": lambda: DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (balanced)": lambda: Pipeline([
            ("preprocess", plain_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)),
        ]),
        "Random Forest (basic)": lambda: Pipeline([
            ("preprocess", plain_preprocessor()),
            ("classifier", RandomForestClassifier(n_estimators=300,
                min_samples_leaf=3, random_state=SEED, n_jobs=-1)),
        ]),
        "Random Forest (balanced)": lambda: Pipeline([
            ("preprocess", plain_preprocessor()),
            ("classifier", RandomForestClassifier(n_estimators=300,
                min_samples_leaf=3, class_weight="balanced", random_state=SEED, n_jobs=-1)),
        ]),
    }
    if SMOTENC is not None:
        factories["Random Forest + SMOTENC"] = lambda: smotenc_pipeline(with_enn=False)
        factories["Random Forest + SMOTEENN"] = smoteenn_pipeline
        factories["Random Forest + SMOTENC + ENN"] = lambda: smotenc_pipeline(with_enn=True)
    else:
        print("NOTE: imbalanced-learn unavailable: skipping resampling models")
    if XGBClassifier is not None:
        factories["XGBoost"] = lambda: Pipeline([
            ("preprocess", plain_preprocessor()),
            ("classifier", XGBClassifier(
                objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.90, colsample_bytree=0.90,
                tree_method="hist", n_jobs=-1, random_state=SEED)),
        ])
    else:
        print("NOTE: xgboost unavailable: skipping XGBoost")
    if LGBMClassifier is not None:
        factories["LightGBM"] = lambda: Pipeline([
            ("preprocess", plain_preprocessor()),
            ("classifier", LGBMClassifier(
                objective="multiclass", num_class=3, n_estimators=300,
                learning_rate=0.05, num_leaves=15,
                min_child_samples=20, verbose=-1, random_state=SEED, n_jobs=-1)),
        ])
    else:
        print("NOTE: lightgbm unavailable: skipping LightGBM")
    return factories


def scores(y, pred, name, window):
    return {
        "Window": window,
        "Model": name,
        "Macro Precision": precision_score(y, pred, labels=[0,1,2], average="macro", zero_division=0),
        "Macro Recall": recall_score(y, pred, labels=[0,1,2], average="macro", zero_division=0),
        "Macro F1": f1_score(y, pred, labels=[0,1,2], average="macro", zero_division=0),
        "Accuracy": accuracy_score(y, pred),
        "High Risk Precision": precision_score(y, pred, labels=[2], average="macro", zero_division=0),
        "High Risk Recall": recall_score(y, pred, labels=[2], average="macro", zero_division=0),
    }


def main():
    df = pd.read_excel(INPUT_FILE, sheet_name="Data")
    required = ["Date-Time", TARGET] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required Excel columns: {missing}")
    df["Date-Time"] = pd.to_datetime(df["Date-Time"], errors="raise")
    df = df.sort_values("Date-Time", kind="mergesort").reset_index(drop=True)
    if len(df) != 1500:
        raise ValueError(f"Expected 1500 records based on this thesis dataset; got {len(df)}")
    if df[TARGET].isna().any() or not df[TARGET].isin(CLASSES).all():
        raise ValueError(f"Target must contain only {CLASSES}")
    for col in CATEGORICAL:
        df[col] = df[col].astype("string").fillna("Unknown").astype(str)
    y_all = df[TARGET].map(TO_NUM).astype(int)
    factories = model_factories()

    print("\nKARACHI TRAFFIC ACCIDENT RISK - RETROSPECTIVE MODEL COMPARISON")
    print("Dataset:", INPUT_FILE.name, "| Total rows:", len(df))
    print("Features:", FEATURES)
    print("Target counts:\n", df[TARGET].value_counts().to_string())
    print("\nDEVELOPMENT EVALUATION: FIRST 1200 ROWS ONLY")
    dev_rows = []
    for wi, (tr_end, va_end) in enumerate(DEVELOPMENT_WINDOWS, 1):
        print(f"\n  Window {wi}: train [0:{tr_end}], validate [{tr_end}:{va_end}]")
        X_train, y_train = df[FEATURES].iloc[:tr_end], y_all.iloc[:tr_end]
        X_val, y_val = df[FEATURES].iloc[tr_end:va_end], y_all.iloc[tr_end:va_end]
        for name, factory in factories.items():
            model = factory()
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            dev_rows.append(scores(y_val, pred, name, wi))
            print(f"    {name}: precision={dev_rows[-1]['Macro Precision']:.4f}, F1={dev_rows[-1]['Macro F1']:.4f}")

    per_window = pd.DataFrame(dev_rows)
    per_window.to_csv(OUTPUT / "development_windows.csv", index=False)
    avg = per_window.groupby("Model", as_index=False)[
        ["Macro Precision", "Macro Recall", "Macro F1", "Accuracy", "High Risk Precision", "High Risk Recall"]
    ].mean()
    avg = avg.sort_values(["Macro F1", "Macro Precision"], ascending=False)
    avg.to_csv(OUTPUT / "development_average.csv", index=False)
    print("\nDEVELOPMENT AVERAGES (selection uses Macro F1 then Macro Precision):")
    print(avg.round(4).to_string(index=False))
    nontrivial = avg[avg["Model"] != "Always Medium (baseline)"]
    selected_name = nontrivial.iloc[0]["Model"]
    print("\nTOP DEVELOPMENT MODEL:", selected_name)

    # The last 300 rows are assessed exactly once in this script, with frozen features/hyperparameters and NO tuning on these later observations.
    print("\nRETROSPECTIVE LATER-PERIOD EVALUATION:")
    X_train, y_train = df[FEATURES].iloc[:FINAL_TRAIN_END], y_all.iloc[:FINAL_TRAIN_END]
    X_test, y_test = df[FEATURES].iloc[FINAL_TRAIN_END:], y_all.iloc[FINAL_TRAIN_END:]
    print("Train date:", df["Date-Time"].iloc[0], "to", df["Date-Time"].iloc[FINAL_TRAIN_END - 1])
    print("Test date:", df["Date-Time"].iloc[FINAL_TRAIN_END], "to", df["Date-Time"].iloc[-1])
    print("Training class counts:", y_train.value_counts().sort_index().to_dict())
    print("Test class counts:", y_test.value_counts().sort_index().to_dict())

    test_rows = []
    predictions = pd.DataFrame({
        "Date-Time": df["Date-Time"].iloc[FINAL_TRAIN_END:].to_numpy(),
        "Actual Accident Risk": df[TARGET].iloc[FINAL_TRAIN_END:].to_numpy(),
    })
    for idx, (name, factory) in enumerate(factories.items(), 1):
        model = factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        entry = scores(y_test, pred, name, "Later 300")
        test_rows.append(entry)
        predictions[name] = [CLASSES[int(p)] for p in pred]
        report = classification_report(y_test, pred, labels=[0,1,2], target_names=CLASSES,
                                       digits=4, zero_division=0)
        cm = confusion_matrix(y_test, pred, labels=[0,1,2])
        slug = ''.join(c if c.isalnum() else '_' for c in name).strip('_')
        (OUTPUT / f"classification_report_{slug}.txt").write_text(report, encoding="utf-8")
        pd.DataFrame(cm, index=["Actual " + s for s in CLASSES],
                     columns=["Predicted " + s for s in CLASSES]).to_csv(
            OUTPUT / f"confusion_matrix_{slug}.csv")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(OUTPUT / f"confusion_matrix_{slug}.png", dpi=150)
        plt.close(fig)
        print(f"\n{name}\n{report}")

    final = pd.DataFrame(test_rows)
    final = final.sort_values(["Macro F1", "Macro Precision"], ascending=False)
    final.to_csv(OUTPUT / "later_period_model_comparison.csv", index=False)
    predictions.to_csv(OUTPUT / "later_period_predictions.csv", index=False)
    print("\nCOMPARISON ON SAME 300 LATER PERIOD RECORDS")
    print(final.round(4).to_string(index=False))
    print("\nOutputs saved to:", OUTPUT)
    print("CAUTION: Target label source is undocumented; past exploration touched parts of this dataset.")
    print("Do NOT describe these test results as external or completely untouched validation.")

if __name__ == "__main__":
    main()
