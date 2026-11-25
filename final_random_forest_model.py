"""
Final Random Forest model for the Karachi traffic-risk thesis.

Selected model:
    RandomForestClassifier(class_weight="balanced")

Evaluation design:
    - Sort all observations by Date-Time.
    - Train on the first 80% of observations.
    - Evaluate once on the final 20% future observations.
    - No random shuffling.
    - No SMOTE, SMOTENC, SMOTEENN, ROS, RUS, or any other resampling.
    - Preprocessing is fitted on training data only.

For the current 1,500-row dataset:
    Training rows = first 1,200
    Final future test rows = last 300

The script saves:
    - final_random_forest_model.joblib
    - final_metrics.csv
    - final_classification_report.csv
    - final_confusion_matrix.csv
    - final_confusion_matrix.png
    - final_test_predictions.csv
    - final_feature_importance.csv
    - final_model_metadata.json

IMPORTANT THESIS NOTE:
The Accident Risk labels are present in the source dataset, but their original
derivation from independently verified crash records is undocumented.
Therefore, performance should be described as prediction of the dataset's
Accident Risk labels, not as independently validated real-world crash
probability.

Required packages:
    pip install pandas openpyxl scikit-learn matplotlib joblib
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. PATHS AND CONSTANTS
# ============================================================

BASE = Path(__file__).resolve().parent

INPUT_CANDIDATES = [
    BASE / "karachi_traffic_feature_engineered_final_v2.xlsx",
    BASE / "karachi_traffic_feature_engineered_final_v2(1).xlsx",
]

OUTPUT_DIR = BASE / "Final_Random_Forest_Output"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Accident Risk"
CLASS_ORDER = ["Low", "Medium", "High"]

# Keep exactly the same frozen feature set used in the final model comparison.
FEATURES = [
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

CATEGORICAL_FEATURES = [
    "Road Type",
    "Road Condition",
    "Area",
    "ML Weather",
]

NUMERICAL_FEATURES = [
    feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES
]

RANDOM_STATE = 42
N_ESTIMATORS = 300
MIN_SAMPLES_LEAF = 3
TEST_FRACTION = 0.20


# ============================================================
# 2. LOAD DATA
# ============================================================

def find_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find the dataset.\n"
        "Place 'karachi_traffic_feature_engineered_final_v2.xlsx' "
        "in the same folder as this script."
    )


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Data")

    required_columns = ["Date-Time", TARGET, *FEATURES]
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            "The dataset is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    df["Date-Time"] = pd.to_datetime(df["Date-Time"], errors="raise")

    # Stable chronological ordering.
    df = (
        df.sort_values("Date-Time", kind="mergesort")
        .reset_index(drop=True)
    )

    if df[TARGET].isna().any():
        raise ValueError("Accident Risk contains missing values.")

    invalid_labels = sorted(
        set(df[TARGET].astype(str)) - set(CLASS_ORDER)
    )
    if invalid_labels:
        raise ValueError(
            f"Unexpected Accident Risk labels found: {invalid_labels}"
        )

    return df


# ============================================================
# 3. CHRONOLOGICAL TRAIN / FINAL FUTURE TEST SPLIT
# ============================================================

def chronological_split(df: pd.DataFrame):
    split_index = int(len(df) * (1 - TEST_FRACTION))

    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Invalid train/test split.")

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # Verify that the test data occur after the training data.
    train_end = train_df["Date-Time"].max()
    test_start = test_df["Date-Time"].min()

    if train_end > test_start:
        raise ValueError(
            "Chronological split check failed: training extends beyond "
            "the start of the future test period."
        )

    return train_df, test_df


# ============================================================
# 4. BUILD FINAL RANDOM FOREST PIPELINE
# ============================================================

def make_one_hot_encoder():
    """
    Compatible with modern scikit-learn.
    If using an older version and sparse_output causes an error,
    replace sparse_output=False with sparse=False.
    """
    return OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )


def build_final_model() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                make_one_hot_encoder(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numeric_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return model


# ============================================================
# 5. EVALUATION
# ============================================================

def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)

    macro_precision = precision_score(
        y_test,
        predictions,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        labels=CLASS_ORDER,
        average="weighted",
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        predictions,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_ORDER,
    )

    metrics = {
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
    }

    return predictions, metrics, report_dict, cm


# ============================================================
# 6. SAVE OUTPUTS
# ============================================================

def save_metrics(metrics: dict):
    metrics_df = pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values()),
        }
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "final_metrics.csv",
        index=False,
    )


def save_classification_report(report_dict: dict):
    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .reset_index()
        .rename(columns={"index": "Class / Average"})
    )

    report_df.to_csv(
        OUTPUT_DIR / "final_classification_report.csv",
        index=False,
    )


def save_confusion_matrix(cm: np.ndarray):
    cm_df = pd.DataFrame(
        cm,
        index=[f"True {label}" for label in CLASS_ORDER],
        columns=[f"Predicted {label}" for label in CLASS_ORDER],
    )

    cm_df.to_csv(
        OUTPUT_DIR / "final_confusion_matrix.csv",
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_ORDER,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        "Final Random Forest Confusion Matrix\n"
        "Chronological Future Test Set"
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "final_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_test_predictions(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
):
    prediction_df = test_df[
        ["Date-Time", TARGET, *FEATURES]
    ].copy()

    prediction_df["Predicted Accident Risk"] = predictions
    prediction_df["Correct Prediction"] = (
        prediction_df[TARGET]
        == prediction_df["Predicted Accident Risk"]
    )

    prediction_df.to_csv(
        OUTPUT_DIR / "final_test_predictions.csv",
        index=False,
    )


def save_feature_importance(model: Pipeline):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed_feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    if len(transformed_feature_names) != len(importances):
        raise RuntimeError(
            "Feature-name count does not match Random Forest importance count."
        )

    importance_df = pd.DataFrame(
        {
            "Transformed Feature": transformed_feature_names,
            "Importance": importances,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    importance_df.to_csv(
        OUTPUT_DIR / "final_feature_importance.csv",
        index=False,
    )


def save_model_metadata(
    input_file: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: dict,
):
    metadata = {
        "model": "RandomForestClassifier",
        "selection": (
            "Final model selected from prior chronological model-comparison "
            "experiments."
        ),
        "class_weight": "balanced",
        "resampling": "None - original training data",
        "n_estimators": N_ESTIMATORS,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "target": TARGET,
        "class_order": CLASS_ORDER,
        "input_file": str(input_file),
        "total_rows": int(len(train_df) + len(test_df)),
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "training_start": str(train_df["Date-Time"].min()),
        "training_end": str(train_df["Date-Time"].max()),
        "test_start": str(test_df["Date-Time"].min()),
        "test_end": str(test_df["Date-Time"].max()),
        "training_class_counts": {
            str(k): int(v)
            for k, v in train_df[TARGET].value_counts().items()
        },
        "test_class_counts": {
            str(k): int(v)
            for k, v in test_df[TARGET].value_counts().items()
        },
        "final_test_metrics": {
            k: float(v) for k, v in metrics.items()
        },
        "important_limitation": (
            "The source dataset already contained Accident Risk labels. "
            "Their derivation from independently observed crash records "
            "has not been established."
        ),
    }

    with open(
        OUTPUT_DIR / "final_model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# 7. OPTIONAL HELPER FOR LATER PREDICTIONS
# ============================================================

def predict_new_observations(
    saved_model_path: Path,
    new_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Example utility for later use.

    new_data must contain exactly the required predictor columns.
    Extra columns are allowed, but all FEATURES must be present.
    """

    missing = [
        feature for feature in FEATURES
        if feature not in new_data.columns
    ]

    if missing:
        raise ValueError(
            "New prediction data are missing required features: "
            + ", ".join(missing)
        )

    loaded_model = joblib.load(saved_model_path)

    result = new_data.copy()
    result["Predicted Accident Risk"] = loaded_model.predict(
        result[FEATURES]
    )

    if hasattr(loaded_model, "predict_proba"):
        probabilities = loaded_model.predict_proba(
            result[FEATURES]
        )

        classes = loaded_model.named_steps[
            "classifier"
        ].classes_

        for index, label in enumerate(classes):
            result[
                f"Probability {label}"
            ] = probabilities[:, index]

    return result


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("FINAL KARACHI TRAFFIC-RISK RANDOM FOREST")
    print("=" * 70)

    input_file = find_input_file()
    df = load_dataset(input_file)

    train_df, test_df = chronological_split(df)

    print(f"\nDataset: {input_file}")
    print(f"Total observations: {len(df)}")
    print(f"Training observations: {len(train_df)}")
    print(f"Final future test observations: {len(test_df)}")

    print(
        "\nTraining period:",
        train_df["Date-Time"].min(),
        "to",
        train_df["Date-Time"].max(),
    )

    print(
        "Future test period:",
        test_df["Date-Time"].min(),
        "to",
        test_df["Date-Time"].max(),
    )

    print("\nTraining class distribution:")
    print(train_df[TARGET].value_counts().to_string())

    print("\nFinal test class distribution:")
    print(test_df[TARGET].value_counts().to_string())

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # --------------------------------------------------------
    # Fit selected final model
    # --------------------------------------------------------

    final_model = build_final_model()

    print("\nTraining final Random Forest...")
    final_model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Evaluate once on future test period
    # --------------------------------------------------------

    predictions, metrics, report_dict, cm = evaluate_model(
        final_model,
        X_test,
        y_test,
    )

    print("\nFINAL FUTURE TEST RESULTS")
    print("-" * 70)

    for metric_name, value in metrics.items():
        print(f"{metric_name:<25}: {value:.4f}")

    print("\nCLASSIFICATION REPORT")
    print("-" * 70)
    print(
        classification_report(
            y_test,
            predictions,
            labels=CLASS_ORDER,
            zero_division=0,
            digits=4,
        )
    )

    print("CONFUSION MATRIX")
    print("-" * 70)
    print(
        pd.DataFrame(
            cm,
            index=[f"True {c}" for c in CLASS_ORDER],
            columns=[f"Pred {c}" for c in CLASS_ORDER],
        ).to_string()
    )

    # --------------------------------------------------------
    # Save model and outputs
    # --------------------------------------------------------

    model_path = OUTPUT_DIR / "final_random_forest_model.joblib"

    joblib.dump(
        final_model,
        model_path,
    )

    save_metrics(metrics)
    save_classification_report(report_dict)
    save_confusion_matrix(cm)
    save_test_predictions(test_df, predictions)
    save_feature_importance(final_model)
    save_model_metadata(
        input_file=input_file,
        train_df=train_df,
        test_df=test_df,
        metrics=metrics,
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL SAVED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nModel file:\n  {model_path}")
    print(f"\nAll outputs:\n  {OUTPUT_DIR}")

    print(
        "\nFor thesis reporting, describe this as:\n"
        "Random Forest with class_weight='balanced', trained on the original "
        "non-resampled training data and evaluated using a chronological "
        "future holdout set."
    )


if __name__ == "__main__":
    main()
