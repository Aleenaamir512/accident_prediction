# Logistic Regression - calibration + class-specific threshold optimization
# + confidence-margin reject option + Macro F0.5 reporting.
#
# IMPORTANT:
# - All feature/model/threshold decisions are made inside the first 80% development period.
# - The last 20% is used only as a chronological evaluation period.
# - Rejected predictions are labeled "Uncertain".
# - Threshold selection maximizes Macro Precision subject to minimum recall and coverage.

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)

# ============================================================
# 1. PATHS / CONSTANTS
# ============================================================

BASE = Path(__file__).resolve().parent
INPUT_FILE = BASE / "karachi_traffic_feature_engineered_final_v2.xlsx"
OUTPUT_FOLDER = BASE / "Model_Output_Logistic_Calibrated_Threshold"
OUTPUT_FOLDER.mkdir(exist_ok=True)

TARGET = "Accident Risk"
CLASS_ORDER = ["Low", "Medium", "High"]
UNCERTAIN = "Uncertain"

# Minimum safeguards for threshold optimization.
# These are applied ONLY on development-period threshold-tuning predictions.
MIN_COVERAGE = 0.80
MIN_MACRO_RECALL = 0.50

# Class-specific probability thresholds tested inside development data only.
THRESHOLD_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

# Required separation between the highest and second-highest class probability.
MARGIN_GRID = [0.00, 0.05, 0.10]

# ============================================================
# 2. FEATURE SETS
# ============================================================

CORE = [
    "Area",
    "Road Type",
    "Road Condition",
    "Vehicle Count",
    "Number of Lanes",
    "Traffic Signal",
    "Hour Sin",
    "Hour Cos",
    "Is Weekend",
    "ML Weather",
]

AUGMENTED = CORE + [
    "Traffic Load per Lane",
    "Congestion Level",
    "Estimated Average Speed",
]

NO_AREA = [
    feature for feature in AUGMENTED
    if feature != "Area"
]

ALL_PREDICTIVE = [
    "Area",
    "Road Type",
    "Road Condition",
    "Vehicle Count",
    "Ride Activity",
    "Bus Activity",
    "Weather",
    "Time of Day",
    "Pickup Hour",
    "Day Name",
    "Month",
    "Is Weekend",
    "Hour Sin",
    "Hour Cos",
    "Rush Hour",
    "Time Period",
    "Construction",
    "Number of Lanes",
    "Near Intersection",
    "Traffic Signal",
    "Traffic Load per Lane",
    "Reference Traffic Load",
    "Traffic Load Ratio",
    "Congestion Level",
    "Rainfall",
    "Visibility",
    "Weather Code",
    "Historical Weather",
    "ML Weather",
    "Is Rainy",
    "Is Foggy",
    "Estimated Average Speed",
]

# FAST VERSION: freeze the previously strongest precision-oriented LR feature set.
# We are now testing post-processing (calibration + thresholds), not redoing model selection.
FEATURE_SETS = {
    "Set_D_All_Predictive_32": ALL_PREDICTIVE,
}

CATEGORICAL_FEATURES = {
    "Area",
    "Road Type",
    "Road Condition",
    "Weather",
    "Time of Day",
    "Day Name",
    "Month",
    "Time Period",
    "Congestion Level",
    "Weather Code",
    "Historical Weather",
    "ML Weather",
}

FREE_FLOW_SPEED = {
    "Highway": 75.0,
    "Main Road": 50.0,
    "Industrial Road": 40.0,
    "Intersection": 30.0,
    "Service Road": 28.0,
    "Narrow Road": 22.0,
}

# Logistic Regression model configurations tested.
C_VALUES = [1.0]  # frozen base LR setting for fast post-processing experiment
WEIGHT_ALPHAS = [0.0]  # no class weighting; previous precision-oriented LR used unweighted fit

# ============================================================
# 3. LEAKAGE-SAFE TRAFFIC FEATURES
# ============================================================

def recompute_traffic_features(reference_train, frame_to_transform):
    """
    Fit train-dependent traffic references using reference_train only,
    then transform frame_to_transform.
    """
    ref = reference_train.copy()
    out = frame_to_transform.copy()

    ref["Traffic Load per Lane"] = (
        ref["Vehicle Count"] / ref["Number of Lanes"]
    )
    out["Traffic Load per Lane"] = (
        out["Vehicle Count"] / out["Number of Lanes"]
    )

    group_ref = (
        ref.groupby(["Area", "Road Type"])["Traffic Load per Lane"]
        .agg(["count", lambda s: s.quantile(0.95)])
        .reset_index()
    )
    group_ref.columns = [
        "Area",
        "Road Type",
        "Group Count",
        "Group Reference",
    ]

    road_ref = (
        ref.groupby("Road Type")["Traffic Load per Lane"]
        .quantile(0.95)
        .to_dict()
    )

    overall_ref = float(
        ref["Traffic Load per Lane"].quantile(0.95)
    )

    out = out.merge(
        group_ref,
        on=["Area", "Road Type"],
        how="left",
    )

    def choose_reference(row):
        if (
            pd.notna(row["Group Count"])
            and row["Group Count"] >= 10
            and pd.notna(row["Group Reference"])
        ):
            return float(row["Group Reference"])

        value = road_ref.get(row["Road Type"])

        if value is not None and pd.notna(value):
            return float(value)

        return overall_ref

    out["Reference Traffic Load"] = out.apply(
        choose_reference,
        axis=1,
    )

    out["Traffic Load Ratio"] = (
        out["Traffic Load per Lane"]
        / out["Reference Traffic Load"]
    )

    def congestion(ratio):
        if ratio <= 0.35:
            return "Low"
        if ratio <= 0.75:
            return "Moderate"
        if ratio <= 0.95:
            return "High"
        return "Severe"

    out["Congestion Level"] = (
        out["Traffic Load Ratio"].apply(congestion)
    )

    def estimate_speed(row):
        free_flow = FREE_FLOW_SPEED.get(
            row["Road Type"],
            40.0,
        )

        ratio = float(row["Traffic Load Ratio"])

        factor = (
            1.0
            - 0.50 * min(ratio, 1.0)
            - 0.15 * max(ratio - 1.0, 0.0)
        )

        factor = np.clip(
            factor,
            0.40,
            1.00,
        )

        speed = free_flow * factor

        weather = str(
            row.get("ML Weather", "")
        ).strip().lower()

        if weather == "rainy":
            speed *= 0.90
        elif weather == "foggy":
            speed *= 0.82

        if int(row.get("Construction", 0)) == 1:
            speed *= 0.80

        if int(row.get("Near Intersection", 0)) == 1:
            speed *= 0.92

        return round(
            float(np.clip(speed, 8.0, 85.0)),
            1,
        )

    out["Estimated Average Speed"] = out.apply(
        estimate_speed,
        axis=1,
    )

    return out.drop(
        columns=["Group Count", "Group Reference"]
    )

# ============================================================
# 4. CLASS WEIGHTS + BASE MODEL
# ============================================================

def make_partial_class_weights(y, alpha):
    """
    alpha=0.0: no weighting
    alpha=0.5: partial balancing
    alpha=1.0: full inverse-frequency balancing
    """
    if alpha == 0.0:
        return None

    counts = y.value_counts()
    n = len(y)
    k = len(counts)

    balanced = {
        cls: n / (k * count)
        for cls, count in counts.items()
    }

    return {
        cls: float(weight ** alpha)
        for cls, weight in balanced.items()
    }


def make_pipeline(features, class_weight, c_value):
    categorical = [
        feature
        for feature in features
        if feature in CATEGORICAL_FEATURES
    ]

    numeric = [
        feature
        for feature in features
        if feature not in categorical
    ]

    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical,
        ),
        (
            "numeric",
            StandardScaler(),
            numeric,
        ),
    ])

    model = LogisticRegression(
        C=c_value,
        max_iter=5000,
        class_weight=class_weight,
        solver="lbfgs",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def ordered_probabilities(pipeline, X):
    """
    Return predict_proba columns in CLASS_ORDER:
    Low, Medium, High.
    """
    raw = pipeline.predict_proba(X)

    model_classes = list(
        pipeline.named_steps["model"].classes_
    )

    indices = [
        model_classes.index(class_name)
        for class_name in CLASS_ORDER
    ]

    return raw[:, indices]

# ============================================================
# 5. SIGMOID PROBABILITY CALIBRATION
# ============================================================

def safe_logit(probabilities):
    clipped = np.clip(
        probabilities,
        1e-6,
        1.0 - 1e-6,
    )

    return np.log(
        clipped / (1.0 - clipped)
    )


def fit_sigmoid_calibrators(raw_proba, y_true):
    """
    One-vs-rest Platt/sigmoid calibration.

    Calibration is fitted only from development OOF predictions.
    """
    calibrators = []

    for class_index, class_name in enumerate(CLASS_ORDER):
        x = safe_logit(
            raw_proba[:, class_index]
        ).reshape(-1, 1)

        y_binary = (
            np.asarray(y_true) == class_name
        ).astype(int)

        calibrator = LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
        )

        calibrator.fit(
            x,
            y_binary,
        )

        calibrators.append(
            calibrator
        )

    return calibrators


def apply_sigmoid_calibration(raw_proba, calibrators):
    calibrated = np.zeros_like(
        raw_proba,
        dtype=float,
    )

    for class_index, calibrator in enumerate(calibrators):
        x = safe_logit(
            raw_proba[:, class_index]
        ).reshape(-1, 1)

        calibrated[:, class_index] = (
            calibrator.predict_proba(x)[:, 1]
        )

    # Normalize back to a multiclass probability distribution.
    row_sums = calibrated.sum(
        axis=1,
        keepdims=True,
    )

    row_sums[row_sums == 0] = 1.0

    calibrated = calibrated / row_sums

    return calibrated

# ============================================================
# 6. SELECTIVE / REJECT PREDICTION
# ============================================================

def selective_predict(
    probabilities,
    thresholds,
    margin_threshold,
):
    """
    Predict the highest-probability class only if:
      1) its class-specific probability threshold is reached, and
      2) it beats the second-best class by margin_threshold.

    Otherwise return "Uncertain".
    """
    top_indices = np.argmax(
        probabilities,
        axis=1,
    )

    sorted_prob = np.sort(
        probabilities,
        axis=1,
    )

    top_prob = sorted_prob[:, -1]
    second_prob = sorted_prob[:, -2]
    margins = top_prob - second_prob

    predictions = np.full(
        len(probabilities),
        UNCERTAIN,
        dtype=object,
    )

    for row_index, class_index in enumerate(top_indices):
        class_threshold = thresholds[class_index]

        if (
            top_prob[row_index] >= class_threshold
            and margins[row_index] >= margin_threshold
        ):
            predictions[row_index] = (
                CLASS_ORDER[class_index]
            )

    return predictions


def argmax_predict(probabilities):
    indices = np.argmax(
        probabilities,
        axis=1,
    )

    return np.asarray([
        CLASS_ORDER[index]
        for index in indices
    ], dtype=object)


def selective_metrics(y_true, predictions):
    """
    Rejected/Uncertain rows count as missed cases for recall.
    Precision is computed for Low/Medium/High only.
    """
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)

    coverage = float(
        np.mean(
            predictions != UNCERTAIN
        )
    )

    return {
        "Accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "Macro Precision": precision_score(
            y_true,
            predictions,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
        "Macro Recall": recall_score(
            y_true,
            predictions,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
        "Macro F1": f1_score(
            y_true,
            predictions,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
        "Macro F0.5": fbeta_score(
            y_true,
            predictions,
            labels=CLASS_ORDER,
            average="macro",
            beta=0.5,
            zero_division=0,
        ),
        "Coverage": coverage,
        "Uncertain Count": int(
            np.sum(
                predictions == UNCERTAIN
            )
        ),
    }

# ============================================================
# 7. LOAD / SPLIT DATA
# ============================================================

df = pd.read_excel(
    INPUT_FILE,
    sheet_name="Data",
)

df["Date-Time"] = pd.to_datetime(
    df["Date-Time"],
    errors="coerce",
)

if df["Date-Time"].isna().any():
    raise ValueError(
        "Some Date-Time values could not be converted."
    )

df = (
    df.sort_values("Date-Time")
    .reset_index(drop=True)
)

outer_split = int(
    len(df) * 0.80
)

development_df = (
    df.iloc[:outer_split]
    .copy()
)

evaluation_raw = (
    df.iloc[outer_split:]
    .copy()
)

print("=" * 98)
print(
    "LOGISTIC REGRESSION — CALIBRATION + CLASS THRESHOLDS "
    "+ REJECT OPTION"
)
print("=" * 98)
print("Total rows:", len(df))
print("Development rows:", len(development_df))
print("Evaluation rows:", len(evaluation_raw))
print(
    "Development period:",
    development_df["Date-Time"].min(),
    "to",
    development_df["Date-Time"].max(),
)
print(
    "Evaluation period:",
    evaluation_raw["Date-Time"].min(),
    "to",
    evaluation_raw["Date-Time"].max(),
)

# ============================================================
# 8. DEVELOPMENT OOF PREDICTIONS
# ============================================================

time_cv = TimeSeriesSplit(
    n_splits=5
)

config_rows = []
search_rows = []
config_artifacts = {}

config_id = 0

total_configs = (
    len(FEATURE_SETS)
    * len(C_VALUES)
    * len(WEIGHT_ALPHAS)
)

print(
    "\nBase LR configurations (FAST):",
    total_configs,
)
print("FAST mode freezes the base LR model and tunes only probability post-processing.")

for set_name, features in FEATURE_SETS.items():

    for alpha in WEIGHT_ALPHAS:

        for c_value in C_VALUES:

            config_id += 1

            print(
                f"Running base configuration "
                f"{config_id}/{total_configs} "
                f"- {set_name} "
                f"- alpha={alpha} "
                f"- C={c_value}",
                flush=True,
            )

            oof_parts = []

            for fold, (
                train_index,
                validation_index,
            ) in enumerate(
                time_cv.split(development_df),
                start=1,
            ):

                train_raw = (
                    development_df
                    .iloc[train_index]
                    .copy()
                )

                validation_raw = (
                    development_df
                    .iloc[validation_index]
                    .copy()
                )

                if (
                    train_raw[TARGET].nunique() < 3
                    or validation_raw[TARGET].nunique() < 3
                ):
                    continue

                train = recompute_traffic_features(
                    train_raw,
                    train_raw,
                )

                validation = recompute_traffic_features(
                    train_raw,
                    validation_raw,
                )

                class_weight = (
                    make_partial_class_weights(
                        train[TARGET],
                        alpha,
                    )
                )

                pipeline = make_pipeline(
                    features,
                    class_weight,
                    c_value,
                )

                pipeline.fit(
                    train[features],
                    train[TARGET],
                )

                raw_proba = ordered_probabilities(
                    pipeline,
                    validation[features],
                )

                fold_oof = pd.DataFrame({
                    "Date-Time":
                        validation["Date-Time"].to_numpy(),
                    "Actual":
                        validation[TARGET].to_numpy(),
                    "P_Low":
                        raw_proba[:, 0],
                    "P_Medium":
                        raw_proba[:, 1],
                    "P_High":
                        raw_proba[:, 2],
                    "Fold":
                        fold,
                })

                oof_parts.append(
                    fold_oof
                )

            oof = (
                pd.concat(
                    oof_parts,
                    ignore_index=True,
                )
                .sort_values("Date-Time")
                .reset_index(drop=True)
            )

            # Temporal split of OOF predictions:
            # earlier 60% -> calibration fit
            # later 40%   -> threshold selection
            calibration_split = int(
                len(oof) * 0.60
            )

            calibration_part = (
                oof.iloc[:calibration_split]
                .copy()
            )

            threshold_part = (
                oof.iloc[calibration_split:]
                .copy()
            )

            calibration_raw = (
                calibration_part[
                    ["P_Low", "P_Medium", "P_High"]
                ]
                .to_numpy()
            )

            threshold_raw = (
                threshold_part[
                    ["P_Low", "P_Medium", "P_High"]
                ]
                .to_numpy()
            )

            calibrators = fit_sigmoid_calibrators(
                calibration_raw,
                calibration_part["Actual"].to_numpy(),
            )

            threshold_calibrated = apply_sigmoid_calibration(
                threshold_raw,
                calibrators,
            )

            # Store artifacts for the final selected configuration.
            config_artifacts[config_id] = {
                "Feature Set": set_name,
                "Features": features,
                "Alpha": alpha,
                "C": c_value,
                "Calibrators": calibrators,
            }

            # Evaluate both raw and calibrated probabilities.
            probability_modes = {
                "Raw": threshold_raw,
                "Calibrated": threshold_calibrated,
            }

            for probability_mode, probability_matrix in probability_modes.items():

                # Ordinary argmax baseline on threshold-tuning portion.
                baseline_pred = argmax_predict(
                    probability_matrix
                )

                baseline_metrics = selective_metrics(
                    threshold_part["Actual"].to_numpy(),
                    baseline_pred,
                )

                config_rows.append({
                    "Config ID": config_id,
                    "Feature Set": set_name,
                    "Weight Alpha": alpha,
                    "C": c_value,
                    "Probability Mode": probability_mode,
                    "Baseline Macro Precision":
                        baseline_metrics["Macro Precision"],
                    "Baseline Macro Recall":
                        baseline_metrics["Macro Recall"],
                    "Baseline Macro F1":
                        baseline_metrics["Macro F1"],
                    "Baseline Macro F0.5":
                        baseline_metrics["Macro F0.5"],
                })

                for low_t, medium_t, high_t in product(
                    THRESHOLD_GRID,
                    THRESHOLD_GRID,
                    THRESHOLD_GRID,
                ):

                    thresholds = [
                        low_t,
                        medium_t,
                        high_t,
                    ]

                    for margin_t in MARGIN_GRID:

                        predictions = selective_predict(
                            probability_matrix,
                            thresholds,
                            margin_t,
                        )

                        metrics = selective_metrics(
                            threshold_part["Actual"].to_numpy(),
                            predictions,
                        )

                        search_rows.append({
                            "Config ID": config_id,
                            "Feature Set": set_name,
                            "Weight Alpha": alpha,
                            "C": c_value,
                            "Probability Mode": probability_mode,
                            "Low Threshold": low_t,
                            "Medium Threshold": medium_t,
                            "High Threshold": high_t,
                            "Margin Threshold": margin_t,
                            **metrics,
                        })

# ============================================================
# 9. SELECT DEVELOPMENT-ONLY THRESHOLDS
# ============================================================

search_df = pd.DataFrame(
    search_rows
)

eligible = search_df[
    (search_df["Coverage"] >= MIN_COVERAGE)
    & (
        search_df["Macro Recall"]
        >= MIN_MACRO_RECALL
    )
].copy()

if eligible.empty:
    print(
        "\nWARNING: No threshold candidate met both safeguards."
    )
    print(
        "Falling back to candidates with Coverage >= 0.70 "
        "and selecting by Macro F0.5."
    )

    eligible = search_df[
        search_df["Coverage"] >= 0.70
    ].copy()

    eligible = (
        eligible
        .sort_values(
            [
                "Macro F0.5",
                "Macro Precision",
                "Macro Recall",
                "Coverage",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )
else:
    # Primary goal: precision.
    # F0.5 is the secondary score because it weights precision
    # more than recall.
    eligible = (
        eligible
        .sort_values(
            [
                "Macro Precision",
                "Macro F0.5",
                "Macro Recall",
                "Coverage",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

best = eligible.iloc[0]

selected_config_id = int(
    best["Config ID"]
)

selected_artifact = (
    config_artifacts[
        selected_config_id
    ]
)

selected_features = (
    selected_artifact["Features"]
)

selected_alpha = float(
    selected_artifact["Alpha"]
)

selected_c = float(
    selected_artifact["C"]
)

selected_calibrators = (
    selected_artifact["Calibrators"]
)

selected_mode = (
    best["Probability Mode"]
)

selected_thresholds = [
    float(best["Low Threshold"]),
    float(best["Medium Threshold"]),
    float(best["High Threshold"]),
]

selected_margin = float(
    best["Margin Threshold"]
)

pd.DataFrame(config_rows).to_excel(
    OUTPUT_FOLDER / "lr_calibration_baselines.xlsx",
    index=False,
)

search_df.to_excel(
    OUTPUT_FOLDER / "lr_threshold_search_all.xlsx",
    index=False,
)

eligible.head(200).to_excel(
    OUTPUT_FOLDER / "lr_threshold_search_eligible_top200.xlsx",
    index=False,
)

print("\n" + "=" * 98)
print("SELECTED DEVELOPMENT-ONLY CALIBRATION / THRESHOLD CONFIGURATION")
print("=" * 98)
print("Feature set:", best["Feature Set"])
print("Weight alpha:", selected_alpha)
print("C:", selected_c)
print("Probability mode:", selected_mode)
print(
    "Thresholds:",
    {
        "Low": selected_thresholds[0],
        "Medium": selected_thresholds[1],
        "High": selected_thresholds[2],
    },
)
print("Margin threshold:", selected_margin)
print("Tuning Macro Precision:", round(float(best["Macro Precision"]), 4))
print("Tuning Macro Recall:", round(float(best["Macro Recall"]), 4))
print("Tuning Macro F1:", round(float(best["Macro F1"]), 4))
print("Tuning Macro F0.5:", round(float(best["Macro F0.5"]), 4))
print("Tuning Coverage:", round(float(best["Coverage"]), 4))

# ============================================================
# 10. FIT FINAL BASE LR ON FULL DEVELOPMENT PERIOD
# ============================================================

development = recompute_traffic_features(
    development_df,
    development_df,
)

evaluation = recompute_traffic_features(
    development_df,
    evaluation_raw,
)

final_class_weight = (
    make_partial_class_weights(
        development[TARGET],
        selected_alpha,
    )
)

final_pipeline = make_pipeline(
    selected_features,
    final_class_weight,
    selected_c,
)

final_pipeline.fit(
    development[selected_features],
    development[TARGET],
)

raw_eval_proba = ordered_probabilities(
    final_pipeline,
    evaluation[selected_features],
)

calibrated_eval_proba = apply_sigmoid_calibration(
    raw_eval_proba,
    selected_calibrators,
)

if selected_mode == "Calibrated":
    selected_eval_proba = calibrated_eval_proba
else:
    selected_eval_proba = raw_eval_proba

# ============================================================
# 11. FINAL EVALUATION:
#     RAW ARGMAX vs CALIBRATED ARGMAX vs SELECTIVE THRESHOLDS
# ============================================================

y_eval = evaluation[TARGET].to_numpy()

raw_argmax_pred = argmax_predict(
    raw_eval_proba
)

calibrated_argmax_pred = argmax_predict(
    calibrated_eval_proba
)

selective_pred = selective_predict(
    selected_eval_proba,
    selected_thresholds,
    selected_margin,
)

raw_metrics = selective_metrics(
    y_eval,
    raw_argmax_pred,
)

calibrated_metrics = selective_metrics(
    y_eval,
    calibrated_argmax_pred,
)

selective_final_metrics = selective_metrics(
    y_eval,
    selective_pred,
)

# ROC-AUC uses probabilities, not the reject decision.
y_eval_numeric = (
    evaluation[TARGET]
    .map({
        "Low": 0,
        "Medium": 1,
        "High": 2,
    })
    .astype(int)
    .to_numpy()
)

selected_auc = roc_auc_score(
    y_eval_numeric,
    selected_eval_proba,
    multi_class="ovr",
    average="macro",
    labels=[0, 1, 2],
)

comparison = pd.DataFrame([
    {
        "Method": "Raw Argmax",
        **raw_metrics,
    },
    {
        "Method": "Calibrated Argmax",
        **calibrated_metrics,
    },
    {
        "Method": "Calibrated/Threshold Selective",
        **selective_final_metrics,
    },
])

comparison.to_excel(
    OUTPUT_FOLDER / "lr_final_method_comparison.xlsx",
    index=False,
)

final_summary = {
    "Selected Feature Set":
        best["Feature Set"],
    "Feature Count":
        len(selected_features),
    "Weight Alpha":
        selected_alpha,
    "C":
        selected_c,
    "Probability Mode":
        selected_mode,
    "Low Threshold":
        selected_thresholds[0],
    "Medium Threshold":
        selected_thresholds[1],
    "High Threshold":
        selected_thresholds[2],
    "Margin Threshold":
        selected_margin,
    "Accuracy":
        selective_final_metrics["Accuracy"],
    "Macro Precision":
        selective_final_metrics["Macro Precision"],
    "Macro Recall":
        selective_final_metrics["Macro Recall"],
    "Macro F1":
        selective_final_metrics["Macro F1"],
    "Macro F0.5":
        selective_final_metrics["Macro F0.5"],
    "Coverage":
        selective_final_metrics["Coverage"],
    "Uncertain Count":
        selective_final_metrics["Uncertain Count"],
    "ROC-AUC Macro OvR":
        selected_auc,
}

pd.DataFrame(
    [final_summary]
).to_excel(
    OUTPUT_FOLDER / "lr_selective_final_metrics.xlsx",
    index=False,
)

print("\n" + "=" * 98)
print("FINAL CHRONOLOGICAL EVALUATION COMPARISON")
print("=" * 98)
print(
    comparison.round(4).to_string(
        index=False
    )
)

print("\nSelected selective model:")
print(
    pd.Series(
        final_summary
    ).to_string()
)

print("\nSelective classification report:")
print(
    classification_report(
        y_eval,
        selective_pred,
        labels=CLASS_ORDER,
        zero_division=0,
    )
)

print(
    "\nUncertain predictions:",
    selective_final_metrics[
        "Uncertain Count"
    ],
    "out of",
    len(y_eval),
)

# ============================================================
# 12. CONFUSION MATRIX INCLUDING UNCERTAIN
# ============================================================

cm_labels = (
    CLASS_ORDER
    + [UNCERTAIN]
)

cm = confusion_matrix(
    y_eval,
    selective_pred,
    labels=cm_labels,
)

fig, ax = plt.subplots(
    figsize=(8, 7)
)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=cm_labels,
).plot(
    ax=ax,
    values_format="d",
)

ax.set_title(
    "Logistic Regression — Selective Prediction Confusion Matrix"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_FOLDER / "lr_selective_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# 13. SAVE ROW-LEVEL PREDICTIONS
# ============================================================

prediction_output = pd.DataFrame({
    "Date-Time":
        evaluation["Date-Time"].to_numpy(),
    "Actual":
        y_eval,
    "P_Low_Raw":
        raw_eval_proba[:, 0],
    "P_Medium_Raw":
        raw_eval_proba[:, 1],
    "P_High_Raw":
        raw_eval_proba[:, 2],
    "P_Low_Calibrated":
        calibrated_eval_proba[:, 0],
    "P_Medium_Calibrated":
        calibrated_eval_proba[:, 1],
    "P_High_Calibrated":
        calibrated_eval_proba[:, 2],
    "Raw Argmax Prediction":
        raw_argmax_pred,
    "Calibrated Argmax Prediction":
        calibrated_argmax_pred,
    "Selective Prediction":
        selective_pred,
})

prediction_output.to_excel(
    OUTPUT_FOLDER / "lr_selective_predictions.xlsx",
    index=False,
)

print("\nOutput folder:")
print(OUTPUT_FOLDER)

print("\nMethodology:")
print(
    "1. Base LR feature set, C and class-weight strength were explored using "
    "TimeSeriesSplit OOF predictions from the development period."
)
print(
    "2. Earlier OOF predictions were used to fit sigmoid probability calibrators."
)
print(
    "3. Later OOF predictions were used to tune class-specific thresholds and "
    "the top-vs-second probability margin."
)
print(
    f"4. Threshold selection targeted Macro Precision with development-only safeguards: "
    f"Macro Recall >= {MIN_MACRO_RECALL:.2f} and Coverage >= {MIN_COVERAGE:.2f}."
)
print(
    "5. Macro F0.5 is reported because it gives more weight to precision than recall."
)
print(
    "6. Predictions failing the probability/margin rules are returned as 'Uncertain' "
    "instead of being forced into Low/Medium/High."
)
print(
    "7. The chronological evaluation period was not used to choose thresholds, "
    "calibration mode, features, C or class weights in this run."
)
