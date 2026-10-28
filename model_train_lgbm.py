#Model LightGBM
# LightGBM - precision-focused, time-aware experiment
# Uses development-only TimeSeriesSplit for feature/weight/hyperparameter selection.
# The last 20% is used only for final evaluation in this run.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)

from lightgbm import LGBMClassifier

# ============================================================
# 1. FILES / CONSTANTS
# ============================================================

SCRIPT_FOLDER = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_FOLDER / "karachi_traffic_feature_engineered_final_v2.xlsx"

OUTPUT_FOLDER = SCRIPT_FOLDER / "Model_Output_LightGBM_Precision_Tuned"
OUTPUT_FOLDER.mkdir(exist_ok=True)

TARGET = "Accident Risk"
CLASS_ORDER = ["Low", "Medium", "High"]

TARGET_MAP = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
}

REVERSE_TARGET_MAP = {
    value: key for key, value in TARGET_MAP.items()
}

# ============================================================
# 2. FEATURE SETS
# ============================================================

CORE_FEATURES = [
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

AUGMENTED_FEATURES = CORE_FEATURES + [
    "Traffic Load per Lane",
    "Congestion Level",
    "Estimated Average Speed",
]

NO_AREA_FEATURES = [
    feature
    for feature in AUGMENTED_FEATURES
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

FEATURE_SETS = {
    "Set_A_Core": CORE_FEATURES,
    "Set_B_Augmented": AUGMENTED_FEATURES,
    "Set_C_Augmented_No_Area": NO_AREA_FEATURES,
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

# ============================================================
# 3. LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found:\n{INPUT_FILE}")

df = pd.read_excel(INPUT_FILE, sheet_name="Data")
df["Date-Time"] = pd.to_datetime(df["Date-Time"], errors="coerce")

if df["Date-Time"].isna().any():
    raise ValueError("Some Date-Time values could not be converted.")

df = df.sort_values("Date-Time").reset_index(drop=True)

outer_split = int(len(df) * 0.80)
development_df = df.iloc[:outer_split].copy()
test_raw = df.iloc[outer_split:].copy()

print("=" * 92)
print("LIGHTGBM — PRECISION-FOCUSED TIME-AWARE TUNING + FINAL EVALUATION")
print("=" * 92)
print("Total rows:", len(df))
print("Development rows:", len(development_df))
print("Evaluation rows:", len(test_raw))
print(
    "Development period:",
    development_df["Date-Time"].min(),
    "to",
    development_df["Date-Time"].max(),
)
print(
    "Evaluation period:",
    test_raw["Date-Time"].min(),
    "to",
    test_raw["Date-Time"].max(),
)

# ============================================================
# 4. LEAKAGE-SAFE TRAFFIC FEATURE RECOMPUTATION
# ============================================================

def recompute_traffic_features(reference_train, frame_to_transform):
    """
    Learn train-dependent traffic-reference values from reference_train only
    and apply them to frame_to_transform.
    """
    ref_train = reference_train.copy()
    out = frame_to_transform.copy()

    ref_train["Traffic Load per Lane"] = (
        ref_train["Vehicle Count"] / ref_train["Number of Lanes"]
    )
    out["Traffic Load per Lane"] = (
        out["Vehicle Count"] / out["Number of Lanes"]
    )

    group_reference = (
        ref_train.groupby(["Area", "Road Type"])["Traffic Load per Lane"]
        .agg(["count", lambda s: s.quantile(0.95)])
        .reset_index()
    )
    group_reference.columns = [
        "Area",
        "Road Type",
        "Group Count",
        "Group Reference",
    ]

    road_type_reference = (
        ref_train.groupby("Road Type")["Traffic Load per Lane"]
        .quantile(0.95)
        .to_dict()
    )

    overall_reference = float(
        ref_train["Traffic Load per Lane"].quantile(0.95)
    )

    out = out.merge(
        group_reference,
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

        road_reference = road_type_reference.get(row["Road Type"])
        if road_reference is not None and pd.notna(road_reference):
            return float(road_reference)

        return overall_reference

    out["Reference Traffic Load"] = out.apply(
        choose_reference,
        axis=1,
    )

    out["Traffic Load Ratio"] = (
        out["Traffic Load per Lane"]
        / out["Reference Traffic Load"]
    )

    def congestion_category(ratio):
        if ratio <= 0.35:
            return "Low"
        if ratio <= 0.75:
            return "Moderate"
        if ratio <= 0.95:
            return "High"
        return "Severe"

    out["Congestion Level"] = (
        out["Traffic Load Ratio"].apply(congestion_category)
    )

    def estimate_speed(row):
        free_flow = FREE_FLOW_SPEED.get(
            row["Road Type"],
            40.0,
        )

        ratio = float(row["Traffic Load Ratio"])

        traffic_factor = (
            1.0
            - 0.50 * min(ratio, 1.0)
            - 0.15 * max(ratio - 1.0, 0.0)
        )
        traffic_factor = np.clip(
            traffic_factor,
            0.40,
            1.00,
        )

        speed = free_flow * traffic_factor

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
# 5. PARTIAL SAMPLE WEIGHTS
# ============================================================

def make_sample_weights(y_numeric, alpha):
    """
    alpha = 0.0 -> no weighting
    alpha = 0.5 -> partial/softer balancing
    alpha = 1.0 -> full inverse-frequency balancing

    Weights are calculated from the current training fold only.
    """
    if alpha == 0.0:
        return None

    y_series = pd.Series(y_numeric)
    counts = y_series.value_counts()
    n = len(y_series)
    k = len(counts)

    balanced = {
        cls: n / (k * count)
        for cls, count in counts.items()
    }

    softened = {
        cls: float(weight ** alpha)
        for cls, weight in balanced.items()
    }

    return y_series.map(softened).to_numpy()

# ============================================================
# 6. PIPELINE
# ============================================================

def make_pipeline(feature_list, params):
    categorical_features = [
        feature
        for feature in feature_list
        if feature in CATEGORICAL_FEATURES
    ]

    numeric_features = [
        feature
        for feature in feature_list
        if feature not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        min_child_samples=params["min_child_samples"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

# ============================================================
# 7. SMALL, DEFENSIBLE LIGHTGBM PARAMETER PROFILES
# ============================================================

PARAMETER_SETS = [
    {
        "name": "P1_Conservative",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_lambda": 1.5,
        "reg_alpha": 0.0,
    },
    {
        "name": "P2_Shallow_Regularized",
        "n_estimators": 450,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "reg_lambda": 2.0,
        "reg_alpha": 0.10,
    },
    {
        "name": "P3_Moderate",
        "n_estimators": 350,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_lambda": 1.0,
        "reg_alpha": 0.10,
    },
    {
        "name": "P4_Regularized",
        "n_estimators": 300,
        "learning_rate": 0.04,
        "num_leaves": 23,
        "max_depth": 6,
        "min_child_samples": 30,
        "subsample": 0.80,
        "colsample_bytree": 0.75,
        "reg_lambda": 2.5,
        "reg_alpha": 0.25,
    },
]

WEIGHT_ALPHAS = [0.0, 0.5, 1.0]

# ============================================================
# 8. TIME-SERIES CV
# ============================================================

time_cv = TimeSeriesSplit(n_splits=5)

cv_rows = []
config_id = 0

total_configs = (
    len(FEATURE_SETS)
    * len(PARAMETER_SETS)
    * len(WEIGHT_ALPHAS)
)

print("\nTotal LightGBM configurations:", total_configs)

for set_name, feature_list in FEATURE_SETS.items():

    for alpha in WEIGHT_ALPHAS:

        for params in PARAMETER_SETS:

            config_id += 1

            print(
                f"Running configuration {config_id}/{total_configs} "
                f"- {set_name} "
                f"- weight_alpha={alpha} "
                f"- {params['name']}",
                flush=True,
            )

            for fold, (
                train_index,
                validation_index,
            ) in enumerate(
                time_cv.split(development_df),
                start=1,
            ):

                fold_train_raw = (
                    development_df
                    .iloc[train_index]
                    .copy()
                )

                fold_validation_raw = (
                    development_df
                    .iloc[validation_index]
                    .copy()
                )

                if (
                    fold_train_raw[TARGET].nunique() < 3
                    or fold_validation_raw[TARGET].nunique() < 3
                ):
                    continue

                fold_train = recompute_traffic_features(
                    fold_train_raw,
                    fold_train_raw,
                )

                fold_validation = recompute_traffic_features(
                    fold_train_raw,
                    fold_validation_raw,
                )

                X_train = fold_train[feature_list]
                X_validation = fold_validation[feature_list]

                y_train_text = fold_train[TARGET]
                y_validation_text = fold_validation[TARGET]

                y_train = (
                    y_train_text
                    .map(TARGET_MAP)
                    .astype(int)
                )

                sample_weights = make_sample_weights(
                    y_train,
                    alpha,
                )

                pipeline = make_pipeline(
                    feature_list,
                    params,
                )

                fit_kwargs = {}
                if sample_weights is not None:
                    fit_kwargs["model__sample_weight"] = sample_weights

                pipeline.fit(
                    X_train,
                    y_train,
                    **fit_kwargs,
                )

                prediction_codes = (
                    pipeline.predict(X_validation)
                    .astype(int)
                )

                predictions = (
                    pd.Series(
                        prediction_codes,
                        index=y_validation_text.index,
                    )
                    .map(REVERSE_TARGET_MAP)
                )

                class_recalls = recall_score(
                    y_validation_text,
                    predictions,
                    labels=CLASS_ORDER,
                    average=None,
                    zero_division=0,
                )

                cv_rows.append({
                    "Config ID": config_id,
                    "Feature Set": set_name,
                    "Parameter Set": params["name"],
                    "Weight Alpha": alpha,
                    "Fold": fold,
                    "Accuracy": accuracy_score(
                        y_validation_text,
                        predictions,
                    ),
                    "Macro Precision": precision_score(
                        y_validation_text,
                        predictions,
                        average="macro",
                        zero_division=0,
                    ),
                    "Weighted Precision": precision_score(
                        y_validation_text,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                    "Macro Recall": recall_score(
                        y_validation_text,
                        predictions,
                        average="macro",
                        zero_division=0,
                    ),
                    "Macro F1": f1_score(
                        y_validation_text,
                        predictions,
                        average="macro",
                        zero_division=0,
                    ),
                    "Low Recall": class_recalls[0],
                    "Medium Recall": class_recalls[1],
                    "High Recall": class_recalls[2],
                })

# ============================================================
# 9. CV SUMMARY + PRECISION-FIRST SELECTION WITH F1 SAFEGUARD
# ============================================================

cv_detail = pd.DataFrame(cv_rows)

cv_summary = (
    cv_detail
    .groupby([
        "Config ID",
        "Feature Set",
        "Parameter Set",
        "Weight Alpha",
    ])
    .agg(
        Folds=("Fold", "count"),
        Mean_Accuracy=("Accuracy", "mean"),
        Mean_Macro_Precision=("Macro Precision", "mean"),
        Mean_Weighted_Precision=("Weighted Precision", "mean"),
        Mean_Macro_Recall=("Macro Recall", "mean"),
        Mean_Macro_F1=("Macro F1", "mean"),
        Mean_Low_Recall=("Low Recall", "mean"),
        Mean_Medium_Recall=("Medium Recall", "mean"),
        Mean_High_Recall=("High Recall", "mean"),
        Std_Macro_Precision=("Macro Precision", "std"),
        Std_Macro_F1=("Macro F1", "std"),
    )
    .reset_index()
)

best_cv_f1 = float(
    cv_summary["Mean_Macro_F1"].max()
)

f1_floor = 0.95 * best_cv_f1

eligible = cv_summary[
    cv_summary["Mean_Macro_F1"] >= f1_floor
].copy()

eligible = (
    eligible
    .sort_values(
        [
            "Mean_Macro_Precision",
            "Mean_Macro_Recall",
            "Mean_Accuracy",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

best = eligible.iloc[0]

cv_detail.to_excel(
    OUTPUT_FOLDER / "lightgbm_precision_cv_detail.xlsx",
    index=False,
)

cv_summary.sort_values(
    "Mean_Macro_Precision",
    ascending=False,
).to_excel(
    OUTPUT_FOLDER / "lightgbm_precision_cv_all_configs.xlsx",
    index=False,
)

eligible.to_excel(
    OUTPUT_FOLDER / "lightgbm_precision_cv_eligible_configs.xlsx",
    index=False,
)

print("\n" + "=" * 92)
print("TOP ELIGIBLE CONFIGURATIONS — PRECISION-FIRST WITH F1 SAFEGUARD")
print("=" * 92)
print("Best CV Macro F1:", round(best_cv_f1, 4))
print("Eligibility floor (95% of best F1):", round(f1_floor, 4))

display_cols = [
    "Config ID",
    "Feature Set",
    "Parameter Set",
    "Weight Alpha",
    "Mean_Accuracy",
    "Mean_Macro_Precision",
    "Mean_Weighted_Precision",
    "Mean_Macro_Recall",
    "Mean_Macro_F1",
    "Mean_Low_Recall",
    "Mean_Medium_Recall",
    "Mean_High_Recall",
]

print(
    eligible[
        display_cols
    ]
    .head(15)
    .round(4)
    .to_string(index=False)
)

selected_set_name = best["Feature Set"]
selected_features = FEATURE_SETS[selected_set_name]
selected_alpha = float(best["Weight Alpha"])
selected_parameter_name = best["Parameter Set"]

selected_params = next(
    p
    for p in PARAMETER_SETS
    if p["name"] == selected_parameter_name
)

print("\nSelected feature set:", selected_set_name)
print("Selected parameter set:", selected_parameter_name)
print("Selected weight alpha:", selected_alpha)
print(
    "CV Macro Precision:",
    round(float(best["Mean_Macro_Precision"]), 4),
)
print(
    "CV Macro Recall:",
    round(float(best["Mean_Macro_Recall"]), 4),
)
print(
    "CV Macro F1:",
    round(float(best["Mean_Macro_F1"]), 4),
)

# ============================================================
# 10. FINAL MODEL
# ============================================================

development = recompute_traffic_features(
    development_df,
    development_df,
)

test = recompute_traffic_features(
    development_df,
    test_raw,
)

X_development = development[selected_features]
X_test = test[selected_features]

y_development_text = development[TARGET]
y_test_text = test[TARGET]

y_development = (
    y_development_text
    .map(TARGET_MAP)
    .astype(int)
)

y_test = (
    y_test_text
    .map(TARGET_MAP)
    .astype(int)
)

final_sample_weights = make_sample_weights(
    y_development,
    selected_alpha,
)

final_pipeline = make_pipeline(
    selected_features,
    selected_params,
)

fit_kwargs = {}
if final_sample_weights is not None:
    fit_kwargs["model__sample_weight"] = final_sample_weights

final_pipeline.fit(
    X_development,
    y_development,
    **fit_kwargs,
)

prediction_codes = (
    final_pipeline.predict(X_test)
    .astype(int)
)

predictions = (
    pd.Series(
        prediction_codes,
        index=y_test_text.index,
    )
    .map(REVERSE_TARGET_MAP)
)

probabilities = final_pipeline.predict_proba(X_test)

# ============================================================
# 11. FINAL METRICS
# ============================================================

final_metrics = {
    "Selected Feature Set": selected_set_name,
    "Feature Count": len(selected_features),
    "Selected Parameter Set": selected_parameter_name,
    "Weight Alpha": selected_alpha,
    "Accuracy": accuracy_score(
        y_test_text,
        predictions,
    ),
    "Macro Precision": precision_score(
        y_test_text,
        predictions,
        average="macro",
        zero_division=0,
    ),
    "Macro Recall": recall_score(
        y_test_text,
        predictions,
        average="macro",
        zero_division=0,
    ),
    "Macro F1": f1_score(
        y_test_text,
        predictions,
        average="macro",
        zero_division=0,
    ),
    "Weighted Precision": precision_score(
        y_test_text,
        predictions,
        average="weighted",
        zero_division=0,
    ),
    "Weighted Recall": recall_score(
        y_test_text,
        predictions,
        average="weighted",
        zero_division=0,
    ),
    "Weighted F1": f1_score(
        y_test_text,
        predictions,
        average="weighted",
        zero_division=0,
    ),
    "ROC-AUC Macro OvR": roc_auc_score(
        y_test,
        probabilities,
        multi_class="ovr",
        average="macro",
        labels=[0, 1, 2],
    ),
    "ROC-AUC Weighted OvR": roc_auc_score(
        y_test,
        probabilities,
        multi_class="ovr",
        average="weighted",
        labels=[0, 1, 2],
    ),
}

pd.DataFrame(
    [final_metrics]
).to_excel(
    OUTPUT_FOLDER / "lightgbm_precision_final_metrics.xlsx",
    index=False,
)

report = classification_report(
    y_test_text,
    predictions,
    labels=CLASS_ORDER,
    zero_division=0,
)

(
    OUTPUT_FOLDER / "lightgbm_precision_classification_report.txt"
).write_text(
    report,
    encoding="utf-8",
)

print("\n" + "=" * 92)
print("FINAL HELD-OUT EVALUATION RESULTS")
print("=" * 92)
print(pd.Series(final_metrics).to_string())
print("\nClassification report:\n")
print(report)

# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

confusion = confusion_matrix(
    y_test_text,
    predictions,
    labels=CLASS_ORDER,
)

fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=CLASS_ORDER,
).plot(
    ax=ax,
    values_format="d",
)

ax.set_title("LightGBM — Precision-Tuned Confusion Matrix")
fig.tight_layout()

fig.savefig(
    OUTPUT_FOLDER / "lightgbm_precision_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# 13. MULTICLASS ROC
# ============================================================

y_test_binary = label_binarize(
    y_test,
    classes=[0, 1, 2],
)

fig, ax = plt.subplots(figsize=(8, 6))

for class_index, class_name in enumerate(CLASS_ORDER):
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test_binary[:, class_index],
        probabilities[:, class_index],
    )

    class_auc = auc(
        false_positive_rate,
        true_positive_rate,
    )

    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{class_name} (AUC={class_auc:.3f})",
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Chance",
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("LightGBM — Multiclass ROC (OvR)")
ax.legend()
fig.tight_layout()

fig.savefig(
    OUTPUT_FOLDER / "lightgbm_precision_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

fitted_preprocessor = final_pipeline.named_steps["preprocessor"]
fitted_model = final_pipeline.named_steps["model"]

expanded_feature_names = (
    fitted_preprocessor.get_feature_names_out()
)

importance_values = fitted_model.feature_importances_

importance_df = (
    pd.DataFrame({
        "Encoded Feature": expanded_feature_names,
        "Importance": importance_values,
    })
    .sort_values(
        "Importance",
        ascending=False,
    )
    .reset_index(drop=True)
)

importance_df.to_excel(
    OUTPUT_FOLDER / "lightgbm_precision_feature_importance.xlsx",
    index=False,
)

top_importance = importance_df.head(20)

fig, ax = plt.subplots(figsize=(9, 8))

ax.barh(
    top_importance["Encoded Feature"][::-1],
    top_importance["Importance"][::-1],
)

ax.set_xlabel("Feature Importance")
ax.set_title("LightGBM — Top 20 Feature Importances")
fig.tight_layout()

fig.savefig(
    OUTPUT_FOLDER / "lightgbm_precision_feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\nSelected features:")
for feature in selected_features:
    print(" -", feature)

print("\nSelected LightGBM parameters:")
for key, value in selected_params.items():
    print(f" - {key}: {value}")

print("\nOutput folder:")
print(OUTPUT_FOLDER)

print("\nMethodology:")
print("1. Data were sorted chronologically and split 80/20.")
print("2. Feature/weight/hyperparameter tuning used only TimeSeriesSplit inside the first 80%.")
print("3. Train-derived traffic features were recomputed separately inside every fold.")
print("4. Four feature sets, three weight strengths and four LightGBM parameter profiles were compared.")
print("5. Selection prioritizes Macro Precision only among models within 95% of the best CV Macro F1.")
print("6. This prevents choosing a high-precision model whose recall/F1 collapses.")
print("7. The evaluation period was not used to select features, weights, or hyperparameters in this run.")
