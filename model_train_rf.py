#Random Forest Model

# Random Forest - precision-focused, time-aware experiment
# Keeps the final 20% completely untouched until the end.
# Feature set + RF hyperparameters are chosen only from TimeSeriesSplit CV
# inside the first 80% of the data.

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score, roc_curve, auc
)

# ============================================================
# PATHS / CONSTANTS
# ============================================================

BASE = Path(__file__).resolve().parent
INPUT = BASE / "karachi_traffic_feature_engineered_final_v2.xlsx"

# New folder so your old Random Forest results are not overwritten.
OUT = BASE / "Model_Output_Random_Forest_Precision"
OUT.mkdir(exist_ok=True)

TARGET = "Accident Risk"
CLASS_ORDER = ["Low", "Medium", "High"]

# ============================================================
# FEATURE SETS
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

# All usable predictive variables.
# Accident Risk is excluded because it is the target.
# Date-Time is used for chronological splitting rather than as a direct predictor.
# Infrastructure Source / Infrastructure Verified / Weather Data Source are metadata.
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
    "Set_A_Core": CORE,
    "Set_B_Augmented": AUGMENTED,
    "Set_D_All_Predictive_32": ALL_PREDICTIVE,
}

# Treat nominal/discrete categories as categorical rather than numeric magnitudes.
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

# ============================================================
# LEAKAGE-SAFE TRAFFIC FEATURES
# ============================================================

FREE_FLOW = {
    "Highway": 75.0,
    "Main Road": 50.0,
    "Industrial Road": 40.0,
    "Intersection": 30.0,
    "Service Road": 28.0,
    "Narrow Road": 22.0,
}


def recompute_features(reference_train, frame):
    """
    Fit Reference Traffic Load from reference_train only, then transform frame.
    This keeps validation/test information out of train-derived traffic features.
    """
    ref = reference_train.copy()
    out = frame.copy()

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
        "Area", "Road Type", "Group Count", "Group Reference"
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
        how="left"
    )

    def choose_ref(row):
        if (
            pd.notna(row["Group Count"])
            and row["Group Count"] >= 10
            and pd.notna(row["Group Reference"])
        ):
            return float(row["Group Reference"])

        road_value = road_ref.get(row["Road Type"])
        if road_value is not None and pd.notna(road_value):
            return float(road_value)

        return overall_ref

    out["Reference Traffic Load"] = out.apply(
        choose_ref, axis=1
    )

    out["Traffic Load Ratio"] = (
        out["Traffic Load per Lane"]
        / out["Reference Traffic Load"]
    )

    def congestion(x):
        if x <= 0.35:
            return "Low"
        if x <= 0.75:
            return "Moderate"
        if x <= 0.95:
            return "High"
        return "Severe"

    out["Congestion Level"] = (
        out["Traffic Load Ratio"].apply(congestion)
    )

    def est_speed(row):
        free = FREE_FLOW.get(row["Road Type"], 40.0)
        ratio = float(row["Traffic Load Ratio"])

        factor = (
            1.0
            - 0.50 * min(ratio, 1.0)
            - 0.15 * max(ratio - 1.0, 0.0)
        )
        factor = np.clip(factor, 0.40, 1.00)
        speed = free * factor

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
            1
        )

    out["Estimated Average Speed"] = out.apply(
        est_speed, axis=1
    )

    return out.drop(
        columns=["Group Count", "Group Reference"]
    )


# ============================================================
# PIPELINE
# ============================================================

def make_pipeline(features, params, n_estimators=200):
    categorical = [
        f for f in features
        if f in CATEGORICAL_FEATURES
    ]
    numeric = [
        f for f in features
        if f not in categorical
    ]

    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        ),
        ("numeric", "passthrough", numeric),
    ])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,

        # Tuned settings:
        class_weight=params["class_weight"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ============================================================
# LOAD / CHECK DATA
# ============================================================

df = pd.read_excel(INPUT, sheet_name="Data")
df["Date-Time"] = pd.to_datetime(
    df["Date-Time"],
    errors="coerce"
)

if df["Date-Time"].isna().any():
    raise ValueError(
        "Some Date-Time values could not be converted."
    )

missing = [
    f for f in ALL_PREDICTIVE
    if f not in df.columns
    and f not in {
        "Traffic Load per Lane",
        "Reference Traffic Load",
        "Traffic Load Ratio",
        "Congestion Level",
        "Estimated Average Speed",
    }
]

if missing:
    raise ValueError(
        "These required features are missing from the dataset: "
        + ", ".join(missing)
    )

df = (
    df.sort_values("Date-Time")
    .reset_index(drop=True)
)

split = int(len(df) * 0.80)

dev_raw = df.iloc[:split].copy()
test_raw = df.iloc[split:].copy()

print("=" * 88)
print(
    "RANDOM FOREST — PRECISION-FOCUSED "
    "TIME-AWARE TUNING + FINAL TEST"
)
print("=" * 88)
print("Total rows:", len(df))
print("Development rows:", len(dev_raw))
print("Final test rows:", len(test_raw))
print(
    "Development period:",
    dev_raw["Date-Time"].min(),
    "to",
    dev_raw["Date-Time"].max()
)
print(
    "Final test period:",
    test_raw["Date-Time"].min(),
    "to",
    test_raw["Date-Time"].max()
)

# ============================================================
# PRE-COMPUTE TIME-SERIES CV FOLDS
# ============================================================

tscv = TimeSeriesSplit(n_splits=5)

cv_folds = []

for fold, (tr_idx, va_idx) in enumerate(
    tscv.split(dev_raw),
    start=1
):
    tr_raw = dev_raw.iloc[tr_idx].copy()
    va_raw = dev_raw.iloc[va_idx].copy()

    if (
        tr_raw[TARGET].nunique() < 3
        or va_raw[TARGET].nunique() < 3
    ):
        print(
            f"Skipping fold {fold}: "
            "not all 3 classes are present."
        )
        continue

    tr = recompute_features(tr_raw, tr_raw)
    va = recompute_features(tr_raw, va_raw)

    cv_folds.append((fold, tr, va))

# ============================================================
# SMALL, DEFENSIBLE PRECISION-FOCUSED PARAMETER GRID
# ============================================================

# balanced_subsample can improve minority recall but may reduce precision.
# Therefore CV is allowed to choose between no weighting and balanced weighting.
parameter_grid = []

for (
    class_weight,
    max_depth,
    min_samples_leaf,
    max_features,
) in product(
    [None, "balanced_subsample"],
    [None, 10],
    [1, 3],
    ["sqrt", 0.70],
):
    parameter_grid.append({
        "class_weight": class_weight,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
    })

print("\nRF parameter combinations:", len(parameter_grid))
print(
    "Total CV configurations:",
    len(parameter_grid) * len(FEATURE_SETS)
)

# ============================================================
# TRAINING-ONLY FEATURE + PARAMETER SELECTION
# ============================================================

cv_detail_rows = []
cv_config_rows = []

config_id = 0

for set_name, features in FEATURE_SETS.items():

    for params in parameter_grid:
        config_id += 1
        print(
    f"Running configuration {config_id}/"
    f"{len(FEATURE_SETS) * len(parameter_grid)} "
    f"- {set_name} - {params}",
    flush=True
)
        fold_metrics = []

        for fold, tr, va in cv_folds:
            pipe = make_pipeline(features, params)

            pipe.fit(
                tr[features],
                tr[TARGET]
            )

            pred = pipe.predict(
                va[features]
            )

            row = {
                "Config ID": config_id,
                "Feature Set": set_name,
                "Fold": fold,
                "class_weight": str(
                    params["class_weight"]
                ),
                "max_depth": str(
                    params["max_depth"]
                ),
                "min_samples_leaf":
                    params["min_samples_leaf"],
                "max_features":
                    params["max_features"],
                "Accuracy": accuracy_score(
                    va[TARGET], pred
                ),
                "Macro Precision":
                    precision_score(
                        va[TARGET],
                        pred,
                        average="macro",
                        zero_division=0,
                    ),
                "Weighted Precision":
                    precision_score(
                        va[TARGET],
                        pred,
                        average="weighted",
                        zero_division=0,
                    ),
                "Macro Recall":
                    recall_score(
                        va[TARGET],
                        pred,
                        average="macro",
                        zero_division=0,
                    ),
                "Macro F1":
                    f1_score(
                        va[TARGET],
                        pred,
                        average="macro",
                        zero_division=0,
                    ),
            }

            cv_detail_rows.append(row)
            fold_metrics.append(row)

        if not fold_metrics:
            continue

        fold_df = pd.DataFrame(
            fold_metrics
        )

        cv_config_rows.append({
            "Config ID": config_id,
            "Feature Set": set_name,
            "class_weight":
                str(params["class_weight"]),
            "max_depth":
                str(params["max_depth"]),
            "min_samples_leaf":
                params["min_samples_leaf"],
            "max_features":
                params["max_features"],
            "Folds": len(fold_df),
            "Mean Accuracy":
                fold_df["Accuracy"].mean(),
            "Mean Macro Precision":
                fold_df["Macro Precision"].mean(),
            "Mean Weighted Precision":
                fold_df[
                    "Weighted Precision"
                ].mean(),
            "Mean Macro Recall":
                fold_df["Macro Recall"].mean(),
            "Mean Macro F1":
                fold_df["Macro F1"].mean(),
            "Std Macro Precision":
                fold_df[
                    "Macro Precision"
                ].std(),
            "Std Macro F1":
                fold_df["Macro F1"].std(),
        })

cv_detail = pd.DataFrame(
    cv_detail_rows
)

cv_summary = pd.DataFrame(
    cv_config_rows
)

# Precision is intentionally the FIRST selection criterion in this experiment.
# Macro F1 is used as a tie-breaker so we do not select an extremely
# precision-heavy but otherwise poor model.
cv_summary = (
    cv_summary.sort_values(
        [
            "Mean Macro Precision",
            "Mean Macro F1",
            "Mean Accuracy",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

cv_detail.to_excel(
    OUT
    / "random_forest_precision_cv_detail.xlsx",
    index=False,
)

cv_summary.to_excel(
    OUT
    / "random_forest_precision_cv_summary.xlsx",
    index=False,
)

print("\n" + "=" * 88)
print(
    "TOP 15 TRAINING-ONLY CONFIGURATIONS "
    "BY MEAN MACRO PRECISION"
)
print("=" * 88)

display_cols = [
    "Config ID",
    "Feature Set",
    "class_weight",
    "max_depth",
    "min_samples_leaf",
    "max_features",
    "Mean Accuracy",
    "Mean Macro Precision",
    "Mean Weighted Precision",
    "Mean Macro Recall",
    "Mean Macro F1",
]

print(
    cv_summary[
        display_cols
    ].head(15).round(4).to_string(
        index=False
    )
)

best = cv_summary.iloc[0]

selected_name = best["Feature Set"]
selected_features = FEATURE_SETS[
    selected_name
]

selected_params = {
    "class_weight":
        None
        if best["class_weight"] == "None"
        else best["class_weight"],
    "max_depth":
        None
        if best["max_depth"] == "None"
        else int(best["max_depth"]),
    "min_samples_leaf":
        int(best["min_samples_leaf"]),
    "max_features":
        float(best["max_features"])
        if str(best["max_features"])
        not in {"sqrt", "log2"}
        else best["max_features"],
}

print("\nSelected by training-only Macro Precision:")
print("Feature set:", selected_name)
print("Parameters:", selected_params)
print(
    "CV Mean Macro Precision:",
    round(
        float(
            best["Mean Macro Precision"]
        ),
        4
    )
)
print(
    "CV Mean Macro F1:",
    round(
        float(best["Mean Macro F1"]),
        4
    )
)

# ============================================================
# FINAL HELD-OUT TEST
# ============================================================

dev = recompute_features(
    dev_raw,
    dev_raw
)

test = recompute_features(
    dev_raw,
    test_raw
)

pipe = make_pipeline(
    selected_features,
    selected_params
)

pipe.fit(
    dev[selected_features],
    dev[TARGET]
)

pred = pipe.predict(
    test[selected_features]
)

proba = pipe.predict_proba(
    test[selected_features]
)

classes = list(
    pipe.named_steps["model"].classes_
)

metrics = {
    "Selected Feature Set":
        selected_name,
    "Feature Count":
        len(selected_features),
    "Selected class_weight":
        str(
            selected_params[
                "class_weight"
            ]
        ),
    "Selected max_depth":
        str(
            selected_params[
                "max_depth"
            ]
        ),
    "Selected min_samples_leaf":
        selected_params[
            "min_samples_leaf"
        ],
    "Selected max_features":
        selected_params[
            "max_features"
        ],
    "Accuracy":
        accuracy_score(
            test[TARGET],
            pred
        ),
    "Macro Precision":
        precision_score(
            test[TARGET],
            pred,
            average="macro",
            zero_division=0,
        ),
    "Macro Recall":
        recall_score(
            test[TARGET],
            pred,
            average="macro",
            zero_division=0,
        ),
    "Macro F1":
        f1_score(
            test[TARGET],
            pred,
            average="macro",
            zero_division=0,
        ),
    "Weighted Precision":
        precision_score(
            test[TARGET],
            pred,
            average="weighted",
            zero_division=0,
        ),
    "Weighted Recall":
        recall_score(
            test[TARGET],
            pred,
            average="weighted",
            zero_division=0,
        ),
    "Weighted F1":
        f1_score(
            test[TARGET],
            pred,
            average="weighted",
            zero_division=0,
        ),
    "ROC-AUC Macro OvR":
        roc_auc_score(
            test[TARGET],
            proba,
            labels=classes,
            multi_class="ovr",
            average="macro",
        ),
    "ROC-AUC Weighted OvR":
        roc_auc_score(
            test[TARGET],
            proba,
            labels=classes,
            multi_class="ovr",
            average="weighted",
        ),
}

pd.DataFrame(
    [metrics]
).to_excel(
    OUT
    / "random_forest_precision_final_metrics.xlsx",
    index=False,
)

report = classification_report(
    test[TARGET],
    pred,
    labels=CLASS_ORDER,
    zero_division=0,
)

(
    OUT
    / "random_forest_precision_classification_report.txt"
).write_text(
    report,
    encoding="utf-8",
)

print("\n" + "=" * 88)
print("FINAL HELD-OUT TEST RESULTS")
print("=" * 88)
print(
    pd.Series(metrics).to_string()
)
print("\nClassification report:\n")
print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test[TARGET],
    pred,
    labels=CLASS_ORDER,
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_ORDER,
).plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "Random Forest — Precision-Focused Confusion Matrix"
)

fig.tight_layout()

fig.savefig(
    OUT
    / "random_forest_precision_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# MULTICLASS ROC
# ============================================================

ybin = label_binarize(
    test[TARGET],
    classes=classes,
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

for i, name in enumerate(classes):
    fpr, tpr, _ = roc_curve(
        ybin[:, i],
        proba[:, i],
    )

    ax.plot(
        fpr,
        tpr,
        label=(
            f"{name} "
            f"(AUC={auc(fpr, tpr):.3f})"
        ),
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Chance",
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.set_title(
    "Random Forest — Multiclass ROC (OvR)"
)

ax.legend()
fig.tight_layout()

fig.savefig(
    OUT
    / "random_forest_precision_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

prep = pipe.named_steps[
    "preprocessor"
]

model = pipe.named_steps[
    "model"
]

importance = pd.DataFrame({
    "Encoded Feature":
        prep.get_feature_names_out(),
    "Importance":
        model.feature_importances_,
}).sort_values(
    "Importance",
    ascending=False,
).reset_index(drop=True)

importance.to_excel(
    OUT
    / "random_forest_precision_feature_importance.xlsx",
    index=False,
)

top = importance.head(20)

fig, ax = plt.subplots(
    figsize=(9, 8)
)

ax.barh(
    top["Encoded Feature"][::-1],
    top["Importance"][::-1],
)

ax.set_xlabel(
    "Feature Importance"
)

ax.set_title(
    "Random Forest — Top 20 Feature Importances"
)

fig.tight_layout()

fig.savefig(
    OUT
    / "random_forest_precision_feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("\nSelected features:")
for feature in selected_features:
    print(" -", feature)

print("\nTop 15 encoded feature importances:")
print(
    importance.head(15)
    .round(4)
    .to_string(index=False)
)

print("\nOutput folder:")
print(OUT)

print("\nMethodology:")
print(
    "1. The last 20% of time-ordered observations "
    "remained untouched until final testing."
)
print(
    "2. TimeSeriesSplit was used only inside the first 80%."
)
print(
    "3. Training-derived traffic reference values were "
    "recomputed independently inside each fold."
)
print(
    "4. Feature set and Random Forest hyperparameters were "
    "selected by mean CV Macro Precision."
)
print(
    "5. Macro F1, recall, accuracy and ROC-AUC are still "
    "reported so the precision-recall trade-off is visible."
)
print(
    "6. class_weight=None is allowed because balanced_subsample "
    "can increase minority recall while reducing precision."
)
print(
    "7. The final test set was never used to choose features, "
    "class weights, or hyperparameters."
)
