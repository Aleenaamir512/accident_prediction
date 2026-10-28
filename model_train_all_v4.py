# ALL-IN-ONE COMPARISON: Logistic Regression, Random Forest, XGBoost, LightGBM
# under IDENTICAL rules:
#   * same rolling-origin backtest (each test window comes AFTER its training data)
#   * same feature sets (including leakage-safe HISTORY features)
#   * same probability handling: raw / sigmoid-calibrated
#   * same class-specific thresholds + "Uncertain" option, chosen by the SAME objective
#     (worst-class precision, small-sample safe), using development data only
#   * same reporting: per-class precision with 95% confidence intervals
#
# Your old RF / XGBoost / LightGBM scripts used plain argmax and a single 20% hold-out,
# while the logistic script used thresholds + reject option. Those numbers cannot be
# compared. This script makes the comparison fair.
#
# Each model is also reported with plain ARGMAX ("Argmax" rows) so you can see what the
# thresholding adds for each model.
#
# Run time: roughly 15-40 minutes for the full run (depends on your CPU).
# Set QUICK_RUN = True for a fast smoke test (about 5x faster, smaller search).

from pathlib import Path
from itertools import product
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

try:
    from xgboost import XGBClassifier
except Exception as error:  # pragma: no cover
    XGBClassifier = None
    print("NOTE: xgboost not available -> XGBoost will be skipped:", error)

try:
    from lightgbm import LGBMClassifier
except Exception as error:  # pragma: no cover
    LGBMClassifier = None
    print("NOTE: lightgbm not available -> LightGBM will be skipped:", error)

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1. SETTINGS
# ============================================================

QUICK_RUN = False  # True = small search for a fast smoke test

MODELS_TO_RUN = ["Logistic", "RandomForest", "XGBoost", "LightGBM"]

BASE = Path(__file__).resolve().parent
INPUT_FILE = BASE / "karachi_traffic_feature_engineered_final_v2.xlsx"
OUTPUT_FOLDER = BASE / "Model_Output_All_Models_v4"
OUTPUT_FOLDER.mkdir(exist_ok=True)

TARGET = "Accident Risk"
CLASS_ORDER = ["Low", "Medium", "High"]
UNCERTAIN = "Uncertain"
class_to_index = {c: i for i, c in enumerate(CLASS_ORDER)}

# ---- Rolling-origin backtest -------------------------------------------
OUTER_STARTS = [0.55, 0.70, 0.85] if not QUICK_RUN else [0.70, 0.85]
WINDOW_FRACTION = 0.15

# ---- Safeguards for threshold selection (tuning data only) -------------
MIN_COVERAGE = 0.70
MIN_MACRO_RECALL = 0.40
MIN_CLASS_RECALL = 0.25
MIN_PRED_PER_CLASS = 15
WILSON_Z = 1.28

# ---- Threshold grids (per class) ---------------------------------------
LOW_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
MEDIUM_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
HIGH_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
MARGIN_GRID = [0.00, 0.05, 0.10, 0.15]

PR_TABLE_THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.91, 0.05)]

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

NO_AREA = [f for f in AUGMENTED if f != "Area"]

INTERACTIONS = [
    "Load x Rush",
    "Load x Rainy",
    "Load x Construction",
    "Load x Intersection",
    "Load Ratio Squared",
    "Rush x Rainy",
]

INTERACTION_SET = (
    AUGMENTED
    + [
        "Traffic Load Ratio",
        "Rush Hour",
        "Construction",
        "Near Intersection",
        "Rainfall",
        "Visibility",
        "Is Rainy",
        "Is Foggy",
    ]
    + INTERACTIONS
)

# History features: built ONLY from earlier rows (no labels, no future data).
HISTORY = [
    "Prev Vehicle Count",
    "Rolling Vehicle Mean (prev 5)",
    "Vehicle Surge",
    "Hours Since Last Record",
    "City Load Last 6 Rows",
    "Rain Last 3 Rows",
]

HISTORY_SET = AUGMENTED + [
    "Traffic Load Ratio",
    "Rush Hour",
    "Construction",
    "Near Intersection",
    "Rainfall",
    "Visibility",
] + HISTORY

# Your ORIGINAL 32-feature list (includes the original "Weather" label,
# "Time of Day" and "Month"). Kept so you can test whether they help.
ALL_ORIGINAL_32 = [
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

# Same list WITHOUT original Weather / Time of Day / Month.
ALL_NO_ORIGINAL_WEATHER = [
    f for f in ALL_ORIGINAL_32 if f not in {"Weather", "Time of Day", "Month"}
]

FEATURE_SETS = {
    "Set_A_Core": CORE,
    "Set_B_Augmented": AUGMENTED,
    "Set_C_Augmented_No_Area": NO_AREA,
    "Set_D_All_No_OrigWeather": ALL_NO_ORIGINAL_WEATHER,
    "Set_E_Interactions": INTERACTION_SET,
    "Set_F_History": HISTORY_SET,
    "Set_G_All_Original32": ALL_ORIGINAL_32,
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
# 3. MODEL SEARCH SPACES
# ============================================================

ALPHAS = {
    "Logistic": [0.0, 0.25, 0.5],
    "RandomForest": [0.0, 0.5],
    "XGBoost": [0.0, 0.5],
    "LightGBM": [0.0, 0.5],
}

MODEL_FEATURE_SETS = {
    "Logistic": [
        "Set_A_Core",
        "Set_B_Augmented",
        "Set_C_Augmented_No_Area",
        "Set_D_All_No_OrigWeather",
        "Set_E_Interactions",
        "Set_F_History",
        "Set_G_All_Original32",
    ],
    "RandomForest": [
        "Set_A_Core",
        "Set_B_Augmented",
        "Set_D_All_No_OrigWeather",
        "Set_F_History",
        "Set_G_All_Original32",
    ],
    "XGBoost": [
        "Set_A_Core",
        "Set_B_Augmented",
        "Set_D_All_No_OrigWeather",
        "Set_F_History",
        "Set_G_All_Original32",
    ],
    "LightGBM": [
        "Set_A_Core",
        "Set_B_Augmented",
        "Set_D_All_No_OrigWeather",
        "Set_F_History",
        "Set_G_All_Original32",
    ],
}

PROFILES = {
    "Logistic": [
        {"name": "C=0.05", "C": 0.05},
        {"name": "C=0.2", "C": 0.20},
        {"name": "C=1", "C": 1.0},
        {"name": "C=5", "C": 5.0},
    ],
    "RandomForest": [
        {"name": "RF1_leaf3", "max_depth": None, "min_samples_leaf": 3, "max_features": "sqrt"},
        {"name": "RF2_depth10", "max_depth": 10, "min_samples_leaf": 3, "max_features": "sqrt"},
        {"name": "RF3_leaf1_mf07", "max_depth": None, "min_samples_leaf": 1, "max_features": 0.70},
        {"name": "RF4_depth8_leaf5", "max_depth": 8, "min_samples_leaf": 5, "max_features": "sqrt"},
    ],
    "XGBoost": [
        {
            "name": "P1_Conservative", "n_estimators": 300, "max_depth": 3,
            "learning_rate": 0.05, "min_child_weight": 2, "subsample": 0.85,
            "colsample_bytree": 0.80, "reg_lambda": 1.5, "reg_alpha": 0.0,
        },
        {
            "name": "P2_Slow_Shallow", "n_estimators": 500, "max_depth": 3,
            "learning_rate": 0.03, "min_child_weight": 1, "subsample": 0.85,
            "colsample_bytree": 0.90, "reg_lambda": 1.0, "reg_alpha": 0.0,
        },
        {
            "name": "P3_Moderate", "n_estimators": 350, "max_depth": 4,
            "learning_rate": 0.05, "min_child_weight": 2, "subsample": 0.80,
            "colsample_bytree": 0.80, "reg_lambda": 1.0, "reg_alpha": 0.10,
        },
        {
            "name": "P4_Deeper_Regularized", "n_estimators": 300, "max_depth": 5,
            "learning_rate": 0.04, "min_child_weight": 3, "subsample": 0.80,
            "colsample_bytree": 0.75, "reg_lambda": 2.0, "reg_alpha": 0.20,
        },
    ],
    "LightGBM": [
        {
            "name": "P1_Conservative", "n_estimators": 300, "learning_rate": 0.05,
            "num_leaves": 15, "max_depth": 5, "min_child_samples": 20,
            "subsample": 0.85, "colsample_bytree": 0.80, "reg_lambda": 1.5, "reg_alpha": 0.0,
        },
        {
            "name": "P2_Shallow_Regularized", "n_estimators": 450, "learning_rate": 0.03,
            "num_leaves": 15, "max_depth": 4, "min_child_samples": 25,
            "subsample": 0.85, "colsample_bytree": 0.90, "reg_lambda": 2.0, "reg_alpha": 0.10,
        },
        {
            "name": "P3_Moderate", "n_estimators": 350, "learning_rate": 0.05,
            "num_leaves": 31, "max_depth": -1, "min_child_samples": 20,
            "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 1.0, "reg_alpha": 0.10,
        },
        {
            "name": "P4_Regularized", "n_estimators": 300, "learning_rate": 0.04,
            "num_leaves": 23, "max_depth": 6, "min_child_samples": 30,
            "subsample": 0.80, "colsample_bytree": 0.75, "reg_lambda": 2.5, "reg_alpha": 0.25,
        },
    ],
}

if QUICK_RUN:
    ALPHAS = {
        "Logistic": [0.0, 0.5],
        "RandomForest": [0.5],
        "XGBoost": [0.5],
        "LightGBM": [0.5],
    }
    MODEL_FEATURE_SETS = {
        "Logistic": ["Set_B_Augmented", "Set_E_Interactions", "Set_F_History"],
        "RandomForest": ["Set_B_Augmented", "Set_F_History", "Set_G_All_Original32"],
        "XGBoost": ["Set_B_Augmented", "Set_F_History", "Set_G_All_Original32"],
        "LightGBM": ["Set_B_Augmented", "Set_F_History", "Set_G_All_Original32"],
    }
    PROFILES = {k: v[:2] for k, v in PROFILES.items()}

# ============================================================
# 4. FEATURE HELPERS
# ============================================================


def add_history_features(frame):
    """
    Leakage-safe history features. Every value uses ONLY earlier rows
    (shift(1) before any rolling window) and never uses Accident Risk.
    Frame must be sorted by Date-Time.
    """
    out = frame.copy()
    group = out.groupby(["Area", "Road Type"], sort=False)

    out["Prev Vehicle Count"] = group["Vehicle Count"].shift(1)
    out["Rolling Vehicle Mean (prev 5)"] = group["Vehicle Count"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    gap_hours = group["Date-Time"].diff().dt.total_seconds() / 3600.0
    out["Hours Since Last Record"] = gap_hours.clip(upper=168.0)

    # First record of a group has no history: neutral fill values.
    out["Prev Vehicle Count"] = out["Prev Vehicle Count"].fillna(out["Vehicle Count"])
    out["Rolling Vehicle Mean (prev 5)"] = out["Rolling Vehicle Mean (prev 5)"].fillna(
        out["Vehicle Count"]
    )
    out["Hours Since Last Record"] = out["Hours Since Last Record"].fillna(168.0)
    out["Vehicle Surge"] = out["Vehicle Count"] - out["Rolling Vehicle Mean (prev 5)"]

    # City-wide context from the previous rows (all areas together).
    out["City Load Last 6 Rows"] = (
        out["Vehicle Count"].shift(1).rolling(6, min_periods=1).mean()
    ).fillna(out["Vehicle Count"])
    if "Rainfall" in out.columns:
        out["Rain Last 3 Rows"] = (
            out["Rainfall"].shift(1).rolling(3, min_periods=1).sum()
        ).fillna(0.0)
    else:
        out["Rain Last 3 Rows"] = 0.0
    return out


def recompute_traffic_features(reference_train, frame_to_transform):
    """Fit traffic references on reference_train only, transform frame_to_transform."""
    ref = reference_train.copy()
    out = frame_to_transform.copy()

    ref["Traffic Load per Lane"] = ref["Vehicle Count"] / ref["Number of Lanes"]
    out["Traffic Load per Lane"] = out["Vehicle Count"] / out["Number of Lanes"]

    group_ref = (
        ref.groupby(["Area", "Road Type"])["Traffic Load per Lane"]
        .agg(["count", lambda s: s.quantile(0.95)])
        .reset_index()
    )
    group_ref.columns = ["Area", "Road Type", "Group Count", "Group Reference"]

    road_ref = ref.groupby("Road Type")["Traffic Load per Lane"].quantile(0.95).to_dict()
    overall_ref = float(ref["Traffic Load per Lane"].quantile(0.95))

    out = out.merge(group_ref, on=["Area", "Road Type"], how="left")

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

    out["Reference Traffic Load"] = out.apply(choose_reference, axis=1)
    out["Traffic Load Ratio"] = out["Traffic Load per Lane"] / out["Reference Traffic Load"]

    def congestion(ratio):
        if ratio <= 0.35:
            return "Low"
        if ratio <= 0.75:
            return "Moderate"
        if ratio <= 0.95:
            return "High"
        return "Severe"

    out["Congestion Level"] = out["Traffic Load Ratio"].apply(congestion)

    def estimate_speed(row):
        free_flow = FREE_FLOW_SPEED.get(row["Road Type"], 40.0)
        ratio = min(float(row["Traffic Load Ratio"]), 1.20)
        factor = 1.0 - 0.50 * min(ratio, 1.0) - 0.15 * max(ratio - 1.0, 0.0)
        factor = np.clip(factor, 0.40, 1.00)
        speed = free_flow * factor
        weather = str(row.get("ML Weather", "")).strip().lower()
        if weather == "rainy":
            speed *= 0.90
        elif weather == "foggy":
            speed *= 0.82
        if int(row.get("Construction", 0)) == 1:
            speed *= 0.80
        if int(row.get("Near Intersection", 0)) == 1:
            speed *= 0.92
        return round(float(np.clip(speed, 8.0, 85.0)), 1)

    out["Estimated Average Speed"] = out.apply(estimate_speed, axis=1)

    ratio = out["Traffic Load Ratio"]
    rush = out["Rush Hour"] if "Rush Hour" in out else 0
    rainy = out["Is Rainy"] if "Is Rainy" in out else 0
    constr = out["Construction"] if "Construction" in out else 0
    inter = out["Near Intersection"] if "Near Intersection" in out else 0

    out["Load x Rush"] = ratio * rush
    out["Load x Rainy"] = ratio * rainy
    out["Load x Construction"] = ratio * constr
    out["Load x Intersection"] = ratio * inter
    out["Load Ratio Squared"] = ratio ** 2
    out["Rush x Rainy"] = rush * rainy

    return out.drop(columns=["Group Count", "Group Reference"])


# ============================================================
# 5. MODELS
# ============================================================


def make_sample_weights(y_idx, alpha):
    """alpha=0: none, alpha=1: full inverse-frequency balancing (from training data only)."""
    if alpha == 0.0:
        return None
    y_series = pd.Series(y_idx)
    counts = y_series.value_counts()
    n, k = len(y_series), len(counts)
    softened = {cls: float((n / (k * cnt)) ** alpha) for cls, cnt in counts.items()}
    return y_series.map(softened).to_numpy()


def build_model(model_name, p):
    if model_name == "Logistic":
        return LogisticRegression(C=p["C"], max_iter=5000, solver="lbfgs")
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"],
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "XGBoost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            min_child_weight=p["min_child_weight"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            reg_lambda=p["reg_lambda"],
            reg_alpha=p["reg_alpha"],
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
    if model_name == "LightGBM":
        return LGBMClassifier(
            objective="multiclass",
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            num_leaves=p["num_leaves"],
            max_depth=p["max_depth"],
            min_child_samples=p["min_child_samples"],
            subsample=p["subsample"],
            subsample_freq=1,
            colsample_bytree=p["colsample_bytree"],
            reg_lambda=p["reg_lambda"],
            reg_alpha=p["reg_alpha"],
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(model_name)


def make_pipeline(model_name, features, params):
    categorical = [f for f in features if f in CATEGORICAL_FEATURES]
    numeric = [f for f in features if f not in categorical]
    numeric_step = StandardScaler() if model_name == "Logistic" else "passthrough"

    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", numeric_step, numeric),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", build_model(model_name, params))])


def fit_pipeline(pipeline, X, y_idx, alpha):
    weights = make_sample_weights(y_idx, alpha)
    kwargs = {"model__sample_weight": weights} if weights is not None else {}
    pipeline.fit(X, y_idx, **kwargs)
    return pipeline


# ============================================================
# 6. CALIBRATION, SELECTIVE PREDICTION, METRICS
# ============================================================


def safe_logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def fit_sigmoid_calibrators(raw_proba, y_true):
    calibrators = []
    for i, name in enumerate(CLASS_ORDER):
        x = safe_logit(raw_proba[:, i]).reshape(-1, 1)
        y_bin = (np.asarray(y_true) == name).astype(int)
        cal = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        cal.fit(x, y_bin)
        calibrators.append(cal)
    return calibrators


def apply_sigmoid_calibration(raw_proba, calibrators):
    out = np.zeros_like(raw_proba, dtype=float)
    for i, cal in enumerate(calibrators):
        x = safe_logit(raw_proba[:, i]).reshape(-1, 1)
        out[:, i] = cal.predict_proba(x)[:, 1]
    sums = out.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return out / sums


def wilson_interval(k, n, z=1.96):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    safe_n = np.where(n > 0, n, 1.0)
    p = k / safe_n
    denom = 1.0 + z * z / safe_n
    centre = p + z * z / (2.0 * safe_n)
    adj = z * np.sqrt(p * (1.0 - p) / safe_n + z * z / (4.0 * safe_n * safe_n))
    lower = np.where(n > 0, (centre - adj) / denom, 0.0)
    upper = np.where(n > 0, (centre + adj) / denom, 0.0)
    return lower, upper


def selective_predict(probabilities, thresholds, margin_threshold):
    top_idx = np.argmax(probabilities, axis=1)
    srt = np.sort(probabilities, axis=1)
    top_prob = srt[:, -1]
    margins = top_prob - srt[:, -2]
    keep = (top_prob >= np.asarray(thresholds)[top_idx]) & (margins >= margin_threshold)
    labels = np.asarray(CLASS_ORDER, dtype=object)
    pred = np.full(len(probabilities), UNCERTAIN, dtype=object)
    pred[keep] = labels[top_idx[keep]]
    return pred


def argmax_predict(probabilities):
    return np.asarray([CLASS_ORDER[i] for i in np.argmax(probabilities, axis=1)], dtype=object)


def selective_metrics(y_true, predictions):
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    covered = predictions != UNCERTAIN

    p_cls = precision_score(y_true, predictions, labels=CLASS_ORDER, average=None, zero_division=0)
    r_cls = recall_score(y_true, predictions, labels=CLASS_ORDER, average=None, zero_division=0)
    n_pred = np.array([np.sum(predictions == c) for c in CLASS_ORDER])
    n_tp = np.array([np.sum((predictions == c) & (y_true == c)) for c in CLASS_ORDER])
    lo, hi = wilson_interval(n_tp, n_pred, 1.96)

    row = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Macro Precision": precision_score(
            y_true, predictions, labels=CLASS_ORDER, average="macro", zero_division=0
        ),
        "Worst-Class Precision": float(np.min(p_cls)),
        "Worst-Class Precision CI95 Low": float(np.min(lo)),
        "Macro Recall": recall_score(
            y_true, predictions, labels=CLASS_ORDER, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_true, predictions, labels=CLASS_ORDER, average="macro", zero_division=0
        ),
        "Macro F0.5": fbeta_score(
            y_true, predictions, labels=CLASS_ORDER, average="macro", beta=0.5, zero_division=0
        ),
        "Coverage": float(np.mean(covered)),
        "Uncertain Count": int(np.sum(~covered)),
        "Accuracy on Covered Rows": (
            float(np.mean(y_true[covered] == predictions[covered])) if covered.any() else 0.0
        ),
    }
    for i, c in enumerate(CLASS_ORDER):
        row[f"Precision {c}"] = float(p_cls[i])
        row[f"Precision {c} CI95 Low"] = float(lo[i])
        row[f"Precision {c} CI95 High"] = float(hi[i])
        row[f"Recall {c}"] = float(r_cls[i])
        row[f"Predicted {c}"] = int(n_pred[i])
    return row


FAST_COLUMNS = [
    "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Macro F0.5",
    "Coverage", "Uncertain Count",
    "Precision Low", "Precision Medium", "Precision High",
    "Recall Low", "Recall Medium", "Recall High",
    "Pred Low", "Pred Medium", "Pred High",
    "LB Low", "LB Medium", "LB High",
]


def fast_metrics(y_idx, pred_idx):
    n = len(y_idx)
    cm = np.bincount(y_idx * 4 + (pred_idx + 1), minlength=12).reshape(3, 4)
    tp = np.diag(cm[:, 1:]).astype(float)
    pred_counts = cm[:, 1:].sum(axis=0).astype(float)
    true_counts = cm.sum(axis=1).astype(float)

    precision = np.divide(tp, pred_counts, out=np.zeros(3), where=pred_counts > 0)
    recall = np.divide(tp, true_counts, out=np.zeros(3), where=true_counts > 0)

    def fbeta(beta):
        b2 = beta * beta
        denom = b2 * precision + recall
        return np.divide((1 + b2) * precision * recall, denom, out=np.zeros(3), where=denom > 0)

    lower, _ = wilson_interval(tp, pred_counts, WILSON_Z)
    uncertain = float(cm[:, 0].sum())
    return [
        tp.sum() / n, precision.mean(), recall.mean(), fbeta(1.0).mean(), fbeta(0.5).mean(),
        1.0 - uncertain / n, uncertain,
        *precision, *recall, *pred_counts, *lower,
    ]


# ============================================================
# 7. LOAD DATA
# ============================================================

df = pd.read_excel(INPUT_FILE, sheet_name="Data")
df["Date-Time"] = pd.to_datetime(df["Date-Time"], errors="coerce")
if df["Date-Time"].isna().any():
    raise ValueError("Some Date-Time values could not be converted.")
df = df.sort_values("Date-Time", kind="mergesort").reset_index(drop=True)
df = add_history_features(df)

print("=" * 98)
print("ALL MODELS - SAME RULES - ROLLING BACKTEST" + ("  [QUICK RUN]" if QUICK_RUN else ""))
print("=" * 98)
print("Total rows:", len(df))
print("Class distribution (all data):")
print(df[TARGET].value_counts(normalize=True).round(3).to_string())

models_to_run = []
for name in MODELS_TO_RUN:
    if name == "XGBoost" and XGBClassifier is None:
        print("Skipping XGBoost (not installed).")
        continue
    if name == "LightGBM" and LGBMClassifier is None:
        print("Skipping LightGBM (not installed).")
        continue
    models_to_run.append(name)

_probe = recompute_traffic_features(df.iloc[: max(200, len(df) // 2)], df.head(50))
available = set(_probe.columns)
for set_name in list(FEATURE_SETS):
    wanted = FEATURE_SETS[set_name]
    kept = [f for f in wanted if f in available]
    dropped = [f for f in wanted if f not in available]
    if dropped:
        print(f"WARNING: {set_name} - dropping missing features: {dropped}")
    FEATURE_SETS[set_name] = kept

threshold_triples = np.array(list(product(LOW_GRID, MEDIUM_GRID, HIGH_GRID)), dtype=float)
combo_thresholds = np.repeat(threshold_triples, len(MARGIN_GRID), axis=0)
combo_margins = np.tile(MARGIN_GRID, len(threshold_triples))

# ============================================================
# 8. SELECTION (development data only)
# ============================================================


def build_fold_cache(development_df, tag):
    cache = []
    for fold, (tr_idx, va_idx) in enumerate(TimeSeriesSplit(n_splits=5).split(development_df), 1):
        train_raw = development_df.iloc[tr_idx].copy()
        valid_raw = development_df.iloc[va_idx].copy()
        if train_raw[TARGET].nunique() < 3 or valid_raw[TARGET].nunique() < 3:
            continue
        cache.append(
            {
                "fold": fold,
                "train": recompute_traffic_features(train_raw, train_raw),
                "validation": recompute_traffic_features(train_raw, valid_raw),
            }
        )
    if not cache:
        raise ValueError(f"[{tag}] No usable time-series folds.")
    return cache


def run_selection(model_name, fold_cache, tag):
    configs = []
    for set_name in MODEL_FEATURE_SETS[model_name]:
        for alpha in ALPHAS[model_name]:
            for params in PROFILES[model_name]:
                configs.append(
                    {"Feature Set": set_name, "Features": FEATURE_SETS[set_name],
                     "Alpha": alpha, "Params": params}
                )

    frames, config_rows, artifacts = [], [], {}
    for config_id, cfg in enumerate(configs, start=1):
        if config_id % 10 == 1 or config_id == len(configs):
            print(f"  [{tag}] {model_name}: configuration {config_id}/{len(configs)}", flush=True)

        parts = []
        for item in fold_cache:
            train, valid = item["train"], item["validation"]
            y_train = train[TARGET].map(class_to_index).astype(int).to_numpy()
            pipe = make_pipeline(model_name, cfg["Features"], cfg["Params"])
            fit_pipeline(pipe, train[cfg["Features"]], y_train, cfg["Alpha"])
            proba = pipe.predict_proba(valid[cfg["Features"]])
            parts.append(
                pd.DataFrame(
                    {
                        "Date-Time": valid["Date-Time"].to_numpy(),
                        "Actual": valid[TARGET].to_numpy(),
                        "P_Low": proba[:, 0],
                        "P_Medium": proba[:, 1],
                        "P_High": proba[:, 2],
                    }
                )
            )

        oof = (
            pd.concat(parts, ignore_index=True)
            .sort_values("Date-Time", kind="mergesort")
            .reset_index(drop=True)
        )
        split = int(len(oof) * 0.60)
        cal_part, thr_part = oof.iloc[:split], oof.iloc[split:]
        cols = ["P_Low", "P_Medium", "P_High"]

        calibrators = fit_sigmoid_calibrators(cal_part[cols].to_numpy(), cal_part["Actual"].to_numpy())
        thr_raw = thr_part[cols].to_numpy()
        thr_cal = apply_sigmoid_calibration(thr_raw, calibrators)
        artifacts[config_id] = {**cfg, "Calibrators": calibrators}

        y_labels = thr_part["Actual"].to_numpy()
        y_idx = np.array([class_to_index[v] for v in y_labels])

        for mode, matrix in {"Raw": thr_raw, "Calibrated": thr_cal}.items():
            base = selective_metrics(y_labels, argmax_predict(matrix))
            config_rows.append(
                {
                    "Model": model_name, "Config ID": config_id,
                    "Feature Set": cfg["Feature Set"], "Weight Alpha": cfg["Alpha"],
                    "Params": cfg["Params"]["name"], "Probability Mode": mode,
                    "Argmax Macro Precision": base["Macro Precision"],
                    "Argmax Worst-Class Precision": base["Worst-Class Precision"],
                    "Argmax Macro Recall": base["Macro Recall"],
                }
            )

            top_idx = np.argmax(matrix, axis=1)
            srt = np.sort(matrix, axis=1)
            top_prob = srt[:, -1]
            margins = top_prob - srt[:, -2]

            results = np.empty((len(combo_margins), len(FAST_COLUMNS)))
            for i in range(len(combo_margins)):
                keep = (top_prob >= combo_thresholds[i][top_idx]) & (margins >= combo_margins[i])
                results[i] = fast_metrics(y_idx, np.where(keep, top_idx, -1))

            frame = pd.DataFrame(results, columns=FAST_COLUMNS)
            frame["Worst-Class Precision"] = frame[
                ["Precision Low", "Precision Medium", "Precision High"]
            ].min(axis=1)
            frame["Worst-Class Precision LB"] = frame[["LB Low", "LB Medium", "LB High"]].min(axis=1)
            frame["Min Class Recall"] = frame[
                ["Recall Low", "Recall Medium", "Recall High"]
            ].min(axis=1)
            frame["Min Class Predictions"] = frame[
                ["Pred Low", "Pred Medium", "Pred High"]
            ].min(axis=1)

            # Keep only rows that can ever be used by any fallback tier (saves memory).
            frame = frame[frame["Coverage"] >= 0.50].copy()
            if frame.empty:
                continue
            frame.insert(0, "Margin Threshold", combo_margins[frame.index])
            frame.insert(0, "High Threshold", combo_thresholds[frame.index, 2])
            frame.insert(0, "Medium Threshold", combo_thresholds[frame.index, 1])
            frame.insert(0, "Low Threshold", combo_thresholds[frame.index, 0])
            frame.insert(0, "Probability Mode", mode)
            frame.insert(0, "Config ID", config_id)
            frames.append(frame)

    search_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def pick(min_cov, min_macro_rec, min_cls_rec, min_preds):
        if search_df.empty:
            return search_df
        return search_df[
            (search_df["Coverage"] >= min_cov)
            & (search_df["Macro Recall"] >= min_macro_rec)
            & (search_df["Min Class Recall"] >= min_cls_rec)
            & (search_df["Min Class Predictions"] >= min_preds)
        ]

    # Graded fallbacks so one weak model can never crash the whole comparison.
    #   level 0 = normal safeguards
    #   level 1 = relaxed safeguards
    #   level 2 = coverage >= 0.50 only (a class may be (almost) never predicted)
    #   level 3 = plain argmax (no thresholds, no Uncertain)
    level = 0
    eligible = pick(MIN_COVERAGE, MIN_MACRO_RECALL, MIN_CLASS_RECALL, MIN_PRED_PER_CLASS)
    if eligible.empty:
        level = 1
        eligible = pick(0.60, 0.30, 0.15, 8)
    if eligible.empty:
        level = 2
        eligible = pick(0.50, 0.0, 0.0, 0)
    sort_columns = ["Worst-Class Precision LB", "Worst-Class Precision", "Coverage"]
    if level == 2:
        sort_columns = ["Macro Precision", "Coverage"]

    if eligible.empty:
        level = 3
        cr = pd.DataFrame(config_rows).sort_values(
            ["Argmax Worst-Class Precision", "Argmax Macro Precision"], ascending=False
        )
        top = cr.iloc[0]
        best = pd.Series(
            {
                "Config ID": top["Config ID"], "Probability Mode": top["Probability Mode"],
                "Low Threshold": 0.0, "Medium Threshold": 0.0, "High Threshold": 0.0,
                "Margin Threshold": 0.0,
                "Worst-Class Precision": top["Argmax Worst-Class Precision"],
                "Worst-Class Precision LB": np.nan, "Coverage": 1.0,
            }
        )
    else:
        eligible = eligible.sort_values(sort_columns, ascending=False).reset_index(drop=True)
        best = eligible.iloc[0]

    if level > 0:
        labels = {1: "safeguards RELAXED", 2: "coverage-only fallback (a class is hard to predict)",
                  3: "plain ARGMAX fallback (no usable thresholds)"}
        print(f"  [{tag}] {model_name}: {labels[level]}.")

    artifact = artifacts[int(best["Config ID"])]

    return {
        "best": best,
        "artifact": artifact,
        "mode": best["Probability Mode"],
        "thresholds": [float(best["Low Threshold"]), float(best["Medium Threshold"]),
                       float(best["High Threshold"])],
        "margin": float(best["Margin Threshold"]),
        "level": level,
        "config_rows": config_rows,
    }


def evaluate_window(model_name, selection, dev_feat, eval_feat):
    art = selection["artifact"]
    features = art["Features"]
    y_dev = dev_feat[TARGET].map(class_to_index).astype(int).to_numpy()

    pipe = make_pipeline(model_name, features, art["Params"])
    fit_pipeline(pipe, dev_feat[features], y_dev, art["Alpha"])

    raw = pipe.predict_proba(eval_feat[features])
    cal = apply_sigmoid_calibration(raw, art["Calibrators"])
    chosen = cal if selection["mode"] == "Calibrated" else raw
    return {
        "raw": raw,
        "selected": chosen,
        "selective_pred": selective_predict(chosen, selection["thresholds"], selection["margin"]),
        "argmax_pred": argmax_predict(raw),
    }


# ============================================================
# 9. ROLLING-ORIGIN BACKTEST FOR ALL MODELS
# ============================================================

n_rows = len(df)
window_rows = []
selected_rows = []
pooled = {m: [] for m in models_to_run}
truth_parts = []

for window_number, start_fraction in enumerate(OUTER_STARTS, start=1):
    start = int(n_rows * start_fraction)
    end = min(int(n_rows * (start_fraction + WINDOW_FRACTION) + 0.5), n_rows)
    if window_number == len(OUTER_STARTS):
        end = n_rows

    development_df = df.iloc[:start].copy()
    evaluation_raw = df.iloc[start:end].copy()
    if len(evaluation_raw) == 0:
        continue

    tag = f"window {window_number}/{len(OUTER_STARTS)}"
    print("\n" + "=" * 98)
    print(
        f"{tag.upper()}: develop on rows 0-{start - 1}, test on rows {start}-{end - 1} "
        f"({evaluation_raw['Date-Time'].min().date()} to {evaluation_raw['Date-Time'].max().date()})"
    )
    print("=" * 98)

    fold_cache = build_fold_cache(development_df, tag)
    dev_feat = recompute_traffic_features(development_df, development_df)
    eval_feat = recompute_traffic_features(development_df, evaluation_raw)
    y_eval = eval_feat[TARGET].to_numpy()

    truth_parts.append(
        pd.DataFrame(
            {"Window": window_number, "Date-Time": eval_feat["Date-Time"].to_numpy(), "Actual": y_eval}
        )
    )

    for model_name in models_to_run:
        selection = run_selection(model_name, fold_cache, tag)
        result = evaluate_window(model_name, selection, dev_feat, eval_feat)
        best = selection["best"]
        art = selection["artifact"]

        sel_m = selective_metrics(y_eval, result["selective_pred"])
        arg_m = selective_metrics(y_eval, result["argmax_pred"])

        print(
            f"  >> {model_name}: {art['Feature Set']} | alpha={art['Alpha']} | {art['Params']['name']} | "
            f"{selection['mode']} | thr L/M/H={selection['thresholds']} | margin={selection['margin']}"
            + (f" (fallback level {selection['level']})" if selection["level"] else "")
        )
        print(
            f"     tuning worst-class precision {best['Worst-Class Precision']:.3f} -> "
            f"TEST Low {sel_m['Precision Low']:.3f} | Medium {sel_m['Precision Medium']:.3f} | "
            f"High {sel_m['Precision High']:.3f} | coverage {sel_m['Coverage']:.3f}"
        )

        selected_rows.append(
            {
                "Window": window_number, "Model": model_name, "Feature Set": art["Feature Set"],
                "Weight Alpha": art["Alpha"], "Params": art["Params"]["name"],
                "Probability Mode": selection["mode"],
                "Low Thr": selection["thresholds"][0], "Medium Thr": selection["thresholds"][1],
                "High Thr": selection["thresholds"][2], "Margin": selection["margin"],
                "Fallback Level (0=normal)": selection["level"],
                "Tuning Worst-Class Precision": float(best["Worst-Class Precision"]),
            }
        )
        window_rows.append(
            {
                "Window": window_number, "Model": model_name, "Test Rows": len(y_eval),
                **{f"Selective {k}": v for k, v in sel_m.items()},
                "Argmax Worst-Class Precision": arg_m["Worst-Class Precision"],
                "Argmax Macro Precision": arg_m["Macro Precision"],
            }
        )
        pooled[model_name].append(
            pd.DataFrame(
                {
                    "P_Low": result["selected"][:, 0],
                    "P_Medium": result["selected"][:, 1],
                    "P_High": result["selected"][:, 2],
                    "Selective Prediction": result["selective_pred"],
                    "Argmax Prediction": result["argmax_pred"],
                }
            )
        )

# ============================================================
# 10. POOLED COMPARISON
# ============================================================

truth = pd.concat(truth_parts, ignore_index=True)
y_pool = truth["Actual"].to_numpy()
y_pool_numeric = pd.Series(y_pool).map(class_to_index).astype(int).to_numpy()
majority_accuracy = float(np.mean(y_pool == "Medium"))

pooled_frames, comparison_rows = {}, []
for model_name in models_to_run:
    frame = pd.concat(pooled[model_name], ignore_index=True)
    pooled_frames[model_name] = frame
    sel = selective_metrics(y_pool, frame["Selective Prediction"].to_numpy())
    arg = selective_metrics(y_pool, frame["Argmax Prediction"].to_numpy())
    auc_value = roc_auc_score(
        y_pool_numeric, frame[["P_Low", "P_Medium", "P_High"]].to_numpy(),
        multi_class="ovr", average="macro", labels=[0, 1, 2],
    )
    comparison_rows.append(
        {
            "Model": model_name,
            **{f"Selective {k}": v for k, v in sel.items()},
            **{f"Argmax {k}": v for k, v in arg.items()
               if k in ("Precision Low", "Precision Medium", "Precision High",
                        "Worst-Class Precision", "Macro Precision", "Accuracy",
                        "Recall Low", "Recall Medium", "Recall High")},
            "ROC-AUC Macro OvR": auc_value,
        }
    )

comparison = pd.DataFrame(comparison_rows)
best_model = comparison.loc[
    comparison["Selective Worst-Class Precision CI95 Low"].idxmax(), "Model"
]

windows_table = pd.DataFrame(window_rows)
selected_table = pd.DataFrame(selected_rows)

# ---- Precision / recall tables ----
pr_rows = []
for model_name in models_to_run:
    prob_matrix = pooled_frames[model_name][["P_Low", "P_Medium", "P_High"]].to_numpy()
    for class_index, class_name in enumerate(CLASS_ORDER):
        is_class = y_pool == class_name
        for threshold in PR_TABLE_THRESHOLDS:
            flagged = prob_matrix[:, class_index] >= threshold
            n_flag = int(flagged.sum())
            tp = int((flagged & is_class).sum())
            lo, _ = wilson_interval(tp, n_flag, 1.96)
            pr_rows.append(
                {
                    "Model": model_name, "Class": class_name,
                    "Probability Threshold": threshold, "Rows Flagged": n_flag,
                    "True Positives": tp,
                    "Precision": tp / n_flag if n_flag else np.nan,
                    "Precision Lower Bound 95%": float(lo) if n_flag else np.nan,
                    "Recall": tp / int(is_class.sum()) if is_class.sum() else np.nan,
                }
            )
pr_table = pd.DataFrame(pr_rows)

# ---- Save ----
with pd.ExcelWriter(OUTPUT_FOLDER / "all_models_comparison_v4.xlsx") as writer:
    comparison.to_excel(writer, sheet_name="Pooled_Comparison", index=False)
    windows_table.to_excel(writer, sheet_name="By_Window", index=False)
    selected_table.to_excel(writer, sheet_name="Selected_Configs", index=False)
    pr_table.to_excel(writer, sheet_name="PR_Tables", index=False)

prediction_output = truth.copy()
for model_name in models_to_run:
    frame = pooled_frames[model_name]
    for column in frame.columns:
        prediction_output[f"{model_name} | {column}"] = frame[column].to_numpy()
prediction_output.to_csv(OUTPUT_FOLDER / "all_models_predictions.csv", index=False)

# ---- Print summary ----
print("\n" + "=" * 98)
print(f"POOLED RESULT OVER ALL TEST WINDOWS - {len(truth)} test rows")
print(f"Always-Medium accuracy (baseline to beat): {majority_accuracy:.3f}")
print("=" * 98)

summary_print = pd.DataFrame(
    {
        "Model": comparison["Model"],
        "P_Low": comparison["Selective Precision Low"].round(3),
        "P_Low 95% CI": [
            f"{a:.2f}-{b:.2f}"
            for a, b in zip(comparison["Selective Precision Low CI95 Low"],
                            comparison["Selective Precision Low CI95 High"])
        ],
        "P_Med": comparison["Selective Precision Medium"].round(3),
        "P_High": comparison["Selective Precision High"].round(3),
        "P_High 95% CI": [
            f"{a:.2f}-{b:.2f}"
            for a, b in zip(comparison["Selective Precision High CI95 Low"],
                            comparison["Selective Precision High CI95 High"])
        ],
        "R_Low": comparison["Selective Recall Low"].round(3),
        "R_High": comparison["Selective Recall High"].round(3),
        "Coverage": comparison["Selective Coverage"].round(3),
        "AccCovered": comparison["Selective Accuracy on Covered Rows"].round(3),
        "AUC": comparison["ROC-AUC Macro OvR"].round(3),
    }
)
print(summary_print.to_string(index=False))

print("\nSame models with PLAIN ARGMAX (how your old scripts predicted):")
argmax_print = pd.DataFrame(
    {
        "Model": comparison["Model"],
        "P_Low": comparison["Argmax Precision Low"].round(3),
        "P_Med": comparison["Argmax Precision Medium"].round(3),
        "P_High": comparison["Argmax Precision High"].round(3),
        "R_Low": comparison["Argmax Recall Low"].round(3),
        "R_High": comparison["Argmax Recall High"].round(3),
        "Accuracy": comparison["Argmax Accuracy"].round(3),
    }
)
print(argmax_print.to_string(index=False))

print("\nWhich settings were chosen in each window:")
print(
    selected_table[
        ["Window", "Model", "Feature Set", "Weight Alpha", "Params", "Probability Mode",
         "Low Thr", "Medium Thr", "High Thr", "Margin"]
    ].to_string(index=False)
)

print(f"\nBest model by conservative worst-class precision (95% lower bound): {best_model}")
for class_name in ["Low", "High"]:
    print(f"\nPrecision / recall table - {best_model} - {class_name} (pooled test windows)")
    print(
        pr_table[(pr_table["Model"] == best_model) & (pr_table["Class"] == class_name)]
        .drop(columns=["Model", "Class"])
        .round(3)
        .to_string(index=False)
    )

# ---- Figures ----
fig, ax = plt.subplots(figsize=(10, 5))
width = 0.8 / max(len(models_to_run), 1)
for k, model_name in enumerate(models_to_run):
    row = comparison[comparison["Model"] == model_name].iloc[0]
    values = [row[f"Selective Precision {c}"] for c in CLASS_ORDER]
    lows = [row[f"Selective Precision {c} CI95 Low"] for c in CLASS_ORDER]
    highs = [row[f"Selective Precision {c} CI95 High"] for c in CLASS_ORDER]
    positions = np.arange(3) + k * width
    ax.bar(positions, values, width, label=model_name,
           yerr=[np.array(values) - np.array(lows), np.array(highs) - np.array(values)], capsize=3)
ax.set_xticks(np.arange(3) + width * (len(models_to_run) - 1) / 2)
ax.set_xticklabels(CLASS_ORDER)
ax.set_ylabel("Precision (pooled test windows)")
ax.set_ylim(0, 1.05)
ax.set_title("Per-class precision with 95% confidence intervals")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_FOLDER / "all_models_precision_by_class.png", dpi=200, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, class_name in zip(axes, ["Low", "High"]):
    for model_name in models_to_run:
        sub = pr_table[
            (pr_table["Model"] == model_name) & (pr_table["Class"] == class_name)
            & (pr_table["Rows Flagged"] >= 10)
        ]
        ax.plot(sub["Recall"], sub["Precision"], marker="o", markersize=3, label=model_name)
    ax.axhline(float(np.mean(y_pool == class_name)), linestyle="--", color="grey", label="base rate")
    ax.set_title(f"{class_name}: precision vs recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_FOLDER / "all_models_precision_recall_curves.png", dpi=200, bbox_inches="tight")
plt.close(fig)

cm = confusion_matrix(
    y_pool, pooled_frames[best_model]["Selective Prediction"].to_numpy(),
    labels=CLASS_ORDER + [UNCERTAIN],
)
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER + [UNCERTAIN]).plot(
    ax=ax, values_format="d"
)
ax.set_title(f"{best_model} - pooled selective confusion matrix")
fig.tight_layout()
fig.savefig(OUTPUT_FOLDER / "best_model_confusion_matrix.png", dpi=250, bbox_inches="tight")
plt.close(fig)

print("\nSelective classification report (best model, pooled):")
print(
    classification_report(
        y_pool, pooled_frames[best_model]["Selective Prediction"].to_numpy(),
        labels=CLASS_ORDER, zero_division=0,
    )
)
print("Output folder:", OUTPUT_FOLDER)
print("\nHow to read this:")
print("1. All models face the SAME test rows and the SAME selection rules -> a fair comparison.")
print("2. If two models' 95% confidence intervals overlap, you cannot say one is better.")
print("3. Compare 'Accuracy on covered' with the always-Medium baseline.")
print("4. Check 'Selected_Configs': if Set_G_All_Original32 wins, the original Weather / Time of Day /")
print("   Month columns carry real signal; if Set_F_History wins, history features help.")
