# ALL 4 MODELS - SAME WORK - FULL COMPARISON (final version)
# Logistic Regression, Random Forest, XGBoost, LightGBM  (+ a 4-model average as 5th candidate)
#
# Every model goes through EXACTLY the same steps:
#   1. Rolling-origin backtest (3 test windows, each AFTER its training data).
#   2. Settings (features, class weighting, hyper-parameters) chosen on development data only.
#   3. Two decision rules are evaluated for every model, every row answered (100% coverage):
#        PLAIN  : pick the class with the highest probability
#        TUNED  : pick argmax( P_low * w_low , P_medium , P_high * w_high ), weights chosen on
#                 development data to maximise macro F1  -> predicted counts move closer to
#                 the REAL class counts and more Low / High cases are found
#   4. Same report for every model: precision, recall, F1 per class, accuracy, balanced accuracy,
#      real rows vs predicted rows, confusion matrix, near-miss vs severe errors.
#   5. Model-vs-model comparison: ranking, block-bootstrap confidence intervals, a "tie or
#      clear winner" verdict, and the best model for each metric.
#   6. Does the tuning really help each model? (bootstrap test) and the price of predicting
#      more Low / more High (trade-off tables).
#
# Fewer answers (the 80% coverage idea of v5) is NOT used here; this script answers every row.
#
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
OUTPUT_FOLDER = BASE / "Model_Output_All_Models_Final"
OUTPUT_FOLDER.mkdir(exist_ok=True)

TARGET = "Accident Risk"
CLASS_ORDER = ["Low", "Medium", "High"]
UNCERTAIN = "Uncertain"
class_to_index = {c: i for i, c in enumerate(CLASS_ORDER)}

# ---- Rolling-origin backtest -------------------------------------------
OUTER_STARTS = [0.55, 0.70, 0.85] if not QUICK_RUN else [0.70, 0.85]
WINDOW_FRACTION = 0.15

# ---- Fixed coverage used for the ranking --------------------------------
TARGET_COVERAGE = 0.80
COVERAGE_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]   # precision-vs-coverage table

# ---- Decision-rule weights (chosen on development data only) -------------
W_LOW_GRID = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
W_HIGH_GRID = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
# Objective used to pick the weights AND the model settings: "macro_f1" or "balanced_accuracy"
OBJECTIVE = "macro_f1"

# ---- Bootstrap for "is the difference real?" ------------------------------
N_BOOTSTRAP = 1000
BLOCK_SIZE = 24          # rows (hourly data -> one day per block)
RANDOM_SEED = 42

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
# 6. DECISION RULE + METRICS (all rows answered)
# ============================================================


def weighted_argmax(proba, w_low, w_high):
    """Class index after multiplying Low / High probabilities by their weights."""
    return np.argmax(proba * np.array([w_low, 1.0, w_high]), axis=1)


def confusion_3x3(y_idx, pred_idx):
    return np.bincount(y_idx * 3 + pred_idx, minlength=9).reshape(3, 3)


def summary_from_cm(cm):
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    real = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(tp, real, out=np.zeros(3), where=real > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(3), where=denom > 0)
    return {
        "accuracy": tp.sum() / cm.sum(),
        "balanced_accuracy": recall.mean(),
        "macro_f1": f1.mean(),
        "precision": precision, "recall": recall, "f1": f1,
        "predicted": predicted, "real": real,
    }


def objective_value(summary):
    return summary["macro_f1"] if OBJECTIVE == "macro_f1" else summary["balanced_accuracy"]


def per_class_table(cm):
    s = summary_from_cm(cm)
    rows = [
        {"Class": c, "Precision": s["precision"][i], "Recall": s["recall"][i], "F1": s["f1"][i],
         "Real rows": int(s["real"][i]), "Predicted rows": int(s["predicted"][i])}
        for i, c in enumerate(CLASS_ORDER)
    ]
    weights = s["real"] / s["real"].sum()
    rows.append({"Class": "Macro average", "Precision": s["precision"].mean(),
                 "Recall": s["recall"].mean(), "F1": s["f1"].mean(),
                 "Real rows": int(s["real"].sum()), "Predicted rows": int(s["predicted"].sum())})
    rows.append({"Class": "Weighted average", "Precision": float((s["precision"] * weights).sum()),
                 "Recall": float((s["recall"] * weights).sum()),
                 "F1": float((s["f1"] * weights).sum()),
                 "Real rows": int(s["real"].sum()), "Predicted rows": int(s["predicted"].sum())})
    rows.append({"Class": "ACCURACY (overall)", "Precision": s["accuracy"], "Recall": s["accuracy"],
                 "F1": s["accuracy"], "Real rows": int(s["real"].sum()),
                 "Predicted rows": int(s["predicted"].sum())})
    return pd.DataFrame(rows)


def error_breakdown(cm):
    """Correct / near-miss (one level off) / severe (Low <-> High) shares."""
    total = cm.sum()
    correct = np.trace(cm)
    severe = cm[0, 2] + cm[2, 0]
    near = total - correct - severe
    return {"Correct": correct / total, "Near miss (one level off)": near / total,
            "Severe (Low <-> High)": severe / total}


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
print("ALL 4 MODELS - FINAL COMPARISON (100% coverage, plain vs tuned decision rule)"
      + ("  [QUICK RUN]" if QUICK_RUN else ""))
print("=" * 98)
print("Total rows:", len(df))
print("Class distribution (all data):")
print(df[TARGET].value_counts(normalize=True).round(3).to_string())
print(f"Objective used to choose settings and weights: {OBJECTIVE}")

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

weight_pairs = [(wl, wh) for wl in W_LOW_GRID for wh in W_HIGH_GRID]

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
    """Choose feature set, class weighting, hyper-parameters AND decision weights jointly."""
    configs = []
    for set_name in MODEL_FEATURE_SETS[model_name]:
        for alpha in ALPHAS[model_name]:
            for params in PROFILES[model_name]:
                configs.append(
                    {"Feature Set": set_name, "Features": FEATURE_SETS[set_name],
                     "Alpha": alpha, "Params": params}
                )

    rows, best = [], None
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
                    {"Date-Time": valid["Date-Time"].to_numpy(),
                     "Actual": valid[TARGET].to_numpy(),
                     "P_Low": proba[:, 0], "P_Medium": proba[:, 1], "P_High": proba[:, 2]}
                )
            )
        oof = (
            pd.concat(parts, ignore_index=True)
            .sort_values("Date-Time", kind="mergesort")
            .reset_index(drop=True)
        )
        oof_proba = oof[["P_Low", "P_Medium", "P_High"]].to_numpy()
        oof_y = np.array([class_to_index[v] for v in oof["Actual"]])

        plain = summary_from_cm(confusion_3x3(oof_y, np.argmax(oof_proba, axis=1)))

        best_weight = None
        for wl, wh in weight_pairs:
            sm = summary_from_cm(confusion_3x3(oof_y, weighted_argmax(oof_proba, wl, wh)))
            key = (objective_value(sm), sm["accuracy"])
            if best_weight is None or key > best_weight[0]:
                best_weight = (key, wl, wh, sm)
        (obj, acc), wl, wh, sm = best_weight

        rows.append(
            {
                "Config ID": config_id, "Feature Set": cfg["Feature Set"],
                "Weight Alpha": cfg["Alpha"], "Params": cfg["Params"]["name"],
                "Best w_low": wl, "Best w_high": wh,
                "Tuning objective (tuned)": obj, "Tuning accuracy (tuned)": acc,
                "Tuning objective (plain)": objective_value(plain),
                "Tuning accuracy (plain)": plain["accuracy"],
            }
        )
        if best is None or (obj, acc) > best["score"]:
            best = {
                "score": (obj, acc),
                "artifact": {**cfg, "Config ID": config_id},
                "w_low": wl, "w_high": wh,
                "plain_score": objective_value(plain),
                "oof_proba": oof_proba, "oof_y": oof_y,
            }

    best["config_table"] = pd.DataFrame(rows)
    return best


def evaluate_window(model_name, selection, dev_feat, eval_feat):
    art = selection["artifact"]
    features = art["Features"]
    y_dev = dev_feat[TARGET].map(class_to_index).astype(int).to_numpy()
    pipe = make_pipeline(model_name, features, art["Params"])
    fit_pipeline(pipe, dev_feat[features], y_dev, art["Alpha"])
    return pipe.predict_proba(eval_feat[features])


# ============================================================
# 9. ROLLING-ORIGIN BACKTEST
# ============================================================

n_rows = len(df)
base_models = list(models_to_run)
selected_rows, config_tables = [], []
store = {m: [] for m in models_to_run}
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
    y_eval = eval_feat[TARGET].map(class_to_index).astype(int).to_numpy()
    truth_parts.append(
        pd.DataFrame({"Window": window_number, "Date-Time": eval_feat["Date-Time"].to_numpy(),
                      "Actual": y_eval})
    )

    window_results = {}
    for model_name in models_to_run:
        selection = run_selection(model_name, fold_cache, tag)
        test_proba = evaluate_window(model_name, selection, dev_feat, eval_feat)
        art = selection["artifact"]

        plain_s = summary_from_cm(confusion_3x3(y_eval, np.argmax(test_proba, axis=1)))
        tuned_s = summary_from_cm(
            confusion_3x3(y_eval, weighted_argmax(test_proba, selection["w_low"], selection["w_high"]))
        )
        print(f"  >> {model_name}: {art['Feature Set']} | alpha={art['Alpha']} | {art['Params']['name']}"
              f" | w_low={selection['w_low']} w_high={selection['w_high']}")
        print(f"     TEST macro F1 plain {plain_s['macro_f1']:.3f} -> tuned {tuned_s['macro_f1']:.3f} | "
              f"accuracy plain {plain_s['accuracy']:.3f} -> tuned {tuned_s['accuracy']:.3f} | "
              f"predicted Low/Med/High tuned = "
              f"{int(tuned_s['predicted'][0])}/{int(tuned_s['predicted'][1])}/{int(tuned_s['predicted'][2])} "
              f"(real {int(tuned_s['real'][0])}/{int(tuned_s['real'][1])}/{int(tuned_s['real'][2])})")

        selected_rows.append(
            {"Window": window_number, "Model": model_name, "Feature Set": art["Feature Set"],
             "Weight Alpha": art["Alpha"], "Params": art["Params"]["name"],
             "w_low": selection["w_low"], "w_high": selection["w_high"],
             "Tuning objective (tuned)": selection["score"][0],
             "Tuning objective (plain)": selection["plain_score"]}
        )
        table = selection["config_table"].copy()
        table.insert(0, "Model", model_name)
        table.insert(0, "Window", window_number)
        config_tables.append(table)
        store[model_name].append(
            {"proba": test_proba, "w_low": selection["w_low"], "w_high": selection["w_high"], "y": y_eval}
        )
        window_results[model_name] = (selection, test_proba)

    # ---- 5th candidate: average of the selected models' probabilities ----
    if len(base_models) >= 2:
        ens_test = np.mean([window_results[m][1] for m in base_models], axis=0)
        oof_list = [window_results[m][0]["oof_proba"] for m in base_models]
        ens_wl, ens_wh, ens_obj = 1.0, 1.0, float("nan")
        if len({a.shape for a in oof_list}) == 1:
            ens_oof = np.mean(oof_list, axis=0)
            oof_y_ref = window_results[base_models[0]][0]["oof_y"]
            best_w = None
            for wl, wh in weight_pairs:
                sm = summary_from_cm(confusion_3x3(oof_y_ref, weighted_argmax(ens_oof, wl, wh)))
                key = (objective_value(sm), sm["accuracy"])
                if best_w is None or key > best_w[0]:
                    best_w = (key, wl, wh)
            ens_obj, ens_wl, ens_wh = best_w[0][0], best_w[1], best_w[2]
        store.setdefault("Average of models", []).append(
            {"proba": ens_test, "w_low": ens_wl, "w_high": ens_wh, "y": y_eval}
        )
        selected_rows.append(
            {"Window": window_number, "Model": "Average of models",
             "Feature Set": "(average of the models above)", "Weight Alpha": np.nan, "Params": "-",
             "w_low": ens_wl, "w_high": ens_wh,
             "Tuning objective (tuned)": ens_obj, "Tuning objective (plain)": np.nan}
        )

# ============================================================
# 10. POOLED RESULTS
# ============================================================

models_to_run = list(store.keys())   # now also contains "Average of models"

truth = pd.concat(truth_parts, ignore_index=True)
y_pool = truth["Actual"].to_numpy().astype(int)
N = len(y_pool)
majority_accuracy = float(np.mean(y_pool == class_to_index["Medium"]))

pooled = {}
for model_name in models_to_run:
    parts = store[model_name]
    proba = np.vstack([p["proba"] for p in parts])
    plain = np.argmax(proba, axis=1)
    tuned = np.concatenate([weighted_argmax(p["proba"], p["w_low"], p["w_high"]) for p in parts])
    pooled[model_name] = {"proba": proba, "plain": plain, "tuned": tuned}

# ---- block bootstrap: does tuning really help? ----
rng = np.random.default_rng(RANDOM_SEED)
n_blocks = int(np.ceil(N / BLOCK_SIZE))
boot_indices = []
for _ in range(N_BOOTSTRAP):
    starts = rng.integers(0, max(N - BLOCK_SIZE, 1), n_blocks)
    boot_indices.append(np.concatenate([np.arange(s, s + BLOCK_SIZE) for s in starts])[:N])


def bootstrap_difference(pred_a, pred_b, key):
    out = np.empty(N_BOOTSTRAP)
    for b, idx in enumerate(boot_indices):
        a = summary_from_cm(confusion_3x3(y_pool[idx], pred_a[idx]))[key]
        c = summary_from_cm(confusion_3x3(y_pool[idx], pred_b[idx]))[key]
        out[b] = a - c
    return out


compare_rows, per_class_frames, cm_frames = [], [], []
for model_name in models_to_run:
    for label in ("plain", "tuned"):
        cm = confusion_3x3(y_pool, pooled[model_name][label])
        s = summary_from_cm(cm)
        eb = error_breakdown(cm)
        compare_rows.append(
            {
                "Model": model_name,
                "Decision rule": "PLAIN argmax" if label == "plain" else "TUNED weights",
                "Accuracy": s["accuracy"], "Balanced accuracy": s["balanced_accuracy"],
                "Macro F1": s["macro_f1"],
                "Precision Low": s["precision"][0], "Recall Low": s["recall"][0], "F1 Low": s["f1"][0],
                "Precision Medium": s["precision"][1], "Recall Medium": s["recall"][1], "F1 Medium": s["f1"][1],
                "Precision High": s["precision"][2], "Recall High": s["recall"][2], "F1 High": s["f1"][2],
                "Predicted Low": int(s["predicted"][0]), "Predicted Medium": int(s["predicted"][1]),
                "Predicted High": int(s["predicted"][2]),
                "Real Low": int(s["real"][0]), "Real Medium": int(s["real"][1]), "Real High": int(s["real"][2]),
                "Share near-miss errors": eb["Near miss (one level off)"],
                "Share severe errors (Low<->High)": eb["Severe (Low <-> High)"],
            }
        )
        pc = per_class_table(cm)
        pc.insert(0, "Decision rule", "PLAIN argmax" if label == "plain" else "TUNED weights")
        pc.insert(0, "Model", model_name)
        per_class_frames.append(pc)
        cmf = pd.DataFrame(cm, index=[f"Real {c}" for c in CLASS_ORDER],
                           columns=[f"Predicted {c}" for c in CLASS_ORDER])
        cmf.insert(0, "Decision rule", "PLAIN argmax" if label == "plain" else "TUNED weights")
        cmf.insert(0, "Model", model_name)
        cm_frames.append(cmf.reset_index().rename(columns={"index": "Real class"}))

comparison = pd.DataFrame(compare_rows)
per_class_all = pd.concat(per_class_frames, ignore_index=True)
confusion_all = pd.concat(cm_frames, ignore_index=True)

boot_rows = []
for model_name in models_to_run:
    for key, label in (("macro_f1", "Macro F1"), ("accuracy", "Accuracy"),
                       ("balanced_accuracy", "Balanced accuracy")):
        diff = bootstrap_difference(pooled[model_name]["tuned"], pooled[model_name]["plain"], key)
        lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
        boot_rows.append(
            {"Model": model_name, "Metric": label,
             "Tuned minus plain": float(
                 summary_from_cm(confusion_3x3(y_pool, pooled[model_name]["tuned"]))[key]
                 - summary_from_cm(confusion_3x3(y_pool, pooled[model_name]["plain"]))[key]),
             "CI95 Low": lo, "CI95 High": hi,
             "Verdict": "tuning HELPS" if lo > 0 else ("tuning HURTS" if hi < 0 else "no clear difference")}
        )
boot_table = pd.DataFrame(boot_rows)

# ---- Model ranking (TUNED rule; headline = macro F1, accuracy as tie-break) ----
def boot_metric(pred_idx, key):
    out = np.empty(N_BOOTSTRAP)
    for b, idx in enumerate(boot_indices):
        out[b] = summary_from_cm(confusion_3x3(y_pool[idx], pred_idx[idx]))[key]
    return out


boot_f1 = {m: boot_metric(pooled[m]["tuned"], "macro_f1") for m in models_to_run}
boot_acc = {m: boot_metric(pooled[m]["tuned"], "accuracy") for m in models_to_run}

ranking = (
    comparison[comparison["Decision rule"] == "TUNED weights"]
    .sort_values(["Macro F1", "Accuracy"], ascending=False)
    .reset_index(drop=True)
)
ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
ranking["Macro F1 CI95 Low"] = [float(np.percentile(boot_f1[m], 2.5)) for m in ranking["Model"]]
ranking["Macro F1 CI95 High"] = [float(np.percentile(boot_f1[m], 97.5)) for m in ranking["Model"]]
ranking["Accuracy CI95 Low"] = [float(np.percentile(boot_acc[m], 2.5)) for m in ranking["Model"]]
ranking["Accuracy CI95 High"] = [float(np.percentile(boot_acc[m], 97.5)) for m in ranking["Model"]]

top_model = ranking.loc[0, "Model"]
pair_rows, all_clear = [], True
for model_name in ranking["Model"].iloc[1:]:
    d_f1 = boot_f1[top_model] - boot_f1[model_name]
    d_acc = boot_acc[top_model] - boot_acc[model_name]
    lo, hi = float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))
    clear = lo > 0
    all_clear = all_clear and clear
    pair_rows.append(
        {"Top model": top_model, "Compared with": model_name,
         "Macro F1 difference": float(ranking.loc[0, "Macro F1"]
                                     - ranking.loc[ranking["Model"] == model_name, "Macro F1"].iloc[0]),
         "Macro F1 diff CI95 Low": lo, "Macro F1 diff CI95 High": hi,
         "Accuracy difference": float(ranking.loc[0, "Accuracy"]
                                      - ranking.loc[ranking["Model"] == model_name, "Accuracy"].iloc[0]),
         "Accuracy diff CI95 Low": float(np.percentile(d_acc, 2.5)),
         "Accuracy diff CI95 High": float(np.percentile(d_acc, 97.5)),
         "Verdict (macro F1)": "top model CLEARLY better" if clear else "TIE (difference within noise)"}
    )
pairwise = pd.DataFrame(pair_rows)

metric_columns = ["Accuracy", "Balanced accuracy", "Macro F1", "Precision Low", "Recall Low", "F1 Low",
                  "Precision Medium", "Recall Medium", "F1 Medium",
                  "Precision High", "Recall High", "F1 High"]
best_rows = []
for rule in ("TUNED weights", "PLAIN argmax"):
    sub = comparison[comparison["Decision rule"] == rule]
    for metric in metric_columns:
        order = sub.sort_values(metric, ascending=False)
        best_rows.append(
            {"Decision rule": rule, "Metric": metric,
             "Best model": order.iloc[0]["Model"], "Best value": order.iloc[0][metric],
             "Runner-up": order.iloc[1]["Model"] if len(order) > 1 else "",
             "Runner-up value": order.iloc[1][metric] if len(order) > 1 else np.nan,
             "Lowest model": order.iloc[-1]["Model"], "Lowest value": order.iloc[-1][metric]}
        )
best_by_metric = pd.DataFrame(best_rows)

# ---- trade-off tables (diagnostic: shows the price of predicting more Low / High) ----
tradeoff_rows = []
for model_name in models_to_run:
    proba = pooled[model_name]["proba"]
    chosen_wh = float(np.median([p["w_high"] for p in store[model_name]]))
    chosen_wl = float(np.median([p["w_low"] for p in store[model_name]]))
    for wl in W_LOW_GRID:
        s = summary_from_cm(confusion_3x3(y_pool, weighted_argmax(proba, wl, chosen_wh)))
        tradeoff_rows.append(
            {"Model": model_name, "Varied": "Low weight", "w_low": wl, "w_high": chosen_wh,
             "Predicted Low": int(s["predicted"][0]), "Real Low": int(s["real"][0]),
             "Precision Low": s["precision"][0], "Recall Low": s["recall"][0], "F1 Low": s["f1"][0],
             "Accuracy": s["accuracy"], "Macro F1": s["macro_f1"]}
        )
    for wh in W_HIGH_GRID:
        s = summary_from_cm(confusion_3x3(y_pool, weighted_argmax(proba, chosen_wl, wh)))
        tradeoff_rows.append(
            {"Model": model_name, "Varied": "High weight", "w_low": chosen_wl, "w_high": wh,
             "Predicted High": int(s["predicted"][2]), "Real High": int(s["real"][2]),
             "Precision High": s["precision"][2], "Recall High": s["recall"][2], "F1 High": s["f1"][2],
             "Accuracy": s["accuracy"], "Macro F1": s["macro_f1"]}
        )
tradeoff = pd.DataFrame(tradeoff_rows)

selected_table = pd.DataFrame(selected_rows)
config_table_all = pd.concat(config_tables, ignore_index=True)

with pd.ExcelWriter(OUTPUT_FOLDER / "all_models_final_comparison.xlsx") as writer:
    ranking.to_excel(writer, sheet_name="Ranking_Tuned", index=False)
    pairwise.to_excel(writer, sheet_name="Top_vs_Others", index=False)
    best_by_metric.to_excel(writer, sheet_name="Best_by_Metric", index=False)
    comparison.to_excel(writer, sheet_name="Plain_vs_Tuned", index=False)
    per_class_all.to_excel(writer, sheet_name="Precision_Recall_F1", index=False)
    confusion_all.to_excel(writer, sheet_name="Confusion_Matrix", index=False)
    boot_table.to_excel(writer, sheet_name="Does_Tuning_Help", index=False)
    tradeoff.to_excel(writer, sheet_name="Tradeoff_Low_High", index=False)
    selected_table.to_excel(writer, sheet_name="Selected_Configs", index=False)
config_table_all.to_csv(OUTPUT_FOLDER / "config_search_scores.csv", index=False)

pred_out = truth.copy()
pred_out["Actual"] = [CLASS_ORDER[i] for i in y_pool]
for model_name in models_to_run:
    proba = pooled[model_name]["proba"]
    pred_out[f"{model_name} | P_Low"] = proba[:, 0]
    pred_out[f"{model_name} | P_Medium"] = proba[:, 1]
    pred_out[f"{model_name} | P_High"] = proba[:, 2]
    pred_out[f"{model_name} | Plain prediction"] = [CLASS_ORDER[i] for i in pooled[model_name]["plain"]]
    pred_out[f"{model_name} | Tuned prediction"] = [CLASS_ORDER[i] for i in pooled[model_name]["tuned"]]
pred_out.to_csv(OUTPUT_FOLDER / "predictions_all_models.csv", index=False)

# ============================================================
# 11. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 98)
print(f"POOLED RESULT - {N} test rows (rolling windows), every row answered")
print(f"Always-Medium accuracy (baseline): {majority_accuracy:.3f}")
print("=" * 98)

def combined_per_class(rule_label):
    frame = per_class_all[per_class_all["Decision rule"] == rule_label]
    frame = frame[frame["Class"].isin(
        ["Low", "Medium", "High", "Macro average", "Weighted average", "ACCURACY (overall)"])]
    return frame.drop(columns="Decision rule")


print("\n" + "#" * 98)
print("A) TUNED decision rule - precision / recall / F1 per class, ALL MODELS")
print("#" * 98)
print(combined_per_class("TUNED weights").round(3).to_string(index=False))

print("\n" + "#" * 98)
print("B) PLAIN argmax - precision / recall / F1 per class, ALL MODELS (what you had before)")
print("#" * 98)
print(combined_per_class("PLAIN argmax").round(3).to_string(index=False))

print("\nConfusion matrices with the TUNED rule (rows = real, columns = predicted):")
for model_name in models_to_run:
    cm = confusion_3x3(y_pool, pooled[model_name]["tuned"])
    eb = error_breakdown(cm)
    print(f"\n{model_name}  (correct {eb['Correct']:.3f} | near-miss {eb['Near miss (one level off)']:.3f}"
          f" | severe Low<->High {eb['Severe (Low <-> High)']:.3f})")
    print(pd.DataFrame(cm, index=[f"Real {c}" for c in CLASS_ORDER],
                       columns=[f"Pred {c}" for c in CLASS_ORDER]).to_string())

print("\nSide by side:")
side = comparison[["Model", "Decision rule", "Accuracy", "Balanced accuracy", "Macro F1",
                   "Recall Low", "Recall High", "F1 Low", "F1 High",
                   "Predicted Low", "Predicted Medium", "Predicted High"]]
print(side.round(3).to_string(index=False))

print("\n" + "=" * 98)
print("MODEL RANKING - TUNED decision rule (headline: macro F1, tie-break: accuracy)")
print("=" * 98)
rank_show = ranking[["Rank", "Model", "Macro F1", "Accuracy", "Balanced accuracy",
                     "Recall Low", "Recall High", "F1 Low", "F1 High",
                     "Precision Low", "Precision High"]].copy()
rank_show.insert(3, "Macro F1 95% CI", [f"{a:.3f}-{b:.3f}" for a, b in
                                         zip(ranking["Macro F1 CI95 Low"], ranking["Macro F1 CI95 High"])])
rank_show.insert(5, "Accuracy 95% CI", [f"{a:.3f}-{b:.3f}" for a, b in
                                         zip(ranking["Accuracy CI95 Low"], ranking["Accuracy CI95 High"])])
print(rank_show.round(3).to_string(index=False))

print(f"\nIs {top_model} REALLY better than the others? (block bootstrap, 95% interval of the difference)")
print(pairwise[["Compared with", "Macro F1 difference", "Macro F1 diff CI95 Low", "Macro F1 diff CI95 High",
                "Accuracy difference", "Verdict (macro F1)"]].round(3).to_string(index=False))
print("-" * 98)
if all_clear and len(pairwise) > 0:
    print(f"RESULT: {top_model} is CLEARLY the best model (macro F1).")
else:
    tied = [top_model] + [r["Compared with"] for r in pair_rows if r["Verdict (macro F1)"].startswith("TIE")]
    behind = [r["Compared with"] for r in pair_rows if not r["Verdict (macro F1)"].startswith("TIE")]
    print(f"RESULT: no single model is clearly best. Statistical TIE between: {', '.join(tied)}.")
    if behind:
        print(f"        Clearly behind {top_model}: {', '.join(behind)}.")
    print("        When models tie, prefer the simplest (Logistic < Random Forest < XGBoost/LightGBM < Average).")
print("-" * 98)

print("\nBest model for each metric (TUNED rule). Caution: Precision of Low rests on few predictions.")
print(best_by_metric[best_by_metric["Decision rule"] == "TUNED weights"][
    ["Metric", "Best model", "Best value", "Runner-up", "Runner-up value", "Lowest model", "Lowest value"]
].round(3).to_string(index=False))

print("\nDoes the tuning REALLY help? (block bootstrap, 95% interval of tuned minus plain)")
print(boot_table.round(3).to_string(index=False))

print("\nPrice of predicting more LOW (pooled test windows, diagnostic only):")
print(tradeoff[tradeoff["Varied"] == "Low weight"][
    ["Model", "w_low", "Predicted Low", "Real Low", "Precision Low", "Recall Low", "F1 Low",
     "Accuracy", "Macro F1"]].round(3).to_string(index=False))
print("\nPrice of predicting more HIGH (pooled test windows, diagnostic only):")
print(tradeoff[tradeoff["Varied"] == "High weight"][
    ["Model", "w_high", "Predicted High", "Real High", "Precision High", "Recall High", "F1 High",
     "Accuracy", "Macro F1"]].round(3).to_string(index=False))

print("\nSettings chosen in each window:")
print(selected_table.round(3).to_string(index=False))

# ---- Figure: confusion matrices plain vs tuned ----
for model_name in models_to_run:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, label, title in zip(axes, ("plain", "tuned"), ("PLAIN argmax", "TUNED weights")):
        cm = confusion_3x3(y_pool, pooled[model_name][label])
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER).plot(
            ax=ax, values_format="d", colorbar=False)
        ax.set_title(f"{model_name} - {title}")
    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / f"confusion_matrix_{model_name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 5))
metrics_plot = ["Accuracy", "Macro F1", "Recall Low", "Recall High"]
width = 0.8 / len(models_to_run)
for k, model_name in enumerate(models_to_run):
    row = comparison[(comparison["Model"] == model_name) & (comparison["Decision rule"] == "TUNED weights")].iloc[0]
    ax.bar(np.arange(len(metrics_plot)) + k * width, [row[m] for m in metrics_plot], width, label=model_name)
ax.set_xticks(np.arange(len(metrics_plot)) + width * (len(models_to_run) - 1) / 2)
ax.set_xticklabels(metrics_plot)
ax.set_ylim(0, 1)
ax.set_title("All models - TUNED decision rule (pooled test windows)")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_FOLDER / "all_models_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("\nOutput folder:", OUTPUT_FOLDER)
print("Main file: all_models_final_comparison.xlsx (sheets: Ranking_Tuned, Top_vs_Others, Best_by_Metric,")
print("           Plain_vs_Tuned, Precision_Recall_F1, Confusion_Matrix, Does_Tuning_Help,")
print("           Tradeoff_Low_High, Selected_Configs)")
print("\nHow to read this:")
print("1. 'Real rows' vs 'Predicted rows': the closer they are, the less the model under- or over-labels a class.")
print("2. Predicting more Low / High lowers their precision - see the trade-off tables.")
print("3. Trust 'Does_Tuning_Help': if the interval includes 0, the gain may be noise.")
