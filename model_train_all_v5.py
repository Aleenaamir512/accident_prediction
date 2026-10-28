# v5 - FAIR RANKING OF 4 MODELS AT FIXED 80% COVERAGE
# Logistic Regression, Random Forest, XGBoost, LightGBM (+ a 4-model average)
#
# Goal: ONE precision number per model that can be compared directly.
#   * every model answers exactly its 80% most-confident rows in each test window
#     (the remaining 20% are "Uncertain") - same coverage for everybody
#   * headline metric = OVERALL precision (share of correct answers among the answered rows);
#     macro and per-class precision are reported too, but macro is NOT used for the ranking
#     because the confident rows contain very few Low cases (a handful of lucky Low
#     predictions would decide the ranking)
#   * settings (features, class weights, hyper-parameters) are chosen on development data
#     only, using the same objective: overall precision at 80% coverage
#   * a block-bootstrap (blocks of 24 hourly rows) says whether a difference between two
#     models is real or just noise -> the script names a winner only if it is clear
#   * a 4-model average is added as a 5th candidate
#   * calibration/threshold search of v4 is dropped (raw probabilities won 12 out of 12 times)
#
# Rules shared by all models:
#   * same rolling-origin backtest (each test window comes AFTER its training data)
#   * same feature sets (including leakage-safe HISTORY features)
#   * same coverage (80%), same objective, same test rows
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
OUTPUT_FOLDER = BASE / "Model_Output_All_Models_v5"
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

# ---- Bootstrap for "is the difference real?" ------------------------------
N_BOOTSTRAP = 2000
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
# 6. METRICS AT FIXED COVERAGE
# ============================================================


def keep_top_fraction(proba, coverage):
    """Answer exactly the `coverage` fraction of rows with the highest top-class probability."""
    confidence = proba.max(axis=1)
    k = int(round(coverage * len(confidence)))
    order = np.argsort(-confidence, kind="stable")
    keep = np.zeros(len(confidence), dtype=bool)
    keep[order[:k]] = True
    return keep


def predictions_from_keep(proba, keep):
    """Class index for kept rows, -1 ("Uncertain") for the rest."""
    return np.where(keep, np.argmax(proba, axis=1), -1)


def metrics_from_indices(y_idx, pred_idx):
    """Macro precision, precision on covered rows (micro), per-class precision/recall/counts."""
    n = len(y_idx)
    cm = np.bincount(y_idx * 4 + (pred_idx + 1), minlength=12).reshape(3, 4)
    tp = np.diag(cm[:, 1:]).astype(float)
    pred_counts = cm[:, 1:].sum(axis=0).astype(float)
    true_counts = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_counts, out=np.zeros(3), where=pred_counts > 0)
    recall = np.divide(tp, true_counts, out=np.zeros(3), where=true_counts > 0)
    covered = float(pred_counts.sum())
    return {
        "Macro Precision": float(precision.mean()),
        "Precision on Covered Rows": float(tp.sum() / covered) if covered else 0.0,
        "Coverage": covered / n,
        "Precision Low": float(precision[0]),
        "Precision Medium": float(precision[1]),
        "Precision High": float(precision[2]),
        "Recall Low": float(recall[0]),
        "Recall Medium": float(recall[1]),
        "Recall High": float(recall[2]),
        "Predicted Low": int(pred_counts[0]),
        "Predicted Medium": int(pred_counts[1]),
        "Predicted High": int(pred_counts[2]),
    }


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
print(
    f"FAIR RANKING AT {TARGET_COVERAGE:.0%} COVERAGE - ROLLING BACKTEST"
    + ("  [QUICK RUN]" if QUICK_RUN else "")
)
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
    """
    Pick feature set / class weight / hyper-parameters by OVERALL PRECISION AT 80% COVERAGE
    on all out-of-fold development predictions (no test data involved).
    """
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
        oof_proba = oof[["P_Low", "P_Medium", "P_High"]].to_numpy()
        oof_y = np.array([class_to_index[v] for v in oof["Actual"]])

        keep = keep_top_fraction(oof_proba, TARGET_COVERAGE)
        at_target = metrics_from_indices(oof_y, predictions_from_keep(oof_proba, keep))
        at_full = metrics_from_indices(oof_y, np.argmax(oof_proba, axis=1))

        rows.append(
            {
                "Config ID": config_id, "Feature Set": cfg["Feature Set"],
                "Weight Alpha": cfg["Alpha"], "Params": cfg["Params"]["name"],
                "Tuning Overall Precision @80%": at_target["Precision on Covered Rows"],
                "Tuning Macro Precision @80%": at_target["Macro Precision"],
                "Tuning Macro Precision @100%": at_full["Macro Precision"],
                "Tuning Predicted Low @80%": at_target["Predicted Low"],
                "Tuning Predicted High @80%": at_target["Predicted High"],
            }
        )

        score = (at_target["Precision on Covered Rows"], at_target["Macro Precision"],
                 at_full["Macro Precision"])
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "artifact": {**cfg, "Config ID": config_id},
                "oof_proba": oof_proba,
                "oof_y": oof_y,
                # Deployable cutoff: the confidence needed to be in the top 80% on tuning data.
                "cutoff": float(np.quantile(oof_proba.max(axis=1), 1.0 - TARGET_COVERAGE)),
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
window_rows, selected_rows, config_tables = [], [], []

# store[model] = list of dicts with test probabilities / cutoff per window
store = {m: [] for m in models_to_run}
truth_parts = []
window_sizes = []

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
        pd.DataFrame(
            {"Window": window_number, "Date-Time": eval_feat["Date-Time"].to_numpy(), "Actual": y_eval}
        )
    )
    window_sizes.append(len(y_eval))

    window_selections = {}
    for model_name in models_to_run:
        selection = run_selection(model_name, fold_cache, tag)
        test_proba = evaluate_window(model_name, selection, dev_feat, eval_feat)
        window_selections[model_name] = (selection, test_proba)

        art = selection["artifact"]
        selected_rows.append(
            {
                "Window": window_number, "Model": model_name, "Feature Set": art["Feature Set"],
                "Weight Alpha": art["Alpha"], "Params": art["Params"]["name"],
                "Tuning Overall Precision @80%": selection["score"][0],
                "Tuning Macro Precision @80%": selection["score"][1],
                "Deployable Confidence Cutoff": selection["cutoff"],
            }
        )
        table = selection["config_table"].copy()
        table.insert(0, "Model", model_name)
        table.insert(0, "Window", window_number)
        config_tables.append(table)

        store[model_name].append(
            {"proba": test_proba, "cutoff": selection["cutoff"], "y": y_eval}
        )

        keep = keep_top_fraction(test_proba, TARGET_COVERAGE)
        m = metrics_from_indices(y_eval, predictions_from_keep(test_proba, keep))
        print(
            f"  >> {model_name}: {art['Feature Set']} | alpha={art['Alpha']} | {art['Params']['name']}"
        )
        print(
            f"     tuning overall precision@80% {selection['score'][0]:.3f} -> TEST@80%: "
            f"overall {m['Precision on Covered Rows']:.3f} | macro {m['Macro Precision']:.3f} | Low {m['Precision Low']:.3f} | "
            f"Medium {m['Precision Medium']:.3f} | High {m['Precision High']:.3f}"
        )

    # ---- 4-model average ("Ensemble") for this window ----
    if len(models_to_run) >= 2:
        ensemble_test = np.mean([window_selections[m][1] for m in models_to_run], axis=0)
        oof_arrays = [window_selections[m][0]["oof_proba"] for m in models_to_run]
        if len({a.shape for a in oof_arrays}) == 1:
            ensemble_oof = np.mean(oof_arrays, axis=0)
            ensemble_cutoff = float(np.quantile(ensemble_oof.max(axis=1), 1.0 - TARGET_COVERAGE))
        else:
            ensemble_cutoff = float("nan")
        store.setdefault("Average of models", []).append(
            {"proba": ensemble_test, "cutoff": ensemble_cutoff, "y": y_eval}
        )

ranked_models = list(store.keys())

# ============================================================
# 10. POOLED RESULTS (all test windows)
# ============================================================

truth = pd.concat(truth_parts, ignore_index=True)
y_pool = truth["Actual"].to_numpy().astype(int)
N = len(y_pool)
majority_accuracy = float(np.mean(y_pool == class_to_index["Medium"]))

pooled_proba, pred_80, pred_fixed, pred_100 = {}, {}, {}, {}
for model_name in ranked_models:
    parts = store[model_name]
    pooled_proba[model_name] = np.vstack([p["proba"] for p in parts])

    # (A) exact 80% coverage in every window  -> used for the RANKING
    pred_80[model_name] = np.concatenate(
        [predictions_from_keep(p["proba"], keep_top_fraction(p["proba"], TARGET_COVERAGE)) for p in parts]
    )
    # (B) deployable: confidence cutoff fixed from development data
    pred_fixed[model_name] = np.concatenate(
        [
            predictions_from_keep(
                p["proba"],
                p["proba"].max(axis=1) >= p["cutoff"] if np.isfinite(p["cutoff"])
                else keep_top_fraction(p["proba"], TARGET_COVERAGE),
            )
            for p in parts
        ]
    )
    # (C) plain argmax, no rejection
    pred_100[model_name] = np.argmax(pooled_proba[model_name], axis=1)

# ---- Block bootstrap ----
rng = np.random.default_rng(RANDOM_SEED)
n_blocks = int(np.ceil(N / BLOCK_SIZE))
boot_indices = []
for _ in range(N_BOOTSTRAP):
    starts = rng.integers(0, max(N - BLOCK_SIZE, 1), n_blocks)
    idx = np.concatenate([np.arange(s, s + BLOCK_SIZE) for s in starts])[:N]
    boot_indices.append(idx)


def bootstrap_metric(pred_idx, which):
    values = np.empty(N_BOOTSTRAP)
    for b, idx in enumerate(boot_indices):
        values[b] = metrics_from_indices(y_pool[idx], pred_idx[idx])[which]
    return values


boot_macro, boot_micro = {}, {}
for model_name in ranked_models:
    boot_macro[model_name] = bootstrap_metric(pred_80[model_name], "Macro Precision")
    boot_micro[model_name] = bootstrap_metric(pred_80[model_name], "Precision on Covered Rows")

ranking_rows = []
for model_name in ranked_models:
    m80 = metrics_from_indices(y_pool, pred_80[model_name])
    mfix = metrics_from_indices(y_pool, pred_fixed[model_name])
    m100 = metrics_from_indices(y_pool, pred_100[model_name])
    auc_value = roc_auc_score(
        y_pool, pooled_proba[model_name], multi_class="ovr", average="macro", labels=[0, 1, 2]
    )
    lows, highs = {}, {}
    for c_index, c_name in enumerate(CLASS_ORDER):
        n_pred = m80[f"Predicted {c_name}"]
        tp = round(m80[f"Precision {c_name}"] * n_pred)
        lows[c_name], highs[c_name] = (float(v) for v in wilson_interval(tp, n_pred, 1.96))
    ranking_rows.append(
        {
            "Model": model_name,
            "Macro Precision @80%": m80["Macro Precision"],
            "Macro Precision CI95 Low": float(np.percentile(boot_macro[model_name], 2.5)),
            "Macro Precision CI95 High": float(np.percentile(boot_macro[model_name], 97.5)),
            "Precision on Covered @80% (overall)": m80["Precision on Covered Rows"],
            "Overall CI95 Low": float(np.percentile(boot_micro[model_name], 2.5)),
            "Overall CI95 High": float(np.percentile(boot_micro[model_name], 97.5)),
            "Precision Low": m80["Precision Low"],
            "Precision Low CI95 Low": lows["Low"], "Precision Low CI95 High": highs["Low"],
            "Precision Medium": m80["Precision Medium"],
            "Precision High": m80["Precision High"],
            "Precision High CI95 Low": lows["High"], "Precision High CI95 High": highs["High"],
            "Recall Low": m80["Recall Low"], "Recall High": m80["Recall High"],
            "Predicted Low": m80["Predicted Low"], "Predicted Medium": m80["Predicted Medium"],
            "Predicted High": m80["Predicted High"],
            "Deployable cutoff: coverage": mfix["Coverage"],
            "Deployable cutoff: Macro Precision": mfix["Macro Precision"],
            "Deployable cutoff: Overall Precision": mfix["Precision on Covered Rows"],
            "No-rejection Macro Precision (argmax)": m100["Macro Precision"],
            "No-rejection Accuracy": m100["Precision on Covered Rows"],
            "ROC-AUC Macro OvR": auc_value,
        }
    )

ranking = (
    pd.DataFrame(ranking_rows)
    .sort_values("Precision on Covered @80% (overall)", ascending=False)
    .reset_index(drop=True)
)
ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))

# ---- Paired comparison against the top model ----
best_name = ranking.loc[0, "Model"]
pair_rows, clear_wins = [], True
for model_name in ranking["Model"].iloc[1:]:
    diff = boot_micro[best_name] - boot_micro[model_name]
    lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
    clear = lo > 0
    clear_wins = clear_wins and clear
    pair_rows.append(
        {
            "Top model": best_name,
            "Compared with": model_name,
            "Overall precision difference": float(
                ranking.loc[0, "Precision on Covered @80% (overall)"]
                - ranking.loc[ranking["Model"] == model_name, "Precision on Covered @80% (overall)"].iloc[0]
            ),
            "Difference CI95 Low": lo,
            "Difference CI95 High": hi,
            "Share of resamples where top model is better": float(np.mean(diff > 0)),
            "Verdict": "top model CLEARLY better" if clear else "TIE (difference is within noise)",
        }
    )
pairwise = pd.DataFrame(pair_rows)

# ---- Precision vs coverage ----
curve_rows = []
for model_name in ranked_models:
    for level in COVERAGE_LEVELS:
        preds = np.concatenate(
            [predictions_from_keep(p["proba"], keep_top_fraction(p["proba"], level))
             for p in store[model_name]]
        )
        m = metrics_from_indices(y_pool, preds)
        curve_rows.append(
            {
                "Model": model_name, "Coverage": level,
                "Macro Precision": m["Macro Precision"],
                "Overall Precision (covered rows)": m["Precision on Covered Rows"],
                "Precision Low": m["Precision Low"], "Precision Medium": m["Precision Medium"],
                "Precision High": m["Precision High"],
                "Predicted Low": m["Predicted Low"], "Predicted High": m["Predicted High"],
            }
        )
curve = pd.DataFrame(curve_rows)

# ---- By window ----
by_window_rows = []
offset = 0
for w_index, size in enumerate(window_sizes, start=1):
    sl = slice(offset, offset + size)
    for model_name in ranked_models:
        m = metrics_from_indices(y_pool[sl], pred_80[model_name][sl])
        by_window_rows.append(
            {"Window": w_index, "Model": model_name, "Test Rows": size,
             "Macro Precision @80%": m["Macro Precision"],
             "Overall Precision @80%": m["Precision on Covered Rows"],
             "Precision Low": m["Precision Low"], "Precision Medium": m["Precision Medium"],
             "Precision High": m["Precision High"]}
        )
    offset += size
by_window = pd.DataFrame(by_window_rows)

selected_table = pd.DataFrame(selected_rows)
config_table_all = pd.concat(config_tables, ignore_index=True)

# ---- Save ----
with pd.ExcelWriter(OUTPUT_FOLDER / "model_ranking_v5.xlsx") as writer:
    ranking.to_excel(writer, sheet_name="Ranking_at_80pct", index=False)
    pairwise.to_excel(writer, sheet_name="Top_vs_Others", index=False)
    curve.to_excel(writer, sheet_name="Precision_vs_Coverage", index=False)
    by_window.to_excel(writer, sheet_name="By_Window", index=False)
    selected_table.to_excel(writer, sheet_name="Selected_Configs", index=False)
config_table_all.to_csv(OUTPUT_FOLDER / "config_search_scores.csv", index=False)

prediction_output = truth.copy()
prediction_output["Actual"] = [CLASS_ORDER[i] for i in y_pool]
for model_name in ranked_models:
    proba = pooled_proba[model_name]
    prediction_output[f"{model_name} | P_Low"] = proba[:, 0]
    prediction_output[f"{model_name} | P_Medium"] = proba[:, 1]
    prediction_output[f"{model_name} | P_High"] = proba[:, 2]
    prediction_output[f"{model_name} | Prediction @80%"] = [
        CLASS_ORDER[i] if i >= 0 else UNCERTAIN for i in pred_80[model_name]
    ]
prediction_output.to_csv(OUTPUT_FOLDER / "all_models_predictions_v5.csv", index=False)

# ============================================================
# 11. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 98)
print(f"RANKING BY OVERALL PRECISION AT {TARGET_COVERAGE:.0%} COVERAGE - {N} test rows (rolling windows)")
print(f"Always-Medium accuracy (baseline): {majority_accuracy:.3f}")
print("=" * 98)

show = pd.DataFrame(
    {
        "Rank": ranking["Rank"],
        "Model": ranking["Model"],
        "OverallPrec": ranking["Precision on Covered @80% (overall)"].round(3),
        "95% CI": [f"{a:.3f}-{b:.3f}" for a, b in
                   zip(ranking["Overall CI95 Low"], ranking["Overall CI95 High"])],
        "MacroPrec": ranking["Macro Precision @80%"].round(3),
        "P_Low": ranking["Precision Low"].round(3),
        "P_Med": ranking["Precision Medium"].round(3),
        "P_High": ranking["Precision High"].round(3),
        "n_Low": ranking["Predicted Low"],
        "n_High": ranking["Predicted High"],
        "AUC": ranking["ROC-AUC Macro OvR"].round(3),
    }
)
print(show.to_string(index=False))

print("\nDeployable version (cutoff fixed from development data; coverage can drift):")
dep = pd.DataFrame(
    {
        "Model": ranking["Model"],
        "Coverage": ranking["Deployable cutoff: coverage"].round(3),
        "OverallPrec": ranking["Deployable cutoff: Overall Precision"].round(3),
        "MacroPrec": ranking["Deployable cutoff: Macro Precision"].round(3),
        "No-rejection accuracy (100%)": ranking["No-rejection Accuracy"].round(3),
    }
)
print(dep.to_string(index=False))

print("\nOverall precision by window (80% coverage):")
print(by_window.pivot(index="Model", columns="Window", values="Overall Precision @80%").round(3).to_string())

print("\nOverall precision vs coverage (pooled):")
print(curve.pivot(index="Model", columns="Coverage", values="Overall Precision (covered rows)").round(3).to_string())

print("\nIs the top model REALLY better? (block bootstrap, 95% interval of the OVERALL precision difference)")
print(
    pairwise[["Compared with", "Overall precision difference", "Difference CI95 Low",
              "Difference CI95 High", "Verdict"]].round(3).to_string(index=False)
)

print("\n" + "=" * 98)
if clear_wins and len(pairwise) > 0:
    print(f"RESULT: {best_name} is CLEARLY the best model at {TARGET_COVERAGE:.0%} coverage.")
else:
    tied = [best_name] + [r["Compared with"] for r in pair_rows if r["Verdict"].startswith("TIE")]
    behind = [r["Compared with"] for r in pair_rows if not r["Verdict"].startswith("TIE")]
    print(f"RESULT: no single model is clearly best. Statistical TIE between: {', '.join(tied)}.")
    if behind:
        print(f"        Clearly behind {best_name}: {', '.join(behind)}.")
    print("        When models tie, use the simplest one (Logistic < Random Forest < XGBoost/LightGBM).")
print("=" * 98)

print("\nSettings chosen in each window:")
print(selected_table.round(3).to_string(index=False))

# ---- Figures ----
fig, ax = plt.subplots(figsize=(8, 5))
for model_name in ranked_models:
    sub = curve[curve["Model"] == model_name]
    ax.plot(sub["Coverage"], sub["Overall Precision (covered rows)"], marker="o", label=model_name)
ax.axvline(TARGET_COVERAGE, linestyle="--", color="grey")
ax.set_xlabel("Coverage (share of rows that get a prediction)")
ax.set_ylabel("Overall precision (answered rows)")
ax.set_title("Overall precision vs coverage (pooled test windows)")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_FOLDER / "overall_precision_vs_coverage.png", dpi=200, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
vals = ranking["Precision on Covered @80% (overall)"].to_numpy()
err = np.vstack([vals - ranking["Overall CI95 Low"].to_numpy(),
                 ranking["Overall CI95 High"].to_numpy() - vals])
ax.bar(ranking["Model"], vals, yerr=err, capsize=5)
ax.set_ylabel("Overall precision @80% coverage")
ax.set_ylim(0, 1)
ax.set_title("Model ranking with 95% bootstrap intervals")
plt.setp(ax.get_xticklabels(), rotation=15)
fig.tight_layout()
fig.savefig(OUTPUT_FOLDER / "model_ranking_with_ci.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("\nOutput folder:", OUTPUT_FOLDER)
print("Main file: model_ranking_v5.xlsx (sheets: Ranking_at_80pct, Top_vs_Others, Precision_vs_Coverage, By_Window, Selected_Configs)")
