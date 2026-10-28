# Logistic Regression v3
# calibration + class-specific thresholds + reject option
# + WORST-CLASS PRECISION objective + rolling-origin backtest + precision/recall tables.
#
# CHANGES compared with model_train_lr_corrected.py
#   1. OBJECTIVE: thresholds are now chosen to maximise the WORST per-class precision
#      (Low, Medium or High - whichever is weakest), using a Wilson lower bound so a
#      class with only a few predictions cannot look good by luck.
#      Macro precision hid a weak Low class behind a strong Medium class.
#   2. CLASS WEIGHTS: alpha grid is now [0.0, 0.25, 0.5] (full balancing, alpha=1,
#      caused Low to be over-predicted -> Low precision ~0.32).
#   3. SEPARATE THRESHOLD GRIDS per class (Low and High can go up to 0.90).
#   4. ROLLING-ORIGIN BACKTEST: the WHOLE selection process (features, C, alpha,
#      calibration, thresholds) is re-run for several expanding windows and each
#      choice is tested on the window that follows it. Results are pooled, with
#      95% confidence intervals for per-class precision.
#   5. PRECISION/RECALL TABLE for each class (what precision is achievable at
#      each recall level).
#
# Nothing in a test window is ever used to choose a setting for that window.

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
OUTPUT_FOLDER = BASE / "Model_Output_Logistic_v3"
OUTPUT_FOLDER.mkdir(exist_ok=True)

TARGET = "Accident Risk"
CLASS_ORDER = ["Low", "Medium", "High"]
UNCERTAIN = "Uncertain"

# ---- Rolling-origin backtest -------------------------------------------
# Each test window is WINDOW_FRACTION of the data; the model is trained and
# tuned ONLY on everything before it.  Last window ends at the last row.
OUTER_STARTS = [0.55, 0.70, 0.85]
WINDOW_FRACTION = 0.15

# ---- Safeguards for threshold selection (tuning data only) -------------
MIN_COVERAGE = 0.70          # share of rows that must receive a prediction
MIN_MACRO_RECALL = 0.40
MIN_CLASS_RECALL = 0.25      # every class must keep at least this recall
MIN_PRED_PER_CLASS = 15      # a class needs at least this many predictions
WILSON_Z = 1.28              # one-sided ~90% lower bound used in the objective

# ---- Separate threshold grids per class ---------------------------------
LOW_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
MEDIUM_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
HIGH_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
MARGIN_GRID = [0.00, 0.05, 0.10, 0.15, 0.20]

# Thresholds shown in the precision/recall table.
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

ALL_PREDICTIVE = [
    "Area",
    "Road Type",
    "Road Condition",
    "Vehicle Count",
    "Ride Activity",
    "Bus Activity",
    "Pickup Hour",
    "Day Name",
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
    "Set_C_Augmented_No_Area": NO_AREA,
    "Set_D_All_Predictive": ALL_PREDICTIVE,
    "Set_E_Interactions": INTERACTION_SET,
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

C_VALUES = [0.05, 0.20, 1.0, 5.0]
WEIGHT_ALPHAS = [0.0, 0.25, 0.5]   # alpha=1.0 removed (over-predicted Low)

# ============================================================
# 3. LEAKAGE-SAFE TRAFFIC FEATURES
# ============================================================


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

    road_ref = (
        ref.groupby("Road Type")["Traffic Load per Lane"].quantile(0.95).to_dict()
    )
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
    out["Traffic Load Ratio"] = (
        out["Traffic Load per Lane"] / out["Reference Traffic Load"]
    )

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
        ratio = min(float(row["Traffic Load Ratio"]), 1.20)  # same cap as feature script
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
# 4. CLASS WEIGHTS + BASE MODEL
# ============================================================


def make_partial_class_weights(y, alpha):
    """alpha=0: none, alpha=1: full inverse-frequency balancing."""
    if alpha == 0.0:
        return None
    counts = y.value_counts()
    n = len(y)
    k = len(counts)
    balanced = {cls: n / (k * count) for cls, count in counts.items()}
    return {cls: float(weight ** alpha) for cls, weight in balanced.items()}


def make_pipeline(features, class_weight, c_value):
    categorical = [f for f in features if f in CATEGORICAL_FEATURES]
    numeric = [f for f in features if f not in categorical]

    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    model = LogisticRegression(
        C=c_value, max_iter=5000, class_weight=class_weight, solver="lbfgs"
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def ordered_probabilities(pipeline, X):
    raw = pipeline.predict_proba(X)
    model_classes = list(pipeline.named_steps["model"].classes_)
    indices = [model_classes.index(c) for c in CLASS_ORDER]
    return raw[:, indices]


# ============================================================
# 5. SIGMOID PROBABILITY CALIBRATION
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


# ============================================================
# 6. SELECTIVE PREDICTION + METRICS
# ============================================================


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a proportion k/n. Works on scalars or arrays."""
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
    """Predict top class only if its class threshold AND the margin are met."""
    top_idx = np.argmax(probabilities, axis=1)
    sorted_prob = np.sort(probabilities, axis=1)
    top_prob = sorted_prob[:, -1]
    margins = top_prob - sorted_prob[:, -2]

    keep = (top_prob >= np.asarray(thresholds)[top_idx]) & (margins >= margin_threshold)

    labels = np.asarray(CLASS_ORDER, dtype=object)
    pred = np.full(len(probabilities), UNCERTAIN, dtype=object)
    pred[keep] = labels[top_idx[keep]]
    return pred


def argmax_predict(probabilities):
    return np.asarray([CLASS_ORDER[i] for i in np.argmax(probabilities, axis=1)], dtype=object)


def selective_metrics(y_true, predictions):
    """Uncertain rows count as missed for recall; precision uses Low/Medium/High only."""
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    covered = predictions != UNCERTAIN

    p_cls = precision_score(y_true, predictions, labels=CLASS_ORDER, average=None, zero_division=0)
    r_cls = recall_score(y_true, predictions, labels=CLASS_ORDER, average=None, zero_division=0)

    # Per-class counts for confidence intervals.
    n_pred = np.array([np.sum(predictions == c) for c in CLASS_ORDER])
    n_tp = np.array([np.sum((predictions == c) & (y_true == c)) for c in CLASS_ORDER])
    lo, hi = wilson_interval(n_tp, n_pred, 1.96)

    row = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Macro Precision": precision_score(
            y_true, predictions, labels=CLASS_ORDER, average="macro", zero_division=0
        ),
        "Worst-Class Precision": float(np.min(p_cls)),
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
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Macro F0.5",
    "Coverage",
    "Uncertain Count",
    "Precision Low",
    "Precision Medium",
    "Precision High",
    "Recall Low",
    "Recall Medium",
    "Recall High",
    "Pred Low",
    "Pred Medium",
    "Pred High",
    "LB Low",
    "LB Medium",
    "LB High",
]


def fast_metrics(y_idx, pred_idx):
    """Numpy equivalent of selective_metrics for the threshold search (-1 = Uncertain)."""
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
        return np.divide(
            (1 + b2) * precision * recall, denom, out=np.zeros(3), where=denom > 0
        )

    lower, _ = wilson_interval(tp, pred_counts, WILSON_Z)
    uncertain = float(cm[:, 0].sum())

    return [
        tp.sum() / n,
        precision.mean(),
        recall.mean(),
        fbeta(1.0).mean(),
        fbeta(0.5).mean(),
        1.0 - uncertain / n,
        uncertain,
        *precision,
        *recall,
        *pred_counts,
        *lower,
    ]


# ============================================================
# 7. LOAD DATA
# ============================================================

df = pd.read_excel(INPUT_FILE, sheet_name="Data")
df["Date-Time"] = pd.to_datetime(df["Date-Time"], errors="coerce")
if df["Date-Time"].isna().any():
    raise ValueError("Some Date-Time values could not be converted.")
df = df.sort_values("Date-Time", kind="mergesort").reset_index(drop=True)

print("=" * 98)
print("LOGISTIC REGRESSION v3 - WORST-CLASS PRECISION + ROLLING BACKTEST")
print("=" * 98)
print("Total rows:", len(df))
print("Class distribution (all data):")
print(df[TARGET].value_counts(normalize=True).round(3).to_string())

# Drop any listed feature that does not exist after feature recomputation.
_probe = recompute_traffic_features(df.iloc[: max(200, len(df) // 2)], df.head(50))
for set_name in list(FEATURE_SETS):
    wanted = FEATURE_SETS[set_name]
    kept = [f for f in wanted if f in _probe.columns]
    dropped = [f for f in wanted if f not in _probe.columns]
    if dropped:
        print(f"WARNING: {set_name} - dropping missing features: {dropped}")
    FEATURE_SETS[set_name] = kept

class_to_index = {c: i for i, c in enumerate(CLASS_ORDER)}

# Threshold combinations (built once).
threshold_triples = np.array(list(product(LOW_GRID, MEDIUM_GRID, HIGH_GRID)), dtype=float)
combo_thresholds = np.repeat(threshold_triples, len(MARGIN_GRID), axis=0)
combo_margins = np.tile(MARGIN_GRID, len(threshold_triples))

# ============================================================
# 8. SELECTION PROCEDURE (runs on DEVELOPMENT data only)
# ============================================================


def run_selection(development_df, tag):
    """Choose features / C / alpha / calibration / thresholds using development data only."""
    time_cv = TimeSeriesSplit(n_splits=5)

    fold_cache = []
    for fold, (tr_idx, va_idx) in enumerate(time_cv.split(development_df), start=1):
        train_raw = development_df.iloc[tr_idx].copy()
        valid_raw = development_df.iloc[va_idx].copy()
        if train_raw[TARGET].nunique() < 3 or valid_raw[TARGET].nunique() < 3:
            continue
        fold_cache.append(
            {
                "fold": fold,
                "train": recompute_traffic_features(train_raw, train_raw),
                "validation": recompute_traffic_features(train_raw, valid_raw),
            }
        )
    if not fold_cache:
        raise ValueError(f"[{tag}] No usable time-series folds.")

    config_rows, search_frames, artifacts = [], [], {}
    config_id = 0
    total = len(FEATURE_SETS) * len(C_VALUES) * len(WEIGHT_ALPHAS)

    for set_name, features in FEATURE_SETS.items():
        for alpha in WEIGHT_ALPHAS:
            for c_value in C_VALUES:
                config_id += 1
                if config_id % 10 == 1:
                    print(f"  [{tag}] configuration {config_id}/{total}", flush=True)

                parts = []
                for item in fold_cache:
                    train, valid = item["train"], item["validation"]
                    pipe = make_pipeline(
                        features, make_partial_class_weights(train[TARGET], alpha), c_value
                    )
                    pipe.fit(train[features], train[TARGET])
                    proba = ordered_probabilities(pipe, valid[features])
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

                calibrators = fit_sigmoid_calibrators(
                    cal_part[cols].to_numpy(), cal_part["Actual"].to_numpy()
                )
                thr_raw = thr_part[cols].to_numpy()
                thr_cal = apply_sigmoid_calibration(thr_raw, calibrators)

                artifacts[config_id] = {
                    "Feature Set": set_name,
                    "Features": features,
                    "Alpha": alpha,
                    "C": c_value,
                    "Calibrators": calibrators,
                }

                y_labels = thr_part["Actual"].to_numpy()
                y_idx = np.array([class_to_index[v] for v in y_labels])

                for mode, matrix in {"Raw": thr_raw, "Calibrated": thr_cal}.items():
                    base = selective_metrics(y_labels, argmax_predict(matrix))
                    config_rows.append(
                        {
                            "Config ID": config_id,
                            "Feature Set": set_name,
                            "Weight Alpha": alpha,
                            "C": c_value,
                            "Probability Mode": mode,
                            "Baseline Macro Precision": base["Macro Precision"],
                            "Baseline Worst-Class Precision": base["Worst-Class Precision"],
                            "Baseline Macro Recall": base["Macro Recall"],
                        }
                    )

                    top_idx = np.argmax(matrix, axis=1)
                    srt = np.sort(matrix, axis=1)
                    top_prob = srt[:, -1]
                    margins = top_prob - srt[:, -2]

                    results = np.empty((len(combo_margins), len(FAST_COLUMNS)))
                    for i in range(len(combo_margins)):
                        keep = (top_prob >= combo_thresholds[i][top_idx]) & (
                            margins >= combo_margins[i]
                        )
                        results[i] = fast_metrics(y_idx, np.where(keep, top_idx, -1))

                    frame = pd.DataFrame(results, columns=FAST_COLUMNS)
                    frame.insert(0, "Margin Threshold", combo_margins)
                    frame.insert(0, "High Threshold", combo_thresholds[:, 2])
                    frame.insert(0, "Medium Threshold", combo_thresholds[:, 1])
                    frame.insert(0, "Low Threshold", combo_thresholds[:, 0])
                    frame.insert(0, "Probability Mode", mode)
                    frame.insert(0, "C", c_value)
                    frame.insert(0, "Weight Alpha", alpha)
                    frame.insert(0, "Feature Set", set_name)
                    frame.insert(0, "Config ID", config_id)
                    search_frames.append(frame)

    search_df = pd.concat(search_frames, ignore_index=True)

    # ---- NEW OBJECTIVE: worst-class precision --------------------------
    search_df["Worst-Class Precision"] = search_df[
        ["Precision Low", "Precision Medium", "Precision High"]
    ].min(axis=1)
    search_df["Worst-Class Precision LB"] = search_df[
        ["LB Low", "LB Medium", "LB High"]
    ].min(axis=1)
    search_df["Min Class Recall"] = search_df[
        ["Recall Low", "Recall Medium", "Recall High"]
    ].min(axis=1)
    search_df["Min Class Predictions"] = search_df[
        ["Pred Low", "Pred Medium", "Pred High"]
    ].min(axis=1)

    def pick(min_cov, min_macro_rec, min_cls_rec, min_preds):
        return search_df[
            (search_df["Coverage"] >= min_cov)
            & (search_df["Macro Recall"] >= min_macro_rec)
            & (search_df["Min Class Recall"] >= min_cls_rec)
            & (search_df["Min Class Predictions"] >= min_preds)
        ]

    eligible = pick(MIN_COVERAGE, MIN_MACRO_RECALL, MIN_CLASS_RECALL, MIN_PRED_PER_CLASS)
    relaxed = False
    if eligible.empty:
        relaxed = True
        print(f"  [{tag}] WARNING: safeguards relaxed (coverage>=0.60, class recall>=0.15, preds>=8).")
        eligible = pick(0.60, 0.30, 0.15, 8)
    if eligible.empty:
        raise RuntimeError(f"[{tag}] No usable threshold candidates even after relaxing safeguards.")

    eligible = eligible.sort_values(
        ["Worst-Class Precision LB", "Worst-Class Precision", "Macro Precision", "Coverage"],
        ascending=False,
    ).reset_index(drop=True)

    best = eligible.iloc[0]
    artifact = artifacts[int(best["Config ID"])]

    return {
        "best": best,
        "artifact": artifact,
        "mode": best["Probability Mode"],
        "thresholds": [
            float(best["Low Threshold"]),
            float(best["Medium Threshold"]),
            float(best["High Threshold"]),
        ],
        "margin": float(best["Margin Threshold"]),
        "relaxed": relaxed,
        "search_df": search_df,
        "eligible": eligible,
        "config_rows": config_rows,
    }


def evaluate_window(development_df, evaluation_raw, selection):
    """Refit the chosen model on all development data and predict the test window."""
    art = selection["artifact"]
    features = art["Features"]

    dev = recompute_traffic_features(development_df, development_df)
    ev = recompute_traffic_features(development_df, evaluation_raw)

    pipe = make_pipeline(
        features, make_partial_class_weights(dev[TARGET], art["Alpha"]), art["C"]
    )
    pipe.fit(dev[features], dev[TARGET])

    raw = ordered_probabilities(pipe, ev[features])
    cal = apply_sigmoid_calibration(raw, art["Calibrators"])
    chosen = cal if selection["mode"] == "Calibrated" else raw

    return {
        "evaluation": ev,
        "raw": raw,
        "calibrated": cal,
        "selected": chosen,
        "selective_pred": selective_predict(chosen, selection["thresholds"], selection["margin"]),
        "raw_argmax_pred": argmax_predict(raw),
    }


# ============================================================
# 9. ROLLING-ORIGIN BACKTEST
# ============================================================

n_rows = len(df)
window_rows = []
pooled_parts = []
last_selection = None

for window_number, start_fraction in enumerate(OUTER_STARTS, start=1):
    start = int(n_rows * start_fraction)
    end = min(int(n_rows * (start_fraction + WINDOW_FRACTION) + 0.5), n_rows)
    if window_number == len(OUTER_STARTS):
        end = n_rows  # last window always runs to the final row

    development_df = df.iloc[:start].copy()
    evaluation_raw = df.iloc[start:end].copy()
    if len(evaluation_raw) == 0:
        continue

    tag = f"window {window_number}/{len(OUTER_STARTS)}"
    print("\n" + "=" * 98)
    print(
        f"{tag.upper()}: develop on rows 0-{start - 1} "
        f"({development_df['Date-Time'].min().date()} to {development_df['Date-Time'].max().date()}), "
        f"test on rows {start}-{end - 1} "
        f"({evaluation_raw['Date-Time'].min().date()} to {evaluation_raw['Date-Time'].max().date()})"
    )
    print("=" * 98)

    selection = run_selection(development_df, tag)
    result = evaluate_window(development_df, evaluation_raw, selection)
    best = selection["best"]

    y_eval = result["evaluation"][TARGET].to_numpy()
    selective = selective_metrics(y_eval, result["selective_pred"])
    raw_argmax = selective_metrics(y_eval, result["raw_argmax_pred"])
    majority_accuracy = float(np.mean(y_eval == "Medium"))

    print(
        f"  Selected: {best['Feature Set']} | alpha={selection['artifact']['Alpha']} | "
        f"C={selection['artifact']['C']} | {selection['mode']} | "
        f"thresholds L/M/H={selection['thresholds']} | margin={selection['margin']}"
        + ("  (RELAXED safeguards)" if selection["relaxed"] else "")
    )
    print(
        f"  Tuning worst-class precision: {best['Worst-Class Precision']:.3f} "
        f"(lower bound {best['Worst-Class Precision LB']:.3f}) | coverage {best['Coverage']:.3f}"
    )
    print(
        f"  TEST  precision  Low {selective['Precision Low']:.3f} | "
        f"Medium {selective['Precision Medium']:.3f} | High {selective['Precision High']:.3f} | "
        f"worst {selective['Worst-Class Precision']:.3f} | coverage {selective['Coverage']:.3f}"
    )

    window_rows.append(
        {
            "Window": window_number,
            "Test Start": evaluation_raw["Date-Time"].min(),
            "Test End": evaluation_raw["Date-Time"].max(),
            "Dev Rows": len(development_df),
            "Test Rows": len(evaluation_raw),
            "Feature Set": best["Feature Set"],
            "Weight Alpha": selection["artifact"]["Alpha"],
            "C": selection["artifact"]["C"],
            "Probability Mode": selection["mode"],
            "Low Thr": selection["thresholds"][0],
            "Medium Thr": selection["thresholds"][1],
            "High Thr": selection["thresholds"][2],
            "Margin": selection["margin"],
            "Safeguards Relaxed": selection["relaxed"],
            "Tuning Worst-Class Precision": float(best["Worst-Class Precision"]),
            "Majority-Class Accuracy (always Medium)": majority_accuracy,
            **{f"Test {k}": v for k, v in selective.items()},
            "Test Raw-Argmax Macro Precision": raw_argmax["Macro Precision"],
            "Test Raw-Argmax Worst-Class Precision": raw_argmax["Worst-Class Precision"],
        }
    )

    pooled_parts.append(
        pd.DataFrame(
            {
                "Window": window_number,
                "Date-Time": result["evaluation"]["Date-Time"].to_numpy(),
                "Actual": y_eval,
                "P_Low": result["selected"][:, 0],
                "P_Medium": result["selected"][:, 1],
                "P_High": result["selected"][:, 2],
                "P_Low_Raw": result["raw"][:, 0],
                "P_Medium_Raw": result["raw"][:, 1],
                "P_High_Raw": result["raw"][:, 2],
                "Raw Argmax Prediction": result["raw_argmax_pred"],
                "Selective Prediction": result["selective_pred"],
            }
        )
    )

    last_selection = selection

# Detailed search results of the LAST window (largest development set).
last_selection["search_df"].to_csv(OUTPUT_FOLDER / "lr_threshold_search_all_last_window.csv", index=False)
last_selection["eligible"].head(200).to_excel(
    OUTPUT_FOLDER / "lr_threshold_search_eligible_top200_last_window.xlsx", index=False
)
pd.DataFrame(last_selection["config_rows"]).to_excel(
    OUTPUT_FOLDER / "lr_calibration_baselines_last_window.xlsx", index=False
)

# ============================================================
# 10. POOLED RESULTS (all test windows together)
# ============================================================

pooled = pd.concat(pooled_parts, ignore_index=True)
y_pool = pooled["Actual"].to_numpy()

pooled_selective = selective_metrics(y_pool, pooled["Selective Prediction"].to_numpy())
pooled_raw = selective_metrics(y_pool, pooled["Raw Argmax Prediction"].to_numpy())

y_pool_numeric = pd.Series(y_pool).map(class_to_index).astype(int).to_numpy()
pooled_auc = roc_auc_score(
    y_pool_numeric,
    pooled[["P_Low", "P_Medium", "P_High"]].to_numpy(),
    multi_class="ovr",
    average="macro",
    labels=[0, 1, 2],
)
pooled_majority_accuracy = float(np.mean(y_pool == "Medium"))

window_table = pd.DataFrame(window_rows)
window_table.to_excel(OUTPUT_FOLDER / "lr_rolling_backtest_by_window.xlsx", index=False)

comparison = pd.DataFrame(
    [
        {"Method": "Raw Argmax (pooled)", **pooled_raw},
        {"Method": "Selective thresholds (pooled)", **pooled_selective},
    ]
)
comparison.to_excel(OUTPUT_FOLDER / "lr_pooled_method_comparison.xlsx", index=False)
pd.DataFrame(
    [
        {
            **pooled_selective,
            "ROC-AUC Macro OvR": pooled_auc,
            "Majority-Class Accuracy (always Medium)": pooled_majority_accuracy,
            "Test Rows": len(pooled),
        }
    ]
).to_excel(OUTPUT_FOLDER / "lr_selective_final_metrics.xlsx", index=False)

print("\n" + "=" * 98)
print("POOLED RESULT OVER ALL TEST WINDOWS (selective model)")
print("=" * 98)
print("Test rows:", len(pooled))
for c in CLASS_ORDER:
    print(
        f"Precision {c:<6}: {pooled_selective[f'Precision {c}']:.3f}  "
        f"(95% CI {pooled_selective[f'Precision {c} CI95 Low']:.3f}"
        f"-{pooled_selective[f'Precision {c} CI95 High']:.3f}, "
        f"{pooled_selective[f'Predicted {c}']} predictions) | "
        f"Recall {pooled_selective[f'Recall {c}']:.3f}"
    )
print(f"Worst-class precision : {pooled_selective['Worst-Class Precision']:.3f}")
print(f"Macro precision       : {pooled_selective['Macro Precision']:.3f}")
print(f"Coverage              : {pooled_selective['Coverage']:.3f}")
print(f"Accuracy on covered   : {pooled_selective['Accuracy on Covered Rows']:.3f}")
print(f"Always-Medium accuracy: {pooled_majority_accuracy:.3f}  (baseline to beat)")
print(f"ROC-AUC (macro OvR)   : {pooled_auc:.3f}")
print(
    "Raw argmax (no thresholds): "
    f"Low {pooled_raw['Precision Low']:.3f} | Medium {pooled_raw['Precision Medium']:.3f} | "
    f"High {pooled_raw['Precision High']:.3f}"
)

print("\nPer-window summary:")
print(
    window_table[
        [
            "Window",
            "Test Rows",
            "Feature Set",
            "Weight Alpha",
            "C",
            "Test Precision Low",
            "Test Precision Medium",
            "Test Precision High",
            "Test Worst-Class Precision",
            "Test Coverage",
        ]
    ]
    .round(3)
    .to_string(index=False)
)

# ============================================================
# 11. PRECISION / RECALL TABLE PER CLASS (pooled test windows)
# ============================================================
# For each class: "predict this class when its probability >= threshold".
# DESCRIPTIVE ONLY - use it to see the trade-off, but pick the final operating
# point from tuning data, not from this table.

pr_rows = []
prob_matrix = pooled[["P_Low", "P_Medium", "P_High"]].to_numpy()
for class_index, class_name in enumerate(CLASS_ORDER):
    is_class = y_pool == class_name
    for threshold in PR_TABLE_THRESHOLDS:
        predicted = prob_matrix[:, class_index] >= threshold
        n_pred = int(predicted.sum())
        tp = int((predicted & is_class).sum())
        lo, _ = wilson_interval(tp, n_pred, 1.96)
        pr_rows.append(
            {
                "Class": class_name,
                "Probability Threshold": threshold,
                "Rows Flagged": n_pred,
                "True Positives": tp,
                "Precision": tp / n_pred if n_pred else np.nan,
                "Precision Lower Bound 95%": float(lo) if n_pred else np.nan,
                "Recall": tp / int(is_class.sum()) if is_class.sum() else np.nan,
            }
        )

pr_table = pd.DataFrame(pr_rows)
pr_table.to_excel(OUTPUT_FOLDER / "lr_precision_recall_table.xlsx", index=False)

for class_name in ["Low", "High"]:
    print(f"\nPrecision / recall table - {class_name} (pooled test windows)")
    print(
        pr_table[pr_table["Class"] == class_name]
        .drop(columns="Class")
        .round(3)
        .to_string(index=False)
    )

# Precision-recall curves figure.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, class_name in zip(axes, CLASS_ORDER):
    sub = pr_table[(pr_table["Class"] == class_name) & (pr_table["Rows Flagged"] > 0)]
    ax.plot(sub["Recall"], sub["Precision"], marker="o")
    base_rate = float(np.mean(y_pool == class_name))
    ax.axhline(base_rate, linestyle="--", color="grey", label=f"base rate {base_rate:.2f}")
    ax.set_title(f"{class_name}: precision vs recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_FOLDER / "lr_precision_recall_curves.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 12. POOLED CONFUSION MATRIX + ROW-LEVEL PREDICTIONS
# ============================================================

cm_labels = CLASS_ORDER + [UNCERTAIN]
cm = confusion_matrix(y_pool, pooled["Selective Prediction"].to_numpy(), labels=cm_labels)
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels).plot(ax=ax, values_format="d")
ax.set_title("Logistic Regression v3 - pooled selective confusion matrix")
fig.tight_layout()
fig.savefig(OUTPUT_FOLDER / "lr_selective_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig)

pooled.to_excel(OUTPUT_FOLDER / "lr_selective_predictions.xlsx", index=False)

print("\nSelective classification report (pooled):")
print(
    classification_report(
        y_pool, pooled["Selective Prediction"].to_numpy(), labels=CLASS_ORDER, zero_division=0
    )
)

print("Output folder:")
print(OUTPUT_FOLDER)
print("\nHow to read this:")
print("1. Test windows are always AFTER the data used to choose that window's settings.")
print("2. Compare 'Accuracy on covered' with 'Always-Medium accuracy' - a model must beat it.")
print("3. If a class CI95 is wide (few predictions), do not trust its precision.")
print("4. If windows disagree on features/thresholds, the choice is unstable - prefer simpler settings.")
