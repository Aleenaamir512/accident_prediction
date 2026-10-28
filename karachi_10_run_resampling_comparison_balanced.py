"""Wang et al. (2022)-inspired, leakage-conscious Karachi traffic-risk benchmarking.

26 configurations = 4 actual classifiers x 6 training-only resampling methods,
plus 2 class-weight-balanced baselines (RF and Logistic Regression on original data).
Evaluates either 10 repeated 80/20 random splits (paper-like) or 10
expanding chronological windows (primary for time-dependent observations).
Saves every run, mean and sample SD, and audit of splits and fitted classifiers.

IMPORTANT: This is an ADAPTATION, not a replication of the published paper:
the data, selected classifiers, features and risk labels differ; the 10 random
splits here are intentionally unstratified (as 'random' described in paper).
The Risk labels' grounding in independently observed crashes is unverified.
Already inspected dataset cannot be described as a pristine holdout.

Required: pip install pandas openpyxl scikit-learn imbalanced-learn xgboost lightgbm
Examples:
  python karachi_10_run_resampling_comparison.py --mode both
  python karachi_10_run_resampling_comparison.py --mode chronological --runs 1 --estimators 30 --output smoke_check
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
INPUT_CANDIDATES = [
    ROOT / 'karachi_traffic_feature_engineered_final_v2.xlsx',
    ROOT / 'karachi_traffic_feature_engineered_final_v2(1).xlsx',
]
TARGET = 'Accident Risk'
LABELS = ['Low', 'Medium', 'High']
NUMBERS = {name: i for i, name in enumerate(LABELS)}
FEATURES = [
    'Vehicle Count', 'Road Type', 'Road Condition', 'Area',
    'Number of Lanes', 'Traffic Signal', 'ML Weather', 'Hour Sin', 'Hour Cos',
]
CATEGORICAL = ['Road Type', 'Road Condition', 'Area', 'ML Weather']
NUMERICAL = [f for f in FEATURES if f not in CATEGORICAL]
CLASSIFIERS = ['Random Forest', 'XGBoost', 'LightGBM', 'Logistic Regression']
SAMPLERS = ['Original', 'RUS', 'ROS', 'ENN', 'SMOTENC', 'SMOTEENN']
BALANCED_CLASSIFIERS = ['Random Forest', 'Logistic Regression']
# Balanced class-weight baselines are evaluated only on Original training data.
# This isolates weighting effects from resampling effects.
CONFIGURATIONS = (
    [(algorithm, sampler, False) for sampler in SAMPLERS for algorithm in CLASSIFIERS]
    + [(algorithm, 'Original', True) for algorithm in BALANCED_CLASSIFIERS]
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['both', 'paper_random', 'chronological'], default='both')
    parser.add_argument('--runs', type=int, default=10, help='Default 10; use 1 for smoke testing.')
    parser.add_argument('--estimators', type=int, default=300,
                        help='Random Forest / XGBoost / LightGBM trees, default 300.')
    parser.add_argument('--input', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=ROOT / 'Karachi_10_Run_Comparison_With_Balanced')
    return parser.parse_args()


def onehot():
    return OneHotEncoder(handle_unknown='ignore', sparse_output=False)


def prep_onehot():
    """Fit imputation, scaling and OHE strictly within each training fold."""
    return ColumnTransformer([
        ('numbers', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), NUMERICAL),
        ('categories', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', onehot()),
        ]), CATEGORICAL),
    ], sparse_threshold=0)


def prep_smotenc():
    """Numerical scaled then ordinal category codes for SMOTENC."""
    return ColumnTransformer([
        ('numbers', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), NUMERICAL),
        ('categories', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(
                handle_unknown='use_encoded_value', unknown_value=-1)),
        ]), CATEGORICAL),
    ], sparse_threshold=0)


def after_smotenc():
    n = len(NUMERICAL)
    return ColumnTransformer([
        ('numbers', 'passthrough', list(range(n))),
        ('categories', onehot(), list(range(n, n + len(CATEGORICAL)))),
    ], sparse_threshold=0)


def classifier_factory(which, seed, estimators, balanced=False):
    weights = 'balanced' if balanced else None
    if which == 'Random Forest':
        return RandomForestClassifier(n_estimators=estimators, min_samples_leaf=3,
                                      class_weight=weights, random_state=seed, n_jobs=-1)
    if which == 'XGBoost':
        return XGBClassifier(objective='multi:softprob', num_class=3,
                             eval_metric='mlogloss', n_estimators=estimators,
                             max_depth=4, learning_rate=.05,
                             subsample=.9, colsample_bytree=.9,
                             tree_method='hist', random_state=seed, n_jobs=-1)
    if which == 'LightGBM':
        return LGBMClassifier(objective='multiclass', num_class=3,
                              n_estimators=estimators, num_leaves=15,
                              min_child_samples=20, learning_rate=.05,
                              random_state=seed, verbose=-1, n_jobs=-1)
    if which == 'Logistic Regression':
        return LogisticRegression(max_iter=2000, class_weight=weights, random_state=seed)
    raise ValueError(f'Unrecognized classifier: {which}')


def build_pipeline(which, technique, seed=42, estimators=300, balanced=False):
    if balanced and (which not in BALANCED_CLASSIFIERS or technique != 'Original'):
        raise ValueError('Balanced weights are included only for RF and Logistic Regression on Original data.')
    clf = classifier_factory(which, seed, estimators, balanced=balanced)
    if technique == 'SMOTENC':
        n = len(NUMERICAL)
        return ImbPipeline([
            ('prepare', prep_smotenc()),
            ('sampling', SMOTENC(categorical_features=list(range(n, n + len(CATEGORICAL))),
                                 k_neighbors=5, random_state=seed)),
            ('post_sample_ohe', after_smotenc()),
            ('classifier', clf),
        ])

    steps = [('prepare', prep_onehot())]
    if technique == 'RUS':
        steps.append(('sampling', RandomUnderSampler(random_state=seed)))
    elif technique == 'ROS':
        steps.append(('sampling', RandomOverSampler(random_state=seed)))
    elif technique == 'ENN':
        # ENN on scaled numerical and one-hot encoded categorical attributes.
        steps.append(('sampling', EditedNearestNeighbours(n_neighbors=3)))
    elif technique == 'SMOTEENN':
        # Standard SMOTEENN interpolates OHE dummies, potentially making
        # fractional category indicators; keep as paper-style benchmark only.
        steps.append(('sampling', SMOTEENN(random_state=seed)))
    elif technique != 'Original':
        raise ValueError(f'Unrecognized resampling method: {technique}')
    steps.append(('classifier', clf))
    return ImbPipeline(steps)


def read_data(path):
    if path is None:
        path = next((p for p in INPUT_CANDIDATES if p.exists()), None)
    if path is None or not path.exists():
        raise FileNotFoundError('Place karachi_traffic_feature_engineered_final_v2.xlsx next to this script.')
    df = pd.read_excel(path, sheet_name='Data')
    cols = ['Date-Time', TARGET, *FEATURES]
    missing = sorted(set(cols).difference(df.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    df['Date-Time'] = pd.to_datetime(df['Date-Time'], errors='raise')
    df = df.sort_values('Date-Time', kind='mergesort').reset_index(drop=True)
    if df['Date-Time'].duplicated().any():
        raise ValueError('Duplicate timestamps found: check grouping/leakage before running.')
    if df[TARGET].isna().any() or not df[TARGET].isin(LABELS).all():
        raise ValueError('Invalid target labels. Expected only Low, Medium, High.')
    for col in CATEGORICAL:
        df[col] = df[col].astype('string').fillna('Unknown').astype(str)
    return df, path


def make_splits(df, mode, runs):
    n = len(df)
    if mode == 'chronological':
        # With 1,500 rows: train 500 -> next 100; ..., train 1,400 -> last 100.
        if n != 1500:
            raise ValueError('Chronological windows are specified for exactly 1,500 rows.')
        if not (1 <= runs <= 10):
            raise ValueError('Chronological mode requires 1 to 10 runs.')
        for k in range(runs):
            train_end = 500 + 100*k
            train_idx = np.arange(train_end)
            test_idx = np.arange(train_end, train_end + 100)
            yield k+1, 42+k, train_idx, test_idx
    elif mode == 'paper_random':
        if runs <= 0:
            raise ValueError('Number of runs must be positive.')
        all_idx = np.arange(n)
        for k in range(runs):
            train_idx, test_idx = train_test_split(all_idx, test_size=.20,
                                                    random_state=42+k, shuffle=True)
            yield k+1, 42+k, train_idx, test_idx
    else:
        raise ValueError(mode)


def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=[0,1,2], average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=[0,1,2], average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=[0,1,2], average=None, zero_division=0)
    metrics = {
        'Accuracy': acc,
        'Macro Precision': precision_score(y_true, y_pred, labels=[0,1,2], average='macro', zero_division=0),
        'Macro Recall': recall_score(y_true, y_pred, labels=[0,1,2], average='macro', zero_division=0),
        'Macro F1': f1_score(y_true, y_pred, labels=[0,1,2], average='macro', zero_division=0),
        # Wang et al. Equation (10), adapted with L1=Low, L2=Medium.
        # This is NOT macro F1 or precision and isn't directly comparable across datasets.
        'Wang Adapted Score': (acc + recall[0] + recall[1]) / 3,
    }
    for i, label in enumerate(LABELS):
        metrics[f'Precision {label}'] = precision[i]
        metrics[f'Recall {label}'] = recall[i]
        metrics[f'F1 {label}'] = f1s[i]
        metrics[f'Support {label}'] = int((np.asarray(y_true) == i).sum())
    return metrics


def evaluate_mode(df, mode, args, out):
    X = df[FEATURES]
    y = df[TARGET].map(NUMBERS).astype(int)
    rows, split_rows, errors = [], [], []
    path_runs = out / f'{mode}_all_runs.csv'
    path_splits = out / f'{mode}_splits.csv'
    path_errors = out / f'{mode}_errors.csv'
    for run, seed, ti, vi in make_splits(df, mode, args.runs):
        overlap = np.intersect1d(ti, vi)
        assert len(overlap) == 0, 'TRAIN/TEST LEAKAGE: split indices overlap'
        X_train, y_train = X.iloc[ti], y.iloc[ti]
        X_val, y_val = X.iloc[vi], y.iloc[vi]
        split_rows.append({
            'Mode': mode, 'Run': run, 'Random Seed': seed,
            'Train Rows': len(ti), 'Test Rows': len(vi),
            'Train Start': str(df['Date-Time'].iloc[ti].min()),
            'Train End': str(df['Date-Time'].iloc[ti].max()),
            'Test Start': str(df['Date-Time'].iloc[vi].min()),
            'Test End': str(df['Date-Time'].iloc[vi].max()),
            'Train Class Counts': json.dumps(df[TARGET].iloc[ti].value_counts().to_dict()),
            'Test Class Counts': json.dumps(df[TARGET].iloc[vi].value_counts().to_dict()),
            'Indices Overlap': len(overlap),
        })
        print(f'\n[{mode}] RUN {run}/{args.runs} | train={len(ti)} test={len(vi)}',flush=True)
        for algorithm, technique, balanced in CONFIGURATIONS:
            model_name = (f'{algorithm} (class_weight=balanced) + Original'
                          if balanced else f'{algorithm} + {technique}')
            model = build_pipeline(algorithm, technique, seed=seed,
                                   estimators=args.estimators, balanced=balanced)
            actual = type(model.named_steps['classifier']).__name__
            expected = {'Random Forest': 'RandomForestClassifier',
                        'XGBoost': 'XGBClassifier',
                        'LightGBM': 'LGBMClassifier',
                        'Logistic Regression': 'LogisticRegression'}[algorithm]
            if actual != expected:
                raise RuntimeError(f'{model_name}: expected {expected}, got {actual}')
            if algorithm in BALANCED_CLASSIFIERS:
                actual_weights = model.named_steps['classifier'].get_params()['class_weight']
                expected_weights = 'balanced' if balanced else None
                if actual_weights != expected_weights:
                    raise RuntimeError(f'{model_name}: class_weight={actual_weights} instead of {expected_weights}')
            started = time.monotonic()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=FutureWarning)
                    model.fit(X_train, y_train)
                pred = model.predict(X_val)
                rec = {
                    'Evaluation Mode': mode, 'Run': run, 'Seed': seed,
                    'Model': model_name, 'Classifier': algorithm,
                    'Resampling': technique,
                    'Class Weighting': 'balanced' if balanced else 'None',
                    'Actual Classifier': actual,
                    'Train Rows': len(ti), 'Test Rows': len(vi),
                    'Training Seconds': round(time.monotonic()-started,3),
                }
                rec.update(calculate_metrics(y_val, pred))
                rows.append(rec)
                print(f'  {len(rows):>3}/{args.runs*len(CONFIGURATIONS)} {model_name:<57} F1={rec["Macro F1"]:.3f} P={rec["Macro Precision"]:.3f}',flush=True)
            except Exception as exc:
                errors.append({'Mode': mode, 'Run': run, 'Model': model_name,
                               'Error': repr(exc), 'Traceback': traceback.format_exc()})
                print(f'  ERROR {model_name}: {exc}',flush=True)
        # Save progress after each run; interrupted jobs preserve complete runs.
        pd.DataFrame(rows).to_csv(path_runs, index=False)
        pd.DataFrame(split_rows).to_csv(path_splits, index=False)
        if errors:
            pd.DataFrame(errors).to_csv(path_errors, index=False)
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f'No model fits succeeded for {mode}.')
    cols = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 'Wang Adapted Score'] + [
        f'{metric} {label}' for label in LABELS for metric in ['Precision','Recall','F1']
    ]
    mean = result.groupby(['Classifier', 'Resampling','Model'])[cols].agg(['mean','std','count'])
    mean.columns = [f'{c} {a.capitalize()}' for c,a in mean.columns]
    mean = mean.reset_index().sort_values(['Macro F1 Mean','Macro Precision Mean'],ascending=False)
    mean.to_csv(out / f'{mode}_mean_sd.csv', index=False)
    print(f'\n[{mode}] TOP 10 BY MEAN MACRO F1:')
    print(mean[['Model','Macro Precision Mean','Macro F1 Mean','Macro F1 Std','Wang Adapted Score Mean']].head(10).round(4).to_string(index=False),flush=True)
    if errors:
        print(f'NOTE: {len(errors)} fits failed. Inspect {path_errors.name}.',flush=True)
    return result, mean, errors


def main():
    args = parse_args()
    df, source = read_data(args.input)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.runs < 1 or args.estimators < 1:
        raise ValueError('runs and estimators must be at least 1.')
    print(f'DATASET: {source} | {len(df)} rows | target counts {df[TARGET].value_counts().to_dict()}',flush=True)
    print(f'FEATURES: {FEATURES}',flush=True)
    print(f'MODE: {args.mode} | RUNS: {args.runs} | trees per ensemble: {args.estimators}',flush=True)
    (out / 'run_settings.json').write_text(json.dumps({
        'input': str(source), 'mode': args.mode, 'runs': args.runs,
        'estimators': args.estimators, 'features': FEATURES, 'categorical': CATEGORICAL,
        'class_order': LABELS, 'models': CLASSIFIERS, 'resamplers': SAMPLERS,
        'balanced_weight_models': BALANCED_CLASSIFIERS,
        'total_configurations': len(CONFIGURATIONS),
        'note': 'Paper-inspired retrospective study; no independent crash-label provenance established.',
    },indent=2),encoding='utf-8')
    all_errors=[]
    modes = ['paper_random','chronological'] if args.mode == 'both' else [args.mode]
    for mode in modes:
        _,_,errs = evaluate_mode(df,mode,args,out)
        all_errors.extend(errs)
    if all_errors:
        raise RuntimeError(f'{len(all_errors)} model evaluations failed; see output error files.')
    print('\nFINISHED. Output:',out,flush=True)


if __name__ == '__main__':
    main()
