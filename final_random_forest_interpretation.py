# ============================================================
# FINAL RANDOM FOREST INTERPRETATION
# Feature Importance + SHAP Analysis
# Karachi Traffic-Risk Thesis
# ============================================================

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(r"D:\accident_prediction")

OUTPUT_DIR = BASE_DIR / "Final_Random_Forest_Output"

MODEL_PATH = (
    OUTPUT_DIR
    / "final_random_forest_model.joblib"
)

DATA_PATH = (
    BASE_DIR
    / "karachi_traffic_feature_engineered_final_v2.xlsx"
)


# ============================================================
# 2. FINAL MODEL FEATURES
# ============================================================

TARGET = "Accident Risk"

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

NUMERICAL_FEATURES = [
    "Vehicle Count",
    "Number of Lanes",
    "Traffic Signal",
    "Hour Sin",
    "Hour Cos",
]

CATEGORICAL_FEATURES = [
    "Road Type",
    "Road Condition",
    "Area",
    "ML Weather",
]

TEST_FRACTION = 0.20

TOP_N = 15


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("=" * 75)
print("FINAL KARACHI RANDOM FOREST INTERPRETATION")
print("=" * 75)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

print("\nLoading final model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

if not hasattr(model, "named_steps"):
    raise TypeError(
        "The saved model is not a sklearn Pipeline."
    )

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

print(
    "Classifier:",
    classifier.__class__.__name__,
)

print(
    "Class weight:",
    classifier.class_weight,
)

print(
    "Classes:",
    list(classifier.classes_),
)


# ============================================================
# 4. LOAD DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}"
    )

print("\nLoading dataset...")

df = pd.read_excel(
    DATA_PATH,
    sheet_name="Data",
)

required_columns = [
    "Date-Time",
    TARGET,
    *FEATURES,
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        "Missing columns:\n"
        + "\n".join(missing)
    )

df["Date-Time"] = pd.to_datetime(
    df["Date-Time"],
    errors="raise",
)

df = (
    df
    .sort_values(
        "Date-Time",
        kind="mergesort",
    )
    .reset_index(drop=True)
)

print(
    "Dataset rows:",
    len(df),
)


# ============================================================
# 5. USE SAME FINAL FUTURE TEST SET
# ============================================================

split_index = int(
    len(df)
    * (1 - TEST_FRACTION)
)

test_df = (
    df
    .iloc[split_index:]
    .copy()
)

X_test = test_df[FEATURES].copy()

y_test = test_df[TARGET].copy()

print(
    "\nFuture evaluation rows:",
    len(test_df),
)

print(
    "Test period:",
    test_df["Date-Time"].min(),
    "to",
    test_df["Date-Time"].max(),
)


# ============================================================
# 6. TRANSFORM FEATURES
# ============================================================

print(
    "\nTransforming features using "
    "the saved preprocessing pipeline..."
)

X_transformed = preprocessor.transform(
    X_test
)

if hasattr(
    X_transformed,
    "toarray",
):
    X_transformed = (
        X_transformed.toarray()
    )

X_transformed = np.asarray(
    X_transformed,
    dtype=float,
)

transformed_feature_names = list(
    preprocessor.get_feature_names_out()
)

print(
    "Number of transformed features:",
    len(transformed_feature_names),
)


# ============================================================
# 7. MAP TRANSFORMED FEATURES BACK TO ORIGINAL FEATURES
# ============================================================

mapping_rows = []

# Numerical features
for feature in NUMERICAL_FEATURES:

    mapping_rows.append(
        {
            "Original Feature": feature,
            "Feature Type": "Numerical",
            "Category": "",
        }
    )


# Categorical one-hot features
categorical_pipeline = (
    preprocessor
    .named_transformers_[
        "categorical"
    ]
)

encoder = (
    categorical_pipeline
    .named_steps[
        "onehot"
    ]
)

for feature, categories in zip(
    CATEGORICAL_FEATURES,
    encoder.categories_,
):

    for category in categories:

        mapping_rows.append(
            {
                "Original Feature": feature,
                "Feature Type":
                    "Categorical",
                "Category":
                    str(category),
            }
        )


if (
    len(mapping_rows)
    != len(
        transformed_feature_names
    )
):

    raise RuntimeError(
        "\nFeature mapping error.\n"
        f"Mapping rows: "
        f"{len(mapping_rows)}\n"
        f"Transformed features: "
        f"{len(transformed_feature_names)}"
    )


feature_mapping = pd.DataFrame(
    mapping_rows
)

feature_mapping.insert(
    0,
    "Transformed Feature",
    transformed_feature_names,
)

feature_mapping.to_csv(
    OUTPUT_DIR
    / "feature_name_mapping.csv",
    index=False,
)


# ============================================================
# 8. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print(
    "\nCalculating Random Forest "
    "feature importance..."
)

rf_importance = (
    feature_mapping.copy()
)

rf_importance[
    "RF Importance"
] = classifier.feature_importances_

rf_importance = (
    rf_importance
    .sort_values(
        "RF Importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


# Save transformed feature importance
rf_importance.to_csv(
    OUTPUT_DIR
    / "rf_feature_importance_transformed.csv",
    index=False,
)


# Aggregate one-hot columns back
# to original 9 features

rf_original = (
    rf_importance
    .groupby(
        "Original Feature",
        as_index=False,
    )[
        "RF Importance"
    ]
    .sum()
    .sort_values(
        "RF Importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


rf_original.to_csv(
    OUTPUT_DIR
    / "rf_feature_importance_original.csv",
    index=False,
)


# ============================================================
# 9. RF FEATURE IMPORTANCE PLOT
# ============================================================

plot_df = (
    rf_original
    .sort_values(
        "RF Importance",
        ascending=True,
    )
)

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    plot_df[
        "Original Feature"
    ],
    plot_df[
        "RF Importance"
    ],
)

plt.xlabel(
    "Random Forest Importance"
)

plt.ylabel(
    "Original Feature"
)

plt.title(
    "Random Forest Feature Importance\n"
    "Final Karachi Traffic-Risk Model"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "rf_feature_importance_original.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 10. PREPARE DATA FOR SHAP
# ============================================================

print(
    "\nCalculating SHAP values..."
)

X_shap = pd.DataFrame(
    X_transformed,
    columns=transformed_feature_names,
    index=X_test.index,
)


# ============================================================
# 11. SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    classifier
)

raw_shap_values = (
    explainer.shap_values(
        X_shap
    )
)


# ============================================================
# 12. NORMALIZE SHAP OUTPUT FORMAT
# ============================================================

number_classes = len(
    classifier.classes_
)

number_samples = len(
    X_shap
)

number_features = len(
    transformed_feature_names
)


def normalize_shap_values(
    values,
):

    # Older SHAP versions:
    # list of arrays,
    # one array per class

    if isinstance(
        values,
        list,
    ):

        return np.stack(
            values,
            axis=2,
        )

    values = np.asarray(
        values
    )

    # Newer SHAP:
    # samples x features x classes

    if (
        values.ndim == 3
        and
        values.shape
        == (
            number_samples,
            number_features,
            number_classes,
        )
    ):

        return values

    # classes x samples x features

    if (
        values.ndim == 3
        and
        values.shape
        == (
            number_classes,
            number_samples,
            number_features,
        )
    ):

        return np.transpose(
            values,
            (1, 2, 0),
        )

    # samples x classes x features

    if (
        values.ndim == 3
        and
        values.shape
        == (
            number_samples,
            number_classes,
            number_features,
        )
    ):

        return np.transpose(
            values,
            (0, 2, 1),
        )

    raise RuntimeError(
        "Unexpected SHAP shape: "
        f"{values.shape}"
    )


shap_values = (
    normalize_shap_values(
        raw_shap_values
    )
)

print(
    "SHAP array shape:",
    shap_values.shape,
)


# ============================================================
# 13. GLOBAL SHAP IMPORTANCE
# ============================================================

# Absolute SHAP averaged across
# observations AND classes

global_mean_abs_shap = (
    np
    .abs(
        shap_values
    )
    .mean(
        axis=(0, 2)
    )
)


shap_global = (
    feature_mapping.copy()
)

shap_global[
    "Mean Absolute SHAP"
] = global_mean_abs_shap


shap_global = (
    shap_global
    .sort_values(
        "Mean Absolute SHAP",
        ascending=False,
    )
    .reset_index(drop=True)
)


shap_global.to_csv(
    OUTPUT_DIR
    / "shap_global_importance_transformed.csv",
    index=False,
)


# ============================================================
# 14. AGGREGATE SHAP TO ORIGINAL FEATURES
# ============================================================

shap_original = (
    shap_global
    .groupby(
        "Original Feature",
        as_index=False,
    )[
        "Mean Absolute SHAP"
    ]
    .sum()
    .sort_values(
        "Mean Absolute SHAP",
        ascending=False,
    )
    .reset_index(drop=True)
)


shap_original.to_csv(
    OUTPUT_DIR
    / "shap_global_importance_original.csv",
    index=False,
)


# ============================================================
# 15. GLOBAL SHAP IMPORTANCE PLOT
# ============================================================

plot_df = (
    shap_original
    .sort_values(
        "Mean Absolute SHAP",
        ascending=True,
    )
)

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    plot_df[
        "Original Feature"
    ],
    plot_df[
        "Mean Absolute SHAP"
    ],
)

plt.xlabel(
    "Mean Absolute SHAP Value"
)

plt.ylabel(
    "Original Feature"
)

plt.title(
    "Global SHAP Feature Importance\n"
    "Final Karachi Traffic-Risk Model"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "shap_global_importance_original.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 16. CLASS-WISE SHAP ANALYSIS
# ============================================================

classwise_rows = []

classes = list(
    classifier.classes_
)


for class_index, class_name in enumerate(
    classes
):

    print(
        "\nProcessing SHAP for class:",
        class_name,
    )

    class_values = (
        shap_values[
            :,
            :,
            class_index,
        ]
    )

    class_mean_abs = (
        np
        .abs(
            class_values
        )
        .mean(
            axis=0
        )
    )

    temp = (
        feature_mapping.copy()
    )

    temp[
        "Mean Absolute SHAP"
    ] = class_mean_abs

    original_class = (
        temp
        .groupby(
            "Original Feature",
            as_index=False,
        )[
            "Mean Absolute SHAP"
        ]
        .sum()
        .sort_values(
            "Mean Absolute SHAP",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    original_class.insert(
        0,
        "Class",
        class_name,
    )

    classwise_rows.append(
        original_class
    )


    # ----------------------------
    # Save class-specific CSV
    # ----------------------------

    safe_class_name = (
        str(class_name)
        .replace(
            " ",
            "_",
        )
    )

    original_class.to_csv(
        OUTPUT_DIR
        / (
            f"shap_importance_"
            f"{safe_class_name}.csv"
        ),
        index=False,
    )


    # ----------------------------
    # Class-specific bar plot
    # ----------------------------

    class_plot = (
        original_class
        .sort_values(
            "Mean Absolute SHAP",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        class_plot[
            "Original Feature"
        ],
        class_plot[
            "Mean Absolute SHAP"
        ],
    )

    plt.xlabel(
        "Mean Absolute SHAP Value"
    )

    plt.title(
        f"SHAP Importance for "
        f"{class_name} Risk"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / (
            f"shap_importance_"
            f"{safe_class_name}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


    # ========================================================
    # 17. SHAP SUMMARY / BEESWARM PLOT
    # ========================================================

    try:

        shap.summary_plot(
            class_values,
            X_shap,
            feature_names=
                transformed_feature_names,
            max_display=TOP_N,
            show=False,
        )

        plt.title(
            f"SHAP Summary Plot - "
            f"{class_name} Risk"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            / (
                f"shap_summary_"
                f"{safe_class_name}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    except Exception as error:

        warnings.warn(
            "Could not generate "
            f"SHAP summary plot for "
            f"{class_name}: "
            f"{error}"
        )

        plt.close(
            "all"
        )


# ============================================================
# 18. SAVE COMBINED CLASS-WISE CSV
# ============================================================

classwise_df = pd.concat(
    classwise_rows,
    ignore_index=True,
)

classwise_df.to_csv(
    OUTPUT_DIR
    / "shap_classwise_importance_original.csv",
    index=False,
)


# ============================================================
# 19. CLASS-WISE COMPARISON PLOT
# ============================================================

classwise_pivot = (
    classwise_df
    .pivot(
        index="Original Feature",
        columns="Class",
        values="Mean Absolute SHAP",
    )
    .fillna(0)
)


# Sort features according to
# overall SHAP importance

feature_order = (
    shap_original[
        "Original Feature"
    ]
    .tolist()
)

classwise_pivot = (
    classwise_pivot
    .reindex(
        feature_order
    )
)


ax = classwise_pivot.plot(
    kind="bar",
    figsize=(12, 7),
)

ax.set_xlabel(
    "Original Feature"
)

ax.set_ylabel(
    "Mean Absolute SHAP Value"
)

ax.set_title(
    "Class-wise SHAP Feature Importance"
)

plt.xticks(
    rotation=45,
    ha="right",
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "shap_classwise_importance_original.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 20. SAVE PER-OBSERVATION ORIGINAL-FEATURE SHAP
# ============================================================

# Map transformed columns
# back to original feature indices

original_mapping = {}

for feature in FEATURES:

    original_mapping[
        feature
    ] = feature_mapping.index[
        feature_mapping[
            "Original Feature"
        ]
        == feature
    ].tolist()


all_observation_rows = []

for class_index, class_name in enumerate(
    classes
):

    class_shap = shap_values[
        :,
        :,
        class_index,
    ]

    for row_position, dataset_index in enumerate(
        X_test.index
    ):

        record = {
            "Dataset Index":
                int(dataset_index),

            "Date-Time":
                str(
                    df.loc[
                        dataset_index,
                        "Date-Time",
                    ]
                ),

            "Actual Risk":
                str(
                    y_test.loc[
                        dataset_index
                    ]
                ),

            "Explained Class":
                str(
                    class_name
                ),
        }

        for feature in FEATURES:

            positions = (
                original_mapping[
                    feature
                ]
            )

            record[
                feature
            ] = float(
                class_shap[
                    row_position,
                    positions,
                ].sum()
            )

        all_observation_rows.append(
            record
        )


per_observation_shap = pd.DataFrame(
    all_observation_rows
)

per_observation_shap.to_csv(
    OUTPUT_DIR
    / (
        "shap_values_original_features_"
        "per_observation.csv"
    ),
    index=False,
)


# ============================================================
# 21. LOCAL EXPLANATION FOR ONE HIGH-RISK PREDICTION
# ============================================================

predictions = model.predict(
    X_test
)

prediction_series = pd.Series(
    predictions,
    index=X_test.index,
)


if "High" in classes:

    high_class_index = (
        classes.index(
            "High"
        )
    )

    correct_high_indices = (
        X_test.index[
            (
                y_test
                == "High"
            )
            &
            (
                prediction_series
                == "High"
            )
        ]
        .tolist()
    )

    if correct_high_indices:

        example_index = (
            correct_high_indices[
                0
            ]
        )

        example_position = (
            X_test.index
            .get_loc(
                example_index
            )
        )

        high_shap_values = (
            shap_values[
                example_position,
                :,
                high_class_index,
            ]
        )

        local_data = (
            feature_mapping.copy()
        )

        local_data[
            "SHAP Contribution"
        ] = high_shap_values

        local_data[
            "Absolute SHAP"
        ] = np.abs(
            high_shap_values
        )

        local_data = (
            local_data
            .sort_values(
                "Absolute SHAP",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        local_data.to_csv(
            OUTPUT_DIR
            / "shap_local_high_risk_example.csv",
            index=False,
        )


        local_plot = (
            local_data
            .head(15)
            .sort_values(
                "SHAP Contribution",
                ascending=True,
            )
        )

        plt.figure(
            figsize=(10, 7)
        )

        plt.barh(
            local_plot[
                "Transformed Feature"
            ],
            local_plot[
                "SHAP Contribution"
            ],
        )

        plt.axvline(
            0,
            linewidth=1,
        )

        plt.xlabel(
            "SHAP Contribution "
            "toward High-risk prediction"
        )

        plt.title(
            "Local SHAP Explanation\n"
            "Example Correct High-risk Prediction"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            / "shap_local_high_risk_example.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# 22. PRINT MAIN FINDINGS
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "RANDOM FOREST FEATURE IMPORTANCE"
)

print(
    "=" * 75
)


print(
    "\nTop original features "
    "by Random Forest importance:"
)

for rank, (_, row) in enumerate(
    rf_original.head(5).iterrows(),
    start=1,
):

    print(
        f"{rank}. "
        f"{row['Original Feature']} "
        f"= "
        f"{row['RF Importance']:.4f}"
    )


print(
    "\n"
    + "=" * 75
)

print(
    "GLOBAL SHAP IMPORTANCE"
)

print(
    "=" * 75
)


print(
    "\nTop original features "
    "by global mean |SHAP|:"
)

for rank, (_, row) in enumerate(
    shap_original.head(5).iterrows(),
    start=1,
):

    print(
        f"{rank}. "
        f"{row['Original Feature']} "
        f"= "
        f"{row['Mean Absolute SHAP']:.6f}"
    )


# ============================================================
# 23. PRINT HIGH-RISK FINDINGS
# ============================================================

if "High" in classes:

    high_results = (
        classwise_df[
            classwise_df[
                "Class"
            ]
            == "High"
        ]
        .sort_values(
            "Mean Absolute SHAP",
            ascending=False,
        )
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "HIGH-RISK SHAP IMPORTANCE"
    )

    print(
        "=" * 75
    )

    print(
        "\nTop features influencing "
        "High-risk predictions:"
    )

    for rank, (_, row) in enumerate(
        high_results.head(5).iterrows(),
        start=1,
    ):

        print(
            f"{rank}. "
            f"{row['Original Feature']} "
            f"= "
            f"{row['Mean Absolute SHAP']:.6f}"
        )


# ============================================================
# 24. FINAL INTERPRETATION MESSAGE
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "INTERPRETATION NOTES"
)

print(
    "=" * 75
)

print(
    """
1. Random Forest importance shows how strongly features
   contributed to splitting decisions across the trees.

2. SHAP shows how strongly each feature influenced
   individual model predictions.

3. Mean absolute SHAP measures importance magnitude.
   It does NOT automatically mean that a larger feature
   value always increases accident risk.

4. Class-wise SHAP helps identify which predictors
   are particularly important for Low, Medium,
   and High-risk classifications.

5. SHAP explains the fitted model. It does not prove
   causal relationships between a feature and accidents.

6. Because the original Accident Risk label methodology
   is undocumented, these explanations should be
   described as explanations of the model's predictions
   of the dataset's existing risk labels.
"""
)


# ============================================================
# 25. OUTPUT FILE LIST
# ============================================================

print(
    "=" * 75
)

print(
    "FILES SAVED"
)

print(
    "=" * 75
)

print(
    f"""
Output folder:

{OUTPUT_DIR}

Important files:

rf_feature_importance_original.csv
rf_feature_importance_original.png

shap_global_importance_original.csv
shap_global_importance_original.png

shap_classwise_importance_original.csv
shap_classwise_importance_original.png

shap_importance_Low.csv
shap_importance_Low.png

shap_importance_Medium.csv
shap_importance_Medium.png

shap_importance_High.csv
shap_importance_High.png

shap_summary_Low.png
shap_summary_Medium.png
shap_summary_High.png

shap_local_high_risk_example.csv
shap_local_high_risk_example.png

shap_values_original_features_per_observation.csv
"""
)

print(
    "\nAnalysis completed successfully."
)