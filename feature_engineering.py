from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import requests



# Original dataset variables are preserved.
#
# Engineered variables are clearly classified as:
#
# 1. Derived
# 2. Historical external weather
# 3. Estimated infrastructure
# 4. Estimated traffic-context variables
#
# ============================================================


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

SCRIPT_FOLDER = Path(__file__).resolve().parent


INPUT_FILE = (
    SCRIPT_FOLDER
    / "realistic_karachi_traffic_risk_dataset_final.xlsx"
)


OUTPUT_FILE = (
    SCRIPT_FOLDER
    / "karachi_traffic_feature_engineered_final_v2.xlsx"
)


# Historical weather cache.

WEATHER_CACHE_FILE = (
    SCRIPT_FOLDER
    / "karachi_historical_weather_cache_v2.csv"
)



VERIFIED_INFRASTRUCTURE_FILE = (
    SCRIPT_FOLDER
    / "karachi_road_infrastructure_verified.xlsx"
)


# The script automatically creates this checklist.

INFRASTRUCTURE_TEMPLATE_FILE = (
    SCRIPT_FOLDER
    / "karachi_infrastructure_to_verify.xlsx"
)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}\n\n"
        "Place the Excel dataset in the same folder "
        "as this Python script."
    )


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_excel(INPUT_FILE)

original_rows = len(df)
original_columns = len(df.columns)


print("\nDataset loaded successfully.")
print("Rows:", original_rows)
print("Columns:", original_columns)


# Preserve original row order.

df["_Original Row Order"] = np.arange(
    len(df)
)


# ============================================================
# 3. REQUIRED ORIGINAL COLUMNS
# ============================================================

required_columns = [
    "Date-Time",
    "Area",
    "Road Type",
    "Road Condition",
    "Vehicle Count",
    "Ride Activity",
    "Bus Activity",
    "Weather",
    "Accident Risk",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )


# ============================================================
# 4. REMOVE PREVIOUS ENGINEERED FEATURES
# ============================================================

old_features = [
    "Pickup Hour",
    "Day Name",
    "Month",
    "Is Weekend",
    "Hour Sin",
    "Hour Cos",
    "Rush Hour",
    "Time Period",
    "Is Hotspot",
    "Number of Lanes",
    "Near Intersection",
    "Traffic Signal",
    "Infrastructure Source",
    "Infrastructure Verified",
    "Construction",
    "Traffic Density",
    "Traffic Load per Lane",
    "Reference Traffic Load",
    "Traffic Load Ratio",
    "Traffic Load Percentile",
    "Congestion Level",
    "Rainfall",
    "Visibility",
    "Weather Code",
    "Historical Weather",
    "ML Weather",
    "Weather Data Source",
    "Is Rainy",
    "Is Foggy",
    "Average Speed",
    "Estimated Average Speed",
]


existing_old_features = [
    column
    for column in old_features
    if column in df.columns
]


if existing_old_features:

    df = df.drop(
        columns=existing_old_features
    )


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def normalize_text(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def stable_fraction(text, salt):
    """
    Reproducible number between 0 and 1.

    Used only for estimated infrastructure.
    """

    value = (
        f"{salt}|{text}"
        .encode("utf-8")
    )

    digest = hashlib.sha256(
        value
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            "big"
        )
        / 2**64
    )


# ============================================================
# 6. NUMERICAL VALIDATION
# ============================================================

numerical_original_columns = [
    "Vehicle Count",
    "Ride Activity",
    "Bus Activity",
]


for column in numerical_original_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


    if df[column].isna().any():

        raise ValueError(
            f"{column} contains missing "
            "or invalid numerical values."
        )


    if (df[column] < 0).any():

        raise ValueError(
            f"{column} contains negative values."
        )


# ============================================================
# 7. DATE-TIME VALIDATION
# ============================================================

df["Date-Time"] = pd.to_datetime(
    df["Date-Time"],
    errors="coerce"
)


if df["Date-Time"].isna().any():

    raise ValueError(
        "Some Date-Time values could not "
        "be converted."
    )


# ============================================================
# 8. NORMALIZED INTERNAL VARIABLES
# ============================================================

area_normalized = normalize_text(
    df["Area"]
)


road_type_normalized = normalize_text(
    df["Road Type"]
)


road_condition_normalized = normalize_text(
    df["Road Condition"]
)


# ============================================================
# 9. TEMPORAL FEATURES
# ============================================================

df["Pickup Hour"] = (
    df["Date-Time"]
    .dt.hour
    .astype(int)
)


df["Day Name"] = (
    df["Date-Time"]
    .dt.day_name()
)


df["Month"] = (
    df["Date-Time"]
    .dt.month
    .astype(int)
)


df["Is Weekend"] = (
    df["Date-Time"]
    .dt.dayofweek >= 5
).astype(int)


# ============================================================
# 10. CYCLIC HOUR FEATURES
# ============================================================

df["Hour Sin"] = np.round(
    np.sin(
        2
        * np.pi
        * df["Pickup Hour"]
        / 24
    ),
    6
)


df["Hour Cos"] = np.round(
    np.cos(
        2
        * np.pi
        * df["Pickup Hour"]
        / 24
    ),
    6
)


# ============================================================
# 11. RUSH HOUR
# ============================================================

def is_rush_hour(hour):

    if 7 <= hour <= 9:
        return 1

    if 17 <= hour <= 20:
        return 1

    return 0


df["Rush Hour"] = (
    df["Pickup Hour"]
    .apply(is_rush_hour)
    .astype(int)
)


# ============================================================
# 12. TIME PERIOD
# ============================================================

def get_time_period(hour):

    if 0 <= hour <= 4:
        return "Late Night"

    elif 5 <= hour <= 6:
        return "Early Morning"

    elif 7 <= hour <= 9:
        return "Morning Rush"

    elif 10 <= hour <= 15:
        return "Daytime"

    elif hour == 16:
        return "Evening Build-up"

    elif 17 <= hour <= 20:
        return "Evening Rush"

    elif 21 <= hour <= 23:
        return "Late Evening"

    return "Unknown"


df["Time Period"] = (
    df["Pickup Hour"]
    .apply(get_time_period)
)


# ============================================================
# 13. CONSTRUCTION
# ============================================================

df["Construction"] = (
    road_condition_normalized
    == "construction"
).astype(int)


# ============================================================
# 14. DEFAULT INFRASTRUCTURE ESTIMATES
# ============================================================
#


default_lane_estimates = {

    "highway": 4,

    "main road": 3,

    "industrial road": 2,

    "intersection": 3,

    "service road": 1,

    "narrow road": 1,
}


estimated_lanes = []
estimated_intersections = []
estimated_signals = []


for area, road_type in zip(
    area_normalized,
    road_type_normalized
):

    context = (
        f"{area}|{road_type}"
    )


    lanes = (
        default_lane_estimates
        .get(
            road_type,
            2
        )
    )


    # -----------------------------------------
    # NEAR INTERSECTION
    # -----------------------------------------

    if road_type == "intersection":

        near_intersection = 1


    elif road_type == "main road":

        near_intersection = int(
            stable_fraction(
                context,
                "intersection"
            )
            < 0.50
        )


    elif road_type == "industrial road":

        near_intersection = int(
            stable_fraction(
                context,
                "intersection"
            )
            < 0.25
        )


    elif road_type in [
        "service road",
        "narrow road",
    ]:

        near_intersection = int(
            stable_fraction(
                context,
                "intersection"
            )
            < 0.10
        )


    else:

        near_intersection = int(
            stable_fraction(
                context,
                "intersection"
            )
            < 0.15
        )


    # -----------------------------------------
    # TRAFFIC SIGNAL
    # -----------------------------------------

    if road_type == "intersection":

        signal_probability = 0.85


    elif (
        road_type == "main road"
        and near_intersection == 1
    ):

        signal_probability = 0.70


    elif near_intersection == 1:

        signal_probability = 0.40


    else:

        signal_probability = 0.05


    traffic_signal = int(
        stable_fraction(
            context,
            "signal"
        )
        < signal_probability
    )


    estimated_lanes.append(
        lanes
    )

    estimated_intersections.append(
        near_intersection
    )

    estimated_signals.append(
        traffic_signal
    )


df["Number of Lanes"] = (
    estimated_lanes
)


df["Near Intersection"] = (
    estimated_intersections
)


df["Traffic Signal"] = (
    estimated_signals
)


df["Infrastructure Source"] = (
    "Rule-based estimate"
)


df["Infrastructure Verified"] = 0


# ============================================================
# 15. CREATE INFRASTRUCTURE VERIFICATION TEMPLATE
# ============================================================
#


verification_template = (
    df[
        [
            "Area",
            "Road Type",
            "Number of Lanes",
            "Near Intersection",
            "Traffic Signal",
        ]
    ]
    .drop_duplicates(
        subset=[
            "Area",
            "Road Type",
        ]
    )
    .sort_values(
        [
            "Area",
            "Road Type",
        ]
    )
    .reset_index(
        drop=True
    )
)


verification_template = (
    verification_template.rename(
        columns={
            "Number of Lanes":
                "Estimated Number of Lanes",

            "Near Intersection":
                "Estimated Near Intersection",

            "Traffic Signal":
                "Estimated Traffic Signal",
        }
    )
)


verification_template[
    "Verified Number of Lanes"
] = np.nan


verification_template[
    "Verified Near Intersection"
] = np.nan


verification_template[
    "Verified Traffic Signal"
] = np.nan


verification_template[
    "Source"
] = ""


verification_template[
    "Notes"
] = ""


if not INFRASTRUCTURE_TEMPLATE_FILE.exists():

    verification_template.to_excel(
        INFRASTRUCTURE_TEMPLATE_FILE,
        index=False
    )

    print(
        "\nInfrastructure verification template created:"
    )

    print(
        INFRASTRUCTURE_TEMPLATE_FILE
    )


# ============================================================
# 16. APPLY VERIFIED INFRASTRUCTURE
# ============================================================
#


if VERIFIED_INFRASTRUCTURE_FILE.exists():

    verified = pd.read_excel(
        VERIFIED_INFRASTRUCTURE_FILE
    )


    required_verified_columns = [
        "Area",
        "Road Type",
        "Number of Lanes",
        "Near Intersection",
        "Traffic Signal",
        "Source",
    ]


    missing_verified = [
        column
        for column
        in required_verified_columns
        if column not in verified.columns
    ]


    if missing_verified:

        raise ValueError(
            "Verified infrastructure file "
            "is missing columns:\n"
            f"{missing_verified}"
        )


    verified["_Area Key"] = normalize_text(
        verified["Area"]
    )


    verified["_Road Key"] = normalize_text(
        verified["Road Type"]
    )


    verified_mapping = {}


    for _, row in verified.iterrows():

        key = (
            row["_Area Key"],
            row["_Road Key"],
        )


        verified_mapping[key] = {

            "lanes":
                int(
                    row["Number of Lanes"]
                ),

            "intersection":
                int(
                    row["Near Intersection"]
                ),

            "signal":
                int(
                    row["Traffic Signal"]
                ),

            "source":
                str(
                    row["Source"]
                ),
        }


    for index in df.index:

        key = (
            area_normalized.loc[index],
            road_type_normalized.loc[index],
        )


        if key in verified_mapping:

            values = (
                verified_mapping[key]
            )


            df.loc[
                index,
                "Number of Lanes"
            ] = values["lanes"]


            df.loc[
                index,
                "Near Intersection"
            ] = values["intersection"]


            df.loc[
                index,
                "Traffic Signal"
            ] = values["signal"]


            df.loc[
                index,
                "Infrastructure Source"
            ] = values["source"]


            df.loc[
                index,
                "Infrastructure Verified"
            ] = 1


    print(
        "\nVerified infrastructure values applied."
    )


else:

    print(
        "\nNo verified infrastructure file found."
    )

    print(
        "Rule-based estimates will remain "
        "clearly labelled as estimates."
    )


# Force integer types.

for column in [
    "Number of Lanes",
    "Near Intersection",
    "Traffic Signal",
    "Infrastructure Verified",
]:

    df[column] = (
        df[column]
        .astype(int)
    )


# ============================================================
# 17. TRAFFIC LOAD PER LANE
# ============================================================
#


df["Traffic Load per Lane"] = np.round(
    df["Vehicle Count"]
    /
    df["Number of Lanes"],
    2
)


# ============================================================
# 18. REFERENCE TRAFFIC LOAD
# ============================================================
#


context_columns = [
    "Area",
    "Road Type",
]


context_size = (
    df.groupby(
        context_columns
    )[
        "Traffic Load per Lane"
    ]
    .transform("size")
)


context_reference = (
    df.groupby(
        context_columns
    )[
        "Traffic Load per Lane"
    ]
    .transform(
        lambda values:
        values.quantile(0.95)
    )
)


road_type_reference = (
    df.groupby(
        "Road Type"
    )[
        "Traffic Load per Lane"
    ]
    .transform(
        lambda values:
        values.quantile(0.95)
    )
)


MIN_CONTEXT_ROWS = 10


df["Reference Traffic Load"] = np.where(

    context_size >= MIN_CONTEXT_ROWS,

    context_reference,

    road_type_reference
)


if (
    df["Reference Traffic Load"]
    .isna()
    .any()
):

    raise ValueError(
        "Reference Traffic Load contains "
        "missing values."
    )


# ============================================================
# 19. TRAFFIC LOAD RATIO
# ============================================================

df["Traffic Load Ratio"] = np.round(

    df["Traffic Load per Lane"]
    /
    df["Reference Traffic Load"],

    3
)


# ============================================================
# 20. IMPROVED CONGESTION LEVEL
# ============================================================
#
# New thresholds:
#
# <= 0.35        Low
# >0.35 - 0.75   Moderate
# >0.75 - 0.95   High
# >0.95          Severe
#


def classify_congestion(
    load_ratio
):

    if load_ratio <= 0.35:

        return "Low"


    elif load_ratio <= 0.75:

        return "Moderate"


    elif load_ratio <= 0.95:

        return "High"


    else:

        return "Severe"


df["Congestion Level"] = (
    df["Traffic Load Ratio"]
    .apply(
        classify_congestion
    )
)


congestion_order = [
    "Low",
    "Moderate",
    "High",
    "Severe",
]


df["Congestion Level"] = (
    pd.Categorical(
        df["Congestion Level"],
        categories=congestion_order,
        ordered=True
    )
)


# ============================================================
# 21. HISTORICAL WEATHER CONFIGURATION
# ============================================================
#
# Historical Forecast API.
#


KARACHI_LATITUDE = 24.8607

KARACHI_LONGITUDE = 67.0011


WEATHER_API_URL = (
    "https://historical-forecast-api.open-meteo.com/"
    "v1/forecast"
)


# ============================================================
# 22. DOWNLOAD HISTORICAL WEATHER
# ============================================================

def fetch_historical_weather(
    start_date,
    end_date
):

    params = {

        "latitude":
            KARACHI_LATITUDE,

        "longitude":
            KARACHI_LONGITUDE,

        "start_date":
            start_date,

        "end_date":
            end_date,

        "hourly":
            "rain,visibility,weather_code",

        "timezone":
            "Asia/Karachi",
    }


    print(
        "\nDownloading historical Karachi weather..."
    )


    response = requests.get(
        WEATHER_API_URL,
        params=params,
        timeout=120
    )


    response.raise_for_status()


    result = response.json()


    if "hourly" not in result:

        raise ValueError(
            "Historical weather API returned "
            "no hourly weather data."
        )


    hourly = result["hourly"]


    weather_df = pd.DataFrame({

        "Weather Match Time":
            pd.to_datetime(
                hourly["time"]
            ),

        "Rainfall":
            pd.to_numeric(
                hourly["rain"],
                errors="coerce"
            ),

        # Visibility returned in metres.
        # Convert metres -> km.

        "Visibility":
            pd.to_numeric(
                hourly["visibility"],
                errors="coerce"
            )
            / 1000.0,

        "Weather Code":
            pd.to_numeric(
                hourly["weather_code"],
                errors="coerce"
            ),
    })


    # Handle occasional missing model values.

    weather_df[
        [
            "Rainfall",
            "Visibility",
        ]
    ] = (
        weather_df[
            [
                "Rainfall",
                "Visibility",
            ]
        ]
        .interpolate(
            limit_direction="both"
        )
    )


    weather_df["Weather Code"] = (
        weather_df["Weather Code"]
        .ffill()
        .bfill()
    )


    if weather_df.isna().any().any():

        raise ValueError(
            "Historical weather still contains "
            "missing values."
        )


    weather_df["Rainfall"] = (
        weather_df["Rainfall"]
        .clip(lower=0)
        .round(2)
    )


    weather_df["Visibility"] = (
        weather_df["Visibility"]
        .clip(lower=0)
        .round(2)
    )


    return weather_df


# ============================================================
# 23. LOAD WEATHER CACHE OR DOWNLOAD
# ============================================================

weather_start_date = (
    df["Date-Time"]
    .min()
    .date()
    .isoformat()
)


weather_end_date = (
    df["Date-Time"]
    .max()
    .date()
    .isoformat()
)


weather_df = None


if WEATHER_CACHE_FILE.exists():

    try:

        cached = pd.read_csv(
            WEATHER_CACHE_FILE
        )


        required_cache_columns = [
            "Weather Match Time",
            "Rainfall",
            "Visibility",
            "Weather Code",
        ]


        if all(
            column in cached.columns
            for column
            in required_cache_columns
        ):

            cached[
                "Weather Match Time"
            ] = pd.to_datetime(
                cached[
                    "Weather Match Time"
                ]
            )


            required_start = (
                df["Date-Time"]
                .min()
                .date()
            )


            required_end = (
                df["Date-Time"]
                .max()
                .date()
            )


            cached_start = (
                cached[
                    "Weather Match Time"
                ]
                .min()
                .date()
            )


            cached_end = (
                cached[
                    "Weather Match Time"
                ]
                .max()
                .date()
            )


            if (
                cached_start <= required_start
                and
                cached_end >= required_end
            ):

                weather_df = cached

                print(
                    "\nUsing historical weather cache."
                )


    except Exception as error:

        print(
            "\nExisting weather cache could "
            "not be used:"
        )

        print(error)


if weather_df is None:

    weather_df = (
        fetch_historical_weather(
            weather_start_date,
            weather_end_date
        )
    )


    weather_df.to_csv(
        WEATHER_CACHE_FILE,
        index=False
    )


    print(
        "\nHistorical weather saved:"
    )

    print(
        WEATHER_CACHE_FILE
    )


# ============================================================
# 24. HISTORICAL / ML WEATHER CATEGORY
# ============================================================
#

RAIN_CODES = {
    51, 53, 55,
    56, 57,
    61, 63, 65,
    66, 67,
    80, 81, 82,
    95, 96, 99,
}


FOG_CODES = {
    45,
    48,
}


def historical_weather_category(
    row
):

    code = int(
        row["Weather Code"]
    )


    if code in FOG_CODES:

        return "Foggy"


    if (
        code in RAIN_CODES
        or row["Rainfall"] > 0
    ):

        return "Rainy"


    return "Dry"


weather_df[
    "Historical Weather"
] = weather_df.apply(
    historical_weather_category,
    axis=1
)


# ============================================================
# 25. MERGE HISTORICAL WEATHER
# ============================================================

df["Weather Match Time"] = (
    df["Date-Time"]
    .dt.floor("h")
)


df = df.merge(

    weather_df[
        [
            "Weather Match Time",
            "Rainfall",
            "Visibility",
            "Weather Code",
            "Historical Weather",
        ]
    ],

    on="Weather Match Time",

    how="left",

    validate="many_to_one",

    sort=False
)


# Restore original traffic-row order.

df = (
    df.sort_values(
        "_Original Row Order"
    )
    .reset_index(
        drop=True
    )
)


if (
    df[
        [
            "Rainfall",
            "Visibility",
            "Historical Weather",
        ]
    ]
    .isna()
    .any()
    .any()
):

    raise ValueError(
        "Some traffic records could not "
        "be matched with historical weather."
    )


df["Weather Data Source"] = (
    "Open-Meteo Historical Forecast"
)


# ============================================================
# 26. ML WEATHER
# ============================================================
#


df["ML Weather"] = (
    df["Historical Weather"]
)


ml_weather_normalized = (
    normalize_text(
        df["ML Weather"]
    )
)


df["Is Rainy"] = (
    ml_weather_normalized
    == "rainy"
).astype(int)


df["Is Foggy"] = (
    ml_weather_normalized
    == "foggy"
).astype(int)


# ============================================================
# 27. ESTIMATED AVERAGE SPEED
# ============================================================
#


free_flow_speed = (
    road_type_normalized
    .map({
        "highway": 75.0,
        "main road": 50.0,
        "industrial road": 40.0,
        "intersection": 30.0,
        "service road": 28.0,
        "narrow road": 22.0,
    })
    .fillna(40.0)
)


speed_load_ratio = np.minimum(
    df["Traffic Load Ratio"],
    1.20
)


traffic_speed_factor = (
    1.0
    -
    (
        0.50
        * np.minimum(
            speed_load_ratio,
            1.0
        )
    )
)


# Additional small penalty if demand
# exceeds reference load.

traffic_speed_factor -= (
    np.maximum(
        speed_load_ratio - 1.0,
        0
    )
    * 0.15
)


traffic_speed_factor = np.clip(
    traffic_speed_factor,
    0.40,
    1.00
)


# Weather factors.

weather_speed_factor = np.ones(
    len(df)
)


weather_speed_factor = np.where(

    df["Is Rainy"] == 1,

    weather_speed_factor * 0.90,

    weather_speed_factor
)


weather_speed_factor = np.where(

    df["Is Foggy"] == 1,

    weather_speed_factor * 0.82,

    weather_speed_factor
)


# Construction.

construction_factor = np.where(

    df["Construction"] == 1,

    0.80,

    1.00
)


# Intersection.

intersection_factor = np.where(

    df["Near Intersection"] == 1,

    0.92,

    1.00
)


estimated_speed = (

    free_flow_speed

    * traffic_speed_factor

    * weather_speed_factor

    * construction_factor

    * intersection_factor
)


df[
    "Estimated Average Speed"
] = np.round(

    np.clip(
        estimated_speed,
        8.0,
        85.0
    ),

    1
)


# ============================================================
# 28. CLEAN INTERNAL COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "Weather Match Time",
        "_Original Row Order",
    ]
)


# ============================================================
# 29. LOGICAL VALIDATION
# ============================================================

binary_columns = [
    "Is Weekend",
    "Rush Hour",
    "Near Intersection",
    "Traffic Signal",
    "Infrastructure Verified",
    "Construction",
    "Is Rainy",
    "Is Foggy",
]


for column in binary_columns:

    assert (
        df[column]
        .isin([0, 1])
        .all()
    ), (
        f"{column} contains "
        "invalid binary values."
    )


assert (
    df["Pickup Hour"]
    .between(0, 23)
    .all()
)


assert (
    df["Number of Lanes"]
    .between(1, 6)
    .all()
)


assert (
    df["Traffic Load per Lane"]
    >= 0
).all()


assert (
    df["Traffic Load Ratio"]
    >= 0
).all()


assert (
    df["Rainfall"]
    >= 0
).all()


assert (
    df["Visibility"]
    >= 0
).all()


assert (
    df["Estimated Average Speed"]
    .between(8, 85)
    .all()
)


# ============================================================
# 30. VALIDATION OUTPUT
# ============================================================

print(
    "\n========================================"
)

print(
    "DATA QUALITY"
)

print(
    "========================================"
)


print(
    "\nDuplicate rows:"
)

print(
    df.duplicated().sum()
)


print(
    "\nMissing values:"
)

print(
    df.isna().sum()
)


# ============================================================
# 31. CONGESTION OUTPUT
# ============================================================

print(
    "\n========================================"
)

print(
    "CONGESTION LEVEL COUNTS"
)

print(
    "========================================"
)


print(
    df["Congestion Level"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 32. INFRASTRUCTURE OUTPUT
# ============================================================

print(
    "\n========================================"
)

print(
    "INFRASTRUCTURE STATUS"
)

print(
    "========================================"
)


print(
    df[
        [
            "Infrastructure Source",
            "Infrastructure Verified",
        ]
    ]
    .value_counts()
)


# ============================================================
# 33. HISTORICAL WEATHER OUTPUT
# ============================================================

print(
    "\n========================================"
)

print(
    "HISTORICAL RAINFALL"
)

print(
    "========================================"
)


print(
    df["Rainfall"]
    .describe()
    .round(2)
)


print(
    "\n========================================"
)

print(
    "HISTORICAL VISIBILITY"
)

print(
    "========================================"
)


print(
    df["Visibility"]
    .describe()
    .round(2)
)


# ============================================================
# 34. ORIGINAL VS ML WEATHER
# ============================================================

weather_validation = (
    df.groupby(
        [
            "Weather",
            "ML Weather",
        ],
        dropna=False
    )
    .size()
    .reset_index(
        name="Rows"
    )
    .sort_values(
        "Rows",
        ascending=False
    )
)


print(
    "\n========================================"
)

print(
    "ORIGINAL WEATHER VS ML WEATHER"
)

print(
    "========================================"
)


print(
    weather_validation.to_string(
        index=False
    )
)


# ============================================================
# 35. ESTIMATED SPEED OUTPUT
# ============================================================

print(
    "\n========================================"
)

print(
    "ESTIMATED AVERAGE SPEED"
)

print(
    "========================================"
)


print(
    df["Estimated Average Speed"]
    .describe()
    .round(2)
)


print(
    "\nAverage Speed by Road Type:"
)


print(
    df.groupby(
        "Road Type"
    )[
        "Estimated Average Speed"
    ]
    .mean()
    .round(2)
)


print(
    "\nAverage Speed by Congestion Level:"
)


print(
    df.groupby(
        "Congestion Level",
        observed=False
    )[
        "Estimated Average Speed"
    ]
    .mean()
    .round(2)
)


# ============================================================
# 36. HOURLY VALIDATION
# ============================================================

hourly_summary = (
    df.groupby(
        "Pickup Hour"
    )[
        [
            "Vehicle Count",
            "Ride Activity",
            "Bus Activity",
            "Traffic Load per Lane",
            "Traffic Load Ratio",
            "Rainfall",
            "Visibility",
            "Estimated Average Speed",
        ]
    ]
    .mean()
    .round(2)
)


print(
    "\n========================================"
)

print(
    "AVERAGE VALUES BY HOUR"
)

print(
    "========================================"
)


print(
    hourly_summary
)


# ============================================================
# 37. NIGHT-TIME VALIDATION
# ============================================================

night_rows = (
    df[
        df["Pickup Hour"]
        .between(0, 4)
    ]
)


print(
    "\n========================================"
)

print(
    "00:00-04:59 TRAFFIC CHECK"
)

print(
    "========================================"
)


print(
    night_rows[
        [
            "Vehicle Count",
            "Ride Activity",
            "Bus Activity",
            "Traffic Load per Lane",
            "Traffic Load Ratio",
            "Estimated Average Speed",
        ]
    ]
    .describe()
    .round(2)
)


# ============================================================
# 38. FEATURE METADATA
# ============================================================

feature_metadata = pd.DataFrame(

    [
        [
            "Pickup Hour",
            "Derived",
            "Hour extracted from Date-Time",
            "Date-Time",
            "Yes",
        ],

        [
            "Day Name",
            "Derived",
            "Day extracted from Date-Time",
            "Date-Time",
            "Yes",
        ],

        [
            "Month",
            "Derived",
            "Month extracted from Date-Time",
            "Date-Time",
            "Yes",
        ],

        [
            "Is Weekend",
            "Derived",
            "Weekend indicator",
            "Date-Time",
            "Yes",
        ],

        [
            "Hour Sin",
            "Derived",
            "Cyclical time feature",
            "Pickup Hour",
            "Yes",
        ],

        [
            "Hour Cos",
            "Derived",
            "Cyclical time feature",
            "Pickup Hour",
            "Yes",
        ],

        [
            "Rush Hour",
            "Derived / assumption",
            "07-09 and 17-20 peak periods",
            "Pickup Hour",
            "Yes",
        ],

        [
            "Number of Lanes",
            "Estimated / verified",
            "Road lane estimate or verified value",
            "Road context",
            "Prefer verified",
        ],

        [
            "Near Intersection",
            "Estimated / verified",
            "Intersection flag",
            "Road context",
            "Prefer verified",
        ],

        [
            "Traffic Signal",
            "Estimated / verified",
            "Traffic signal flag",
            "Road context",
            "Prefer verified",
        ],

        [
            "Traffic Load per Lane",
            "Derived proxy",
            "Vehicle Count / Number of Lanes",
            "Vehicle Count + lanes",
            "Yes",
        ],

        [
            "Reference Traffic Load",
            "Derived reference",
            "95th percentile contextual load",
            "Full dataset for EDA",
            "RECOMPUTE after train/test split",
        ],

        [
            "Traffic Load Ratio",
            "Derived proxy",
            "Current load / reference load",
            "Traffic load",
            "RECOMPUTE after train/test split",
        ],

        [
            "Congestion Level",
            "Derived proxy",
            "Congestion category from Load Ratio",
            "Traffic Load Ratio",
            "RECOMPUTE after train/test split",
        ],

        [
            "Rainfall",
            "Historical external data",
            "Hourly historical modelled rain (mm)",
            "Open-Meteo",
            "Yes",
        ],

        [
            "Visibility",
            "Historical external data",
            "Hourly historical visibility (km)",
            "Open-Meteo",
            "Yes",
        ],

        [
            "Weather",
            "Original dataset",
            "Original weather label",
            "Original dataset",
            "NO - audit only",
        ],

        [
            "ML Weather",
            "Historical external data",
            "Historical Dry/Rainy/Foggy category",
            "Open-Meteo",
            "YES - replace original Weather",
        ],

        [
            "Estimated Average Speed",
            "Estimated",
            "Traffic/load/context-based speed estimate",
            "Road + Traffic + Weather",
            "RECOMPUTE after train/test split",
        ],
    ],

    columns=[
        "Feature",
        "Feature Type",
        "Definition",
        "Source / Method",
        "Final ML Guidance",
    ]
)


# ============================================================
# 39. MODELING GUIDANCE
# ============================================================

modeling_guidance = pd.DataFrame(

    [
        [
            "Weather",
            "Exclude",
            "Original weather does not agree well with historical weather.",
        ],

        [
            "ML Weather",
            "Use",
            "Historical external weather category.",
        ],

        [
            "Rainfall",
            "Use",
            "Historical rain variable.",
        ],

        [
            "Visibility",
            "Use/Test",
            "Keep initially; feature analysis may show limited value.",
        ],

        [
            "Traffic Load per Lane",
            "Use",
            "Directly derived traffic-load proxy.",
        ],

        [
            "Reference Traffic Load",
            "Recompute",
            "Fit only on training data.",
        ],

        [
            "Traffic Load Ratio",
            "Recompute",
            "Use training reference values.",
        ],

        [
            "Congestion Level",
            "Recompute",
            "Calculate after train/test split.",
        ],

        [
            "Estimated Average Speed",
            "Recompute",
            "Depends on Traffic Load Ratio.",
        ],

        [
            "Infrastructure Source",
            "Do not use",
            "Metadata only.",
        ],

        [
            "Infrastructure Verified",
            "Do not use as predictor",
            "Data-quality metadata.",
        ],
    ],

    columns=[
        "Feature",
        "ML Action",
        "Reason",
    ]
)


# ============================================================
# 40. SAVE FINAL WORKBOOK
# ============================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:


    df.to_excel(
        writer,
        sheet_name="Data",
        index=False
    )


    feature_metadata.to_excel(
        writer,
        sheet_name="Feature_Metadata",
        index=False
    )


    modeling_guidance.to_excel(
        writer,
        sheet_name="ML_Guidance",
        index=False
    )


    hourly_summary.to_excel(
        writer,
        sheet_name="Hourly_Validation"
    )


    weather_validation.to_excel(
        writer,
        sheet_name="Weather_Validation",
        index=False
    )


# ============================================================
# 41. FINAL SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "FINAL FEATURE ENGINEERING COMPLETED"
)

print(
    "========================================"
)


print(
    "\nOriginal rows:",
    original_rows
)


print(
    "Final rows:",
    len(df)
)


print(
    "Original columns:",
    original_columns
)


print(
    "Final columns:",
    len(df.columns)
)


print(
    "\nOutput file:"
)

print(
    OUTPUT_FILE
)


print(
    "\nIMPORTANT:"
)


print(
    "\n1. Traffic Load per Lane replaces "
    "the misleading Traffic Density name."
)


print(
    "\n2. Congestion Level uses revised "
    "0.35 / 0.75 / 0.95 thresholds."
)


print(
    "\n3. Original Weather is retained for audit "
    "but should NOT be used in final ML."
)


print(
    "\n4. ML Weather, Rainfall and Visibility "
    "come from historical Open-Meteo weather."
)


print(
    "\n5. Infrastructure remains estimated until "
    "verified external values are supplied."
)


print(
    "\n6. Estimated Average Speed is NOT measured speed."
)


print(
    "\n7. Reference Traffic Load, Traffic Load Ratio, "
    "Congestion Level and Estimated Average Speed "
    "must be recomputed from TRAINING DATA "
    "during final ML."
)


print(
    "\n8. Accident Risk was never used to "
    "construct predictor features."
)