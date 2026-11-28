from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import osmnx as ox


# ============================================================
# KARACHI OSM ROAD / INFRASTRUCTURE EXTRACTION
# RESUMABLE + MULTI-SERVER RETRY VERSION
# ============================================================
#
# PURPOSE
# -------
# - Read the Karachi thesis dataset.
# - Download candidate real roads from OpenStreetMap.
# - Extract road name, OSM class, lanes, speed, intersections,
#   signals, length, OSM IDs, etc.
# - Preserve areas that were already successfully downloaded.
# - Retry ONLY failed/missing areas.
# - Automatically switch Overpass servers on timeout/failure.
#
# IMPORTANT
# ---------
# This script DOES NOT modify the 1,500-row thesis dataset.
# It only creates/updates the OSM candidate-road workbook.
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
    / "karachi_osm_road_candidates.xlsx"
)

# Used only if the main workbook is open/locked in Excel.
FALLBACK_OUTPUT_FILE = (
    SCRIPT_FOLDER
    / "karachi_osm_road_candidates_recovered.xlsx"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}\n"
    )


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEARCH_RADIUS_METRES = 2500
USE_HISTORICAL_SNAPSHOT = True

# If True, read OUTPUT_FILE (when it exists), preserve successful
# areas, and retry only failed/missing areas.
RESUME_EXISTING_RESULTS = True

# Retry each area across these public global Overpass instances.
# OSMnx expects the base API URL and appends the interpreter path.
OVERPASS_SERVERS = [
    # Keep this first because your previous successful PECHS/Saddar/
    # Sohrab Goth responses were downloaded from this server and may
    # already exist in the local OSMnx cache.
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass-api.de/api",
    "https://overpass.private.coffee/api",
]

# HTTP wait time. A slow/unresponsive public server will be abandoned
# sooner instead of holding the script for many minutes.
REQUEST_TIMEOUT_SECONDS = 90

# IMPORTANT: keep the Overpass QUERY timeout at 300 so the query text
# matches your previous cached historical requests. This lets OSMnx
# reuse those cached responses if they are still on your computer.
OVERPASS_QUERY_TIMEOUT_SECONDS = 300

SECONDS_BETWEEN_RETRIES = 2

# Rewriting a large Excel workbook after every area is slow.
# Save once at the end instead.
SAVE_AFTER_EACH_SUCCESS = False

# Raw edge-level OSM data is huge and is not needed for selecting
# representative road candidates. Keeping it out of the workbook
# makes the script and Excel output much faster/smaller.
SAVE_RAW_EDGES = False


# ============================================================
# 3. LOAD ORIGINAL THESIS DATASET
# ============================================================

df = pd.read_excel(INPUT_FILE)

required_columns = [
    "Date-Time",
    "Area",
    "Road Type",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["Date-Time"] = pd.to_datetime(
    df["Date-Time"],
    errors="coerce"
)

if df["Date-Time"].isna().any():
    raise ValueError(
        "Some Date-Time values could not be converted."
    )


# ============================================================
# 4. STUDY PERIOD / HISTORICAL SNAPSHOT
# ============================================================

study_start = df["Date-Time"].min()
study_end = df["Date-Time"].max()

print("\nStudy period:")
print(study_start, "to", study_end)

snapshot_date = study_end.date().isoformat()
snapshot_timestamp = f"{snapshot_date}T23:59:59Z"

print("\nOSM snapshot date:")
print(snapshot_timestamp)


# ============================================================
# 5. OSMNX SETTINGS
# ============================================================

ox.settings.use_cache = True

# Reuse the cache folder from the directory where you launch the script
# (this is where your earlier OSMnx run saved cache\\*.json files).
# If no cache exists there, fall back to a cache beside this script.
CURRENT_CACHE = Path.cwd() / "cache"
SCRIPT_CACHE = SCRIPT_FOLDER / "cache"

if CURRENT_CACHE.exists():
    ox.settings.cache_folder = CURRENT_CACHE
else:
    ox.settings.cache_folder = SCRIPT_CACHE

# Reduce thousands of OSMnx log lines in the terminal.
ox.settings.log_console = False

# HTTP request timeout only. The Overpass query timeout is set separately
# below so cached 2024 queries can still match your previous requests.
ox.settings.requests_timeout = REQUEST_TIMEOUT_SECONDS

print("\nOSMnx cache folder:")
print(ox.settings.cache_folder)

required_way_tags = [
    "access",
    "area",
    "bridge",
    "est_width",
    "highway",
    "junction",
    "landuse",
    "lanes",
    "maxspeed",
    "name",
    "oneway",
    "ref",
    "service",
    "surface",
    "tunnel",
    "width",
]

for tag in required_way_tags:
    if tag not in ox.settings.useful_tags_way:
        ox.settings.useful_tags_way.append(tag)

required_node_tags = [
    "highway",
    "junction",
    "railway",
    "ref",
]

for tag in required_node_tags:
    if tag not in ox.settings.useful_tags_node:
        ox.settings.useful_tags_node.append(tag)

if USE_HISTORICAL_SNAPSHOT:
    ox.settings.overpass_settings = (
        '[out:json]'
        f'[timeout:{OVERPASS_QUERY_TIMEOUT_SECONDS}]'
        f'[date:"{snapshot_timestamp}"]'
        '{maxsize}'
    )
    print("\nHistorical OSM mode enabled.")
else:
    print("\nCurrent OSM mode enabled.")


# ============================================================
# 6. AREA NAMES
# ============================================================

areas = sorted(
    df["Area"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

print("\nAreas found in dataset:")
for area in areas:
    print(" -", area)


# ============================================================
# 7. GEOCODING ALIASES
# ============================================================
#
# These are only for finding the general area centre.
# They do NOT assign a particular road to thesis rows.
# ============================================================

AREA_SEARCH_QUERIES = {
    "DHA":
        "Defence Housing Authority, Karachi, Pakistan",

    "Clifton":
        "Clifton, Karachi, Pakistan",

    "Gulshan":
        "Gulshan-e-Iqbal, Karachi, Pakistan",

    "Karimabad":
        "Karimabad, Karachi, Pakistan",

    "Korangi":
        "Korangi, Karachi, Pakistan",

    "Lyari":
        "Lyari, Karachi, Pakistan",

    "Malir":
        "Malir, Karachi, Pakistan",

    "NIPA Road":
        "NIPA Chowrangi, Karachi, Pakistan",

    "Nazimabad":
        "Nazimabad, Karachi, Pakistan",

    "North Nazimabad":
        "North Nazimabad, Karachi, Pakistan",

    "PECHS":
        "PECHS, Karachi, Pakistan",

    "Saddar":
        "Saddar, Karachi, Pakistan",

    "Sohrab Goth":
        "Sohrab Goth, Karachi, Pakistan",
}


# Exact centres already returned successfully by Nominatim in your
# previous run. Using them avoids another geocoding request and helps
# reproduce the same cached Overpass query geometry.
KNOWN_AREA_CENTRES = {
    "PECHS": (24.8684802, 67.0587242),
    "Saddar": (24.8605712, 67.0317406),
    "Sohrab Goth": (24.9505443, 67.0897399),
}


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================

def clean_osm_value(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple, set)):
        cleaned = [
            str(item)
            for item in value
            if item is not None
        ]
        return " | ".join(sorted(set(cleaned)))

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value)


def extract_osm_ids(value):
    if isinstance(value, (list, tuple, set)):
        return " | ".join(str(item) for item in value)

    if value is None:
        return ""

    return str(value)


def parse_lane_value(value):
    """
    Parse only simple numeric lane tags.
    Complex/ambiguous tags stay missing rather than being guessed.
    """
    if value is None:
        return np.nan

    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = [value]

    values = []

    for raw in raw_values:
        text = str(raw).strip()

        if re.fullmatch(r"\d+(\.\d+)?", text):
            values.append(float(text))

    if not values:
        return np.nan

    return float(np.median(values))


def contains_traffic_signal(value):
    return "traffic_signals" in clean_osm_value(value).lower()


def suggested_dataset_category(highway_value):
    """
    Broad helper mapping only.
    It is NOT the final thesis Road Type assignment.
    """
    text = str(highway_value).lower()

    if "motorway" in text or "trunk" in text:
        return "Highway"

    if any(
        road_class in text
        for road_class in [
            "primary",
            "secondary",
            "tertiary",
        ]
    ):
        return "Main Road"

    if "service" in text:
        return "Service Road"

    if any(
        road_class in text
        for road_class in [
            "residential",
            "unclassified",
            "living_street",
        ]
    ):
        return "Smaller Road - review manually"

    return "Needs Review"


def mode_numeric(series):
    cleaned = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
    )

    if cleaned.empty:
        return np.nan

    modes = cleaned.mode()

    if not modes.empty:
        return float(modes.iloc[0])

    return float(cleaned.median())


def unique_join(series):
    values = []

    for value in series:
        text = clean_osm_value(value)
        if text:
            values.append(text)

    return " | ".join(
        sorted(set(values))
    )


def safe_read_sheet(path, sheet_name):
    """
    Read a sheet from an existing workbook.
    Return an empty DataFrame if the sheet does not exist.
    """
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_excel(
            path,
            sheet_name=sheet_name
        )
    except Exception:
        return pd.DataFrame()


def download_graph_with_retry(
    centre_latitude,
    centre_longitude,
    area
):
    """
    Try the same road-network request on several Overpass servers.
    Return (graph, server_used).
    """
    last_error = None

    for attempt, server in enumerate(
        OVERPASS_SERVERS,
        start=1
    ):
        ox.settings.overpass_url = server

        print(
            f"\n{area}: Overpass attempt "
            f"{attempt}/{len(OVERPASS_SERVERS)}"
        )
        print("Server:", server)

        try:
            graph = ox.graph.graph_from_point(
                (
                    centre_latitude,
                    centre_longitude,
                ),
                dist=SEARCH_RADIUS_METRES,
                network_type="drive_service",
                simplify=True,
                retain_all=True,
                truncate_by_edge=True,
            )

            print(
                f"{area}: download succeeded "
                f"using {server}"
            )

            return graph, server

        except Exception as error:
            last_error = error

            print(
                f"{area}: server failed:"
            )
            print(error)

            if attempt < len(OVERPASS_SERVERS):
                print(
                    f"Trying next server in "
                    f"{SECONDS_BETWEEN_RETRIES} seconds..."
                )
                time.sleep(SECONDS_BETWEEN_RETRIES)

    raise RuntimeError(
        f"All Overpass servers failed for {area}. "
        f"Last error: {last_error}"
    )


# ============================================================
# 9. DATASET ROAD-TYPE REQUIREMENTS
# ============================================================

dataset_contexts = (
    df.groupby(
        [
            "Area",
            "Road Type",
        ]
    )
    .size()
    .reset_index(
        name="Dataset Row Count"
    )
    .sort_values(
        [
            "Area",
            "Road Type",
        ]
    )
)


# ============================================================
# 10. LOAD EXISTING RESULTS TO RESUME
# ============================================================

existing_candidates = pd.DataFrame()
existing_raw_edges = pd.DataFrame()
existing_coverage = pd.DataFrame()
existing_errors = pd.DataFrame()

if (
    RESUME_EXISTING_RESULTS
    and OUTPUT_FILE.exists()
):
    print(
        "\nExisting OSM workbook found. "
        "Successful areas will be preserved."
    )

    existing_candidates = safe_read_sheet(
        OUTPUT_FILE,
        "Road_Candidates"
    )

    existing_raw_edges = safe_read_sheet(
        OUTPUT_FILE,
        "Raw_OSM_Edges"
    )

    existing_coverage = safe_read_sheet(
        OUTPUT_FILE,
        "Area_Coverage"
    )

    existing_errors = safe_read_sheet(
        OUTPUT_FILE,
        "Errors"
    )


if (
    not existing_coverage.empty
    and "Status" in existing_coverage.columns
    and "Area" in existing_coverage.columns
):
    successful_areas = set(
        existing_coverage.loc[
            existing_coverage["Status"].eq("Success"),
            "Area",
        ]
        .astype(str)
        .tolist()
    )
else:
    successful_areas = set()


areas_to_process = [
    area
    for area in areas
    if area not in successful_areas
]

print("\nAlready successful areas:")
if successful_areas:
    for area in sorted(successful_areas):
        print(" -", area)
else:
    print(" None")

print("\nAreas to process/retry:")
if areas_to_process:
    for area in areas_to_process:
        print(" -", area)
else:
    print(" None")


# ============================================================
# 11. STORAGE FOR NEW/RETRIED AREAS
# ============================================================

new_candidate_rows = []
new_raw_edges = []
new_coverage_rows = []
new_error_rows = []


# ============================================================
# 12. BUILD FINAL WORKBOOK FUNCTION
# ============================================================

def build_combined_frames():
    new_candidates_df = pd.DataFrame(
        new_candidate_rows
    )

    if new_raw_edges:
        new_raw_edges_df = pd.concat(
            new_raw_edges,
            ignore_index=True
        )
    else:
        new_raw_edges_df = pd.DataFrame()

    new_coverage_df = pd.DataFrame(
        new_coverage_rows
    )

    new_errors_df = pd.DataFrame(
        new_error_rows
    )

    # Areas retried in this run replace their previous failed/error rows.
    retried_areas = set(areas_to_process)

    base_candidates = existing_candidates.copy()
    base_raw_edges = existing_raw_edges.copy()
    base_coverage = existing_coverage.copy()
    base_errors = existing_errors.copy()

    if (
        not base_candidates.empty
        and "Area" in base_candidates.columns
    ):
        base_candidates = base_candidates[
            ~base_candidates["Area"].isin(
                retried_areas
            )
        ]

    if (
        not base_raw_edges.empty
        and "Area" in base_raw_edges.columns
    ):
        base_raw_edges = base_raw_edges[
            ~base_raw_edges["Area"].isin(
                retried_areas
            )
        ]

    if (
        not base_coverage.empty
        and "Area" in base_coverage.columns
    ):
        base_coverage = base_coverage[
            ~base_coverage["Area"].isin(
                retried_areas
            )
        ]

    if (
        not base_errors.empty
        and "Area" in base_errors.columns
    ):
        base_errors = base_errors[
            ~base_errors["Area"].isin(
                retried_areas
            )
        ]

    candidate_parts = [
        part
        for part in [
            base_candidates,
            new_candidates_df,
        ]
        if not part.empty
    ]

    if candidate_parts:
        combined_candidates = pd.concat(
            candidate_parts,
            ignore_index=True
        )
    else:
        combined_candidates = pd.DataFrame()

    raw_parts = [
        part
        for part in [
            base_raw_edges,
            new_raw_edges_df,
        ]
        if not part.empty
    ]

    if raw_parts:
        combined_raw_edges = pd.concat(
            raw_parts,
            ignore_index=True
        )
    else:
        combined_raw_edges = pd.DataFrame()

    coverage_parts = [
        part
        for part in [
            base_coverage,
            new_coverage_df,
        ]
        if not part.empty
    ]

    if coverage_parts:
        combined_coverage = pd.concat(
            coverage_parts,
            ignore_index=True
        )
    else:
        combined_coverage = pd.DataFrame()

    error_parts = [
        part
        for part in [
            base_errors,
            new_errors_df,
        ]
        if not part.empty
    ]

    if error_parts:
        combined_errors = pd.concat(
            error_parts,
            ignore_index=True
        )
    else:
        combined_errors = pd.DataFrame()

    return (
        combined_candidates,
        combined_raw_edges,
        combined_coverage,
        combined_errors,
    )


def save_workbook_to_path(output_path):
    (
        candidate_df,
        raw_edges_df,
        coverage_df,
        errors_df,
    ) = build_combined_frames()

    # Rank candidates after combining all areas.
    if not candidate_df.empty:
        candidate_df = (
            candidate_df.sort_values(
                [
                    "Area",
                    "Total Retrieved Length m",
                ],
                ascending=[
                    True,
                    False,
                ]
            )
            .reset_index(drop=True)
        )

        candidate_df[
            "Candidate Rank in Area"
        ] = (
            candidate_df.groupby("Area")
            .cumcount()
            + 1
        )

        # Keep rank near the beginning if possible.
        preferred_columns = [
            "Area",
            "Candidate Rank in Area",
            "Road Name",
            "Suggested Dataset Category",
            "OSM Highway Class",
            "Representative OSM Lanes",
            "OSM Lane Values",
            "Lane Tag Coverage Percent",
            "Has Intersection",
            "Intersection Node Count",
            "Has Traffic Signal",
            "Traffic Signal Node Count",
            "OSM Maxspeed Values",
            "OSM Surface Values",
            "OSM Width Values",
            "One-Way Edge Percent",
            "Total Retrieved Length m",
            "Retrieved Edge Count",
            "OSM Way IDs",
            "Area Centre Latitude",
            "Area Centre Longitude",
            "OSM Snapshot Date",
            "Infrastructure Source",
            "Area Search Query",
            "Overpass Server Used",
        ]

        present = [
            column
            for column in preferred_columns
            if column in candidate_df.columns
        ]

        remaining = [
            column
            for column in candidate_df.columns
            if column not in present
        ]

        candidate_df = candidate_df[
            present + remaining
        ]

        top_candidates_df = candidate_df[
            candidate_df[
                "Candidate Rank in Area"
            ]
            <= 20
        ].copy()

    else:
        top_candidates_df = pd.DataFrame()

    readme_df = pd.DataFrame(
        [
            [
                "Purpose",
                (
                    "Candidate real Karachi road segments "
                    "from OpenStreetMap for infrastructure "
                    "enrichment."
                ),
            ],
            [
                "Historical snapshot",
                snapshot_timestamp,
            ],
            [
                "Search radius",
                (
                    f"{SEARCH_RADIUS_METRES} metres around "
                    "each geocoded area centre"
                ),
            ],
            [
                "Resume behavior",
                (
                    "Previously successful areas are kept. "
                    "Only failed/missing areas are retried."
                ),
            ],
            [
                "Fast mode",
                (
                    "Raw edge-level OSM data is not written to Excel because "
                    "it is very large and not required for representative-road "
                    "selection. Road_Candidates keeps the required summary data."
                ),
            ],
            [
                "Overpass retry",
                (
                    "If one public Overpass server fails, "
                    "the script tries another configured server."
                ),
            ],
            [
                "Representative OSM Lanes",
                (
                    "Most common parsed OSM lane value among "
                    "retrieved edges for the named road. Blank "
                    "means no usable lane tag was available."
                ),
            ],
            [
                "Has Intersection",
                (
                    "1 if the retrieved road touches at least "
                    "one node with 3 or more physical streets."
                ),
            ],
            [
                "Has Traffic Signal",
                (
                    "1 if the retrieved road touches at least "
                    "one node tagged highway=traffic_signals."
                ),
            ],
            [
                "Important limitation",
                (
                    "The thesis rows do not contain precise "
                    "road coordinates. OSM roads are candidate "
                    "representative segments, not confirmed "
                    "row-level observation locations."
                ),
            ],
        ],
        columns=[
            "Item",
            "Explanation",
        ]
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl"
    ) as writer:
        candidate_df.to_excel(
            writer,
            sheet_name="Road_Candidates",
            index=False
        )

        top_candidates_df.to_excel(
            writer,
            sheet_name="Top_Candidates",
            index=False
        )

        dataset_contexts.to_excel(
            writer,
            sheet_name="Dataset_Contexts",
            index=False
        )

        coverage_df.to_excel(
            writer,
            sheet_name="Area_Coverage",
            index=False
        )

        if SAVE_RAW_EDGES:
            raw_edges_df.to_excel(
                writer,
                sheet_name="Raw_OSM_Edges",
                index=False
            )

        errors_df.to_excel(
            writer,
            sheet_name="Errors",
            index=False
        )

        readme_df.to_excel(
            writer,
            sheet_name="README",
            index=False
        )


def save_workbook_safely(final_save=False):
    """
    Save without treating an Excel file lock as an OSM extraction failure.

    If the main workbook is open in Excel, Windows may raise PermissionError.
    During intermediate saves we simply warn and keep all downloaded data in
    memory. During the final save we write a recovered workbook instead.
    """
    try:
        save_workbook_to_path(OUTPUT_FILE)
        print(f"Workbook saved: {OUTPUT_FILE}")
        return OUTPUT_FILE

    except PermissionError:
        print("\nWARNING: The main Excel workbook is currently locked/open:")
        print(OUTPUT_FILE)
        print("Close it in Excel before the next run.")

        if not final_save:
            print(
                "OSM extraction is still successful. "
                "Continuing without marking the area as failed."
            )
            return None

        print("Saving the completed results to a recovery workbook instead...")
        save_workbook_to_path(FALLBACK_OUTPUT_FILE)
        print(f"Recovery workbook saved: {FALLBACK_OUTPUT_FILE}")
        return FALLBACK_OUTPUT_FILE


# ============================================================
# 13. PROCESS EACH MISSING/FAILED AREA
# ============================================================

for area in areas_to_process:
    print("\n" + "=" * 70)
    print(f"PROCESSING AREA: {area}")
    print("=" * 70)

    search_query = AREA_SEARCH_QUERIES.get(
        area,
        f"{area}, Karachi, Pakistan"
    )

    print("Search query:", search_query)

    try:
        # ----------------------------------------------------
        # GEOCODE AREA CENTRE
        # ----------------------------------------------------
        if area in KNOWN_AREA_CENTRES:
            centre_latitude, centre_longitude = KNOWN_AREA_CENTRES[area]
            print("Using previously verified centre from earlier run.")
        else:
            centre_latitude, centre_longitude = (
                ox.geocoder.geocode(
                    search_query
                )
            )

        print(
            "Centre:",
            centre_latitude,
            centre_longitude
        )

        # ----------------------------------------------------
        # DOWNLOAD ROAD NETWORK WITH SERVER RETRY
        # ----------------------------------------------------
        G, server_used = download_graph_with_retry(
            centre_latitude,
            centre_longitude,
            area
        )

        # ----------------------------------------------------
        # CONVERT GRAPH TO GEODATAFRAMES
        # ----------------------------------------------------
        nodes, edges = ox.convert.graph_to_gdfs(
            G,
            nodes=True,
            edges=True,
        )

        print("Network nodes:", len(nodes))
        print("Network edges:", len(edges))

        # ----------------------------------------------------
        # IDENTIFY INTERSECTIONS
        # ----------------------------------------------------
        if "street_count" in nodes.columns:
            intersection_nodes = set(
                nodes[
                    pd.to_numeric(
                        nodes["street_count"],
                        errors="coerce"
                    )
                    >= 3
                ].index
            )
        else:
            street_counts = (
                ox.stats.count_streets_per_node(G)
            )

            intersection_nodes = {
                node_id
                for node_id, count
                in street_counts.items()
                if count >= 3
            }

        # ----------------------------------------------------
        # IDENTIFY TRAFFIC SIGNAL NODES
        # ----------------------------------------------------
        if "highway" in nodes.columns:
            signal_mask = (
                nodes["highway"]
                .apply(
                    contains_traffic_signal
                )
            )

            signal_nodes = set(
                nodes[
                    signal_mask
                ].index
            )
        else:
            signal_nodes = set()

        print(
            "Intersection nodes:",
            len(intersection_nodes)
        )

        print(
            "Traffic signal nodes:",
            len(signal_nodes)
        )

        # ----------------------------------------------------
        # PREPARE EDGE DATA
        # ----------------------------------------------------
        edges_reset = (
            edges
            .reset_index()
            .copy()
        )

        optional_columns = [
            "name",
            "highway",
            "lanes",
            "maxspeed",
            "oneway",
            "surface",
            "width",
            "est_width",
            "junction",
            "service",
            "ref",
            "osmid",
            "length",
        ]

        for column in optional_columns:
            if column not in edges_reset.columns:
                edges_reset[column] = np.nan

        edges_reset["Area"] = area
        edges_reset["Area Search Query"] = search_query
        edges_reset[
            "Area Centre Latitude"
        ] = centre_latitude
        edges_reset[
            "Area Centre Longitude"
        ] = centre_longitude
        edges_reset[
            "Overpass Server Used"
        ] = server_used

        edges_reset["Road Name"] = (
            edges_reset["name"]
            .apply(clean_osm_value)
        )

        edges_reset["OSM Highway Class"] = (
            edges_reset["highway"]
            .apply(clean_osm_value)
        )

        edges_reset["OSM Lanes Raw"] = (
            edges_reset["lanes"]
            .apply(clean_osm_value)
        )

        edges_reset["OSM Lanes Parsed"] = (
            edges_reset["lanes"]
            .apply(parse_lane_value)
        )

        edges_reset["OSM Maxspeed"] = (
            edges_reset["maxspeed"]
            .apply(clean_osm_value)
        )

        edges_reset["OSM Surface"] = (
            edges_reset["surface"]
            .apply(clean_osm_value)
        )

        edges_reset["OSM Width"] = (
            edges_reset["width"]
            .apply(clean_osm_value)
        )

        edges_reset["OSM Estimated Width"] = (
            edges_reset["est_width"]
            .apply(clean_osm_value)
        )

        edges_reset["OSM Way IDs"] = (
            edges_reset["osmid"]
            .apply(extract_osm_ids)
        )

        edges_reset["Touches Intersection"] = (
            edges_reset.apply(
                lambda row:
                int(
                    row["u"] in intersection_nodes
                    or row["v"] in intersection_nodes
                ),
                axis=1
            )
        )

        edges_reset["Touches Traffic Signal"] = (
            edges_reset.apply(
                lambda row:
                int(
                    row["u"] in signal_nodes
                    or row["v"] in signal_nodes
                ),
                axis=1
            )
        )

        edges_reset[
            "Suggested Dataset Category"
        ] = (
            edges_reset[
                "OSM Highway Class"
            ]
            .apply(
                suggested_dataset_category
            )
        )

        # ----------------------------------------------------
        # RAW EDGE OUTPUT
        # ----------------------------------------------------
        raw_columns = [
            "Area",
            "Area Search Query",
            "Area Centre Latitude",
            "Area Centre Longitude",
            "Overpass Server Used",
            "u",
            "v",
            "key",
            "Road Name",
            "OSM Highway Class",
            "OSM Lanes Raw",
            "OSM Lanes Parsed",
            "OSM Maxspeed",
            "OSM Surface",
            "OSM Width",
            "OSM Estimated Width",
            "oneway",
            "junction",
            "service",
            "ref",
            "OSM Way IDs",
            "length",
            "Touches Intersection",
            "Touches Traffic Signal",
            "Suggested Dataset Category",
        ]

        if SAVE_RAW_EDGES:
            raw_area_edges = (
                edges_reset[
                    raw_columns
                ]
                .copy()
            )

            new_raw_edges.append(
                raw_area_edges
            )

        # ----------------------------------------------------
        # NAMED ROAD CANDIDATES
        # ----------------------------------------------------
        named_edges = (
            edges_reset[
                edges_reset[
                    "Road Name"
                ]
                .str.strip()
                .ne("")
            ]
            .copy()
        )

        candidate_count = 0

        group_columns = [
            "Road Name",
            "OSM Highway Class",
        ]

        for (
            road_name,
            highway_class
        ), road_group in (
            named_edges.groupby(
                group_columns,
                dropna=False
            )
        ):
            road_nodes = (
                set(
                    road_group["u"]
                    .tolist()
                )
                |
                set(
                    road_group["v"]
                    .tolist()
                )
            )

            road_intersections = (
                road_nodes
                & intersection_nodes
            )

            road_signals = (
                road_nodes
                & signal_nodes
            )

            lane_mode = mode_numeric(
                road_group[
                    "OSM Lanes Parsed"
                ]
            )

            lane_coverage = (
                road_group[
                    "OSM Lanes Parsed"
                ]
                .notna()
                .mean()
                * 100
            )

            oneway_values = (
                road_group["oneway"]
                .astype(str)
                .str.lower()
            )

            one_way_share = (
                oneway_values
                .isin(
                    [
                        "true",
                        "1",
                        "yes",
                    ]
                )
                .mean()
                * 100
            )

            total_length = (
                pd.to_numeric(
                    road_group["length"],
                    errors="coerce"
                )
                .sum()
            )

            candidate_record = {
                "Area":
                    area,

                "Area Search Query":
                    search_query,

                "Area Centre Latitude":
                    centre_latitude,

                "Area Centre Longitude":
                    centre_longitude,

                "OSM Snapshot Date":
                    snapshot_date,

                "Road Name":
                    road_name,

                "OSM Highway Class":
                    highway_class,

                "Suggested Dataset Category":
                    suggested_dataset_category(
                        highway_class
                    ),

                "OSM Lane Values":
                    unique_join(
                        road_group[
                            "OSM Lanes Raw"
                        ]
                    ),

                "Representative OSM Lanes":
                    lane_mode,

                "Lane Tag Coverage Percent":
                    round(
                        lane_coverage,
                        1
                    ),

                "OSM Maxspeed Values":
                    unique_join(
                        road_group[
                            "OSM Maxspeed"
                        ]
                    ),

                "OSM Surface Values":
                    unique_join(
                        road_group[
                            "OSM Surface"
                        ]
                    ),

                "OSM Width Values":
                    unique_join(
                        road_group[
                            "OSM Width"
                        ]
                    ),

                "One-Way Edge Percent":
                    round(
                        one_way_share,
                        1
                    ),

                "Intersection Node Count":
                    len(
                        road_intersections
                    ),

                "Has Intersection":
                    int(
                        len(
                            road_intersections
                        )
                        > 0
                    ),

                "Traffic Signal Node Count":
                    len(
                        road_signals
                    ),

                "Has Traffic Signal":
                    int(
                        len(
                            road_signals
                        )
                        > 0
                    ),

                "Total Retrieved Length m":
                    round(
                        total_length,
                        1
                    ),

                "Retrieved Edge Count":
                    len(
                        road_group
                    ),

                "OSM Way IDs":
                    unique_join(
                        road_group[
                            "OSM Way IDs"
                        ]
                    ),

                "Infrastructure Source":
                    "OpenStreetMap via OSMnx",

                "Overpass Server Used":
                    server_used,
            }

            new_candidate_rows.append(
                candidate_record
            )

            candidate_count += 1

        # ----------------------------------------------------
        # COVERAGE SUMMARY
        # ----------------------------------------------------
        lane_tagged_edges = (
            edges_reset[
                "OSM Lanes Parsed"
            ]
            .notna()
            .sum()
        )

        new_coverage_rows.append({
            "Area":
                area,

            "Search Query":
                search_query,

            "Centre Latitude":
                centre_latitude,

            "Centre Longitude":
                centre_longitude,

            "Network Nodes":
                len(nodes),

            "Network Edges":
                len(edges_reset),

            "Named Road Edges":
                len(named_edges),

            "Unique Named Road Candidates":
                candidate_count,

            "Edges With Lane Data":
                int(
                    lane_tagged_edges
                ),

            "Lane Data Coverage Percent":
                round(
                    (
                        lane_tagged_edges
                        / len(edges_reset)
                        * 100
                    )
                    if len(edges_reset)
                    else 0,
                    1
                ),

            "Intersection Nodes":
                len(intersection_nodes),

            "Traffic Signal Nodes":
                len(signal_nodes),

            "Overpass Server Used":
                server_used,

            "Status":
                "Success",
        })

        print(
            "Candidate named roads:",
            candidate_count
        )

        # Save immediately after success. A locked Excel workbook is
        # handled separately and does NOT make the OSM area a failure.
        if SAVE_AFTER_EACH_SUCCESS:
            save_workbook_safely(final_save=False)

    except Exception as error:
        print(
            "\nERROR for area:",
            area
        )
        print(error)

        new_error_rows.append({
            "Area":
                area,

            "Search Query":
                search_query,

            "Error":
                str(error),
        })

        new_coverage_rows.append({
            "Area":
                area,

            "Search Query":
                search_query,

            "Centre Latitude":
                np.nan,

            "Centre Longitude":
                np.nan,

            "Network Nodes":
                0,

            "Network Edges":
                0,

            "Named Road Edges":
                0,

            "Unique Named Road Candidates":
                0,

            "Edges With Lane Data":
                0,

            "Lane Data Coverage Percent":
                0,

            "Intersection Nodes":
                0,

            "Traffic Signal Nodes":
                0,

            "Overpass Server Used":
                "",

            "Status":
                "Failed",
        })


# ============================================================
# 14. FINAL SAVE
# ============================================================

ACTUAL_OUTPUT_FILE = save_workbook_safely(final_save=True)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

final_result_file = (
    ACTUAL_OUTPUT_FILE
    if ACTUAL_OUTPUT_FILE is not None
    else OUTPUT_FILE
)

final_coverage = safe_read_sheet(
    final_result_file,
    "Area_Coverage"
)

final_candidates = safe_read_sheet(
    final_result_file,
    "Road_Candidates"
)

final_errors = safe_read_sheet(
    final_result_file,
    "Errors"
)

print("\n" + "=" * 70)
print("OSM ROAD EXTRACTION COMPLETED")
print("=" * 70)

print("\nAreas in thesis dataset:")
print(len(areas))

if not final_coverage.empty:
    print("\nAreas successfully processed:")
    print(
        (
            final_coverage["Status"]
            == "Success"
        ).sum()
    )

    print("\nAreas still failed:")
    print(
        (
            final_coverage["Status"]
            == "Failed"
        ).sum()
    )

    failed_areas = (
        final_coverage.loc[
            final_coverage[
                "Status"
            ].eq("Failed"),
            "Area",
        ]
        .astype(str)
        .tolist()
    )

    if failed_areas:
        print("\nStill failed:")
        for area in failed_areas:
            print(" -", area)

print("\nTotal real named-road candidates:")
print(len(final_candidates))

print("\nOutput file:")
print(final_result_file)

print(
    "\nThe workbook is resumable. "
    "Running this script again will preserve successful "
    "areas and retry only failed/missing areas."
)

print(
    "\nDo NOT merge candidate roads into the thesis "
    "dataset yet. First review Top_Candidates, "
    "Dataset_Contexts, Area_Coverage and Errors."
)
