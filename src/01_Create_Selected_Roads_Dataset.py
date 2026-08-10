import csv
import gc
import json
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PROJECT_FOLDER = Path(r"E:\FYP")

ZIP_PATH = (
    PROJECT_FOLDER
    / "data"
    / "raw"
    / "Main_dataset.zip"
)

OUTPUT_PATH = (
    PROJECT_FOLDER
    / "data"
    / "processed"
    / "selected_delhi_roads_hourly.csv"
)

SUMMARY_PATH = (
    PROJECT_FOLDER
    / "outputs"
    / "selected_roads_dataset_summary.txt"
)

SELECTED_ROADS = [
    "Mahatma Gandhi Marg",
    "Mathura Road",
    "Outer Ring Road",
    "Aurobindo Marg",
    "Mehrauli Badarpur Road",
    "Barapullah Road",
    "Vikas Marg",
    "Africa Avenue",
    "Sardar Patel Marg",
    "Noida Link Road",
]

DATE_PATTERN = re.compile(
    r"new_delhi__(\d{4}-\d{2}-\d{2})_to_",
    re.IGNORECASE,
)

OUTPUT_COLUMNS = [
    "date",
    "street_name",
    "hour",
    "day_of_week",
    "day_number",
    "is_weekend",
    "is_peak_hour",
    "probe_count",
    "segment_count",
    "average_speed_limit",
    "average_frc",
    "total_distance",
]


def safe_number(value, default=0.0):
    """Convert a value to a number without stopping the program."""

    try:
        if value is None:
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def main():
    """Create an hourly dataset for ten selected New Delhi roads."""

    print("=" * 72)
    print("CREATE SELECTED NEW DELHI ROADS DATASET")
    print("=" * 72)

    print("\n1. INPUT AND OUTPUT PATHS")
    print("Input ZIP:", ZIP_PATH)
    print("Input exists:", ZIP_PATH.exists())
    print("Output CSV:", OUTPUT_PATH)

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            r"Main_dataset.zip was not found in E:\FYP\data\raw"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_rows = []

    road_observation_counts = {
        road: 0
        for road in SELECTED_ROADS
    }

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:

        daily_files = sorted(
            file_name
            for file_name in zip_file.namelist()
            if "probe_counts/geojson/" in file_name.lower()
            and file_name.lower().endswith(".geojson")
        )

        print("\n2. DAILY FILES")
        print("Number of daily files:", len(daily_files))

        if not daily_files:
            raise ValueError(
                "No daily probe-count GeoJSON files were found."
            )

        for file_number, file_name in enumerate(
            daily_files,
            start=1,
        ):
            date_match = DATE_PATTERN.search(file_name)

            if not date_match:
                print(
                    "Skipped because the date could not be read:",
                    file_name,
                )
                continue

            observation_date = date_match.group(1)

            date_object = datetime.strptime(
                observation_date,
                "%Y-%m-%d",
            )

            print(
                f"Processing {file_number}/{len(daily_files)}:",
                observation_date,
            )

            with zip_file.open(file_name) as file_object:
                daily_data = json.load(file_object)

            hourly_aggregates = defaultdict(
                lambda: {
                    "probe_count": 0,
                    "segment_count": 0,
                    "speed_limit_total": 0.0,
                    "frc_total": 0.0,
                    "distance_total": 0.0,
                }
            )

            for feature in daily_data.get(
                "features",
                [],
            ):
                properties = (
                    feature.get("properties") or {}
                )

                street_name = properties.get(
                    "streetName"
                )

                if street_name not in road_observation_counts:
                    continue

                speed_limit = safe_number(
                    properties.get("speedLimit")
                )

                road_class = safe_number(
                    properties.get("frc")
                )

                segment_distance = safe_number(
                    properties.get("distance")
                )

                hourly_records = properties.get(
                    "segmentProbeCounts"
                ) or []

                for hourly_record in hourly_records:
                    time_set = hourly_record.get(
                        "timeSet"
                    )

                    if not isinstance(
                        time_set,
                        (int, float),
                    ):
                        continue

                    # The dataset uses timeSet 2 for 00:00–01:00.
                    # Subtracting 2 converts the values to hours 0–23.
                    hour = int(time_set) - 2

                    if hour < 0 or hour > 23:
                        continue

                    probe_count = int(
                        safe_number(
                            hourly_record.get(
                                "probeCount"
                            ),
                            default=0,
                        )
                    )

                    aggregation_key = (
                        street_name,
                        hour,
                    )

                    values = hourly_aggregates[
                        aggregation_key
                    ]

                    values["probe_count"] += (
                        probe_count
                    )

                    values["segment_count"] += 1

                    values["speed_limit_total"] += (
                        speed_limit
                    )

                    values["frc_total"] += (
                        road_class
                    )

                    values["distance_total"] += (
                        segment_distance
                    )

            for street_name in SELECTED_ROADS:
                for hour in range(24):

                    values = hourly_aggregates.get(
                        (street_name, hour)
                    )

                    if not values:
                        continue

                    segment_count = values[
                        "segment_count"
                    ]

                    output_rows.append({
                        "date": observation_date,
                        "street_name": street_name,
                        "hour": hour,
                        "day_of_week": (
                            date_object.strftime("%A")
                        ),
                        "day_number": (
                            date_object.weekday()
                        ),
                        "is_weekend": (
                            1
                            if date_object.weekday() >= 5
                            else 0
                        ),
                        "is_peak_hour": (
                            1
                            if hour in {
                                7,
                                8,
                                9,
                                17,
                                18,
                                19,
                                20,
                            }
                            else 0
                        ),
                        "probe_count": values[
                            "probe_count"
                        ],
                        "segment_count": (
                            segment_count
                        ),
                        "average_speed_limit": round(
                            values[
                                "speed_limit_total"
                            ]
                            / segment_count,
                            4,
                        ),
                        "average_frc": round(
                            values["frc_total"]
                            / segment_count,
                            4,
                        ),
                        "total_distance": round(
                            values["distance_total"],
                            4,
                        ),
                    })

                    road_observation_counts[
                        street_name
                    ] += 1

            del daily_data
            del hourly_aggregates
            gc.collect()

    print("\n3. SAVING THE PROCESSED DATASET")

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=OUTPUT_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(output_rows)

    dates = sorted({
        row["date"]
        for row in output_rows
    })

    summary_lines = [
        "=" * 72,
        "SELECTED ROADS DATASET SUMMARY",
        "=" * 72,
        "",
        f"Output file: {OUTPUT_PATH}",
        f"Total rows: {len(output_rows):,}",
        f"Total columns: {len(OUTPUT_COLUMNS)}",
        f"Selected roads: {len(SELECTED_ROADS)}",
        f"First date: {dates[0] if dates else 'Not available'}",
        f"Last date: {dates[-1] if dates else 'Not available'}",
        f"Number of dates: {len(dates)}",
        "",
        "OBSERVATIONS PER ROAD",
    ]

    for road_name in SELECTED_ROADS:
        summary_lines.append(
            f"{road_name}: "
            f"{road_observation_counts[road_name]:,}"
        )

    expected_rows = (
        len(SELECTED_ROADS)
        * len(dates)
        * 24
    )

    summary_lines.extend([
        "",
        f"Expected complete rows: {expected_rows:,}",
        f"Actual rows: {len(output_rows):,}",
        (
            "Completeness check: PASSED"
            if len(output_rows) == expected_rows
            else "Completeness check: REVIEW REQUIRED"
        ),
        "",
        "=" * 72,
        "DATASET CREATION COMPLETED",
        "=" * 72,
    ])

    summary_text = "\n".join(
        summary_lines
    )

    SUMMARY_PATH.write_text(
        summary_text,
        encoding="utf-8",
    )

    print(summary_text)


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\nERROR:", error)
        sys.exit(1)