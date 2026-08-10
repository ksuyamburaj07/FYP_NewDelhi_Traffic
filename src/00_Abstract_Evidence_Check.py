import json
import re
import sys
import zipfile
from pathlib import Path


PROJECT_FOLDER = Path(r"E:\FYP")
ZIP_PATH = PROJECT_FOLDER / "data" / "raw" / "Main_dataset.zip"

DATE_PATTERN = re.compile(
    r"new_delhi__(\d{4}-\d{2}-\d{2})_to_",
    re.IGNORECASE,
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


def main():
    print("=" * 70)
    print("FYP ABSTRACT DATASET EVIDENCE CHECK")
    print("=" * 70)

    print("\n1. DATASET FILE")
    print("Dataset path:", ZIP_PATH)
    print("Dataset exists:", ZIP_PATH.exists())

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            r"Main_dataset.zip was not found in E:\FYP\data\raw"
        )

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
        all_files = zip_file.namelist()

        daily_files = sorted(
            file_name
            for file_name in all_files
            if "probe_counts/geojson/" in file_name.lower()
            and file_name.lower().endswith(".geojson")
        )

        print("\n2. DAILY GEOJSON FILES")
        print("Number of daily GeoJSON files:", len(daily_files))

        if not daily_files:
            raise ValueError(
                "No daily probe-count GeoJSON files were found."
            )

        dates = []

        for file_name in daily_files:
            match = DATE_PATTERN.search(file_name)

            if match:
                dates.append(match.group(1))

        print("\n3. DATE COVERAGE")
        print("First date:", min(dates))
        print("Last date:", max(dates))
        print("Number of extracted dates:", len(dates))

        first_file = daily_files[0]

        print("\n4. FIRST DAILY FILE")
        print("Filename:", first_file)

        with zip_file.open(first_file) as file_object:
            first_day_data = json.load(file_object)

    all_features = first_day_data.get("features", [])

    segment_features = [
        feature
        for feature in all_features
        if "segmentProbeCounts"
        in (feature.get("properties") or {})
    ]

    print("\n5. ROAD-SEGMENT COVERAGE")
    print("All features in first file:", len(all_features))
    print(
        "Road-segment features:",
        len(segment_features),
    )

    if not segment_features:
        raise ValueError(
            "No road-segment probe-count records were found."
        )

    sample_properties = (
        segment_features[0].get("properties") or {}
    )

    hourly_records = sample_properties.get(
        "segmentProbeCounts",
        [],
    )

    print("\n6. SAMPLE ROAD SEGMENT")
    print("Segment ID:", sample_properties.get("segmentId"))
    print("Street name:", sample_properties.get("streetName"))
    print("Speed limit:", sample_properties.get("speedLimit"))
    print("Road class:", sample_properties.get("frc"))
    print("Distance:", sample_properties.get("distance"))
    print("Hourly records:", len(hourly_records))

    print("\nFirst three hourly records:")

    for record in hourly_records[:3]:
        print(record)

    number_of_days = len(daily_files)
    segments_on_first_day = len(segment_features)
    hours_per_segment = len(hourly_records)

    approximate_records = (
        number_of_days
        * segments_on_first_day
        * hours_per_segment
    )

    print("\n7. APPROXIMATE RAW DATA SIZE")
    print("Number of days:", number_of_days)
    print("Segments on first day:", segments_on_first_day)
    print("Hours per sample segment:", hours_per_segment)
    print(
        "Approximate potential segment-hour records:",
        f"{approximate_records:,}",
    )

    unique_named_roads = {
        (feature.get("properties") or {}).get("streetName")
        for feature in segment_features
        if (feature.get("properties") or {}).get("streetName")
    }

    print("\n8. ROAD-NAME COVERAGE")
    print(
        "Unique named roads in first daily file:",
        len(unique_named_roads),
    )

    found_roads = [
        road
        for road in SELECTED_ROADS
        if road in unique_named_roads
    ]

    missing_roads = [
        road
        for road in SELECTED_ROADS
        if road not in unique_named_roads
    ]

    print("\nSelected project roads found:")

    for road in found_roads:
        print("[FOUND]", road)

    if missing_roads:
        print("\nSelected roads not found exactly:")

        for road in missing_roads:
            print("[NOT FOUND]", road)

    print("\n9. VERIFIED DATASET SUMMARY")
    print("Location: New Delhi")
    print("Coverage:", min(dates), "to", max(dates))
    print("Daily files:", len(daily_files))
    print("Road segments per daily file:", len(segment_features))
    print("Hourly records per segment:", len(hourly_records))
    print("Unique named roads:", len(unique_named_roads))
    print("Selected project roads found:", len(found_roads))
    print(
        "Approximate segment-hour observations:",
        f"{approximate_records:,}",
    )

    print("\n" + "=" * 70)
    print("EVIDENCE CHECK COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR:", error)
        sys.exit(1)