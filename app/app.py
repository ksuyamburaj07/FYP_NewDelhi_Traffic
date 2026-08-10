
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Application configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title=(
        "New Delhi Traffic Probe Count Forecasting"
    ),
    page_icon="🚦",
    layout="wide"
)


BASE_DIRECTORY = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIRECTORY
    / "random_forest_next_hour_traffic_model.joblib"
)

DATASET_PATH = (
    BASE_DIRECTORY
    / "selected_delhi_roads_hourly.csv"
)


FEATURE_COLUMNS = [
    "street_name",
    "probe_count",
    "lag_1",
    "lag_24",
    "rolling_mean_3",
    "rolling_mean_24",
    "target_hour",
    "target_day_number",
    "target_is_weekend",
    "target_is_peak_hour",
    "segment_count",
    "average_speed_limit",
    "average_frc",
    "total_distance"
]


PEAK_HOURS = [
    7,
    8,
    9,
    17,
    18,
    19,
    20
]


# ---------------------------------------------------------
# Loading functions
# ---------------------------------------------------------

@st.cache_resource
def load_model(model_path):
    """Load the saved Random Forest pipeline."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {model_path.name}"
        )

    return joblib.load(model_path)


@st.cache_data
def prepare_dataset(dataset_path):
    """
    Load the processed dataset and recreate the historical
    forecasting features required by the trained model.
    """

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {dataset_path.name}"
        )

    data = pd.read_csv(dataset_path)

    data["datetime"] = (
        pd.to_datetime(data["date"])
        + pd.to_timedelta(
            data["hour"],
            unit="h"
        )
    )

    data = (
        data
        .sort_values(
            ["street_name", "datetime"]
        )
        .reset_index(drop=True)
    )

    grouped_data = data.groupby(
        "street_name",
        group_keys=False
    )

    data["lag_1"] = (
        grouped_data["probe_count"]
        .shift(1)
    )

    data["lag_24"] = (
        grouped_data["probe_count"]
        .shift(24)
    )

    data["rolling_mean_3"] = (
        grouped_data["probe_count"]
        .transform(
            lambda series:
            series.shift(1).rolling(3).mean()
        )
    )

    data["rolling_mean_24"] = (
        grouped_data["probe_count"]
        .transform(
            lambda series:
            series.shift(1).rolling(24).mean()
        )
    )

    data["target_next_hour"] = (
        grouped_data["probe_count"]
        .shift(-1)
    )

    data["target_datetime"] = (
        data["datetime"]
        + pd.Timedelta(hours=1)
    )

    data["target_hour"] = (
        data["target_datetime"].dt.hour
    )

    data["target_day_number"] = (
        data["target_datetime"].dt.dayofweek
    )

    data["target_is_weekend"] = (
        data["target_day_number"] >= 5
    ).astype(int)

    data["target_is_peak_hour"] = (
        data["target_hour"]
        .isin(PEAK_HOURS)
        .astype(int)
    )

    model_ready_data = (
        data
        .dropna(
            subset=[
                "lag_1",
                "lag_24",
                "rolling_mean_3",
                "rolling_mean_24",
                "target_next_hour"
            ]
        )
        .copy()
    )

    model_ready_data[
        "observation_date"
    ] = (
        model_ready_data[
            "datetime"
        ].dt.date
    )

    return model_ready_data


# ---------------------------------------------------------
# Load project resources
# ---------------------------------------------------------

try:
    forecasting_model = load_model(
        MODEL_PATH
    )

    forecasting_data = prepare_dataset(
        DATASET_PATH
    )

except Exception as error:
    st.error(
        "The application could not load its required "
        "project resources."
    )

    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Page heading
# ---------------------------------------------------------

st.title(
    "New Delhi Next-Hour Traffic Probe Count Forecasting"
)

st.caption(
    "Random Forest forecasting prototype for ten selected "
    "New Delhi roads"
)

st.info(
    "This prototype operates in historical evaluation mode. "
    "It uses an observation from the processed dataset to "
    "forecast the following hour and compare the prediction "
    "with the recorded historical value."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("Project Information")

    st.write(
        "**Student:** Suyambu Raj"
    )

    st.write(
        "**Model:** Random Forest Regressor"
    )

    st.write(
        "**Forecast horizon:** One hour"
    )

    st.write(
        "**Dataset coverage:** 11–30 August 2024"
    )

    st.write(
        "**Selected roads:** 10"
    )

    st.divider()

    st.warning(
        "Probe count represents aggregated probe activity. "
        "It is not an exact vehicle count or a direct "
        "congestion percentage."
    )


# ---------------------------------------------------------
# User selection controls
# ---------------------------------------------------------

st.subheader(
    "Select a historical road observation"
)

available_roads = sorted(
    forecasting_data[
        "street_name"
    ].unique()
)

selected_road = st.selectbox(
    "Road",
    options=available_roads
)

road_data = (
    forecasting_data[
        forecasting_data[
            "street_name"
        ] == selected_road
    ]
    .copy()
)

available_dates = sorted(
    road_data[
        "observation_date"
    ].unique()
)

selected_date = st.selectbox(
    "Observation date",
    options=available_dates,
    format_func=lambda value:
    pd.Timestamp(value).strftime(
        "%d %B %Y"
    )
)

date_data = (
    road_data[
        road_data[
            "observation_date"
        ] == selected_date
    ]
    .copy()
)

available_hours = sorted(
    date_data["hour"].astype(int).unique()
)

selected_hour = st.selectbox(
    "Current observation hour",
    options=available_hours,
    format_func=lambda hour:
    f"{int(hour):02d}:00"
)

matching_rows = date_data[
    date_data["hour"].astype(int)
    == int(selected_hour)
]

if matching_rows.empty:
    st.error(
        "No model-ready observation was found for "
        "the selected road, date and hour."
    )

    st.stop()

selected_row = matching_rows.iloc[0]


# ---------------------------------------------------------
# Current observation summary
# ---------------------------------------------------------

st.subheader(
    "Selected observation"
)

summary_column_1, summary_column_2, summary_column_3 = (
    st.columns(3)
)

summary_column_1.metric(
    "Current probe count",
    f'{selected_row["probe_count"]:,.0f}'
)

summary_column_2.metric(
    "Current time",
    selected_row[
        "datetime"
    ].strftime("%d %b %Y, %H:%M")
)

summary_column_3.metric(
    "Prediction target",
    selected_row[
        "target_datetime"
    ].strftime("%d %b %Y, %H:%M")
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button(
    "Predict next-hour probe count",
    type="primary",
    use_container_width=True
):

    model_input = pd.DataFrame(
        [
            selected_row[
                FEATURE_COLUMNS
            ].to_dict()
        ]
    )

    predicted_value = (
        forecasting_model.predict(
            model_input
        )[0]
    )

    actual_value = (
        selected_row[
            "target_next_hour"
        ]
    )

    absolute_error = abs(
        actual_value
        - predicted_value
    )

    percentage_error = (
        absolute_error
        / actual_value
        * 100
        if actual_value != 0
        else np.nan
    )

    st.success(
        "The next-hour forecast was generated successfully."
    )

    result_column_1, result_column_2 = (
        st.columns(2)
    )

    result_column_1.metric(
        "Predicted next-hour probe count",
        f"{predicted_value:,.0f}"
    )

    result_column_2.metric(
        "Recorded next-hour probe count",
        f"{actual_value:,.0f}"
    )

    error_column_1, error_column_2 = (
        st.columns(2)
    )

    error_column_1.metric(
        "Absolute error",
        f"{absolute_error:,.0f}"
    )

    error_column_2.metric(
        "Percentage error",
        f"{percentage_error:.2f}%"
    )

    comparison_data = pd.DataFrame({
        "Measure": [
            "Predicted next-hour probe count",
            "Recorded next-hour probe count"
        ],
        "Probe count": [
            predicted_value,
            actual_value
        ]
    })

    st.subheader(
        "Prediction comparison"
    )

    st.bar_chart(
        comparison_data.set_index(
            "Measure"
        )
    )

    with st.expander(
        "View model-input details"
    ):

        input_display = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Value": [
                model_input.iloc[0][feature]
                for feature in FEATURE_COLUMNS
            ]
        })

        st.dataframe(
            input_display,
            use_container_width=True,
            hide_index=True
        )


# ---------------------------------------------------------
# Recent road activity
# ---------------------------------------------------------

st.subheader(
    "Recent probe-count pattern"
)

historical_window = (
    road_data[
        road_data["datetime"]
        <= selected_row["datetime"]
    ]
    .tail(24)
    .set_index("datetime")[
        ["probe_count"]
    ]
)

historical_window = historical_window.rename(
    columns={
        "probe_count":
        "Aggregated probe count"
    }
)

st.line_chart(
    historical_window
)


# ---------------------------------------------------------
# Methodological notice
# ---------------------------------------------------------

st.divider()

st.caption(
    "The prototype demonstrates the integration of a saved "
    "machine-learning pipeline with a simple interactive "
    "interface. Results are restricted to the selected roads "
    "and the available 20-day historical dataset. The system "
    "does not currently include live traffic, weather, incident "
    "or seasonal information."
)
