from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="New Delhi Traffic Forecasting",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIRECTORY / "random_forest_next_hour_traffic_model.joblib"
DATASET_PATH = BASE_DIRECTORY / "selected_delhi_roads_hourly.csv"

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
    "total_distance",
]

PEAK_HOURS = [7, 8, 9, 17, 18, 19, 20]

FINAL_MAE = 6915.02
FINAL_RMSE = 12052.55
FINAL_R2 = 0.9912
FINAL_WAPE = 6.1640
MAE_IMPROVEMENT = 52.10
RMSE_IMPROVEMENT = 53.42

VALIDATION_RESULTS = pd.DataFrame({
    "Validation period": ["Validation 1", "Validation 2", "Final validation"],
    "MAE": [7981.16, 7035.60, 6915.02],
    "RMSE": [15259.18, 12586.91, 12052.55],
    "R²": [0.9823, 0.9905, 0.9912],
    "WAPE (%)": [7.96, 6.42, 6.16],
})

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    .app-header {
        padding: 1.2rem 1.4rem;
        border: 1px solid #334155;
        border-radius: 14px;
        background: #0f172a;
        margin-bottom: 1rem;
    }
    .app-header h1 { margin: 0; font-size: 2rem; }
    .app-header p { margin: 0.45rem 0 0 0; color: #cbd5e1; }
    .tag {
        display: inline-block;
        margin: 0.6rem 0.35rem 0 0;
        padding: 0.25rem 0.55rem;
        border: 1px solid #475569;
        border-radius: 999px;
        font-size: 0.78rem;
        color: #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.name}")
    return joblib.load(path)

@st.cache_data
def prepare_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path.name}")

    data = pd.read_csv(path)
    data["datetime"] = (
        pd.to_datetime(data["date"])
        + pd.to_timedelta(data["hour"], unit="h")
    )

    data = data.sort_values(
        ["street_name", "datetime"]
    ).reset_index(drop=True)

    grouped = data.groupby("street_name", group_keys=False)

    data["lag_1"] = grouped["probe_count"].shift(1)
    data["lag_24"] = grouped["probe_count"].shift(24)
    data["rolling_mean_3"] = grouped["probe_count"].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )
    data["rolling_mean_24"] = grouped["probe_count"].transform(
        lambda s: s.shift(1).rolling(24).mean()
    )

    data["target_next_hour"] = grouped["probe_count"].shift(-1)
    data["target_datetime"] = data["datetime"] + pd.Timedelta(hours=1)
    data["target_hour"] = data["target_datetime"].dt.hour
    data["target_day_number"] = data["target_datetime"].dt.dayofweek
    data["target_is_weekend"] = (
        data["target_day_number"] >= 5
    ).astype(int)
    data["target_is_peak_hour"] = (
        data["target_hour"].isin(PEAK_HOURS)
    ).astype(int)

    model_ready = data.dropna(
        subset=[
            "lag_1",
            "lag_24",
            "rolling_mean_3",
            "rolling_mean_24",
            "target_next_hour",
        ]
    ).copy()

    model_ready["observation_date"] = model_ready["datetime"].dt.date
    return model_ready

try:
    model = load_model(MODEL_PATH)
    data = prepare_dataset(DATASET_PATH)
except Exception as exc:
    st.error("The application could not load the project resources.")
    st.exception(exc)
    st.stop()

st.markdown(
    """
    <div class="app-header">
        <h1>🚦 New Delhi Traffic Forecasting</h1>
        <p>Machine-learning based next-hour forecasting of aggregated traffic probe activity.</p>
        <span class="tag">Random Forest Regressor</span>
        <span class="tag">10 Selected Roads</span>
        <span class="tag">Next-Hour Forecast</span>
        <span class="tag">Historical Evaluation</span>
    </div>
    """,
    unsafe_allow_html=True,
)

roads = sorted(data["street_name"].unique())

with st.sidebar:
    st.header("Forecast Controls")

    selected_road = st.selectbox("Road", roads)

    road_data = data[
        data["street_name"] == selected_road
    ].copy()

    dates = sorted(road_data["observation_date"].unique())

    selected_date = st.selectbox(
        "Observation date",
        dates,
        format_func=lambda d: pd.Timestamp(d).strftime("%d %B %Y"),
    )

    date_data = road_data[
        road_data["observation_date"] == selected_date
    ].copy()

    hours = sorted(date_data["hour"].astype(int).unique())

    selected_hour = st.selectbox(
        "Current observation hour",
        hours,
        format_func=lambda h: f"{int(h):02d}:00",
    )

    st.divider()
    st.subheader("Project Scope")
    st.write("**Model:** Random Forest Regressor")
    st.write("**Forecast horizon:** 1 hour")
    st.write("**Roads:** 10")
    st.write("**Data:** 11–30 August 2024")

    st.divider()
    st.warning(
        "Probe count is aggregated traffic probe activity. "
        "It is not an exact vehicle count or a direct congestion percentage."
    )

selected_rows = date_data[
    date_data["hour"].astype(int) == int(selected_hour)
]

if selected_rows.empty:
    st.error("No model-ready record was found for this selection.")
    st.stop()

selected = selected_rows.iloc[0]

selection_key = (
    selected_road,
    str(selected_date),
    int(selected_hour),
)

if (
    "forecast_result" in st.session_state
    and st.session_state["forecast_result"]["selection_key"] != selection_key
):
    del st.session_state["forecast_result"]

forecast_tab, pattern_tab, performance_tab, about_tab = st.tabs(
    ["Forecast", "Traffic Pattern", "Model Performance", "About"]
)

with forecast_tab:
    st.subheader("Selected Observation")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Current probe count",
        f'{selected["probe_count"]:,.0f}',
        border=True,
    )

    c2.metric(
        "Current time",
        selected["datetime"].strftime("%d %b, %H:%M"),
        border=True,
    )

    c3.metric(
        "Forecast target",
        selected["target_datetime"].strftime("%d %b, %H:%M"),
        border=True,
    )

    peak_text = (
        "Peak-period target"
        if int(selected["target_is_peak_hour"]) == 1
        else "Off-peak target"
    )
    weekend_text = (
        "Weekend"
        if int(selected["target_is_weekend"]) == 1
        else "Weekday"
    )

    st.caption(f"{selected_road} · {weekend_text} · {peak_text}")

    if st.button(
        "Generate next-hour forecast",
        type="primary",
        use_container_width=True,
    ):
        model_input = pd.DataFrame(
            [selected[FEATURE_COLUMNS].to_dict()]
        )

        prediction = float(model.predict(model_input)[0])
        actual = float(selected["target_next_hour"])
        abs_error = abs(actual - prediction)
        pct_error = (
            abs_error / actual * 100
            if actual != 0
            else np.nan
        )

        st.session_state["forecast_result"] = {
            "selection_key": selection_key,
            "prediction": prediction,
            "actual": actual,
            "abs_error": abs_error,
            "pct_error": pct_error,
            "model_input": model_input,
        }

    result = st.session_state.get("forecast_result")

    if result:
        st.success("Forecast generated successfully.")

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Predicted next hour",
            f'{result["prediction"]:,.0f}',
            border=True,
        )
        r2.metric(
            "Recorded next hour",
            f'{result["actual"]:,.0f}',
            border=True,
        )
        r3.metric(
            "Absolute error",
            f'{result["abs_error"]:,.0f}',
            border=True,
        )

        pct_text = (
            f'{result["pct_error"]:.2f}%'
            if not np.isnan(result["pct_error"])
            else "N/A"
        )

        r4.metric(
            "Percentage error",
            pct_text,
            border=True,
        )

        comparison = pd.DataFrame(
            {
                "Probe count": [
                    result["prediction"],
                    result["actual"],
                ]
            },
            index=["Predicted", "Recorded"],
        )

        st.subheader("Prediction Comparison")
        st.bar_chart(comparison)

        with st.expander("View model-input details"):
            input_table = pd.DataFrame(
                {
                    "Feature": FEATURE_COLUMNS,
                    "Value": [
                        result["model_input"].iloc[0][feature]
                        for feature in FEATURE_COLUMNS
                    ],
                }
            )
            st.dataframe(
                input_table,
                use_container_width=True,
                hide_index=True,
            )

with pattern_tab:
    st.subheader(f"Recent Traffic Activity — {selected_road}")

    history = (
        road_data[
            road_data["datetime"] <= selected["datetime"]
        ]
        .tail(24)
        .set_index("datetime")[["probe_count"]]
        .rename(columns={"probe_count": "Aggregated probe count"})
    )

    st.line_chart(history)

    h1, h2, h3 = st.columns(3)

    h1.metric(
        "Current activity",
        f'{selected["probe_count"]:,.0f}',
        border=True,
    )
    h2.metric(
        "Recent average",
        f'{history["Aggregated probe count"].mean():,.0f}',
        border=True,
    )
    h3.metric(
        "Recent maximum",
        f'{history["Aggregated probe count"].max():,.0f}',
        border=True,
    )

    st.caption(
        "The peak/off-peak feature is a calendar indicator used by the model. "
        "It is not a direct congestion classification."
    )

with performance_tab:
    st.subheader("Final Random Forest Performance")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("MAE", f"{FINAL_MAE:,.2f}", border=True)
    m2.metric("RMSE", f"{FINAL_RMSE:,.2f}", border=True)
    m3.metric("R²", f"{FINAL_R2:.4f}", border=True)
    m4.metric("WAPE", f"{FINAL_WAPE:.2f}%", border=True)

    i1, i2 = st.columns(2)

    i1.metric(
        "MAE improvement vs baseline",
        f"{MAE_IMPROVEMENT:.2f}%",
        border=True,
    )
    i2.metric(
        "RMSE improvement vs baseline",
        f"{RMSE_IMPROVEMENT:.2f}%",
        border=True,
    )

    st.caption(
        "R² is a regression goodness-of-fit measure, not classification accuracy."
    )

    st.subheader("Chronological Validation")
    st.dataframe(
        VALIDATION_RESULTS,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Random Forest outperformed the persistence baseline across all "
        "three chronological validation periods."
    )

with about_tab:
    st.subheader("Project Overview")

    st.write(
        "This Final Year Project forecasts the next hour's aggregated traffic "
        "probe activity for ten selected roads in New Delhi."
    )

    a1, a2, a3, a4 = st.columns(4)

    a1.metric("Processed records", "4,800", border=True)
    a2.metric("Model-ready records", "4,550", border=True)
    a3.metric("Final test records", "960", border=True)
    a4.metric("Forecast horizon", "1 hour", border=True)

    st.subheader("Workflow")
    st.markdown(
        """
        **Raw daily GeoJSON files**  
        ↓  
        **Selected-road hourly aggregation**  
        ↓  
        **Lag, rolling and calendar features**  
        ↓  
        **Random Forest regression pipeline**  
        ↓  
        **Next-hour probe-count forecast**
        """
    )

    st.subheader("Interpretation")
    st.warning(
        "The model predicts aggregated road-level probe activity. "
        "The output should not be interpreted as an exact unique-vehicle count "
        "or a direct congestion percentage."
    )

st.divider()
st.caption(
    "Suyambu Raj · BSc (Hons) Computer Science Final Year Project · "
    "Historical evaluation prototype"
)
