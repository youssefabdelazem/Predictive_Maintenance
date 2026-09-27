# =========================================================
# 155 — Complete Streamlit Predictive Maintenance Dashboard
# AI-Powered Predictive Maintenance & Machine Health Monitoring
# =========================================================


# =========================================================
# 1. Import Libraries
# =========================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model


# =========================================================
# 2. Page Configuration
# =========================================================

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 3. Custom Dashboard Styling
# =========================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
    }

    h2 {
        font-size: 1.8rem !important;
        font-weight: 650 !important;
    }

    h3 {
        font-weight: 600 !important;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 15px;
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.05);
    }

    div[data-testid="stMetricValue"] {
        font-weight: 700;
    }

    .dashboard-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        padding: 20px;
        opacity: 0.65;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 4. Define Project Paths
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

RAW_DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)


# =========================================================
# 5. Load Saved Models
# =========================================================

@st.cache_resource
def load_models():

    rul_model = load_model(
        os.path.join(
            MODELS_DIR,
            "final_rul_gru_cap100.keras"
        )
    )

    risk_model = load_model(
        os.path.join(
            MODELS_DIR,
            "failure_risk_gru.keras"
        )
    )

    dl_scaler = joblib.load(
        os.path.join(
            MODELS_DIR,
            "dl_scaler.pkl"
        )
    )

    anomaly_scaler = joblib.load(
        os.path.join(
            MODELS_DIR,
            "anomaly_scaler.pkl"
        )
    )

    isolation_forest = joblib.load(
        os.path.join(
            MODELS_DIR,
            "isolation_forest.pkl"
        )
    )

    return (
        rul_model,
        risk_model,
        dl_scaler,
        anomaly_scaler,
        isolation_forest
    )


# =========================================================
# 6. Load Engine Summary
# =========================================================

@st.cache_data
def load_engine_summary():

    return pd.read_csv(
        os.path.join(
            DATA_DIR,
            "engine_health_summary.csv"
        )
    )


# =========================================================
# 7. Load Test Sensor Data
# =========================================================

@st.cache_data
def load_test_data():

    test_path = os.path.join(
        RAW_DATA_DIR,
        "test_FD001.txt"
    )

    columns = [
        "Engine_ID",
        "Cycle",
        "Setting_1",
        "Setting_2",
        "Setting_3",
        "Sensor_1",
        "Sensor_2",
        "Sensor_3",
        "Sensor_4",
        "Sensor_5",
        "Sensor_6",
        "Sensor_7",
        "Sensor_8",
        "Sensor_9",
        "Sensor_10",
        "Sensor_11",
        "Sensor_12",
        "Sensor_13",
        "Sensor_14",
        "Sensor_15",
        "Sensor_16",
        "Sensor_17",
        "Sensor_18",
        "Sensor_19",
        "Sensor_20",
        "Sensor_21"
    ]

    return pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        names=columns
    )


# =========================================================
# 8. Load All Required Data
# =========================================================

(
    rul_model,
    risk_model,
    dl_scaler,
    anomaly_scaler,
    isolation_forest
) = load_models()

engine_summary = load_engine_summary()

test_data = load_test_data()


# =========================================================
# 9. Sidebar
# =========================================================

with st.sidebar:

    st.title(
        "⚙️ Predictive Maintenance"
    )

    st.markdown(
        """
        ### Navigation

        **Machine Monitoring**
        - Fleet Overview
        - Engine Status
        - Maintenance Priority
        - Sensor Analysis

        **AI Insights**
        - Explainability
        - Model Performance

        **Reports**
        - Fleet Monitoring Report
        """
    )

    st.divider()

    st.info(
        "NASA C-MAPSS FD001\n\n"
        "GRU + Isolation Forest"
    )


# =========================================================
# 10. Dashboard Header
# =========================================================

st.title(
    "⚙️ AI-Powered Predictive Maintenance"
)

st.markdown(
    """
    ### Machine Health Monitoring System

    An AI-based monitoring platform that combines
    **Deep Learning, supervised classification, and
    unsupervised anomaly detection** to monitor machine health.
    """
)


# =========================================================
# 11. Engine Selection
# =========================================================

engine_ids = sorted(
    engine_summary["Engine_ID"].unique()
)

selected_engine = st.selectbox(
    "🔧 Select Engine",
    engine_ids
)


# =========================================================
# 12. Fleet Overview
# =========================================================

st.divider()

st.subheader(
    "🚀 Fleet Overview"
)

total_engines = len(
    engine_summary
)

high_risk_count = (
    engine_summary["Risk_Level"]
    == "High Risk"
).sum()

medium_risk_count = (
    engine_summary["Risk_Level"]
    == "Medium Risk"
).sum()

low_risk_count = (
    engine_summary["Risk_Level"]
    == "Low Risk"
).sum()

anomaly_count = (
    engine_summary["Anomaly_Status"]
    == "Anomaly"
).sum()


fleet_col1, fleet_col2, fleet_col3, fleet_col4 = st.columns(4)


with fleet_col1:

    st.metric(
        "Total Engines",
        total_engines
    )


with fleet_col2:

    st.metric(
        "High Risk Engines",
        high_risk_count
    )


with fleet_col3:

    st.metric(
        "Medium Risk Engines",
        medium_risk_count
    )


with fleet_col4:

    st.metric(
        "Anomalous Engines",
        anomaly_count
    )


# =========================================================
# 13. Fleet Risk Distribution
# =========================================================

st.subheader(
    "⚠️ Fleet Risk Distribution"
)

risk_distribution = (
    engine_summary["Risk_Level"]
    .value_counts()
    .reindex(
        [
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ],
        fill_value=0
    )
)

st.bar_chart(
    risk_distribution
)


# =========================================================
# 14. Fleet Analytics
# =========================================================

st.subheader(
    "📊 Fleet Analytics"
)

analytics_col1, analytics_col2 = st.columns(2)


with analytics_col1:

    st.markdown(
        "#### ❤️ Health Score Distribution"
    )

    health_distribution = (
        engine_summary[
            "Health_Score"
        ]
        .round()
        .value_counts()
        .sort_index()
    )

    st.line_chart(
        health_distribution
    )


with analytics_col2:

    st.markdown(
        "#### 🔍 Top 10 Engines by Anomaly Rate"
    )

    top_anomaly_engines = (
        engine_summary[
            [
                "Engine_ID",
                "Anomaly_Rate"
            ]
        ]
        .sort_values(
            "Anomaly_Rate",
            ascending=False
        )
        .head(10)
        .set_index("Engine_ID")
    )

    top_anomaly_engines[
        "Anomaly_Rate"
    ] = (
        top_anomaly_engines[
            "Anomaly_Rate"
        ] * 100
    )

    st.bar_chart(
        top_anomaly_engines
    )


# =========================================================
# 15. Get Selected Engine Information
# =========================================================

engine_data = engine_summary[
    engine_summary["Engine_ID"]
    == selected_engine
].iloc[0]


predicted_rul = engine_data[
    "Predicted_RUL"
]

risk_level = engine_data[
    "Risk_Level"
]

health_score = engine_data[
    "Health_Score"
]

anomaly_status = engine_data[
    "Anomaly_Status"
]

anomaly_rate = engine_data[
    "Anomaly_Rate"
]

latest_cycle = engine_data[
    "Latest_Cycle"
]


# =========================================================
# 16. Main Engine KPIs
# =========================================================

st.divider()

st.subheader(
    f"🔧 Engine {selected_engine} Current Status"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Predicted RUL",
        f"{predicted_rul:.1f} cycles"
    )


with col2:

    st.metric(
        "Failure Risk",
        risk_level
    )


with col3:

    st.metric(
        "Health Score",
        f"{health_score:.1f} / 100"
    )


with col4:

    st.metric(
        "Anomaly Status",
        anomaly_status
    )


# =========================================================
# 17. Engine Information
# =========================================================

st.divider()

st.subheader(
    "📋 Engine Information"
)

info_col1, info_col2, info_col3 = st.columns(3)


with info_col1:

    st.write(
        f"**Engine ID:** {selected_engine}"
    )


with info_col2:

    st.write(
        f"**Latest Cycle:** "
        f"{int(latest_cycle)}"
    )


with info_col3:

    st.write(
        f"**Anomaly Rate:** "
        f"{anomaly_rate:.2%}"
    )


# =========================================================
# 18. Maintenance Priority
# =========================================================

st.divider()

st.subheader(
    "🚨 Maintenance Priority"
)


def calculate_maintenance_priority(
    risk_level,
    anomaly_rate,
    health_score,
    predicted_rul
):

    if (
        risk_level == "High Risk"
        and anomaly_rate > 0
        and health_score < 40
    ):
        return "Critical"

    elif (
        risk_level == "High Risk"
        or predicted_rul <= 20
    ):
        return "High"

    elif (
        risk_level == "Medium Risk"
        and (
            anomaly_rate > 2
            or health_score < 60
        )
    ):
        return "High"

    elif risk_level == "Medium Risk":
        return "Medium"

    elif (
        anomaly_rate > 5
        or health_score < 50
    ):
        return "Medium"

    else:

        return "Low"


maintenance_priority = calculate_maintenance_priority(
    risk_level=risk_level,
    anomaly_rate=anomaly_rate * 100,
    health_score=health_score,
    predicted_rul=predicted_rul
)


priority_col1, priority_col2, priority_col3 = st.columns(3)


with priority_col1:

    st.metric(
        "Maintenance Priority",
        maintenance_priority
    )


with priority_col2:

    st.metric(
        "Predicted RUL",
        f"{predicted_rul:.1f} cycles"
    )


with priority_col3:

    st.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.2%}"
    )


if maintenance_priority == "Critical":

    st.error(
        "🚨 CRITICAL: Immediate maintenance inspection "
        "is recommended based on the combined monitoring indicators."
    )

elif maintenance_priority == "High":

    st.warning(
        "⚠️ HIGH PRIORITY: Maintenance inspection "
        "should be scheduled based on the current monitoring indicators."
    )

elif maintenance_priority == "Medium":

    st.warning(
        "🟡 MEDIUM PRIORITY: Continue close monitoring "
        "and plan maintenance activities."
    )

else:

    st.success(
        "🟢 LOW PRIORITY: Continue routine monitoring."
    )


st.caption(
    "Maintenance Priority is a dashboard decision-support heuristic "
    "based on RUL, risk, health score, and anomaly rate."
)


# =========================================================
# 19. Health Interpretation
# =========================================================

st.divider()

st.subheader(
    "📊 Health Interpretation"
)


if risk_level == "High Risk":

    st.error(
        "High failure-risk level detected. "
        "This engine requires close attention."
    )

elif risk_level == "Medium Risk":

    st.warning(
        "Medium failure-risk level detected. "
        "This engine should be monitored closely."
    )

else:

    st.success(
        "Low failure-risk level detected. "
        "The engine currently shows a lower predicted risk."
    )


if anomaly_status == "Anomaly":

    st.warning(
        "Unusual sensor behavior has been detected "
        "by the anomaly detection model."
    )

else:

    st.info(
        "No significant anomalous sensor behavior "
        "was detected at the latest observation."
    )


# =========================================================
# 20. Maintenance Recommendation
# =========================================================

st.subheader(
    "💡 Maintenance Recommendation"
)


if maintenance_priority == "Critical":

    recommendation = (
        "Perform a detailed inspection of the engine and "
        "investigate abnormal sensor behavior. "
        "The current indicators suggest elevated maintenance priority."
    )

elif maintenance_priority == "High":

    recommendation = (
        "Schedule a maintenance inspection and continue "
        "monitoring the engine's RUL, risk level, and sensor behavior."
    )

elif maintenance_priority == "Medium":

    recommendation = (
        "Continue enhanced monitoring and consider planning "
        "maintenance before the predicted RUL enters the high-risk range."
    )

else:

    recommendation = (
        "Continue routine monitoring. The current model indicators "
        "do not show a high maintenance priority."
    )


st.info(
    recommendation
)

st.caption(
    "Recommendation is generated from model outputs and dashboard rules; "
    "it is not a certified engineering maintenance decision."
)


# =========================================================
# 21. Engine Comparison
# =========================================================

st.divider()

st.subheader(
    "🔄 Engine Comparison"
)

comparison_engine = st.selectbox(
    "Compare Selected Engine With",
    [
        engine_id
        for engine_id in engine_ids
        if engine_id != selected_engine
    ]
)


engine_a = engine_summary[
    engine_summary["Engine_ID"]
    == selected_engine
].iloc[0]


engine_b = engine_summary[
    engine_summary["Engine_ID"]
    == comparison_engine
].iloc[0]


comparison_table = pd.DataFrame(
    {
        f"Engine {selected_engine}": [
            engine_a["Predicted_RUL"],
            engine_a["Health_Score"],
            engine_a["Anomaly_Rate"] * 100,
            engine_a["Risk_Level"],
            engine_a["Anomaly_Status"]
        ],

        f"Engine {comparison_engine}": [
            engine_b["Predicted_RUL"],
            engine_b["Health_Score"],
            engine_b["Anomaly_Rate"] * 100,
            engine_b["Risk_Level"],
            engine_b["Anomaly_Status"]
        ]
    },

    index=[
        "Predicted RUL (cycles)",
        "Health Score",
        "Anomaly Rate (%)",
        "Risk Level",
        "Anomaly Status"
    ]
)


st.dataframe(
    comparison_table,
    use_container_width=True
)


# =========================================================
# 22. Fleet Monitoring Table
# =========================================================

st.divider()

st.subheader(
    "🏭 Fleet Monitoring"
)

fleet_table = engine_summary[
    [
        "Engine_ID",
        "Predicted_RUL",
        "Risk_Level",
        "Health_Score",
        "Anomaly_Status",
        "Anomaly_Rate",
        "Latest_Cycle"
    ]
].copy()


fleet_table = fleet_table.rename(
    columns={
        "Engine_ID": "Engine",
        "Predicted_RUL": "Predicted RUL",
        "Risk_Level": "Risk Level",
        "Health_Score": "Health Score",
        "Anomaly_Status": "Anomaly Status",
        "Anomaly_Rate": "Anomaly Rate",
        "Latest_Cycle": "Latest Cycle"
    }
)


# =========================================================
# 23. Fleet Search and Filtering
# =========================================================

search_engine = st.text_input(
    "🔎 Search Engine",
    placeholder="Enter Engine ID..."
)


risk_filter = st.multiselect(
    "⚠️ Filter by Risk Level",
    options=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ],
    default=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)


sort_option = st.selectbox(
    "📊 Sort Engines By",
    options=[
        "Engine ID",
        "Predicted RUL",
        "Health Score",
        "Anomaly Rate"
    ]
)


filtered_fleet = fleet_table[
    fleet_table["Risk Level"].isin(
        risk_filter
    )
].copy()


if search_engine:

    filtered_fleet = filtered_fleet[
        filtered_fleet["Engine"]
        .astype(str)
        .str.contains(
            search_engine,
            case=False,
            na=False
        )
    ]


# =========================================================
# 24. Sort Fleet Table
# =========================================================

sort_mapping = {

    "Engine ID": "Engine",

    "Predicted RUL": "Predicted RUL",

    "Health Score": "Health Score",

    "Anomaly Rate": "Anomaly Rate"

}


filtered_fleet = filtered_fleet.sort_values(
    by=sort_mapping[sort_option],
    ascending=True
)


# =========================================================
# 25. Format Fleet Table
# =========================================================

filtered_fleet[
    "Predicted RUL"
] = filtered_fleet[
    "Predicted RUL"
].round(1)


filtered_fleet[
    "Health Score"
] = filtered_fleet[
    "Health Score"
].round(1)


filtered_fleet[
    "Anomaly Rate"
] = (
    filtered_fleet[
        "Anomaly Rate"
    ] * 100
).round(2)


# =========================================================
# 26. Display Fleet Table
# =========================================================

st.dataframe(
    filtered_fleet,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 27. Download Fleet Report
# =========================================================

st.subheader(
    "📥 Fleet Report"
)


download_data = fleet_table.copy()

download_data[
    "Predicted RUL"
] = download_data[
    "Predicted RUL"
].round(2)

download_data[
    "Health Score"
] = download_data[
    "Health Score"
].round(2)

download_data[
    "Anomaly Rate"
] = (
    download_data[
        "Anomaly Rate"
    ] * 100
).round(2)


csv_data = download_data.to_csv(
    index=False
).encode(
    "utf-8"
)


st.download_button(
    label="📥 Download Fleet Monitoring Report",
    data=csv_data,
    file_name="fleet_monitoring_report.csv",
    mime="text/csv"
)


# =========================================================
# 28. Sensor Trend Analysis
# =========================================================

st.divider()

st.subheader(
    "📈 Sensor Trend Analysis"
)


engine_sensor_data = (
    test_data[
        test_data["Engine_ID"]
        == selected_engine
    ]
    .sort_values("Cycle")
    .copy()
)


available_sensors = [
    f"Sensor_{i}"
    for i in range(1, 22)
]


selected_sensor = st.selectbox(
    "Select Sensor",
    available_sensors
)


st.line_chart(
    engine_sensor_data.set_index(
        "Cycle"
    )[
        selected_sensor
    ]
)


# =========================================================
# 29. Anomaly Score Trend
# =========================================================

st.subheader(
    "🔍 Anomaly Score Trend"
)


engine_anomaly_data = (
    test_data[
        test_data["Engine_ID"]
        == selected_engine
    ]
    .sort_values("Cycle")
    .copy()
)


sensor_cols = [
    f"Sensor_{i}"
    for i in range(1, 22)
]


constant_sensors = [
    "Sensor_1",
    "Sensor_5",
    "Sensor_10",
    "Sensor_16",
    "Sensor_18",
    "Sensor_19"
]


variable_sensors = [
    sensor
    for sensor in sensor_cols
    if sensor not in constant_sensors
]


selected_anomaly_scaled = anomaly_scaler.transform(
    engine_anomaly_data[
        variable_sensors
    ]
)


selected_anomaly_scores = (
    isolation_forest.decision_function(
        selected_anomaly_scaled
    )
)


anomaly_chart_data = pd.DataFrame(
    {
        "Anomaly_Score": selected_anomaly_scores
    },
    index=engine_anomaly_data[
        "Cycle"
    ]
)


st.line_chart(
    anomaly_chart_data
)


st.caption(
    "Higher anomaly scores indicate more normal sensor behavior, "
    "while lower scores indicate more anomalous behavior."
)


# =========================================================
# 30. Sensor Statistics
# =========================================================

st.subheader(
    "📡 Sensor Statistics"
)


sensor_statistics = (
    engine_sensor_data[
        variable_sensors
    ]
    .describe()
    .T[
        [
            "mean",
            "std",
            "min",
            "max"
        ]
    ]
)


display_columns = {

    "mean": "Mean",

    "std": "Std",

    "min": "Minimum",

    "max": "Maximum"

}


sensor_statistics = sensor_statistics.rename(
    columns=display_columns
)


st.dataframe(
    sensor_statistics,
    use_container_width=True
)


# =========================================================
# 31. Model Explainability
# =========================================================

st.divider()

st.subheader(
    "💡 Model Explainability"
)

st.markdown(
    """
    This section highlights sensor signals that show relatively
    large changes between earlier and recent observations for
    the selected engine.
    """
)


# =========================================================
# 31.1 Calculate Sensor Change Scores
# =========================================================

explainability_data = engine_sensor_data[
    variable_sensors
].copy()


sensor_change_scores = {}


for sensor in variable_sensors:

    sensor_values = (
        explainability_data[
            sensor
        ]
        .dropna()
        .values
    )

    if len(sensor_values) > 1:

        quarter_size = max(
            1,
            len(sensor_values) // 4
        )

        early_mean = np.mean(
            sensor_values[
                :quarter_size
            ]
        )

        recent_mean = np.mean(
            sensor_values[
                -quarter_size:
            ]
        )

        overall_std = np.std(
            sensor_values
        )

        if overall_std > 0:

            change_score = abs(
                recent_mean - early_mean
            ) / overall_std

        else:

            change_score = 0

        sensor_change_scores[
            sensor
        ] = change_score


# =========================================================
# 31.2 Rank Sensors
# =========================================================

sensor_importance = (
    pd.DataFrame(
        list(
            sensor_change_scores.items()
        ),
        columns=[
            "Sensor",
            "Change_Score"
        ]
    )
    .sort_values(
        "Change_Score",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


sensor_importance[
    "Change_Score"
] = sensor_importance[
    "Change_Score"
].round(3)


# =========================================================
# 31.3 Display Explainability Results
# =========================================================

explain_col1, explain_col2 = st.columns(2)


with explain_col1:

    st.markdown(
        "#### 🔎 Most Changing Sensors"
    )

    st.dataframe(
        sensor_importance.head(5),
        use_container_width=True,
        hide_index=True
    )


with explain_col2:

    st.markdown(
        "#### 📊 Sensor Change Ranking"
    )

    chart_data = (
        sensor_importance
        .head(10)
        .set_index(
            "Sensor"
        )[
            "Change_Score"
        ]
    )

    st.bar_chart(
        chart_data
    )


# =========================================================
# 31.4 Explain Current Engine Condition
# =========================================================

st.markdown(
    "#### 🧠 Current Engine Explanation"
)


top_sensors = (
    sensor_importance
    .head(3)
    [
        "Sensor"
    ]
    .tolist()
)


if maintenance_priority == "Critical":

    explanation_text = (
        f"Engine {selected_engine} currently has a "
        f"Critical maintenance priority. "
        f"The strongest relative sensor changes are observed in "
        f"{', '.join(top_sensors)}. "
        f"These signals should be investigated together with "
        f"the RUL, risk, health, and anomaly indicators."
    )

elif maintenance_priority == "High":

    explanation_text = (
        f"Engine {selected_engine} currently has a "
        f"High maintenance priority. "
        f"The sensors showing the largest relative changes are "
        f"{', '.join(top_sensors)}. "
        f"These signals should be monitored alongside the "
        f"predicted RUL and failure-risk classification."
    )

elif maintenance_priority == "Medium":

    explanation_text = (
        f"Engine {selected_engine} currently has a "
        f"Medium maintenance priority. "
        f"The largest relative sensor changes are observed in "
        f"{', '.join(top_sensors)}. "
        f"Continued monitoring of these signals may help "
        f"identify changes in machine behavior."
    )

else:

    explanation_text = (
        f"Engine {selected_engine} currently has a "
        f"Low maintenance priority. "
        f"The sensors with the largest relative changes are "
        f"{', '.join(top_sensors)}. "
        f"Current model indicators should continue to be "
        f"monitored during normal operation."
    )


st.info(
    explanation_text
)


st.caption(
    "Sensor change ranking is an interpretability aid based on "
    "relative changes within the selected engine. It does not "
    "establish that a specific sensor caused a failure."
)


# =========================================================
# 32. Final Model Performance
# =========================================================

st.divider()

st.subheader(
    "🎯 Final Model Performance"
)

st.markdown(
    """
    The following metrics summarize the final models selected
    during the training and evaluation stages of the project.
    """
)


# =========================================================
# 32.1 RUL Regression Performance
# =========================================================

st.markdown(
    "#### 🤖 RUL Prediction — GRU"
)

rul_col1, rul_col2, rul_col3 = st.columns(3)


with rul_col1:

    st.metric(
        "Test MAE",
        "4.90 cycles"
    )


with rul_col2:

    st.metric(
        "Test RMSE",
        "8.15 cycles"
    )


with rul_col3:

    st.metric(
        "Test R²",
        "0.8462"
    )


st.caption(
    "Final RUL model: GRU with maintenance-relevant RUL capped at 100 cycles."
)


# =========================================================
# 32.2 Risk Classification Performance
# =========================================================

st.markdown(
    "#### ⚠️ Failure Risk Classification — GRU"
)

risk_col1, risk_col2, risk_col3 = st.columns(3)


with risk_col1:

    st.metric(
        "Test Accuracy",
        "96.72%"
    )


with risk_col2:

    st.metric(
        "Macro F1",
        "85.95%"
    )


with risk_col3:

    st.metric(
        "High Risk Recall",
        "87.80%"
    )


risk_performance = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Macro F1",
            "High Risk Recall"
        ],

        "Score": [
            0.9672,
            0.8595,
            0.8780
        ]
    }
)


risk_performance[
    "Score"
] = (
    risk_performance[
        "Score"
    ] * 100
).round(2)


st.dataframe(
    risk_performance,
    use_container_width=True,
    hide_index=True
)


st.caption(
    "High Risk Recall indicates the proportion of actual high-risk "
    "test sequences correctly identified by the classifier."
)


# =========================================================
# 32.3 Model Comparison
# =========================================================

st.markdown(
    "#### 📈 RUL Model Comparison"
)


model_comparison = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Random Forest",
            "XGBoost",
            "DNN",
            "LSTM",
            "GRU"
        ],

        "MAE": [
            25.1809,
            23.4580,
            24.2519,
            18.1155,
            17.3256,
            15.7597
        ],

        "RMSE": [
            31.6945,
            31.7311,
            32.3899,
            25.4758,
            24.3495,
            22.7694
        ],

        "R2": [
            0.7669,
            0.7664,
            0.7566,
            0.8087,
            0.8252,
            0.8472
        ]
    }
)


display_model_comparison = model_comparison.copy()

display_model_comparison[
    "MAE"
] = display_model_comparison[
    "MAE"
].round(2)

display_model_comparison[
    "RMSE"
] = display_model_comparison[
    "RMSE"
].round(2)

display_model_comparison[
    "R2"
] = display_model_comparison[
    "R2"
].round(4)


st.dataframe(
    display_model_comparison,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 33. AI System Components
# =========================================================

st.divider()

st.subheader(
    "🧠 AI System Components"
)


component_col1, component_col2 = st.columns(2)


with component_col1:

    st.markdown(
        """
        **Supervised / Deep Learning**

        - GRU-based RUL prediction
        - Failure risk classification
        - Engine-level health monitoring
        - Maintenance priority analysis
        - Sensor trend analysis
        """
    )


with component_col2:

    st.markdown(
        """
        **Unsupervised Learning**

        - Isolation Forest
        - Sensor anomaly detection
        - Anomaly rate analysis
        - Fleet-level anomaly monitoring
        - Relative machine health scoring
        """
    )


# =========================================================
# 34. System Notes
# =========================================================

st.divider()

st.subheader(
    "ℹ️ System Notes"
)

st.markdown(
    """
    **RUL Prediction**

    The final GRU model predicts maintenance-relevant Remaining
    Useful Life with an upper cap of 100 cycles.

    **Failure Risk**

    Risk levels are derived from RUL thresholds:

    - High Risk: 0–20 cycles
    - Medium Risk: 21–50 cycles
    - Low Risk: above 50 cycles

    **Anomaly Detection**

    Isolation Forest identifies sensor observations that differ
    from learned normal patterns.

    **Health Score**

    The current health score is a relative indicator derived from
    anomaly scores within the evaluated fleet.

    **Explainability**

    Sensor ranking highlights relative changes between earlier
    and recent observations. It does not prove that a specific
    sensor caused a failure.

    **Important**

    An anomaly does not necessarily mean engine failure.
    The system combines multiple indicators to support maintenance
    monitoring and decision-making.
    """
)


# =========================================================
# 35. Project Footer
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">

    <strong>
    AI-Powered Predictive Maintenance & Machine Health Monitoring
    </strong>

    <br>

    NASA C-MAPSS FD001 | GRU | Isolation Forest | Streamlit

    </div>
    """,
    unsafe_allow_html=True
)