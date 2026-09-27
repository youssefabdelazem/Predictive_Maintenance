# AI-Powered Predictive Maintenance & Machine Health Monitoring System

An end-to-end Artificial Intelligence system for **predictive maintenance and machine health monitoring** using NASA's C-MAPSS simulated turbofan engine dataset.

The system combines **Machine Learning, Deep Learning, Time-Series Modeling, and Unsupervised Anomaly Detection** to estimate Remaining Useful Life (RUL), classify failure risk, detect abnormal sensor behavior, and provide an interactive machine health monitoring dashboard.

---

## 🚀 Project Overview

Traditional maintenance strategies often rely on fixed maintenance schedules or reactive repairs after a machine failure occurs.

This project aims to build a data-driven predictive maintenance system that continuously analyzes machine sensor data and provides:

* **Remaining Useful Life (RUL) estimation**
* **Failure Risk Classification**
* **Unsupervised Anomaly Detection**
* **Machine Health Score**
* **Maintenance Priority**
* **Interactive Fleet Monitoring Dashboard**
* **Sensor Trend Analysis**
* **Model Performance Monitoring**

The final system transforms raw sensor measurements into actionable machine-health insights.

---

## 🎯 Problem Statement

The project addresses four main predictive maintenance problems:

### 1. Remaining Useful Life Prediction

Estimate how many operational cycles remain before an engine reaches its end of life.

> **Question:**
> "How many cycles does this engine approximately have left?"

### 2. Failure Risk Classification

Convert the estimated RUL into maintenance-oriented risk categories:

|           RUL | Risk Level     |
| ------------: | -------------- |
|   0–20 cycles | 🔴 High Risk   |
|  21–50 cycles | 🟠 Medium Risk |
| 51–100 cycles | 🟢 Low Risk    |

### 3. Anomaly Detection

Identify sensor behavior that significantly differs from learned normal patterns.

This component uses **Isolation Forest** and operates without requiring labeled failure/anomaly data.

> An anomaly does **not** automatically mean that a machine has failed.

### 4. Machine Health Monitoring

Combine anomaly information and machine-level statistics into a relative **Health Score** that can be monitored through the dashboard.

---

# 🧠 System Architecture

```text
                 NASA C-MAPSS Dataset
                         │
                         ▼
              Data Understanding
                         │
                         ▼
                Data Preprocessing
                         │
                         ▼
            Exploratory Data Analysis
                         │
                         ▼
               Feature Engineering
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       RUL Prediction          Anomaly Detection
             │                       │
             ▼                       ▼
       Deep Learning          Isolation Forest
             │                       │
             ▼                       ▼
       GRU Time-Series        Anomaly Score
             │                       │
             ▼                       ▼
      Failure Risk Class.      Health Score
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
              Maintenance Priority
                         │
                         ▼
              Streamlit Dashboard
                         │
                         ▼
              Fleet Monitoring System
```

---

# 📊 Dataset

The project uses the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset.

The current implementation focuses on:

**FD001**

The dataset contains simulated turbofan engine degradation trajectories generated under controlled conditions.

Each engine contains multiple time-series observations recorded over operational cycles.

### Main Data Components

* Engine ID
* Operational Cycle
* Operational settings
* Multiple sensor measurements
* Remaining Useful Life (RUL)

### Dataset Files

```text
data/
└── raw/
    ├── train_FD001.txt
    ├── test_FD001.txt
    └── RUL_FD001.txt
```

---

# 🔍 Data Preprocessing

The preprocessing pipeline includes:

* Dataset loading
* Column naming
* Data inspection
* Missing-value analysis
* Duplicate detection
* Constant-feature removal
* Sensor selection
* RUL calculation
* Feature engineering
* Scaling
* Sequence generation

### Constant Sensors Removed

The following sensors were constant in FD001 and therefore removed:

```text
Sensor_1
Sensor_5
Sensor_10
Sensor_16
Sensor_18
Sensor_19
```

This reduced the sensor set to **15 variable sensors**.

---

# ⚙️ Feature Engineering

To capture machine degradation behavior, several time-series features were created.

### Rolling Features

For selected sensors:

* Rolling Mean
* Rolling Standard Deviation

### Temporal Features

* Sensor Difference
* Trend Deviation
* Cycle information

These features help the models capture not only the current sensor value but also how machine behavior changes over time.

The engineered training dataset reached:

```text
20,631 rows
81 features
```

---

# 🤖 Machine Learning Models

Several regression algorithms were evaluated for RUL prediction.

### Models Evaluated

* Linear Regression
* Random Forest
* XGBoost

Validation results:

| Model             |   MAE |  RMSE |     R² |
| ----------------- | ----: | ----: | -----: |
| Linear Regression | 25.18 | 31.69 | 0.7669 |
| Random Forest     | 23.46 | 31.73 | 0.7664 |
| XGBoost           | 24.25 | 32.39 | 0.7566 |

These models established a traditional Machine Learning baseline before moving to sequence-based Deep Learning.

---

# 🧠 Deep Learning

Because predictive maintenance data is sequential, Deep Learning models were introduced to capture temporal dependencies between machine cycles.

The project evaluated:

* DNN
* LSTM
* GRU

### Validation Comparison

| Model |   MAE |  RMSE |     R² |
| ----- | ----: | ----: | -----: |
| DNN   | 18.12 | 25.48 | 0.8087 |
| LSTM  | 17.33 | 24.35 | 0.8252 |
| GRU   | 15.76 | 22.77 | 0.8472 |

The GRU architecture provided the strongest validation performance among the tested sequence models.

---

# 🏆 Final RUL Model

The final RUL system uses a **GRU-based Deep Learning model** trained on sequences of machine sensor measurements.

A RUL cap of **100 cycles** was applied to focus the prediction task on the maintenance-relevant operating range.

### Final Test Performance

| Metric |          Result |
| ------ | --------------: |
| MAE    | **4.90 cycles** |
| RMSE   | **8.15 cycles** |
| R²     |      **0.8462** |

### Interpretation

The final model predicts the maintenance-relevant remaining life of an engine with an average absolute error of approximately **4.9 cycles** on the test set.

Because RUL was capped at 100 cycles, the model should be interpreted as a **capped maintenance-oriented RUL estimator**, rather than an exact prediction of very large RUL values.

---

# ⚠️ Failure Risk Classification

A separate GRU-based classification model was developed to classify machine sequences into three maintenance-oriented risk levels.

### Risk Categories

```text
Low Risk       → RUL > 50
Medium Risk    → 21 ≤ RUL ≤ 50
High Risk      → RUL ≤ 20
```

Class weights were used during training to address class imbalance.

### Test Performance

| Metric           |     Result |
| ---------------- | ---------: |
| Accuracy         | **96.72%** |
| Macro F1         | **85.95%** |
| High Risk Recall | **87.80%** |

The High Risk recall is particularly useful for monitoring because it measures how many actual high-risk sequences were identified by the classifier.

---

# 🚨 Unsupervised Anomaly Detection

The project also includes an independent anomaly detection layer using:

**Isolation Forest**

Configuration:

```text
Estimators: 200
Contamination: 5%
Random State: 42
```

The model learns patterns from sensor behavior and identifies observations that differ from the learned normal distribution.

### Anomaly Output

Each observation receives:

* Anomaly Prediction
* Anomaly Score
* Anomaly Status

Interpretation:

```text
Normal     → sensor behavior is consistent with learned patterns
Anomaly    → sensor behavior differs from learned patterns
```

An anomaly should be treated as an **early warning signal**, not direct evidence of engine failure.

---

# ❤️ Machine Health Score

A relative Machine Health Score was created from the anomaly score.

The dashboard converts anomaly scores into a normalized:

```text
0 – 100
```

scale.

Higher values indicate more normal sensor behavior relative to the evaluated fleet.

The score is intended for **monitoring and visualization**, not as an absolute engineering measurement of physical engine health.

---

# 🔧 Maintenance Priority

The dashboard includes a rule-based maintenance priority layer.

It considers:

* Failure Risk
* Predicted RUL
* Anomaly Rate
* Health Score

Possible priorities:

```text
Critical
High
Medium
Low
```

This layer is a **decision-support heuristic** designed for the dashboard. It is not a separately trained predictive model and should not be interpreted as a guaranteed maintenance decision.

---

# 📈 Interactive Streamlit Dashboard

The project includes a complete Streamlit dashboard for machine and fleet monitoring.

### Dashboard Features

#### Fleet Overview

Displays:

* Total Engines
* High-Risk Engines
* Medium-Risk Engines
* Low-Risk Engines
* Average Health Score

#### Engine Monitoring

Users can select an individual engine and inspect:

* Predicted RUL
* Failure Risk
* Health Score
* Anomaly Status
* Latest Cycle
* Anomaly Rate
* Maintenance Priority

#### Sensor Analysis

The dashboard provides:

* Sensor trend visualization
* Anomaly score trend
* Sensor statistics
* Engine-level comparisons

#### Fleet Monitoring

Users can:

* Search engines
* Filter by risk
* Sort by RUL
* Compare machine health
* Download the monitoring report

---

# 🔎 Explainability

The dashboard includes an explainability layer that highlights sensors showing relatively large changes throughout the observed engine trajectory.

The system analyzes:

* Sensor variation
* Temporal change
* Relative deviation

and identifies sensors that may deserve further investigation.

This is intended as an **interpretability aid**, not causal analysis.

A high sensor change does not prove that the sensor caused degradation or failure.

---

# 📁 Project Structure

```text
Predictive_Maintenance/
│
├── data/
│   ├── raw/
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   └── RUL_FD001.txt
│   │
│   └── processed/
│       └── engine_health_summary.csv
│
├── notebooks/
│   └── 01_Data_Understanding_and_Preprocessing.ipynb
│
├── models/
│   ├── final_rul_gru_cap100.keras
│   ├── failure_risk_gru.keras
│   ├── dl_scaler.pkl
│   ├── anomaly_scaler.pkl
│   └── isolation_forest.pkl
│
├── src/
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

---

# 🛠️ Technologies Used

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Deep Learning

* TensorFlow
* Keras

### Deployment / Dashboard

* Streamlit

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Predictive_Maintenance.git
```

Navigate to the project:

```bash
cd Predictive_Maintenance
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

If the `streamlit` command is not recognized on Windows, use:

```powershell
python -m streamlit run app.py
```

The application will open in your browser.

---

# 🧪 Model Development Workflow

The complete workflow used in the project is:

```text
1. Load Dataset
        ↓
2. Understand Data Structure
        ↓
3. Clean Data
        ↓
4. Remove Constant Sensors
        ↓
5. Exploratory Data Analysis
        ↓
6. Generate RUL
        ↓
7. Feature Engineering
        ↓
8. Engine-Level Train/Validation Split
        ↓
9. Train ML Baselines
        ↓
10. Train DNN
        ↓
11. Train LSTM
        ↓
12. Train GRU
        ↓
13. RUL Capping
        ↓
14. Final GRU Training
        ↓
15. Failure Risk Classification
        ↓
16. Isolation Forest
        ↓
17. Health Score
        ↓
18. Explainability
        ↓
19. Streamlit Dashboard
```

---

# 🔐 Avoiding Data Leakage

Because each engine contains a sequence of observations, randomly splitting individual rows could cause information from the same engine trajectory to appear in both training and validation sets.

To reduce trajectory leakage, the project performs the main split at the **Engine ID level**.

This allows complete engine trajectories to remain separated between training and validation.

---

# 📊 Model Artifacts

The trained models and preprocessing objects are saved for later inference.

```text
models/
├── final_rul_gru_cap100.keras
├── failure_risk_gru.keras
├── dl_scaler.pkl
├── anomaly_scaler.pkl
└── isolation_forest.pkl
```

This allows the Streamlit application to load the trained system without retraining the models every time.

---

# 🚀 Future Improvements

The project can be extended significantly.

### Dataset Expansion

Move from:

```text
FD001
```

to:

```text
FD004
```

FD004 introduces substantially more complex operating conditions and fault scenarios.

### Advanced Deep Learning

Potential future architectures:

* Bidirectional LSTM
* Bidirectional GRU
* CNN-LSTM
* Temporal Convolutional Networks
* Transformer-based Time-Series Models

### Advanced Anomaly Detection

Future versions can include:

* Autoencoders
* LSTM Autoencoders
* Variational Autoencoders
* Deep SVDD

### Explainable AI

Possible additions:

* SHAP
* Integrated Gradients
* Feature attribution
* Temporal attention visualization

### Production Monitoring

Future deployment could include:

* Real-time sensor streaming
* Database integration
* Automated alerts
* Maintenance logs
* Model monitoring
* Data drift detection
* Cloud deployment

---

# ⚠️ Limitations

This project uses the **FD001 subset** of NASA C-MAPSS.

Therefore:

* The data is simulated rather than collected from real operational aircraft.
* FD001 represents a simplified operating environment.
* The dataset does not provide a direct real-world failure label for every observation.
* Anomaly detection identifies unusual sensor behavior, not confirmed failures.
* The Health Score is relative to the evaluated dataset.
* Maintenance Priority is a rule-based decision-support layer.
* The final RUL model uses a 100-cycle cap.

These limitations should be considered before applying the system to real industrial equipment.

---

# 📌 Key Results

The final system combines multiple AI techniques into one predictive-maintenance pipeline:

| Component           | Technique                | Purpose                             |
| ------------------- | ------------------------ | ----------------------------------- |
| RUL Prediction      | GRU                      | Estimate remaining useful life      |
| Risk Classification | GRU                      | Classify maintenance risk           |
| Anomaly Detection   | Isolation Forest         | Detect unusual sensor behavior      |
| Health Monitoring   | Normalized anomaly score | Relative machine health             |
| Explainability      | Sensor-change analysis   | Highlight important sensor behavior |
| Dashboard           | Streamlit                | Interactive fleet monitoring        |

---

# 💡 What This Project Demonstrates

This project demonstrates practical experience with:

* Time-series data
* Predictive maintenance
* Regression
* Classification
* Unsupervised learning
* Feature engineering
* Deep Learning
* GRU and LSTM architectures
* Model evaluation
* Class imbalance
* Anomaly detection
* Data leakage prevention
* Model persistence
* Explainability
* Interactive dashboards
* End-to-end ML system development

---

# 👨‍💻 Author

**Youssef Mohamed Abdelazem**

Computer Science Student
Machine Learning & Deep Learning Enthusiast

---

# ⭐ Project Goal

The ultimate goal is to transform machine sensor data into a complete AI-driven monitoring system that can move from:

```text
Raw Sensor Data
        ↓
Machine Understanding
        ↓
Prediction
        ↓
Risk Detection
        ↓
Anomaly Detection
        ↓
Health Monitoring
        ↓
Maintenance Decision Support
```

into a practical and deployable **Predictive Maintenance Platform**.
