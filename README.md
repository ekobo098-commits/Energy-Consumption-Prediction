# ⚡ Machine Learning Framework for Industrial Energy Consumption Prediction

### AI Decision Support System for Smart Manufacturing

An Industrial Artificial Intelligence application that predicts electrical energy consumption in manufacturing environments using **Random Forest Regression**. The project combines machine learning with an interactive Streamlit dashboard to support industrial energy forecasting, operational analytics, and engineering decision support.

---

## Overview

Industrial facilities continuously consume large amounts of electrical energy during production. Accurately forecasting energy demand enables engineers and plant managers to improve operational planning, monitor electrical performance, and support energy-efficient manufacturing.

This project presents a Machine Learning framework that predicts industrial electrical energy consumption (Usage_kWh) using operational electrical parameters collected from manufacturing environments.

The developed application includes:

- Industrial energy consumption prediction
- Interactive Streamlit dashboard
- Operational prediction ledger
- Historical prediction tracking
- CSV report generation
- Engineering analytics visualization

---

## Project Objectives

- Predict industrial electrical energy consumption
- Analyze operational electrical parameters
- Support manufacturing decision-making
- Maintain historical prediction records
- Visualize prediction trends
- Demonstrate practical AI applications in Smart Manufacturing

---

## Machine Learning Model

Model Used

- Random Forest Regression

Reason for Selection

- Robust performance on tabular industrial datasets
- Handles nonlinear relationships
- Resistant to overfitting
- Provides feature importance analysis
- Well suited for manufacturing data

---

## Dataset

Energy Consumption Dataset

Target Variable

```
Usage_kWh
```

Input Features

- Lagging Current Reactive Power
- Leading Current Reactive Power
- Carbon Emission (CO₂)
- Lagging Current Power Factor
- Leading Current Power Factor
- Seconds from Midnight (NSM)
- Load Type
- Week Status
- Day of Week

---

## Machine Learning Workflow

```
Industrial Dataset

        │

        ▼

Data Cleaning

        │

        ▼

Feature Engineering

        │

        ▼

Train/Test Split

        │

        ▼

Random Forest Regression

        │

        ▼

Model Evaluation

        │

        ▼

Industrial Dashboard

        │

        ▼

Prediction History

        │

        ▼

CSV Reporting
```

---

## Software Features

- Predict industrial energy consumption
- Interactive operational parameter controls
- Real-time prediction engine
- Historical prediction ledger
- CSV export
- Prediction trend visualization
- Engineering dashboard
- Operational analytics

---

## Dashboard

### Industrial Energy Prediction Dashboard
<img width="1920" height="821" alt="energy_con_streamlit1" src="https://github.com/user-attachments/assets/4465a807-cbac-4c62-b430-ec6f4694db0c" />

- Live operational input panel
- Predicted energy demand
- Power factor monitoring
- Operational prediction ledger
- Historical prediction visualization
- CSV export
- Maintenance utilities

---

## Model Performance

| Metric | Result |
|---------|---------|
| Mean Absolute Error (MAE) | **0.336 kWh** |
| Root Mean Squared Error (RMSE) | **1.057 kWh** |
| R² Score | **0.9990** |

The model demonstrates excellent predictive capability for estimating industrial electrical energy consumption based on operational electrical parameters.

---

## Performance Evaluation

The model was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Actual vs Predicted Analysis
- Residual Analysis
- Feature Importance
- Distribution Analysis

---

## Project Structure

```
Industrial-Energy-Prediction/

│

├── app.py

├── model.py

├── Steel_industry_data.csv

├── energy_predictions_ledger.csv

├── steel_energy_rf_model.pkl

├── requirements.txt

├── README.md

└── images/
```

---

## Technologies Used

Programming Language

- Python

Machine Learning

- Scikit-learn
- Random Forest Regression

Data Analysis

- Pandas
- NumPy

Visualization

- Matplotlib
- Seaborn

Web Application

- Streamlit

Model Storage

- Joblib

---

## Practical Industrial Applications

The proposed framework can support decision-making in industries including:

- Steel Manufacturing
- Automotive Manufacturing
- Chemical Processing
- Food Manufacturing
- Heavy Industry
- Industrial Parks
- Energy Management
- Smart Manufacturing

---

## Future Improvements

Potential future enhancements include:

- Live SCADA integration
- OPC-UA communication
- MQTT support
- Industrial IoT integration
- Explainable AI (SHAP)
- Multi-factory deployment
- Cloud deployment
- Digital Twin integration
- Predictive maintenance integration

---

## Results

The developed framework successfully demonstrates the practical application of Machine Learning for industrial electrical energy consumption prediction.

The software integrates predictive analytics with an interactive engineering dashboard, enabling manufacturing personnel to estimate future electrical demand, maintain historical prediction records, and support operational planning.

---

## Research Areas

- Artificial Intelligence
- Industrial AI
- Smart Manufacturing
- Machine Learning
- Predictive Analytics
- Industrial Energy Management
- Industry 4.0
- Energy Consumption Prediction

---

## Author

**Sarthak Salve**

Industrial Artificial Intelligence • Smart Manufacturing • Machine Learning • Predictive Analytics

---

## License

This project is intended for educational, research, and portfolio purposes.
