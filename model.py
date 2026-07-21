# ==========================================================
# STEEL ENERGY CONSUMPTION PREDICTION
# RANDOM FOREST REGRESSOR
# ==========================================================

import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("Steel_industry_data.csv")

# ==========================================================
# DEFINE TARGET
# ==========================================================

target = "Usage_kWh"

X = df.drop(columns=[target])

y = df[target]

# ==========================================================
# REMOVE DATE COLUMN
# ==========================================================

X = X.drop(columns=["date"], errors="ignore")

# ==========================================================
# FEATURE TYPES
# ==========================================================

numeric_features = X.select_dtypes(include=np.number).columns

categorical_features = X.select_dtypes(include="object").columns

# ==========================================================
# NUMERIC PIPELINE
# ==========================================================

numeric_pipeline = Pipeline(

    steps=[

        ("imputer", SimpleImputer(strategy="median")),

        ("scaler", StandardScaler())

    ]

)

# ==========================================================
# CATEGORICAL PIPELINE
# ==========================================================

categorical_pipeline = Pipeline(

    steps=[

        ("imputer", SimpleImputer(strategy="most_frequent")),

        ("encoder", OneHotEncoder(handle_unknown="ignore"))

    ]

)

# ==========================================================
# PREPROCESSOR
# ==========================================================

preprocessor = ColumnTransformer(

    transformers=[

        ("num", numeric_pipeline, numeric_features),

        ("cat", categorical_pipeline, categorical_features)

    ]

)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

# ==========================================================
# RANDOM FOREST PIPELINE
# ==========================================================

model = Pipeline(

    steps=[

        ("preprocessor", preprocessor),

        ("regressor",

            RandomForestRegressor(

                n_estimators=300,

                random_state=42,

                n_jobs=-1

            )

        )

    ]

)

# ==========================================================
# TRAIN MODEL
# ==========================================================

print("Training Random Forest...")

model.fit(

    X_train,

    y_train

)

print("Training Complete.")

# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = model.predict(

    X_test

)

# ==========================================================
# EVALUATION
# ==========================================================

mae = mean_absolute_error(

    y_test,

    predictions

)

rmse = np.sqrt(

    mean_squared_error(

        y_test,

        predictions

    )

)

r2 = r2_score(

    y_test,

    predictions

)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"MAE  : {mae:.4f}")

print(f"RMSE : {rmse:.4f}")

print(f"R²   : {r2:.4f}")

# ==========================================================
# ACTUAL VS PREDICTED
# ==========================================================

plt.figure(figsize=(7,7))

plt.scatter(

    y_test,

    predictions,

    alpha=0.5

)

plt.plot(

    [y_test.min(), y_test.max()],

    [y_test.min(), y_test.max()],

    color="red",

    linewidth=2

)

plt.xlabel("Actual Usage_kWh")

plt.ylabel("Predicted Usage_kWh")

plt.title("Actual vs Predicted")

plt.show()

# ==========================================================
# RESIDUAL PLOT
# ==========================================================

residuals = y_test - predictions

plt.figure(figsize=(8,5))

sns.scatterplot(

    x=predictions,

    y=residuals

)

plt.axhline(

    y=0,

    color="red"

)

plt.xlabel("Predicted")

plt.ylabel("Residual")

plt.title("Residual Plot")

plt.show()

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

feature_names = model.named_steps["preprocessor"].get_feature_names_out()

importance = model.named_steps["regressor"].feature_importances_

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)

print("\nTop 20 Important Features")

print(importance_df.head(20))

# ==========================================================
# FEATURE IMPORTANCE PLOT
# ==========================================================

plt.figure(figsize=(10,8))

sns.barplot(

    data=importance_df.head(20),

    x="Importance",

    y="Feature"

)

plt.title("Top 20 Feature Importance")

plt.show()

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(

    model,

    "steel_energy_rf_model.pkl"

)

print("\nModel saved successfully!")

print("Filename: steel_energy_rf_model.pkl")