"""
Train the sklearn Pipeline and write joblib to MODEL_PATH.

Reproduces the notebook: BMI to Nutritional_Status, zero-as-missing on selected
columns, ColumnTransformer + RandomForest, GridSearchCV with recall scoring.
Run from ml-service root after ``pip install -e .``; override output with
OUTPUT_MODEL_PATH.

Set ``ML_SERVICE_FAST_TRAIN=1`` to skip GridSearchCV (faster CI); same pipeline,
default RandomForest hyperparameters.
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_service.config import DATA_PATH, MODEL_PATH

RANDOM_STATE = 42
COLUMNS_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin"]
SKEWED_NUMERIC = ["Pregnancies", "SkinThickness", "Insulin", "DiabetesPedigreeFunction", "Age"]
SYMMETRIC_NUMERIC = ["Glucose", "BloodPressure"]
CATEGORICAL = ["Nutritional_Status"]


def calculate_nutritional_status(x: float) -> str | float:
    """BMI bucket labels; 0 or NaN become NaN for imputation."""
    if x == 0.0 or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if x < 18.5:
        return "Underweight"
    if x < 25:
        return "Normal"
    if x < 30:
        return "Overweight"
    return "Obese"


def load_and_preprocess(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV, derive Nutritional_Status from BMI, treat 0 as missing on key columns."""
    if not os.path.isfile(data_path):
        print(f"Data file not found: {data_path}", file=sys.stderr)
        print("Place diabetes.csv in ml-service/data/ or set DATA_PATH.", file=sys.stderr)
        sys.exit(1)
    data = pd.read_csv(data_path)
    data["Nutritional_Status"] = data["BMI"].apply(calculate_nutritional_status)
    data.drop("BMI", axis=1, inplace=True)
    data[COLUMNS_ZERO_AS_MISSING] = data[COLUMNS_ZERO_AS_MISSING].replace(0, np.nan)
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    return X, y


def build_pipeline() -> Pipeline:
    """Preprocessor + RandomForestClassifier (GridSearchCV tunes hyperparameters)."""
    skewed_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    symmetric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("skewed_num", skewed_transformer, SKEWED_NUMERIC),
        ("symmetric_num", symmetric_transformer, SYMMETRIC_NUMERIC),
        ("cat", cat_transformer, CATEGORICAL),
    ])
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def _fast_train_enabled() -> bool:
    return os.getenv("ML_SERVICE_FAST_TRAIN", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def main() -> None:
    """Fit model (full grid search or fast path), print report, persist joblib."""
    X, y = load_and_preprocess(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pipeline = build_pipeline()
    if _fast_train_enabled():
        print("ML_SERVICE_FAST_TRAIN: fitting pipeline without GridSearchCV")
        pipeline.fit(X_train, y_train)
        best_model = pipeline
    else:
        param_grid = {
            "classifier__n_estimators": [50, 100, 150],
            "classifier__max_depth": [5, 10, 15, None],
            "classifier__min_samples_split": [2, 3, 5],
            "classifier__min_samples_leaf": [1, 2],
            "classifier__class_weight": [None, "balanced"],
        }
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print("Best params:", grid_search.best_params_)
    print("\nTest Classification Report:")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    out_path = Path(os.getenv("OUTPUT_MODEL_PATH", MODEL_PATH))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_path)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()   
