"""
Feature construction for inference (aligned with notebook and training).

Builds a one-row DataFrame with column names and order expected by the saved
sklearn Pipeline. BMI can be turned into Nutritional_Status; BMI is not a model
column after training (it was dropped post feature engineering).
"""
from typing import Any, Optional

import numpy as np
import pandas as pd

from ml_service.config import FEATURE_COLUMNS, VALID_NUTRITIONAL_STATUS


def calculate_nutritional_status(bmi: float) -> Optional[str]:
    """
    Map BMI to a categorical label. Zero or NaN BMI yields None (missing).
    """
    if bmi is None or (isinstance(bmi, (int, float)) and (bmi == 0.0 or np.isnan(bmi))):
        return None
    bmi = float(bmi)
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def build_input_df(
    pregnancies: Optional[float] = None,
    glucose: Optional[float] = None,
    blood_pressure: Optional[float] = None,
    skin_thickness: Optional[float] = None,
    insulin: Optional[float] = None,
    diabetes_pedigree_function: Optional[float] = None,
    age: Optional[float] = None,
    bmi: Optional[float] = None,
    nutritional_status: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build one model input row. If only BMI is set, Nutritional_Status is derived.
    Invalid status strings become None (imputed inside the pipeline).
    """
    if nutritional_status is None and bmi is not None:
        nutritional_status = calculate_nutritional_status(bmi)
    if nutritional_status is not None and nutritional_status not in VALID_NUTRITIONAL_STATUS:
        nutritional_status = None

    row = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age,
        "Nutritional_Status": nutritional_status,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def build_input_df_from_dict(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Map JSON keys (snake_case or PascalCase) to build_input_df arguments.
    """

    def get(key_aliases: tuple[str, ...]) -> Any:
        """Match payload keys case-insensitively with hyphen/underscore normalization."""
        for alias in key_aliases:
            if alias in payload:
                return payload[alias]
            low = alias.lower().replace("-", "_")
            for k, v in payload.items():
                if k.lower().replace("-", "_") == low:
                    return v
        return None

    return build_input_df(
        pregnancies=get(("Pregnancies", "pregnancies")),
        glucose=get(("Glucose", "glucose")),
        blood_pressure=get(("BloodPressure", "blood_pressure")),
        skin_thickness=get(("SkinThickness", "skin_thickness")),
        insulin=get(("Insulin", "insulin")),
        diabetes_pedigree_function=get(("DiabetesPedigreeFunction", "diabetes_pedigree_function")),
        age=get(("Age", "age")),
        bmi=get(("BMI", "bmi")),
        nutritional_status=get(("Nutritional_Status", "nutritional_status")),
    )
