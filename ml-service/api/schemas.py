"""
Request and response bodies for POST /predict.

Field names accept PascalCase (notebook-style) or snake_case aliases from JSON.
"""
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Send 7 numerics + BMI, or 8 fields with Nutritional_Status."""

    model_config = {"populate_by_name": True}

    Pregnancies: int | None = Field(None, ge=0, le=20, alias="pregnancies")
    Glucose: float | None = Field(None, ge=0, le=199, alias="glucose")
    BloodPressure: float | None = Field(None, ge=0, le=122, alias="blood_pressure")
    SkinThickness: float | None = Field(None, ge=0, le=99, alias="skin_thickness")
    Insulin: float | None = Field(None, ge=0, le=846, alias="insulin")
    BMI: float | None = Field(None, ge=0, le=67.1, alias="bmi")
    DiabetesPedigreeFunction: float | None = Field(
        None, ge=0.078, le=2.42, alias="diabetes_pedigree_function"
    )
    Age: int | None = Field(None, ge=21, le=81, alias="age")
    Nutritional_Status: str | None = Field(
        None,
        pattern="^(Underweight|Normal|Overweight|Obese)$",
        alias="nutritional_status",
    )


class PredictResponse(BaseModel):
    """Binary prediction plus per-class probabilities (four decimal places in API)."""

    outcome: int
    probability_no_diabetes: float
    probability_diabetes: float
