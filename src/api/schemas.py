from typing import Optional
from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    # Demográficas
    gender: str = Field(..., examples=["Male"], description="Male ou Female")
    senior_citizen: str = Field(..., examples=["No"], description="Yes ou No")
    partner: str = Field(..., examples=["Yes"], description="Yes ou No")
    dependents: str = Field(..., examples=["No"], description="Yes ou No")

    # Contrato
    tenure_months: int = Field(
        ..., ge=0, examples=[24], description="Meses de contrato (≥ 0)"
    )
    contract: str = Field(..., examples=["Month-to-month"])
    paperless_billing: str = Field(..., examples=["Yes"], description="Yes ou No")
    payment_method: str = Field(..., examples=["Electronic check"])

    # Financeiro
    monthly_charges: float = Field(
        ..., gt=0, examples=[65.50], description="Cobrança mensal (> 0)"
    )

    # Serviços
    phone_service: str = Field(..., examples=["Yes"], description="Yes ou No")
    internet_service: str = Field(..., examples=["Fiber optic"])

    # Adicionar depois de internet_service
    multiple_lines: str = Field(..., examples=["Yes"], description="Yes ou No")
    online_security: str = Field(..., examples=["No"], description="Yes ou No")
    online_backup: str = Field(..., examples=["No"], description="Yes ou No")
    device_protection: str = Field(..., examples=["No"], description="Yes ou No")
    tech_support: str = Field(..., examples=["No"], description="Yes ou No")
    streaming_tv: str = Field(..., examples=["No"], description="Yes ou No")
    streaming_movies: str = Field(..., examples=["No"], description="Yes ou No")

    #  Validadores
    @field_validator("gender")      # ← adicionar esse
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in {"Male", "Female"}:
            raise ValueError("gender deve ser 'Male' ou 'Female'")
        return v

    @field_validator(
    "senior_citizen", "partner", "dependents", "paperless_billing",
    "phone_service", "multiple_lines", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies"
)
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        if v not in {"Yes", "No"}:
            raise ValueError("campo deve ser 'Yes' ou 'No'")
        return v

    @field_validator("contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        valid = {"Month-to-month", "One year", "Two year"}
        if v not in valid:
            raise ValueError(f"contract deve ser um de: {valid}")
        return v

    @field_validator("internet_service")
    @classmethod
    def validate_internet(cls, v: str) -> str:
        valid = {"DSL", "Fiber optic", "No"}
        if v not in valid:
            raise ValueError(f"internet_service deve ser um de: {valid}")
        return v

    @field_validator("payment_method")
    @classmethod
    def validate_payment(cls, v: str) -> str:
        valid = {
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        }
        if v not in valid:
            raise ValueError(f"payment_method deve ser um de: {valid}")
        return v


class PredictionResponse(BaseModel):
    churn: bool = Field(..., description="True se probabilidade >= threshold")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilidade de churn"
    )
    threshold: float = Field(..., description="Limiar usado na binarização")
    model_version: Optional[str] = Field(None, description="run_id do modelo no MLflow")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_metadata: Optional[dict] = None
