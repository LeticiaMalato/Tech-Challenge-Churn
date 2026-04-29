# Recebe features de um cliente e retorna predição de churn.

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import CustomerFeatures, PredictionResponse
from src.api.dependencies import get_pipeline, get_threshold, get_metadata
from src.api.logging import build_logger

router = APIRouter()
logger = build_logger("churn_api.predict")

# Mapeamento campo Pydantic → nome de coluna esperado pelo pipeline sklearn
_FIELD_TO_COL = {
    "gender":             "Gender",
    "senior_citizen":     "Senior Citizen",
    "partner":            "Partner",
    "dependents":         "Dependents",
    "tenure_months":      "Tenure Months",
    "contract":           "Contract",
    "paperless_billing":  "Paperless Billing",
    "payment_method":     "Payment Method",
    "monthly_charges":    "Monthly Charges",
    "phone_service":      "Phone Service",
    "internet_service":   "Internet Service",
    "multiple_lines":    "Multiple Lines",
    "online_security":   "Online Security",
    "online_backup":     "Online Backup",
    "device_protection": "Device Protection",
    "tech_support":      "Tech Support",
    "streaming_tv":      "Streaming TV",
    "streaming_movies":  "Streaming Movies",
}


def _to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
   #Converte o schema Pydantic para DataFrame de uma linha
    row = {col: getattr(customer, field) for field, col in _FIELD_TO_COL.items()}
    return pd.DataFrame([row])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    customer:  CustomerFeatures,
    pipeline=Depends(get_pipeline),       
    threshold: float = Depends(get_threshold),
    metadata:  dict  = Depends(get_metadata),
) -> PredictionResponse:
   
    df_input = _to_dataframe(customer)

    try:
        proba = float(pipeline.predict_proba(df_input)[0, 1])
    except Exception as exc:
        logger.error("Erro na inferência", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail=f"Erro na inferência: {exc}") from exc

    churn = proba >= threshold

    logger.info(
        "Predição realizada",
        extra={"probability": proba, "threshold": threshold, "churn": churn},
    )

    return PredictionResponse(
        churn=churn,
        probability=round(proba, 4),
        threshold=threshold,
        model_version=metadata.get("run_id"),
    )