# tests/conftest.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

def _make_rows(n):
    #Gera n linhas com estrutura idêntica ao dataset real
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "CustomerID":        [str(i).zfill(3) for i in range(n)],
        "Churn Value":       [i % 2 for i in range(n)],
        "Churn Label":       ["Yes" if i % 2 else "No" for i in range(n)],
        "Churn Score":       rng.integers(10, 95, n).tolist(),
        "CLTV":              rng.integers(1000, 4000, n).tolist(),
        "Churn Reason":      [None] * n,
        "Count":             [1] * n,
        "Country":           ["United States"] * n,
        "State":             ["California"] * n,
        "City":              ["Los Angeles"] * n,
        "Zip Code":          rng.integers(90001, 90099, n).tolist(),
        "Lat Long":          ["34.05, -118.24"] * n,
        "Latitude":          [34.05] * n,
        "Longitude":         [-118.24] * n,
        "Gender":            ["Male" if i % 2 else "Female" for i in range(n)],
        "Senior Citizen":    ["No"] * n,
        "Partner":           ["Yes" if i % 2 else "No" for i in range(n)],
        "Dependents":        ["No"] * n,
        "Tenure Months":     rng.integers(1, 72, n).tolist(),
        "Monthly Charges":   rng.uniform(20.0, 110.0, n).tolist(),
        "Total Charges":     rng.uniform(100.0, 5000.0, n).tolist(),
        "Phone Service":     ["Yes"] * n,
        "Internet Service":  ["DSL" if i % 3 == 0 else "Fiber optic" if i % 3 == 1 else "No" for i in range(n)],
        "Contract":          ["Month-to-month" if i % 3 == 0 else "One year" if i % 3 == 1 else "Two year" for i in range(n)],
        "Paperless Billing": ["Yes" if i % 2 else "No" for i in range(n)],
        "Payment Method":    ["Electronic check" if i % 4 == 0 else "Mailed check" if i % 4 == 1 else "Bank transfer" if i % 4 == 2 else "Credit card" for i in range(n)],
    })


@pytest.fixture
def sample_df():
    #Amostra pequena para testes unitários rápidos
    return _make_rows(5)


@pytest.fixture
def full_df():
    #Dataset completo para testes que precisam de train_test_split com stratify
    return _make_rows(50)


@pytest.fixture
def processed_df(sample_df):
    #DataFrame já com pré-processamento básico aplicado
    from src.data.preprocess import convert_yes_no, rename_target, drop_columns, encoding
    df = convert_yes_no(sample_df.copy())
    df = rename_target(df)
    df = drop_columns(df)
    df = encoding(df)
    return df


@pytest.fixture
def X_y_split(processed_df):
    #X e y prontos para treino, usados nos testes de modelo
    from src.config import TARGET
    X = processed_df.drop(columns=[TARGET], errors="ignore")
    y = processed_df[TARGET]
    return X, y


@pytest.fixture
def empty_dataframe():
    #Fixture com DataFrame vazio para testar edge cases
    return pd.DataFrame()


@pytest.fixture(scope="session")
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.predict_proba.return_value = [[0.2, 0.8]]
    return pipeline

@pytest.fixture
def api_client(mock_pipeline):
    from src.api.app import app
    from src.api import dependencies
    dependencies.MODEL_ARTIFACTS["pipeline"]  = mock_pipeline
    dependencies.MODEL_ARTIFACTS["threshold"] = 0.5
    dependencies.MODEL_ARTIFACTS["metadata"]  = {"run_id": "test-run-001"}
    with TestClient(app) as client:
        yield client
    dependencies.MODEL_ARTIFACTS.clear()

@pytest.fixture
def valid_customer_payload():
    return {
        "gender": "Male", "senior_citizen": "No", "partner": "Yes",
        "dependents": "No", "tenure_months": 24, "contract": "Month-to-month",
        "paperless_billing": "Yes", "payment_method": "Electronic check",
        "monthly_charges": 75.50, "phone_service": "Yes",
        "internet_service": "Fiber optic",
    }