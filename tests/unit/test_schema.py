import pytest
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from src.config import TARGET

schema_bruto = DataFrameSchema(
    columns={
        "Churn Value": Column(int, checks=Check.isin([0, 1]), nullable=False),
        "Tenure Months": Column(
            int, checks=Check.greater_than_or_equal_to(0), nullable=False
        ),  # CORRIGIDO: float → int
        "Monthly Charges": Column(float, checks=Check.greater_than(0), nullable=False),
        "Gender": Column(str, checks=Check.isin(["Male", "Female"]), nullable=False),
    },
    strict=False,
)

schema_processado = DataFrameSchema(
    columns={
        TARGET: Column(int, checks=Check.isin([0, 1]), nullable=False),
        "Tenure Months": Column(
            int, checks=Check.greater_than_or_equal_to(0), nullable=False
        ),  # CORRIGIDO: float → int
        "Monthly Charges": Column(float, checks=Check.greater_than(0), nullable=False),
    },
    strict=False,
)


class TestRawSchema:
    def test_valid_schema(self, sample_df):
        df = sample_df.copy()
        df["Churn Value"] = df["Churn Value"].astype(int)
        schema_bruto.validate(df)

    def test_rejects_invalid_churn(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "Churn Value"] = 99
        with pytest.raises(pa.errors.SchemaError):
            schema_bruto.validate(df)

    def test_rejects_negative_tenure(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "Tenure Months"] = -5
        with pytest.raises(pa.errors.SchemaError):
            schema_bruto.validate(df)

    def test_rejects_zero_monthly_charges(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "Monthly Charges"] = 0.0
        with pytest.raises(pa.errors.SchemaError):
            schema_bruto.validate(df)


class TestProcessedSchema:
    def test_valid_schema_after_preprocessing(self, processed_df):
        schema_processado.validate(processed_df)

    def test_no_nulls_in_target(self, processed_df):
        assert processed_df[TARGET].isnull().sum() == 0

    def test_binary_values_in_target(self, processed_df):
        assert set(processed_df[TARGET].unique()).issubset({0, 1})

    def test_no_object_columns(self, processed_df):
        object_columns = processed_df.select_dtypes(include=["object"]).columns
        assert len(object_columns) == 0
