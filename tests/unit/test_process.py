# tests/test_preprocess.py

import pytest
import pandas as pd
from src.config import TARGET
from src.data.preprocess import (
    convert_yes_no,
    rename_target,
    drop_columns,
    encoding,
)


class TestConverterYesNo:
    #Agrupa todos os testes de convert_yes_no.

    def test_converts_yes_to_1(self, sample_df):
        result = convert_yes_no(sample_df.copy())
        # Senior Citizen tinha "Yes"/"No" → deve virar 1/0
        assert set(result["Senior Citizen"].unique()).issubset({0, 1})

    def test_converts_no_to_0(self, sample_df):
        result = convert_yes_no(sample_df.copy())
        assert 0 in result["Senior Citizen"].values

    def test_does_not_modify_numeric_columns(self, sample_df):
        #Colunas já numéricas não devem ser tocadas
        result = convert_yes_no(sample_df.copy())
        assert result["Tenure Months"].dtype in ["int64", "float64"]

    def test_does_not_modify_multiclass_categorical_columns(self, sample_df):
        #Colunas com mais de dois valores não devem ser alteradas.
        result = convert_yes_no(sample_df.copy())
        # Internet Service tem 3 valores → não deve virar 0/1
        assert "Fiber optic" in result["Internet Service"].values


class TestRenameTarget:

    def test_renames_churn_value_to_target(self, sample_df):
        result = rename_target(sample_df.copy())
        assert TARGET in result.columns

    def test_removes_old_target_name(self, sample_df):
        result = rename_target(sample_df.copy())
        assert "Churn Value" not in result.columns

    def test_does_not_fail_if_already_renamed(self, sample_df):
        #Se já estiver com o nome certo, não deve lançar erro
        df = sample_df.rename(columns={"Churn Value": TARGET})
        result = rename_target(df)
        assert TARGET in result.columns


class TestDropColumns:

    def test_removes_configured_columns(self, sample_df):
        df = rename_target(sample_df.copy())
        result = drop_columns(df)
        # CustomerID deve ter sido removido (está em COLUNAS_CAT_REMOVER)
        assert "CustomerID" not in result.columns

    def test_does_not_fail_if_column_missing(self, sample_df):
        #Não deve lançar erro se uma coluna configurada não existir.
        df = sample_df.drop(columns=["CustomerID"])
        df = rename_target(df)
        result = drop_columns(df)  # não deve estourar KeyError
        assert result is not None

    def test_target_column_is_preserved(self, sample_df):
        #A coluna target não deve ser removida pelo drop_columns.
        df = rename_target(sample_df.copy())
        result = drop_columns(df)
        assert TARGET in result.columns


class TestEncoding:

    def test_no_object_columns_after_encoding(self, sample_df):
        #Após encoding, não deve restar nenhuma coluna do tipo object.
        df = convert_yes_no(sample_df.copy())
        df = rename_target(df)
        df = drop_columns(df)
        result = encoding(df)
        colunas_object = result.select_dtypes(include=["object"]).columns
        assert len(colunas_object) == 0

    def test_increases_number_of_columns(self, sample_df):
        #One-hot encoding sempre gera mais colunas do que havia antes.
        df = convert_yes_no(sample_df.copy())
        df = rename_target(df)
        df = drop_columns(df)
        n_colunas_antes = df.shape[1]
        result = encoding(df)
        assert result.shape[1] >= n_colunas_antes

    def test_preserves_number_of_rows(self, sample_df):
        #Encoding não deve alterar o número de linhas.
        df = convert_yes_no(sample_df.copy())
        df = rename_target(df)
        df = drop_columns(df)
        resultado = encoding(df)
        assert resultado.shape[0] == sample_df.shape[0]