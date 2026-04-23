# tests/test_load.py

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.load import load_data


def test_load_data_returns_dataframe():
    
    #Deve retornar um dataframe
    mock_df  = pd.DataFrame({"col1": [1, 2], "Churn Value": [0, 1]})

    # patch substitui pd.read_excel por uma função falsa durante o teste
    with patch("src.data.load.pd.read_excel", return_value=mock_df ):
        resultado = load_data("caminho/falso.xlsx")

    assert isinstance(resultado, pd.DataFrame)


def test_load_data_not_empty():
    
    #DataFrame carregado não pode estar vazio
    mock_df  = pd.DataFrame({"col1": [1, 2], "Churn Value": [0, 1]})

    with patch("src.data.load.pd.read_excel", return_value=mock_df ):
        resultado = load_data("caminho/falso.xlsx")

    assert len(resultado) > 0


def test_load_data_file_not_found():
    #Deve lançar erro se o arquivo não existir
    with pytest.raises(Exception):
        load_data("arquivo_que_nao_existe.xlsx")