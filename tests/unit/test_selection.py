import pandas as pd
from src.features.selection import get_num_columns, get_cat_columns


class TestGetNumColumns:

    def test_returns_numeric_columns(self):
        df = pd.DataFrame({
            "age":    [25, 30],
            "salary": [1000.0, 2000.0],
            "name":   ["Ana", "João"],
        })
        result = get_num_columns(df)
        assert "age" in result
        assert "salary" in result

    def test_excludes_target(self):
        from src.config import TARGET
        df = pd.DataFrame({
            "age":  [25, 30],
            TARGET: [0, 1],
        })
        result = get_num_columns(df)
        assert TARGET not in result

    def test_excludes_categorical(self):
        df = pd.DataFrame({
            "age":  [25, 30],
            "name": ["Ana", "João"],
        })
        result = get_num_columns(df)
        assert "name" not in result

    def test_empty_dataframe_returns_empty_list(self):
        result = get_num_columns(pd.DataFrame())
        assert result == []


class TestGetCatColumns:

    def test_returns_categorical_columns(self):
        df = pd.DataFrame({
            "name":    ["Ana", "João"],
            "city":    ["SP", "RJ"],
            "age":     [25, 30],
        })
        result = get_cat_columns(df)
        assert "name" in result
        assert "city" in result

    def test_excludes_numeric_columns(self):
        df = pd.DataFrame({
            "name": ["Ana", "João"],
            "age":  [25, 30],
        })
        result = get_cat_columns(df)
        assert "age" not in result

    def test_empty_dataframe_returns_empty_list(self):
        result = get_cat_columns(pd.DataFrame())
        assert result == []