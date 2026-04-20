# Identificar quais colunas numéricas existem e quais são relevantes para análise.

import numpy as np
import pandas as pd
from src.config import TARGET


def get_num_columns(df: pd.DataFrame) -> list:
    #Retorna lista de colunas numéricas
    return [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col != TARGET
    ]


def get_cat_columns(df: pd.DataFrame) -> list:
    #Retorna lista de colunas categóricas
    return df.select_dtypes(include=["object"]).columns.tolist()