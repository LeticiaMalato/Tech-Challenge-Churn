# src/data/pipeline.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    TARGET,
    NUM_COLS_TO_DROP,
    CAT_COLS_TO_DROP,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
)

class RenameTargetTransformer(BaseEstimator, TransformerMixin):
    # Renomeia 'Churn Value' para 'Churn Target'.
    def fit(self, X, y=None):

        return self

    def transform(self, X):
        X = X.copy()
        if "Churn Value" in X.columns:
            X = X.rename(columns={"Churn Value": TARGET})
        return X



class ConvertYesNoTransformer(BaseEstimator, TransformerMixin):
    # Converte colunas com 'Yes'/'No' para 1/0.
    def __init__(self):
        self.cols_yes_no_ = []  # será preenchido no fit()

    def fit(self, X, y=None):
        # Aprende quais colunas têm APENAS Yes/No no treino
        self.cols_yes_no_ = [
            col for col in X.columns if X[col].dropna().isin(["Yes", "No"]).all()
        ]
        return self

    def transform(self, X):
        X = X.copy()
        cols_presentes = [c for c in self.cols_yes_no_ if c in X.columns]
        X[cols_presentes] = X[cols_presentes].replace({"Yes": 1, "No": 0})
        return X


class RemoveColumnsTransformer(BaseEstimator, TransformerMixin):
    # Remove colunas sem valor preditivo e com risco de data leakage.

    def __init__(self, colunas_remover: list):
        self.colunas_remover = colunas_remover

    def fit(self, X, y=None):

        self.colunas_existentes_ = [c for c in self.colunas_remover if c in X.columns]
        return self

    def transform(self, X):
        X = X.copy()
        cols_presentes = [c for c in self.colunas_existentes_ if c in X.columns]
        return X.drop(columns=cols_presentes)


class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    # Aplica One-Hot Encoding nas colunas categóricas (tipo object).
    def __init__(self):
        self.cat_cols_ = []
        self.columns_fit_ = []

    def fit(self, X, y=None):
        self.cat_cols_ = X.select_dtypes(include=["object"]).columns.tolist()
        # Faz o encoding no treino e memoriza as colunas resultantes
        X_encoded = pd.get_dummies(X, columns=self.cat_cols_, drop_first=True)
        self.columns_fit_ = X_encoded.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        X_encoded = pd.get_dummies(X, columns=self.cat_cols_, drop_first=True)
        X_encoded = X_encoded.reindex(columns=self.columns_fit_, fill_value=0)
        return X_encoded


class SeparateTargetTransformer(BaseEstimator, TransformerMixin):
    # Remove a coluna target do DataFrame de features.
    def __init__(self, target: str = TARGET):
        self.target = target

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Remove o target se ainda estiver presente
        if self.target in X.columns:
            X = X.drop(columns=[self.target])
        return X

    def fit_transform_com_y(self, X):

        y = X[self.target].copy() if self.target in X.columns else None
        X_sem_target = self.fit_transform(X)
        return X_sem_target, y



def create_preprocessing_pipeline(model=None) -> Pipeline:
    #Monta e retorna a pipeline
    colunas_para_remover = NUM_COLS_TO_DROP + CAT_COLS_TO_DROP

    steps = [
        ("renomear_target", RenameTargetTransformer()),
        ("converter_yes_no", ConvertYesNoTransformer()),
        ("remover_colunas", RemoveColumnsTransformer(
            colunas_remover=colunas_para_remover
        )),
        ("separar_target", SeparateTargetTransformer(target=TARGET)),
        ("encoding", OneHotEncoderTransformer()),
        ("scaler", StandardScaler()),
    ]

    if model is not None:
        steps.append(("model", model))

    return Pipeline(steps=steps)
