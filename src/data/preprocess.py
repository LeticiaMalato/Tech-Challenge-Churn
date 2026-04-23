# limpeza e preparação dos dados

import hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    TARGET, NUM_COLS_TO_DROP, CAT_COLS_TO_DROP,
    TEST_SIZE, VAL_SIZE, RANDOM_STATE
)


def convert_yes_no(df: pd.DataFrame) -> pd.DataFrame:

    # Identifica colunas com valores "Yes"/"No" e converte para 1/0
    yes_no_cols= [
        col for col in df.columns
        if df[col].dropna().isin(["Yes", "No"]).all()
    ]
    df[yes_no_cols] = df[yes_no_cols].replace({"Yes": 1, "No": 0})
    print(f"  Convertidas {len(yes_no_cols)} colunas Yes/No → 1/0")
    return df


def rename_target(df: pd.DataFrame) -> pd.DataFrame:
    # Renomeia a variavel chun
    if "Churn Value" in df.columns:
        df = df.rename(columns={"Churn Value": TARGET})
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    #Remove as colunas listadas em config.py
    cols_to_drop  = [c for c in NUM_COLS_TO_DROP + CAT_COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop )
    print(f"  Removidas {len(cols_to_drop )} colunas. Shape atual: {df.shape}")
    return df


def encoding(df: pd.DataFrame) -> pd.DataFrame:
    # One Hot Encoding

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"  Encoding aplicado. Shape final: {df_encoded.shape}")
    return df_encoded


def hash(df: pd.DataFrame) -> str:
    #Gera um hash do dataset para controle de versão e rastreabilidade
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()


def split_data(df: pd.DataFrame):
    #split dos dados
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

   
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
   
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"  Treino:    {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test):
   #Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # aprende a média/desvio aqui
    X_val_scaled   = scaler.transform(X_val)          # apenas aplica
    X_test_scaled  = scaler.transform(X_test)         # apenas aplica
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def preprocessing_pipeline(df: pd.DataFrame):
    
    df = convert_yes_no(df)
    df = rename_target(df)
    df = drop_columns(df)
    df = encoding(df)

    dataset_hash = hash(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_sc, X_val_sc, X_test_sc, scaler = scale_data(X_train, X_val, X_test)

    dataset_meta = {
        "dataset_rows":     len(df),
        "dataset_features": X_train.shape[1],
        "train_size":       len(X_train),
        "test_size":        len(X_test),
        "churn_rate_pct":   round(df[TARGET].mean() * 100, 2),
        "dataset_hash":     dataset_hash,
    }

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_scaled": X_train_sc,
        "X_val_scaled":   X_val_sc,
        "X_test_scaled":  X_test_sc,
        "scaler": scaler,
        "df_encoded": df,
        "dataset_meta": dataset_meta,
    }