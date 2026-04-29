# Calcular métricas de classificação
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score
)


def calculate_metrics(y_true, y_pred, y_proba) -> dict:
    
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_score":  f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_proba),
        "pr_auc":    average_precision_score(y_true, y_proba),
    }

def compare_models_metrics(resultados: list[dict]) -> pd.DataFrame:
    
    cols = ["model", "accuracy", "f1_score", "precision", "recall", "roc_auc"]
    df = pd.DataFrame(resultados)[cols].sort_values("f1_score", ascending=False)

    print("\n" + "=" * 70)
    print("COMPARAÇÃO DE MODELOS — MÉTRICAS DE CLASSIFICAÇÃO")
    print("=" * 70)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 70)

    melhor = df.iloc[0]
    print(f"\n  Melhor modelo por F1: {melhor['model']}")
    print(f"    F1:        {melhor['f1_score']:.4f}")
    print(f"    ROC-AUC:   {melhor['roc_auc']:.4f}")
    print(f"    Precision: {melhor['precision']:.4f}")
    print(f"    Recall:    {melhor['recall']:.4f}")
    print(f"    Accuracy:  {melhor['accuracy']:.4f}")

    return df