# Calcular métricas de classificação

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