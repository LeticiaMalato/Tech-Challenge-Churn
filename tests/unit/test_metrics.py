# tests/test_metrics.py

import numpy as np
import pandas as pd
from src.evaluation.metrics import calculate_metrics, compare_models_metrics


class TestCalculateMetrics:
    def test_perfect_model(self):
        # Quando y_pred == y_true, todas as métricas devem ser 1.0.

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.9, 0.9])

        metrics = calculate_metrics(y_true, y_pred, y_proba)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_returns_all_keys(self):
        # O dicionário de métricas deve sempre ter as 6 chaves esperadas.
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_proba = np.array([0.2, 0.8, 0.6, 0.4])

        metrics = calculate_metrics(y_true, y_pred, y_proba)
        chaves_esperadas = {
            "accuracy",
            "f1_score",
            "precision",
            "recall",
            "roc_auc",
            "pr_auc",
        }

        assert chaves_esperadas == set(metrics.keys())

    def test_metrics_values_between_0_and_1(self):
        # Todas as métricas devem estar no intervalo [0, 1]
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        y_proba = np.array([0.1, 0.7, 0.8, 0.3, 0.2, 0.9])

        metrics = calculate_metrics(y_true, y_pred, y_proba)

        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{name} fora do intervalo: {value}"

    def test_high_recall_with_many_true_positives(self): 

        # Quando o modelo acerta todos os churners (nenhum FN),o recall deve ser 1.0.

        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 0])  # acerta todos os 1s, um FP
        y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.1])

        metrics = calculate_metrics(y_true, y_pred, y_proba)

        assert metrics["recall"] == 1.0

    def _make_results(self):
        return [
            {
                "model": "model_a",
                "accuracy": 0.80, "f1_score": 0.75,
                "precision": 0.70, "recall": 0.80,
                "roc_auc": 0.85, "pr_auc": 0.80,
            },
            {
                "model": "model_b",
                "accuracy": 0.70, "f1_score": 0.65,
                "precision": 0.60, "recall": 0.70,
                "roc_auc": 0.75, "pr_auc": 0.70,
            },
        ]

    def test_returns_dataframe(self):
        result = compare_models_metrics(self._make_results())
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_has_expected_columns(self):
        result = compare_models_metrics(self._make_results())
        assert {"model", "accuracy", "f1_score", "precision", "recall", "roc_auc"}.issubset(
            result.columns
        )

    def test_dataframe_has_correct_number_of_rows(self):
        result = compare_models_metrics(self._make_results())
        assert len(result) == 2

    def test_sorted_by_f1_descending(self):
        result = compare_models_metrics(self._make_results())
        f1_values = result["f1_score"].tolist()
        assert f1_values == sorted(f1_values, reverse=True)
    

    
