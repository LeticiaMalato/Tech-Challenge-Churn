# src/models/baseline.py

import mlflow
import mlflow.sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import MLFLOW_EXPERIMENT
from src.evaluation.metrics import calculate_metrics


def dummy_classifier(X_train, X_test, y_train, y_test, dataset_meta: dict):
    #Dummy Classifier
    mlflow.end_run()
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="dummy_baseline"):
        mlflow.log_params(dataset_meta)

        model = DummyClassifier(strategy="most_frequent", random_state=42)
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print("\n=== DUMMY CLASSIFIER ===")
        for name, value in metrics.items():
            print(f"  {name:<16}: {value:.4f}")

    return model, y_pred, y_proba


def logistic_regression(X_train, X_test, y_train, y_test, dataset_meta: dict):
    # Regressão Logística
    mlflow.end_run()
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="logistic_regression_baseline"):
        mlflow.log_params(dataset_meta)

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred       = model.predict(X_test)
        y_proba      = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_proba)
        metrics["train_accuracy"] = accuracy_score(y_train, y_pred_train)
        metrics["overfitting"]    = metrics["train_accuracy"] - metrics["accuracy"]

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print("\n=== LOGISTIC REGRESSION ===")
        for name, value in metrics.items():
            print(f"  {name:<16}: {value:.4f}")

    return model, y_pred, y_proba