# src/models/tree_models.py

import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from src.config import MLFLOW_EXPERIMENT
from src.evaluation.metrics import calculate_metrics


def find_best_depth(X_train, y_train) -> dict:
    # Usa GridSearchCV para encontrar o melhor max_depth para árvore de decisão e random fores
    param_grid = {"max_depth": [3, 5, 7, 10, 15, None]}

    # Decision Tree
    gs_tree = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=1,
    )
    gs_tree.fit(X_train, y_train)

    print(
        f"  Melhor Decision Tree → {gs_tree.best_params_} | ROC-AUC: {gs_tree.best_score_:.4f}"
    )

    # Random Forest
    gs_forest = GridSearchCV(
        RandomForestClassifier(n_estimators=40, random_state=42),
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=1,
    )
    gs_forest.fit(X_train, y_train)
    print(
        f"  Melhor Random Forest → {gs_forest.best_params_} | ROC-AUC: {gs_forest.best_score_:.4f}"
    )

    return {
        "tree_depth": gs_tree.best_params_["max_depth"],
        "forest_depth": gs_forest.best_params_["max_depth"],
    }


def decision_tree(X_train, X_test, y_train, y_test, dataset_meta: dict, max_depth=7):
    # Decision Tree

    mlflow.end_run()
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="decision_tree_baseline"):
        mlflow.log_params(dataset_meta)

        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_proba)
        metrics["train_accuracy"] = accuracy_score(y_train, y_pred_train)
        metrics["overfitting"] = metrics["train_accuracy"] - metrics["accuracy"]

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print("\n=== DECISION TREE ===")
        for name, value in metrics.items():
            print(f"  {name:<16}: {value:.4f}")

    return model, y_pred, y_proba


def random_forest(X_train, X_test, y_train, y_test, dataset_meta: dict, max_depth=15):
    # Random Forest

    mlflow.end_run()
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="random_forest_baseline"):
        mlflow.log_params(dataset_meta)

        model = RandomForestClassifier(
            n_estimators=40, max_depth=max_depth, criterion="entropy", random_state=42
        )
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_proba)
        metrics["train_accuracy"] = accuracy_score(y_train, y_pred_train)
        metrics["overfitting"] = metrics["train_accuracy"] - metrics["accuracy"]

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print("\n=== RANDOM FOREST ===")
        for name, value in metrics.items():
            print(f"  {name:<16}: {value:.4f}")

    return model, y_pred, y_proba
