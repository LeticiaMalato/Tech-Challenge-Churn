# main.py

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.evaluation.metrics import calculate_metrics
from src.models.neural import mlp
import shutil, os

from src.config import MODEL_ARTIFACT_PATH
from src.data.load import load_data
from src.models.train import prepare_data, split_data, train_pipeline, get_preprocessed_data
from src.visualization.eda import (
    plot_target_distribution,
    plot_boxplots,
    plot_histograms,
    plot_bivariate,
    plot_confusion_matrix,
)
from src.evaluation.metrics import compare_models_metrics
from src.features.selection import get_num_columns
from src.evaluation.business import calculate_financial_result, compare_models_financial
import hashlib


def main():
    # Carregar Dados
    df = load_data()
    df_eda = df.rename(columns={"Churn Value": "Churn Target"})

    #EDA 
    num_cols = get_num_columns(df_eda)
    plot_target_distribution(df_eda)
    plot_boxplots(df_eda, num_cols)
    plot_histograms(df_eda, num_cols)
    plot_bivariate(df_eda, num_cols)

    #  Preparar X e y 
    X, y = prepare_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Metadados para o MLflow
    dataset_meta = {
        "dataset_rows":   len(df),
        "train_size":     len(X_train),
        "test_size":      len(X_test),
        "churn_rate_pct": round(y.mean() * 100, 2),
        "dataset_hash":   hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest(),
    }

    # Treinar models via pipeline 
    resultados_business = []
    resultados_metricas = [] 

    models = [
        (DummyClassifier(strategy="most_frequent", random_state=42), "dummy_baseline",               False),
        (LogisticRegression(random_state=42, max_iter=1000),          "logistic_regression_baseline", True),
        (DecisionTreeClassifier(max_depth=7, random_state=42),        "decision_tree_baseline",       True),
        (RandomForestClassifier(n_estimators=40, max_depth=15,
                                criterion="entropy", random_state=42),"random_forest_baseline",       True),
    ]

    for model, nome_run, fazer_cv in models:
        pipeline, y_pred, y_proba = train_pipeline(
            model=model,
            nome_run=nome_run,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            dataset_meta=dataset_meta,
            fazer_cv=fazer_cv,
        )
        plot_confusion_matrix(y_test, y_pred, nome_run)

        r = calculate_financial_result(y_test, y_pred)
        r["model"] = nome_run
        resultados_business.append(r)

        m = calculate_metrics(y_test, y_pred, y_proba)     # ← coleta métricas
        m["model"] = nome_run
        resultados_metricas.append(m)
    
    #Treinar Mlp
    X_train_sc, X_val_sc, X_test_sc = get_preprocessed_data(
    X_train, X_val, X_test, y_train, y_val, y_test
)
    _, y_pred_mlp, y_proba_mlp = mlp(
        X_train_sc=X_train_sc,
        X_val_sc=X_val_sc,
        X_test_sc=X_test_sc,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        dataset_meta=dataset_meta,
    )
    m_mlp = calculate_metrics(y_test, y_pred_mlp, y_proba_mlp)  
    m_mlp["model"] = "mlp_pytorch"
    resultados_metricas.append(m_mlp)

    plot_confusion_matrix(y_test, y_pred_mlp, "mlp_pytorch")

    r_mlp = calculate_financial_result(y_test, y_pred_mlp)
    r_mlp["model"] = "mlp_pytorch"
    resultados_business.append(r_mlp)

   

    #Comparação de métricas
    compare_models_metrics(resultados_metricas)  
    # Análise financeira 
    compare_models_financial(resultados_business)

    melhor = max(resultados_business, key=lambda r: r["net_result"])
    nome_melhor = melhor["model"]
    src_path = MODEL_ARTIFACT_PATH.replace(".joblib", f"_{nome_melhor}.joblib")

    if os.path.exists(src_path):
        shutil.copy(src_path, MODEL_ARTIFACT_PATH)
        print(f"Artefato '{nome_melhor}' promovido para {MODEL_ARTIFACT_PATH}")
    else:
        print(f"Artefato '{src_path}' não encontrado")


if __name__ == "__main__":
    main()