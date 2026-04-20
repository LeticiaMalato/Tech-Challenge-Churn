# main.py

from src.data.load import load_data
from src.data.preprocess import preprocessing_pipeline
from src.features.selection import get_num_columns

from src.visualization.eda import (
    plot_target_distribution,
    plot_boxplots,
    plot_histograms,
    plot_bivariate,
    plot_confusion_matrix,
)

from src.models.baseline import dummy_classifier, logistic_regression
from src.models.tree_models import find_best_depth, decision_tree, random_forest
from src.models.neural import mlp

from src.evaluation.business import calculate_financial_result, compare_models_financial


def main():
    # Carregar Dataset
    df = load_data()

    df = df.rename(columns={"Churn Value": "Churn Target"})
    #  EDA  
    num_cols = get_num_columns(df)
    plot_target_distribution(df)
    plot_boxplots(df, num_cols)
    plot_histograms(df, num_cols)
    plot_bivariate(df, num_cols)

    # Pré-processamento 
    dados = preprocessing_pipeline(df)

    X_train      = dados["X_train"]
    X_val        = dados["X_val"]
    X_test       = dados["X_test"]
    y_train      = dados["y_train"]
    y_val        = dados["y_val"]
    y_test       = dados["y_test"]
    X_train_sc   = dados["X_train_scaled"]
    X_val_sc     = dados["X_val_scaled"]
    X_test_sc    = dados["X_test_scaled"]
    dataset_meta = dados["dataset_meta"]

    # Modelos Baseline 
    _, y_pred_dummy, _ = dummy_classifier(
        X_train_sc, X_test_sc, y_train, y_test, dataset_meta
    )
    plot_confusion_matrix(y_test, y_pred_dummy, "Dummy")

    _, y_pred_lr, _ = logistic_regression(
        X_train_sc, X_test_sc, y_train, y_test, dataset_meta
    )
    plot_confusion_matrix(y_test, y_pred_lr, "Regressão Logística")

    #  Modelos de Árvore 
    melhores = find_best_depth(X_train, y_train)

    _, y_pred_tree, _ = decision_tree(
        X_train, X_test, y_train, y_test, dataset_meta,
        max_depth=melhores["tree_depth"]
    )
    plot_confusion_matrix(y_test, y_pred_tree, "Decision Tree")

    _, y_pred_forest, _ = random_forest(
        X_train, X_test, y_train, y_test, dataset_meta,
        max_depth=melhores["forest_depth"]
    )
    plot_confusion_matrix(y_test, y_pred_forest, "Random Forest")

    #  6. Rede Neural 
    _, y_pred_mlp, _ = mlp(
        X_train_sc, X_val_sc, X_test_sc,
        y_train, y_val, y_test, dataset_meta
    )
    plot_confusion_matrix(y_test, y_pred_mlp, "MLP PyTorch")

    #  7. Análise de Negócio 
    modelos_resultado = []
    for nome, y_pred in [
        ("Dummy",               y_pred_dummy),
        ("Regressão Logística", y_pred_lr),
        ("Decision Tree",       y_pred_tree),
        ("Random Forest",       y_pred_forest),
        ("MLP PyTorch",         y_pred_mlp),
    ]:
        r = calculate_financial_result(y_test, y_pred)
        r["model"] = nome
        modelos_resultado.append(r)

    compare_models_financial(modelos_resultado)


if __name__ == "__main__":
    main()