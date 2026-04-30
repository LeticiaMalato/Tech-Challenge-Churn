# src/visualization/eda.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.config import TARGET

# Paleta de cores
COR_CHURN = "#E24B4A"
COR_NO_CHURN = "#3B8BD4"
COLORS = [COR_NO_CHURN, COR_CHURN]
LABELS = ["Não Churn", "Churn"]
sns.set_theme(style="whitegrid", palette="muted")


# Distribuição do Target


def plot_target_distribution(df: pd.DataFrame):
    # Mostrar o desbalanceamento da variável target.
    counts = df[TARGET].value_counts().sort_index()
    pcts = df[TARGET].value_counts(normalize=True).sort_index() * 100

    print(f"Ratio churn: {pcts[1]:.1f}%  |  não-churn: {pcts[0]:.1f}%")
    ratio = counts[0] / counts[1]
    if ratio > 3:
        print(
            f"⚠️  Dataset desbalanceado ({ratio:.1f}:1) → usar PR-AUC e Recall como métricas principais."
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(LABELS, counts.values, color=COLORS, width=0.5, edgecolor="white")
    for i, (v, p) in enumerate(zip(counts.values, pcts.values)):
        axes[0].text(i, v + 30, f"{v}\n({p:.1f}%)", ha="center", fontsize=10)
    axes[0].set_title("Contagem por classe")
    axes[0].set_ylabel("Qtd clientes")

    axes[1].pie(
        counts.values,
        labels=LABELS,
        colors=COLORS,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    axes[1].set_title("Proporção das classes")

    plt.suptitle("Desbalanceamento — Churn Target", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


# Outliers


def detect_outliers(serie: pd.Series):
    # Conta outliers pelo método IQR e Z-score.
    Q1, Q3 = serie.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    n_iqr = ((serie < Q1 - 1.5 * IQR) | (serie > Q3 + 1.5 * IQR)).sum()
    n_z = (np.abs(stats.zscore(serie)) > 3).sum()
    return n_iqr, n_z


def plot_boxplots(df: pd.DataFrame, num_cols: list):
    # Boxplots para identificar outliers em todas as colunas numéricas.
    ncols = 3
    nrows = int(np.ceil(len(num_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))

    box_style = dict(
        patch_artist=True,
        boxprops=dict(facecolor="#AED6F1"),
        medianprops=dict(color="#E24B4A", linewidth=2),
        flierprops=dict(marker="o", color="#E24B4A", alpha=0.4, markersize=4),
    )
    for ax, col in zip(axes.ravel(), num_cols):
        ax.boxplot(df[col].dropna(), **box_style)
        ax.set_title(col, fontweight="bold")
    for ax in axes.ravel()[len(num_cols) :]:
        ax.set_visible(False)

    plt.suptitle("Boxplots — Detecção de Outliers", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


#  Distribuições


def plot_histograms(df: pd.DataFrame, num_cols: list):
    # Histogramas com KDE, média e mediana para cada variável numérica.
    ncols = 3
    nrows = int(np.ceil(len(num_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))

    for ax, col in zip(axes.ravel(), num_cols):
        serie = df[col].dropna()
        sns.histplot(serie, kde=True, ax=ax, color="#3B8BD4", edgecolor="white")
        ax.set_title(col, fontweight="bold")
        media, mediana = serie.mean(), serie.median()
        ax.axvline(
            media,
            color="#E24B4A",
            linestyle="--",
            linewidth=1.5,
            label=f"Média={media:.1f}",
        )
        ax.axvline(
            mediana,
            color="#F39C12",
            linestyle=":",
            linewidth=1.5,
            label=f"Mediana={mediana:.1f}",
        )
        ax.legend(fontsize=7)

    for ax in axes.ravel()[len(num_cols) :]:
        ax.set_visible(False)

    plt.suptitle("Distribuição das Variáveis Numéricas", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


#  Análise Bivariada


def plot_bivariate(df: pd.DataFrame, num_cols: list):

    # Identifica quais variáveis discriminam melhor os churners.
    ncols = 3
    nrows = int(np.ceil(len(num_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))

    for ax, col in zip(axes.ravel(), num_cols):
        g0 = df.loc[df[TARGET] == 0, col].dropna()
        g1 = df.loc[df[TARGET] == 1, col].dropna()
        bp = ax.boxplot(
            [g0, g1],
            patch_artist=True,
            labels=["Não Churn", "Churn"],
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor(COR_NO_CHURN)
        bp["boxes"][1].set_facecolor(COR_CHURN)
        ax.set_title(col, fontweight="bold")
        _, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        sig = "✅ sig." if p < 0.05 else "❌ n.s."
        ax.set_xlabel(f"p={p:.3f} {sig}", fontsize=8)

    for ax in axes.ravel()[len(num_cols) :]:
        ax.set_visible(False)

    plt.suptitle(
        "Distribuição por Grupo (Churn vs. Não Churn)\nTeste Mann-Whitney",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, titulo: str = "Matriz de Confusão"):

    from sklearn.metrics import confusion_matrix

    matriz = confusion_matrix(y_true, y_pred)
    sns.heatmap(matriz, annot=True, fmt="d")
    plt.title(titulo)
    plt.show()
