import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix

from src.config import LTV, RETENTION_COST, RETENTION_RATE


def calculate_financial_result(
    y_true,
    y_pred,
    ltv=LTV,
    retention_cost=RETENTION_COST,
    retention_rate=RETENTION_RATE,
) -> dict:

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tp_gain = tp * (retention_rate * ltv - retention_cost)
    fp_cost = fp * retention_cost
    fn_cost = fn * ltv
    net_result = tp_gain - fp_cost - fn_cost

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "tp_gain": round(tp_gain, 2),
        "fp_cost": round(fp_cost, 2),
        "fn_cost": round(fn_cost, 2),
        "net_result": round(net_result, 2),
    }


def compare_models_financial(results: list):
    col_model = 30
    col_num = 6
    col_cost = 13

    header = (
        f"{'Modelo':<{col_model}} {'FP':>{col_num}} {'FN':>{col_num}}"
        f"  {'Custo FP':>{col_cost}}  {'Custo FN':>{col_cost}}  {'Resultado Líquido':>{col_cost}}"
    )

    separator = "═" * len(header)
    best_net = max(r["net_result"] for r in results)

    print(f"\n{separator}")
    print(header)
    print("─" * len(header))
    for r in results:
        marker = " ★" if r["net_result"] == best_net else "  "
        fp_col = f"R$ {r['fp_cost']:>9,.0f}"
        fn_col = f"R$ {r['fn_cost']:>9,.0f}"
        net_col = f"R$ {r['net_result']:>9,.0f}"
        print(
            f"{r['model']:<{col_model}} {r['FP']:>{col_num}} {r['FN']:>{col_num}}"
            f"  {fp_col:>{col_cost}}"
            f"  {fn_col:>{col_cost}}"
            f"  {net_col:>{col_cost}}"
            f"{marker}"
        )

    names = [r["model"] for r in results]
    fp_cost = [r["fp_cost"] for r in results]
    fn_cost = [r["fn_cost"] for r in results]
    net_result = [r["net_result"] for r in results]

    x = np.arange(len(names))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.bar(
        x - w / 2, fp_cost, w, label="FP Cost (unnecessary action)", color="#F39C12"
    )
    ax1.bar(x + w / 2, fn_cost, w, label="FN Cost (lost customer)", color="#E74C3C")
    ax1.set_title("Cost per error type", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Cost (R$)")
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"R${v:,.0f}"))

    ax2 = axes[1]
    colors = ["#27AE60" if v >= 0 else "#E74C3C" for v in net_result]
    ax2.bar(names, net_result, color=colors, edgecolor="white")
    ax2.set_title("Net result per model", fontsize=13, fontweight="bold")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("R$")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"R${v:,.0f}"))
    ax2.legend(
        handles=[
            mpatches.Patch(color="#27AE60", label="Gain"),
            mpatches.Patch(color="#E74C3C", label="Loss"),
        ]
    )

    plt.tight_layout()
    plt.show()

    best_model = max(results, key=lambda r: r["net_result"])
    print(f"\n Melhor modelo financeiramente: {best_model['model']}")
    print(f"   Resultado líquido estimado : R$ {best_model['net_result']:,.2f}")
    print(f"   FP (ações desnecessárias): {best_model['FP']} clientes")
    print(f"   FN (clientes perdidos)     : {best_model['FN']} clientes")
