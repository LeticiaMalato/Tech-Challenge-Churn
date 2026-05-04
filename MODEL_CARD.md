#  Model Card — Churn Predictor MLP

> Documento de transparência do modelo `churn-mlp-v1.0`, conforme boas práticas de ML responsável.

---

## 1. Identidade do Modelo

| Campo | Detalhe |
|---|---|
| **Nome** | `churn-mlp-v1.0` |
| **Tipo** | MLP (Multi-Layer Perceptron) — PyTorch |
| **Tarefa** | Classificação binária: churn (1) / não-churn (0) |
| **Versão** | 1.0.0 |
| **Responsável** | Time de ML — Fase 1 |
| **Rastreamento** | MLflow — Experiment: `churn_mlp_prod` |

---

## 2. Dados de Treinamento

- **Dataset:** Telco Customer Churn (IBM / Kaggle) — ~7.000 clientes, 21 features originais
- **Split:** 70% treino / 15% validação / 15% teste (estratificado por classe)
- **Desbalanceamento:** ~26% churn vs ~74% não-churn → tratado com `pos_weight` e SMOTE

### Features utilizadas

| Tipo | Features |
|---|---|
| Originais | `tenure`, `monthly_charges`, `total_charges`, `contract`, `internet_service`, `tech_support`, `paperless_billing`, `payment_method` |
| Engenhadas | `charges_per_month_ratio`, `tenure_bucket`, `num_services` |

### Pré-processamento

- `StandardScaler` nas variáveis numéricas
- `OneHotEncoder` nas variáveis categóricas
- Pipeline serializado com `joblib` e registrado no MLflow

---

## 3. Arquitetura

Input (N features)
    → Linear(N, N//2) + ReLU
    → Linear(N//2, N//2) + ReLU
    → Linear(N//2, 1) + Sigmoid

| Hiperparâmetro | Valor |
|---|---|
| Loss | BCELoss com `pos_weight` |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Early stopping | patience=10 epochs |
| Threshold de decisão | 0.4 |

---

## 4. Performance

### Comparação com Baselines

| Modelo | AUC-ROC | F1 (churn) | Precision | Recall | Accuracy |
|---|---|---|---|---|---|

> **Métrica principal:** AUC-ROC + F1 (classe churn). O custo de um falso negativo (churn não detectado) é maior que o de um falso positivo (campanha desnecessária).



---

## 5. Limitações

- **Dados estáticos:** o modelo não captura séries temporais de comportamento (ex: queda de uso nos últimos 30 dias)
- **Viés geográfico:** dataset IBM/EUA — padrões de churn podem diferir em operadoras brasileiras
- **Drift de conceito:** contratos e planos mudam; features como `contract_type` podem perder poder preditivo
- **Sem variáveis externas:** não inclui reclamações no Anatel, NPS ou interações com suporte
- **Threshold fixo:** o 0.45 pode precisar de recalibração por segmento de cliente

---

## 6. Vieses Identificados

| Viés | Impacto | Mitigação |
|---|---|---|
| Desbalanceamento de classe | Subestima churn | `pos_weight` + SMOTE + threshold calibrado |
| Clientes novos (tenure < 3 meses) | Pouco histórico → predições menos confiáveis | Regra de negócio paralela para este segmento |
| Planos legados | Sub-representados no treino | Monitorar recall separado por tipo de contrato |

---

## 7. Cenários de Falha

**Quando o modelo falha:**

- Cliente mudou de plano recentemente → features desatualizadas no banco
- Campanha de retenção ativa → labels de treino futuro distorcidos artificialmente
- Sazonalidade → fim de ano e Black Friday alteram padrões de cancelamento
- Novos produtos → features one-hot não vistas no treino causam comportamento imprevisível

**O que NÃO fazer com este modelo:**

-  Usar como único critério de bloqueio de cancelamento
-  Aplicar em clientes corporativos (B2B) — dataset é 100% B2C
-  Confiar em predições com probabilidade entre 0.40–0.55 (zona de incerteza)
-  Ignorar alertas de drift por mais de 7 dias

---

## 8. Histórico de Versões

| Versão | Data | Mudança |
|---|---|---|
| 1.0.0 | — | Versão inicial — MLP PyTorch |