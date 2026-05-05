#  Model Card — Churn Predictor MLP

> Documento de transparência do modelo `tech-challenge-churn-v1.0 `, conforme boas práticas de ML responsável.

---

## 1. Identidade do Modelo

| Campo | Detalhe |
|---|---|
| **Nome** | `tech-challenge-churn-v1.0` |
| **Tipo** | MLP (Multi-Layer Perceptron) — PyTorch |
| **Tarefa** | Classificação binária: churn (1) / não-churn (0) |
| **Versão** | 1.0.0 |
| **Threshold de decisão** | 0.4 |
| **Rastreamento** | MLflow — Experiment: `Telco-Churn` |
| **Artefato** | `artifacts/churn_pipeline.joblib` |

---

## 2. Dados de Treinamento

- **Dataset:** Telco Customer Churn (IBM Sample Data) — ~7.043 clientes, 33 colunas originais
- **Split:** 75% treino / 12.5% validação / 12.5% teste (estratificado por classe)
- **Desbalanceamento:** ~26,5% churn vs ~73,5% não-churn

### Features utilizadas após pré-processamento

| Tipo | Features |
|---|---|
| Numéricas | `Tenure Months`, `Monthly Charges` |
| Categóricas |  `Gender`, `Senior Citizen`, `Partner`, `Dependents`, `Phone Service`, `Multiple Lines`,` Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`, `Contract`, `Paperless Billing`, `Payment Metho` |

### Colunas removidas e motivo
| Coluna | Motivo |
|---|---|
| `Churn Score`, `Churn Label`, `Churn Reason` | Data leakage  |
| `Total Charges` | Derivada de Monthly Charges × Tenure Months  |
| `CLTV` | Calculado com histórico completo, indisponível em produção  |
| `CustomerID`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude` | Alta cardinalidade ou redundância geográfica  |
| `Count`, `Country`, `State`, `City` | Constantes ou sem variância útil  |

### Pré-processamento

- `StandardScaler` nas variáveis numéricas
- `OneHotEncoder` nas variáveis categóricas
- Conversão de `Yes/No` para `1/0`
- Pipeline serializado com `joblib` e registrado no MLflow

---

## 3. Arquitetura

Input (N features)
    → Linear(N, N//2) + ReLU
    → Linear(N//2, N//2) + ReLU
    → Linear(N//2, 1) + Sigmoid


| Hiperparâmetro | Valor |
|---|---|
| Loss | `BCELoss` |
| Optimizer | Adam (lr=0.001) |
| Batch size | 10 |
| Max epochs | 10.000 |
| Early stopping | patience=10 (monitorando val_loss) |
| Threshold de decisão | 0.4 |

---


## 4. Performance

> Devido ao desbalanceamento típico em problemas de churn, adotamos **PR-AUC** como métrica principal, pois ela fornece uma avaliação mais fiel da performance na classe minoritária (Saito & Rehmsmeier, 2015; Davis & Goadrich, 2006).
>
> Complementarmente, utilizamos o **F1-score** para otimização do threshold, uma vez que essa métrica equilibra precision e recall, sendo amplamente recomendada em cenários com classes desbalanceadas (He & Garcia, 2009).
>
> O **ROC-AUC** foi mantido como métrica secundária para comparação com a literatura, dado seu uso consolidado na avaliação de classificadores binários (Fawcett, 2006), apesar de suas limitações em cenários desbalanceados.
>
> Por fim, traduzimos o desempenho do modelo em valor de negócio por meio do cálculo de churn evitado, alinhando a solução com impacto financeiro direto:
>
> `Valor de Negócio = (TP × LTV × taxa_retenção) − ((TP + FP) × custo_retenção)`


### Comparação com Baselines

| Modelo                       | accuracy | f1_score | precision | recall | roc_auc | pr_auc |
| ---------------------------- | -------- | -------- | --------- | ------ | ------- | ------ |
| mlp_pytorch                  | 0.8173   | 0.6786   | 0.6367    | 0.7265 | 0.8773  | 0.6919 |
| logistic_regression_baseline | 0.8241   | 0.6453   | 0.6946    | 0.6026 | 0.8791  | 0.6974 |
| random_forest_baseline       | 0.8116   | 0.6121   | 0.6753    | 0.5598 | 0.8695  | 0.7018 |
| decision_tree_baseline       | 0.8104   | 0.6033   | 0.6791    | 0.5427 | 0.8446  | 0.6632 |
| dummy_baseline               | 0.7344   | 0.0000   | 0.0000    | 0.0000 | 0.5000  | 0.2566 |


> **Métrica principal:** AUC-ROC + F1 (classe churn). O custo de um falso negativo (churn não detectado) é maior que o de um falso positivo (campanha desnecessária).
### Análise de custo (threshold=0.4)

| Modelo              | TP | FP | FN  | Resultado Líquido |
| ------------------- | -- | -- | --- | ----------------- |
| Dummy               | —  | 0  | 234 | R$ -117.000       |
| Logistic Regression | —  | 62 | 93  | R$ -35.500        |
| Decision Tree       | —  | 60 | 107 | R$ -43.800        |
| Random Forest       | —  | 63 | 103 | R$ -41.550        |
| **MLP PyTorch**     | —  | 97 | 64  | R$ -19.850        |


| Parâmetro                       | Valor                 |
| ------------------------------- | --------------------- |
| LTV (receita perdida por churn) | R$ 500                |
| Custo de abordagem              | R$ 50                 |
| Taxa de retenção com campanha   | 30%                   |
| Resultado líquido estimado      | R$ -19.850  |



---

## 5. Limitações

**Dados e representatividade:**
- Dataset de origem norte-americana (IBM/EUA) — padrões de churn podem diferir em operadoras brasileiras com regulação Anatel, portabilidade e perfis de contrato distintos
- Dados estáticos: o modelo não captura séries temporais de comportamento (ex: queda de uso nos últimos 30 dias, histórico de chamadas ao suporte)
- Ausência de variáveis externas relevantes: NPS, reclamações regulatórias, promoções de concorrentes

**Modelo:**
- Threshold fixo em 0.4 — pode precisar de recalibração por segmento (ex: clientes corporativos vs. residenciais)
- Predições com probabilidade entre 0.35–0.50 estão na zona de incerteza — não devem ser usadas como único critério de ação
- Sem calibração de probabilidade (Platt Scaling / Isotonic Regression) — as probabilidades brutas podem não refletir frequências reais

---

## 6. Vieses Identificados

| Viés | Impacto | Mitigação |
|---|---|---|
|Desbalanceamento de classe (26,5% churn) | Modelo tende a prever não-churn | Threshold calibrado em 0.4; PR-AUC como métrica principal |
| Clientes novos (tenure < 3 meses) | Subrepresentados, predições instáveis | Regra de negócio paralela para este segmento |
| Planos legados / descontinuados | Sub-representados no treino | Monitorar recall separado por tipo de contrato |


---

## 7. Cenários de Falha

**Quando o modelo falha:**

| Cenário                                 | Sintoma                                |
| --------------------------------------- | -------------------------------------- | 
| Cliente mudou de plano recentemente     | Features desatualizadas no banco       | 
| Campanha de retenção ativa              | Labels de treino distorcidos           | 
| Sazonalidade (fim de ano, Black Friday) | Padrões fora da distribuição histórica |                                            
| Novos produtos ou planos                | Features one-hot não vistas no treino  | 
| Migração de sistema de billing          | Mudança no formato de Payment Method   |                                                 


**O que NÃO fazer com este modelo:**

-  Usar como único critério de bloqueio de cancelamento
-  Aplicar em clientes corporativos (B2B) 
-  Confiar em predições com probabilidade entre 0.35–0.50  sem revisão humana
-  Ignorar alertas de drift por mais de 7 dias
-  Retreinar sem validar o hash do dataset de origem

---

## 8. Histórico de Versões

| Versão | Data | Mudança |
|---|---|---|
| 1.0.0 | — | Versão inicial — MLP PyTorch |