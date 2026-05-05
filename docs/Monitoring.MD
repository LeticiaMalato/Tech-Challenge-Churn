# Plano de Monitoramento — Churn Predictor

---

## Visão Geral

O modelo em produção é monitorado em três dimensões: **saúde da API**, **qualidade das predições** e **drift de dados/conceito**. Este documento define métricas, alertas e o playbook de resposta para cada cenário de falha.

---

## 1. Métricas de Monitoramento

### 1.1 Saúde da API (Infraestrutura)

| Métrica | Descrição | Coleta |
|---|---|---|
| `latency_p95_ms` | Latência do percentil 95 em `/predict` | Header `X-Process-Time` |
| `latency_p99_ms` | Latência do percentil 99 | Header `X-Process-Time` |
| `error_rate_5xx` | % de respostas 5xx nas últimas 1h | Logs estruturados |
| `error_rate_422` | % de respostas 422 (payload inválido) | Logs estruturados |
| `model_loaded` | Booleano — modelo carregado em memória | `/health` endpoint |
| `requests_per_minute` | Volume de requisições | Logs estruturados |

### 1.2 Qualidade das Predições (Modelo)

| Métrica | Descrição | Frequência |
|---|---|---|
| `churn_rate_predicted` | % de predições com `churn=True` nas últimas 24h | Diária |
| `avg_probability` | Média das probabilidades retornadas | Diária |
| `probability_distribution` | Histograma das probabilidades (decis) | Semanal |
| `predictions_in_uncertainty_zone` | % de predições com probabilidade entre 0.35–0.55 | Diária |

### 1.3 Drift de Dados (Features)

| Métrica | Descrição | Frequência |
|---|---|---|
| `PSI_tenure_months` | Population Stability Index de `Tenure Months` | Semanal |
| `PSI_monthly_charges` | PSI de `Monthly Charges` | Semanal |
| `pct_fiber_optic` | % de clientes com `Internet Service = Fiber optic` | Semanal |
| `pct_month_to_month` | % de clientes com `Contract = Month-to-month` | Semanal |
| `missing_rate_per_feature` | % de valores ausentes por feature | Diária |

### 1.4 Drift de Conceito (Quando labels disponíveis)

| Métrica | Descrição | Frequência |
|---|---|---|
| `f1_rolling_30d` | F1 da classe churn nos últimos 30 dias | Mensal |
| `recall_rolling_30d` | Recall nos últimos 30 dias | Mensal |
| `precision_rolling_30d` | Precision nos últimos 30 dias | Mensal |

> Labels reais ficam disponíveis com atraso de 30–60 dias após a predição. Agendar avaliação retroativa mensal.

---

## 2. Alertas

### Nível CRÍTICO — Ação imediata (< 15 minutos)

| Alerta | Condição | Canal |
|---|---|---|
| `API_DOWN` | `/health` retorna não-200 por mais de 2 minutos | PagerDuty + Slack #alertas-criticos |
| `MODEL_NOT_LOADED` | `model_loaded = false` por mais de 1 minuto | PagerDuty + Slack #alertas-criticos |
| `HIGH_ERROR_RATE` | `error_rate_5xx > 5%` nas últimas 5 min | PagerDuty + Slack #alertas-criticos |
| `LATENCY_SPIKE` | `latency_p99 > 2000ms` por mais de 5 minutos | PagerDuty + Slack #alertas-criticos |

### Nível ALTO — Investigar em < 2 horas

| Alerta | Condição | Canal |
|---|---|---|
| `LATENCY_DEGRADATION` | `latency_p95 > 500ms` por mais de 15 minutos | Slack #alertas-ml |
| `CHURN_RATE_SPIKE` | `churn_rate_predicted > 50%` nas últimas 24h | Slack #alertas-ml |
| `CHURN_RATE_DROP` | `churn_rate_predicted < 5%` nas últimas 24h | Slack #alertas-ml |
| `HIGH_422_RATE` | `error_rate_422 > 10%` na última hora | Slack #alertas-ml |

### Nível MÉDIO — Investigar em < 24 horas

| Alerta | Condição | Canal |
|---|---|---|
| `DATA_DRIFT_PSI` | Qualquer PSI de feature > 0.2 | Slack #alertas-ml |
| `HIGH_UNCERTAINTY` | `predictions_in_uncertainty_zone > 30%` | Slack #alertas-ml |
| `MISSING_FEATURES` | `missing_rate > 5%` em qualquer feature | Slack #alertas-ml |
| `PERFORMANCE_DEGRADATION` | `f1_rolling_30d` cai mais de 10% vs baseline | Slack #alertas-ml |

---

## 3. Playbook de Resposta

### PLAY-001: API fora do ar (`API_DOWN`)

```
1. Verificar logs do processo uvicorn
   → docker logs churn-api --tail 100

2. Checar uso de memória e CPU
   → docker stats

3. Reiniciar o serviço
   → docker restart churn-api

4. Se persistir: verificar se o artefato .joblib existe e não está corrompido
   → python -c "import joblib; joblib.load('artifacts/churn_pipeline.joblib')"

5. Escalar para engenharia se não resolvido em 15 min
```

### PLAY-002: Modelo não carregado (`MODEL_NOT_LOADED`)

```
1. Verificar se o artefato existe no caminho configurado
   → ls -la artifacts/churn_pipeline.joblib

2. Se ausente: promover a partir do último run registrado no MLflow
   → python main.py   (executa treino completo e promove o melhor modelo)

3. Reiniciar a API para acionar o lifespan de carregamento
   → docker restart churn-api

4. Confirmar via /health que model_loaded = true
```

### PLAY-003: Alta taxa de erros 5xx (`HIGH_ERROR_RATE`)

```
1. Checar logs estruturados para identificar o erro
   → grep '"levelname": "ERROR"' /var/log/churn-api.json | tail -50

2. Erros mais comuns e causas:
   a) "Erro na inferência" → pipeline incompatível com features recebidas
      → Verificar se houve mudança no schema de entrada
   b) "Modelo não carregado" → artefato ausente (ver PLAY-002)
   c) Erro de memória → reiniciar com mais memória alocada

3. Se causa for schema: acionar time de dados para verificar mudanças
   no sistema de origem
```

### PLAY-004: Drift de dados detectado (`DATA_DRIFT_PSI`)

```
1. Identificar qual feature está em drift (PSI > 0.2)

2. Investigar causa:
   a) Mudança de produto/plano → esperada, documentar
   b) Bug no pipeline de ingestão → corrigir urgentemente
   c) Sazonalidade → monitorar por 7 dias antes de retreinar

3. Se drift confirmado e persistente (> 7 dias):
   a) Coletar novos dados de treino
   b) Executar python main.py com dataset atualizado
   c) Comparar métricas do novo modelo vs. atual no MLflow
   d) Promover novo modelo se métricas >= atual
   e) Reiniciar API para carregar novo artefato

4. Documentar o incidente no histórico de versões do Model Card
```

### PLAY-005: Degradação de performance (`PERFORMANCE_DEGRADATION`)

```
1. Confirmar que o cálculo usa labels reais (não predições anteriores)

2. Verificar se há viés de seleção nos dados rotulados
   (ex: só temos labels de clientes que foram abordados)

3. Analisar por segmento:
   → recall por tipo de contrato
   → recall por faixa de tenure
   → recall por tipo de internet

4. Se degradação generalizada → retreinar (ver PLAY-004, passo 3)
5. Se degradação segmentada → avaliar regra de negócio específica
   para o segmento afetado (ex: threshold diferente por segmento)
```

---

## 4. Frequência de Revisão

| Atividade | Frequência | Responsável |
|---|---|---|
| Revisão de alertas ativos | Diária | On-call |
| Análise de distribuição de predições | Semanal | Time ML |
| Cálculo de PSI das features | Semanal | Time ML |
| Avaliação de performance com labels reais | Mensal | Time ML |
| Revisão do Model Card | A cada retreino | Time ML |
| Revisão de threshold | Trimestral ou após mudança de produto | Time ML + Negócio |

---

## 5. Referência de PSI

| PSI | Interpretação | Ação |
|---|---|---|
| < 0.1 | Sem drift significativo | Nenhuma |
| 0.1 – 0.2 | Drift moderado | Monitorar de perto |
| > 0.2 | Drift severo | Investigar e considerar retreino |