# Arquitetura de Deploy — Churn Predictor

---

## Decisão: Real-Time via API REST

### Justificativa

| Critério | Batch | Real-Time |
|---|---|---|
| Latência tolerada | Horas / dias | Segundos |
| Trigger de predição | Agendado (cron) | Evento de negócio |
| Custo de infraestrutura | Menor | Maior |
| Integração com CRM/atendimento | Difícil (arquivo) | Nativa (API) |
| Atualização de features | Snapshot diário | Dado mais recente disponível |

**Escolhemos real-time pelos seguintes motivos:**

1. **Integração com atendimento ao cliente:** quando um agente abre o perfil de um cliente, o sistema de CRM consulta a API e exibe o score de churn em tempo real — permitindo ação imediata durante a ligação
2. **Trigger por evento:** ações específicas (abertura de reclamação, consulta de cancelamento, downgrade de plano) podem acionar a predição automaticamente
3. **Latência aceitável:** a inferência de um MLP com ~30 features leva menos de 50ms, dentro do SLO de 200ms definido para a API
4. **Dataset de tamanho moderado:** ~7.000 clientes não justifica infraestrutura de batch distribuído (Spark, Dataflow)

> **Nota:** Uma execução batch diária complementar pode ser usada para gerar rankings de risco para campanhas de retenção proativas — os dois modos não são excludentes.

---

## Diagrama de Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                          PRODUÇÃO                               │
│                                                                 │
│  ┌──────────┐    POST /predict     ┌─────────────────────────┐  │
│  │  CRM /   │ ────────────────────▶│   FastAPI (uvicorn)     │  │
│  │ Frontend │                      │                         │  │
│  └──────────┘ ◀────────────────────│  /predict    /health    │  │
│                churn_probability   └───────────┬─────────────┘  │
│                                               │                 │
│                                   ┌───────────▼─────────────┐   │
│                                   │   sklearn Pipeline      │   │
│                                   │   (joblib artifact)     │   │
│                                   │                         │   │
│                                   │   Preprocessor →        │   │
│                                   │   MLP PyTorch           │   │
│                                   └─────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     MLFLOW SERVER                        │   │
│  │  Experiment tracking │ Model Registry │ Artifact Store   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Componentes

### API (FastAPI + Uvicorn)

| Endpoint | Método | Descrição |
|---|---|---|
| `/predict` | POST | Recebe features do cliente, retorna probabilidade e classe |
| `/health` | GET | Status da API e confirmação de modelo carregado |
| `/docs` | GET | Swagger UI gerado automaticamente |

**Características:**
- Validação de entrada via Pydantic (422 em payload inválido)
- Logging estruturado em JSON (python-json-logger)
- Middleware de latência com header `X-Process-Time`
- Graceful degradation: API sobe sem modelo, retorna 503 em `/predict`

### Artefato de Modelo

```
artifacts/
└── churn_pipeline.joblib
    ├── pipeline      # sklearn Pipeline completo (preprocessamento + modelo)
    ├── threshold     # 0.4 (limiar de decisão)
    └── metadata      # run_id MLflow, run_name, dataset_hash, churn_rate
```

O artefato é carregado uma única vez no startup (lifespan) e mantido em memória.

### MLflow

Usado para:
- Rastreamento de experimentos (parâmetros, métricas, artefatos)
- Versionamento de modelos (Model Registry)
- Comparação de runs entre baselines e MLP

---

## Ambiente de Execução

### Local / Desenvolvimento

```bash
uvicorn src.api.app:app --reload --port 8000
```

### Produção

```bash
uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --log-level info
```


## SLOs (Service Level Objectives)

| Métrica | Objetivo |
|---|---|
| Latência p95 `/predict` | < 200ms |
| Latência p99 `/predict` | < 500ms |
| Disponibilidade | > 99% |
| Taxa de erro 5xx | < 1% |

---

## Decisões Técnicas Complementares

**Por que joblib e não MLflow Model Serving?**
MLflow serve modelos sklearn nativamente, mas adiciona overhead de infraestrutura (servidor MLflow em produção). Para este escopo, serializar o pipeline com joblib e carregá-lo na API FastAPI é mais simples, testável e sem dependência de serviço externo em runtime.

**Por que um único worker em desenvolvimento?**
PyTorch e sklearn não são thread-safe em todos os contextos. Em produção, usar `--workers 2` com processos separados garante isolamento. Para escala horizontal, replicar containers é preferível a aumentar workers por instância.

**Por que não Kubernetes/Serverless agora?**
O volume atual (~7.000 clientes, inferência sob demanda) não justifica a complexidade operacional. A arquitetura escolhida é evoluível: o container Docker pode ser deployado em qualquer orquestrador sem mudanças no código.