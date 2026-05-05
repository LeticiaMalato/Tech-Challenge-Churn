# Churn Predictor — Telecom

Sistema completo de predição de churn para operadora de telecomunicações, cobrindo da ingestão de dados até o modelo servido via API REST. O modelo central é uma MLP treinada com PyTorch, comparada com baselines Scikit-Learn e rastreada com MLflow.

---

##  Estrutura do Projeto

---

## Setup

### Pré-requisitos

- Python 3.10+
- Docker (opcional, recomendado para produção)
- Git

### Instalação local

```bash
# 1. Clone o repositório
git clone https://github.com/seu-time/churn-predictor.git
cd churn-predictor

# 2. Crie o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure variáveis de ambiente
cp .env.example .env
# Edite .env com suas configs (MLFLOW_TRACKING_URI, etc.)
```

### Instalação com Docker

```bash
docker-compose up --build
# API disponível em http://localhost:8000/docs
```

---

## Executando o Pipeline Completo

```bash
# 1. Ingestão e pré-processamento
python src/data/ingestion.py --input data/raw/telco.csv

# 2. Treinar baselines (Scikit-Learn)
python src/models/baselines.py --experiment churn_baselines

# 3. Treinar MLP (PyTorch)
python src/models/train.py \
    --epochs 100 \
    --lr 0.001 \
    --hidden 128 64 32 \
    --dropout 0.3 \
    --experiment churn_mlp_prod

# 4. Comparar experimentos no MLflow
mlflow ui --port 5000

# 5. Registrar melhor modelo
python src/models/register_model.py --run-id <RUN_ID> --name churn-mlp-prod

# 6. Servir via API
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

---

##  Usando a API

### Predição

```bash
POST https://tech-challenge-churn.onrender.com//predict
Content-Type: application/json

{
  "gender": "Male",
  "senior_citizen": "No",
  "partner": "Yes",
  "dependents": "No",
  "tenure_months": 24,
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 65.5,
  "phone_service": "Yes",
  "internet_service": "Fiber optic",
  "multiple_lines": "Yes",
  "online_security": "No",
  "online_backup": "No",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "No",
  "streaming_movies": "No"
}
```

**Resposta:**

```json
{
  "churn_probability": 0.73,
  "churn_prediction": 1,
  "risk_level": "high",
  "model_version": "1.0.0"
}
```

### Health Check

```bash
GET https://tech-challenge-churn.onrender.com//health
→ {"status": "ok", "model": "churn-mlp-v1.0", "uptime": "2h 34m"}
```

Documentação interativa disponível em: `http://localhost:8000/docs`

---

## Testes

```bash
# Todos os testes
pytest tests/ -v

# Com cobertura
pytest tests/ --cov=src --cov-report=html

# Apenas API
pytest tests/test_api.py -v
```

---

## Decisões de Arquitetura

**Por que MLP e não LSTM/Transformer?**
Os dados são tabulares e estáticos, sem sequência temporal explícita. MLP com regularização adequada supera modelos mais complexos nesse contexto, com menor custo de treinamento e inferência.

**Por que FastAPI?**
Validação automática com Pydantic, documentação Swagger gerada em `/docs`, async nativo e performance superior ao Flask para I/O intensivo.

**Por que MLflow e não W&B?**
MLflow é open source, pode ser auto-hospedado sem custo e integra nativamente com o Model Registry para controle de versão e stage de modelos (Staging → Production → Archived).

**Por que threshold 0.45 e não 0.5?**
Calibrado para maximizar F1 da classe churn na validação. O negócio aceita mais falsos positivos (contatar cliente que não ia sair) do que falsos negativos (perder cliente sem agir).

---

## Time

| Nome |
|---|---|
| — | Letícia Malato|
| — | Rafael Maranhão |

---

##  Licença

MIT
