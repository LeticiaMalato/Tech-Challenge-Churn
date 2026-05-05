# Churn Predictor — Telecom

Sistema completo de predição de churn para operadora de telecomunicações, cobrindo da ingestão de dados até o modelo servido via API REST. O modelo central é uma MLP treinada com PyTorch, comparada com baselines Scikit-Learn e rastreada com MLflow.

---

##  Estrutura do Projeto

---

## Setup

### Pré-requisitos

- Python 3.10+
- Git

### Instalação local

```bash
# 1. Clone o repositório
git clone https://github.com/LeticiaMalato/Tech-Challenge-Churn.git
cd Tech-Challenge-Churn

# 2. Crie o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Instale as dependências
pip install -r requirements.txt

```

---

## Executando o Pipeline Completo

1. Treinar todos os modelos 
```bash
python main.py
```
O script executa em sequência:

- EDA com visualizações (distribuição do target, outliers, análise bivariada)
- Treino de baselines: DummyClassifier, Logistic Regression, Decision Tree, Random Forest
- Treino da MLP PyTorch com early stopping
- Comparação de métricas e análise financeira (FP/FN/resultado líquido)
- Promoção do melhor modelo para artifacts/churn_pipeline.joblib


2. Acompanhar experimentos no MLflow
```bash
mlflow ui --port 5000
# Acesse http://localhost:5000
```

3. Servir o modelo via API
```bash
uvicorn src.api.app:app --reload --port 8000
# Documentação interativa: http://localhost:8000/docs
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
pytest

# Com cobertura
task test-cov

# Apenas smoke tests da API
pytest tests/unit/test_api_smoke.py -v

# Lint + format + testes (CI completo)
task check
```

---

## Decisões de Arquitetura

**Por que MLP e não LSTM/Transformer?**
Os dados são tabulares e estáticos, sem sequência temporal explícita. MLP com regularização adequada supera modelos mais complexos nesse contexto, com menor custo de treinamento e inferência.

**Por que real-time e não batch?**
Permite integração direta com CRM e ação imediata durante atendimento. A latência de inferência é menor que 50ms, dentro do SLO de 200ms. Ver docs/DEPLOY_ARCHITECTURE.md.

**Por que FastAPI?**
Validação automática com Pydantic, documentação Swagger gerada em `/docs`, async nativo e performance superior ao Flask para I/O intensivo.

**Por que MLflow?**
MLflow é open source, pode ser auto-hospedado sem custo e integra nativamente com o Model Registry para controle de versão e stage de modelos.
O custo de um falso negativo (cliente que cancela sem ação) é maior que o de um falso positivo (campanha desnecessária). o threshold foi calibrado para aumentar recall na classe chun. Ver docs/MODEK_CARD.md .

**Por que threshold 0.4?**
O custo de um falso negativo (cliente que cancela sem ação) é maior que o de um falso positivo (campanha desnecessária). o threshold foi calibrado para aumentar recall na classe chun. Ver docs/MODEK_CARD.md .

## Time

| Nome |
|---|---|
| — | Letícia Malato|
| — | Rafael Maranhão |

---

##  Licença

MIT
