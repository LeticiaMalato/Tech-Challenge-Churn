# src/config.py -> centraliza as configurações do projeto (caminhos, parâmetros, etc)

#  Caminhos 
DATA_PATH = "data/Telco_customer_churn.xlsx"

#  Variável target 
TARGET = "Churn Target"

#  Colunas para remover
NUM_COLS_TO_DROP  = [
    "Count",        # constante = 1 para todos os registros
    "Zip Code",     # alta cardinalidade, redundante
    "Latitude",
    "Longitude",
    "Churn Score",  # derivado do churn — não disponível em produção
    "CLTV",         # calculado com histórico completo
    "Total Charges",# derivada de Monthly Charges × Tenure Months
    "Churn Label",  # repetição do target
]

CAT_COLS_TO_DROP  = [
    "CustomerID",
    "Country",
    "State",
    "City",
    "Lat Long",
    "Churn Reason",  # preenchido só para quem já cancelou — leakage
]

# Parâmetros de split 
TEST_SIZE       = 0.25
VAL_SIZE        = 0.50   # % do test_size que vira validação
RANDOM_STATE    = 42

#  Parâmetros do MLP
EPOCHS    = 10_000
PATIENCE  = 10
LR        = 0.001
BATCH_SIZE = 10

#  Parâmetros de negócio  
LTV             = 500    # receita perdida por cliente que cancela
RETENTION_COST  = 50     # custo de abordar um cliente
RETENTION_RATE    = 0.30   # % de clientes que ficam quando abordados

#  MLflow 
MLFLOW_EXPERIMENT = "Telco-Churn"

# API 
MODEL_ARTIFACT_PATH = "artifacts/churn_pipeline.joblib"
INFERENCE_THRESHOLD = 0.4  
API_LOG_LEVEL       = "INFO"