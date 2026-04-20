

import pandas as pd
from src.config import DATA_PATH

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    
    df = pd.read_excel(path)
    print(df.shape)
    return df