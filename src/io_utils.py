import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

def load_patient_df(filename: str = "rrp_synthetic.csv") -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    return df

def load_hpo_mapping(filename: str = "hpo_mapping.json") -> dict:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)