# src/data_utils.py
import json
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    # Ensure consistent column names
    expected_cols = ['User ID','Gender','Age','Interests','Swiping History','Looking For']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    # Clean Interests column: convert JSON-like strings to lists
    def parse_interests(x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return x
        try:
            return list(json.loads(x))
        except Exception:
            # fallback: split by comma
            return [s.strip() for s in str(x).split(',') if s.strip()]
    df['Interests'] = df['Interests'].apply(parse_interests)
    # Fill NaNs for numeric columns
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].median())
    df['Swiping History'] = pd.to_numeric(df['Swiping History'], errors='coerce').fillna(0)
    df['Looking For'] = df['Looking For'].fillna('Unknown')
    return df
