from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_PATH = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw_data(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the IBM Telco Customer Churn CSV at data/raw/"
        )
    return pd.read_csv(path)


def main() -> None:
    df = load_raw_data()
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns from {DATASET_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
