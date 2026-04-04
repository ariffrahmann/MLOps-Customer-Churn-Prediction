import pandas as pd
import os
from datetime import datetime

RAW_DIR = "data/raw"
INTERIM_DIR = "data/interim"
PROCESSED_DIR = "data/processed"


def get_latest_raw():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]

    if not files:
        raise FileNotFoundError("Tidak ada file di data/raw")

    files.sort()
    return os.path.join(RAW_DIR, files[-1])


def preprocess():
    # LOAD RAW DATA
    raw_path = get_latest_raw()
    print(f"📥 Load RAW: {raw_path}")

    df = pd.read_csv(raw_path)

    # CLEANING
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df = df.drop_duplicates()

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    # HANDLE MISSING VALUE

    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # numerik → median
    for col in num_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna("unknown")

    # SAVE INTERIM
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    interim_path = os.path.join(INTERIM_DIR, f"interim_{timestamp}.csv")

    df.to_csv(interim_path, index=False)
    print(f"🧹 Interim saved: {interim_path}")

    df = pd.read_csv(interim_path)

    # FINAL PROCESSING
    # encoding target
    if "churn" in df.columns:
        df["churn"] = df["churn"].astype(str).str.lower()
        df["churn"] = df["churn"].map({
            "yes": 1,
            "no": 0
        })

    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 60],
            labels=["low", "medium", "high"]
        )

    # SAVE PROCESSED
    processed_path = os.path.join(PROCESSED_DIR, f"clean_{timestamp}.csv")

    df.to_csv(processed_path, index=False)

    print(f"✅ Processed saved: {processed_path}")
    print(f"📊 Final shape: {df.shape}")


if __name__ == "__main__":
    preprocess()