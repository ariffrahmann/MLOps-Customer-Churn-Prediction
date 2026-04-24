# src/data/preprocess.py

import pandas as pd
import os
from datetime import datetime

RAW_DIR       = "data/raw"
INTERIM_DIR   = "data/interim"
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
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # Numerik → isi dengan median
    for col in num_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Kategorikal → isi dengan "unknown"
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna("unknown")

    # SAVE INTERIM 
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    interim_path = os.path.join(INTERIM_DIR, f"interim_{timestamp}.csv")
    df.to_csv(interim_path, index=False)
    print(f"🧹 Interim saved: {interim_path}")

    df = pd.read_csv(interim_path)


    # Encoding target churn — handle angka (0/1) DAN string (yes/no)
    if "churn" in df.columns:
        if df["churn"].dtype in ["int64", "float64"]:
            # Sudah numerik, langsung convert ke int
            df["churn"] = df["churn"].astype(int)
        else:
            # Bertipe string, map ke 0/1
            df["churn"] = (
                df["churn"]
                .astype(str)
                .str.lower()
                .map({"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
            )

        print(f"✅ Churn encoded : {df['churn'].value_counts().to_dict()}")
        print(f"⚠️  Churn NaN    : {df['churn'].isnull().sum()}")

    # Tenure grouping
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
    print(f"📊 Final shape    : {df.shape}")


if __name__ == "__main__":
    preprocess()