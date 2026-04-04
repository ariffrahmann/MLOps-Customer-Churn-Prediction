import pandas as pd
from datetime import datetime
import os

RAW_DIR = "data/raw"

def ingest_data(source_path):
    # baca data
    if source_path.endswith(".csv"):
        df = pd.read_csv(source_path)
    elif source_path.endswith(".xlsx"):
        df = pd.read_excel(source_path)
    else:
        raise ValueError("Format file tidak didukung")

    # timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RAW_DIR, f"data_{timestamp}.csv")

    df.to_csv(output_path, index=False)

    print(f"Data berhasil disimpan ke: {output_path}")

if __name__ == "__main__":
    ingest_data("data/external/ecommerce_churn_kaggle.csv")