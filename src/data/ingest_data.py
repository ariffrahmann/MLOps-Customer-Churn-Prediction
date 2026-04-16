import pandas as pd
import os
from datetime import datetime

RAW_DIR = "data/raw"
CHUNK_SIZE = 100  

def ingest_data(source_path):
    os.makedirs(RAW_DIR, exist_ok=True)

    # cari total data yang sudah pernah diambil
    existing_files = sorted([
        f for f in os.listdir(RAW_DIR) if f.endswith(".csv")
    ])

    total_rows_taken = 0
    for file in existing_files:
        file_path = os.path.join(RAW_DIR, file)
        df = pd.read_csv(file_path)
        total_rows_taken += len(df)

    print(f"Total data sebelumnya: {total_rows_taken} baris")

    # ambil batch baru
    if source_path.endswith(".csv"):
        new_df = pd.read_csv(
            source_path,
            skiprows=range(1, total_rows_taken + 1),
            nrows=CHUNK_SIZE
        )
    elif source_path.endswith(".xlsx"):
        new_df = pd.read_excel(
            source_path,
            skiprows=range(1, total_rows_taken + 1),
            nrows=CHUNK_SIZE
        )
    else:
        raise ValueError("Format file tidak didukung")

    # kalau sudah habis
    if new_df.empty:
        print("Tidak ada data baru untuk diambil.")
        return

    # simpan pakai timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RAW_DIR, f"data_{timestamp}.csv")

    new_df.to_csv(output_path, index=False)

    print(f"Data batch baru: {len(new_df)} baris")
    print(f"Disimpan ke: {output_path}")


if __name__ == "__main__":
    ingest_data("data/external/ecommerce_churn_kaggle.csv")