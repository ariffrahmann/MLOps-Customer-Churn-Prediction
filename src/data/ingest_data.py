import pandas as pd
import os

RAW_DIR = "data/raw"
OUTPUT_FILE = "dataset.csv"
# jumlah data per ingest
CHUNK_SIZE = 100  

def ingest_data(source_path):
    # baca data baru
    if source_path.endswith(".csv"):
        new_df = pd.read_csv(source_path, nrows=CHUNK_SIZE)
    elif source_path.endswith(".xlsx"):
        new_df = pd.read_excel(source_path, nrows=CHUNK_SIZE)
    else:
        raise ValueError("Format file tidak didukung")

    output_path = os.path.join(RAW_DIR, OUTPUT_FILE)

    # kalau dataset sudah ada → append
    if os.path.exists(output_path):
        old_df = pd.read_csv(output_path)
        combined_df = pd.concat([old_df, new_df], ignore_index=True)

        print(f"Dataset lama: {len(old_df)} baris")
        print(f"Data baru (batch): {len(new_df)} baris")
        print(f"Total setelah update: {len(combined_df)} baris")
    else:
        combined_df = new_df
        print(f"Dataset baru dibuat: {len(new_df)} baris")

    # simpan
    combined_df.to_csv(output_path, index=False)

    print(f"Data berhasil disimpan ke: {output_path}")


if __name__ == "__main__":
    ingest_data("data/external/ecommerce_churn_kaggle.csv")