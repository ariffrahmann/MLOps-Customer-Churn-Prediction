import pandas as pd
from pathlib import Path

# konfigurasi
BATCH_SIZE = 200
DATA_EXTERNAL = "data/external/ecommerce_churn_kaggle.csv"
RAW_FOLDER = Path("data/raw")

RAW_FOLDER.mkdir(parents=True, exist_ok=True)


def ingest_batch(batch_number: int):
    """
    Mengambil batch data dari dataset eksternal
    lalu menyimpannya ke folder data/raw
    """

    offset = batch_number * BATCH_SIZE

    df = pd.read_csv(
        DATA_EXTERNAL,
        skiprows=range(1, offset + 1),
        nrows=BATCH_SIZE
    )

    # metadata batch
    df["batch_id"] = batch_number
    df["ingestion_date"] = pd.Timestamp.today().date()

    output_file = RAW_FOLDER / f"batch_{batch_number:03d}_raw.csv"

    df.to_csv(output_file, index=False)

    print("===== INGESTION BERHASIL =====")
    print("Batch:", batch_number)
    print("Jumlah data:", len(df))
    print("File tersimpan:", output_file)
    print(df.head())


if __name__ == "__main__":
    ingest_batch(batch_number=1)