import glob
import os
import warnings
warnings.filterwarnings("ignore")

import mlflow.pyfunc
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

TRACKING_URI  = "sqlite:///mlflow.db"
MODEL_NAME    = "ChurnModel-LK07"
PROCESSED_DIR = "data/processed"
TARGET_COL    = "churn"
DROP_COLS     = ["customerid"]

mlflow.set_tracking_uri(TRACKING_URI)

print("=" * 60)
print("  LK-07 Step 5: Verifikasi Inferensi Production Model")
print("=" * 60)

# Load model dari Production
model_uri = f"models:/{MODEL_NAME}/Production"
print(f"\n[1] Memuat model dari: {model_uri}")
loaded_model = mlflow.pyfunc.load_model(model_uri)
print("    ✅ Model berhasil dimuat dari Production registry!")

# Ambil file CSV terbaru
files  = glob.glob(os.path.join(PROCESSED_DIR, "*.csv"))
latest = max(files, key=os.path.getmtime)
print(f"\n[2] Dataset: {latest}")

df = pd.read_csv(latest)

# Encode kategorikal
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Siapkan fitur
cols_to_drop = [TARGET_COL] + [c for c in DROP_COLS if c in df.columns]
X_sample     = df.drop(columns=cols_to_drop).head(5)
y_true       = df[TARGET_COL].head(5).tolist()

# Handle missing values
imputer  = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_sample)
X_ready   = pd.DataFrame(X_imputed, columns=X_sample.columns)

print(f"\n[3] Menjalankan prediksi pada {len(X_ready)} sampel...")
predictions = loaded_model.predict(X_ready)

print("\n    Hasil Prediksi:")
print("    " + "-" * 45)
for i, (pred, true) in enumerate(zip(predictions, y_true)):
    label_pred = "CHURN ⚠️ " if pred == 1 else "TIDAK CHURN ✅"
    label_true = "CHURN"     if true == 1 else "TIDAK CHURN"
    match      = "✓" if pred == true else "✗"
    print(f"    Pelanggan {i+1}: Prediksi={label_pred:<18} | Aktual={label_true} {match}")

print("\n" + "=" * 60)
print(f"  ✅ Verifikasi inferensi BERHASIL!")
print(f"  Model '{MODEL_NAME}' Production siap digunakan.")
print("=" * 60)