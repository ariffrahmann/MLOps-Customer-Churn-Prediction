import glob
import os
import datetime
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import pandas as pd
import yaml
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score

TRACKING_URI    = "sqlite:///mlflow.db"   # sesuai repo kamu
EXPERIMENT_NAME = "Customer-Churn-Prediction"
PROCESSED_DIR   = "data/processed"
TARGET_COLUMN   = "churn"                 
DROP_COLUMNS    = ["customerid"]          
MODEL_NAME_LK07 = "ChurnModel-LK07"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

print("=" * 60)
print("  LK-07: MLflow Model Registry")
print("=" * 60)

def get_latest_file(directory):
    files = glob.glob(os.path.join(directory, "*.csv"))
    if not files:
        raise FileNotFoundError(f"Tidak ada CSV di: {directory}")
    latest = max(files, key=os.path.getmtime)
    print(f"   📂 Dataset digunakan: {latest}")
    return latest

def load_data():
    df = pd.read_csv(get_latest_file(PROCESSED_DIR))

    # Encode kolom kategorikal
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Drop kolom bukan fitur
    cols_to_drop = [TARGET_COLUMN] + DROP_COLUMNS
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COLUMN]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# STEP 1
print("\n[STEP 1] Mendaftarkan model terbaik LK-06 ke registry...")

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' tidak ditemukan!")

# Cari run terbaik berdasarkan f1_score tertinggi
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_score DESC"],
    max_results=10
)

print(f"\n   Daftar run yang ditemukan di LK-06:")
for r in runs:
    print(f"   • {r.info.run_name:<20} | "
          f"F1={r.data.metrics.get('f1_score', 0):.4f} | "
          f"Acc={r.data.metrics.get('accuracy', 0):.4f} | "
          f"ID={r.info.run_id[:8]}...")

best_run    = runs[0]
best_run_id = best_run.info.run_id
best_f1     = best_run.data.metrics.get("f1_score", 0)
best_acc    = best_run.data.metrics.get("accuracy", 0)
best_recall = best_run.data.metrics.get("recall", 0)

print(f"\n   ✅ Run terbaik: '{best_run.info.run_name}'")
print(f"      Run ID  : {best_run_id}")
print(f"      F1-Score: {best_f1:.4f} | Accuracy: {best_acc:.4f}")

# Daftarkan ke registry LK-07 sebagai Version 1
result_v1 = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=MODEL_NAME_LK07
)
print(f"\n   ✅ Terdaftar sebagai '{MODEL_NAME_LK07}' Version {result_v1.version}")

# STEP 2 — Training ulang
print("\n[STEP 2] Training model v2 dengan parameter berbeda...")

X_train, X_test, y_train, y_test = load_data()

# Handle missing values
imputer     = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="lk07-model-v2") as run:
    # Parameter BERBEDA dari run_3_aggressive LK-06
    # LK-06 terbaik: n_estimators=300, lr=0.01, max_depth=8
    # LK-07 v2     : n_estimators=200, lr=0.05, max_depth=6
    params_v2 = {
        "n_estimators" : 200,
        "learning_rate": 0.05,
        "max_depth"    : 6,
        "random_state" : 42,
        "eval_metric"  : "logloss"
    }

    model_v2 = XGBClassifier(**params_v2)
    model_v2.fit(X_train_imp, y_train)

    y_pred   = model_v2.predict(X_test_imp)
    acc_v2   = accuracy_score(y_test, y_pred)
    f1_v2    = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_v2= recall_score(y_test, y_pred, average="weighted", zero_division=0)

    mlflow.log_param("n_estimators",  params_v2["n_estimators"])
    mlflow.log_param("learning_rate", params_v2["learning_rate"])
    mlflow.log_param("max_depth",     params_v2["max_depth"])
    mlflow.log_metric("accuracy",     acc_v2)
    mlflow.log_metric("f1_score",     f1_v2)
    mlflow.log_metric("recall",       recall_v2)

    # Daftar ke registry dengan nama SAMA → jadi Version 2
    mlflow.sklearn.log_model(
        sk_model=model_v2,
        artifact_path="model",
        registered_model_name=MODEL_NAME_LK07
    )
    run_id_v2 = run.info.run_id

print(f"   ✅ Model v2 terdaftar!")
print(f"      Run ID  : {run_id_v2}")
print(f"      F1-Score: {f1_v2:.4f} | Accuracy: {acc_v2:.4f} | Recall: {recall_v2:.4f}")


# STEP 3 — Transisi Stage: None → Staging → Production
print("\n[STEP 3] Transisi Stage...")

# v1 → Archived (model lama)
client.transition_model_version_stage(
    name=MODEL_NAME_LK07, version=1, stage="Archived"
)
print(f"   ✅ Version 1 (dari LK-06) → Archived")

# v2 → Staging
client.transition_model_version_stage(
    name=MODEL_NAME_LK07, version=2, stage="Staging"
)
print(f"   ✅ Version 2 → Staging")

# v2 → Production
client.transition_model_version_stage(
    name=MODEL_NAME_LK07, version=2, stage="Production"
)
print(f"   ✅ Version 2 → Production")

# STEP 4 — Buat metadata YAML untuk DVC
print("\n[STEP 4] Membuat file model_metadata.yaml untuk DVC...")

prod_models = client.get_latest_versions(MODEL_NAME_LK07, stages=["Production"])
prod_model  = prod_models[0]

metadata = {
    "model_name"            : MODEL_NAME_LK07,
    "version_production"    : int(prod_model.version),
    "stage"                 : prod_model.current_stage,
    "run_id_production"     : prod_model.run_id,
    "run_id_lk06_source"    : best_run_id,
    "run_name_lk06_source"  : best_run.info.run_name,
    "registered_at"         : datetime.datetime.now().isoformat(),
    "algorithm"             : "XGBClassifier",
    "parameters_v2"         : {
        "n_estimators" : 200,
        "learning_rate": 0.05,
        "max_depth"    : 6
    },
    "metrics_v2_production" : {
        "accuracy": round(acc_v2, 4),
        "f1_score": round(f1_v2, 4),
        "recall"  : round(recall_v2, 4)
    },
    "metrics_v1_lk06_best"  : {
        "accuracy": round(best_acc, 4),
        "f1_score": round(best_f1, 4),
        "recall"  : round(best_recall, 4)
    },
    "data_lineage": {
        "processed_dir"  : PROCESSED_DIR,
        "file_selection" : "latest file by modification time (glob)",
        "target_column"  : TARGET_COLUMN
    }
}

os.makedirs("models", exist_ok=True)
with open("models/model_metadata.yaml", "w") as f:
    yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print("   ✅ File 'models/model_metadata.yaml' berhasil dibuat!")
print("\n   Preview isi metadata:")
print("   " + "-" * 45)
for k, v in metadata.items():
    print(f"   {k}: {v}")

print("\n" + "=" * 60)
print("  ✅ Step 1–4 selesai!")
print("=" * 60)