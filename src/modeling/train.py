# src/modeling/train.py

import mlflow
import mlflow.sklearn
import pandas as pd
import glob
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.impute import SimpleImputer

PROCESSED_DIR   = "data/processed"
MODELS_DIR      = "models"
EXPERIMENT_NAME = "Customer-Churn-Prediction"
os.makedirs(MODELS_DIR, exist_ok=True)


def get_latest_file(directory):
    files = glob.glob(os.path.join(directory, "*.csv"))
    if not files:
        raise FileNotFoundError(f"Tidak ada CSV di: {directory}")
    latest = max(files, key=os.path.getmtime)
    print(f"📂 Dataset: {latest}")
    return latest


def load_data():
    df = pd.read_csv(get_latest_file(PROCESSED_DIR))

    # Deteksi kolom target Churn
    target_col = next(
        (c for c in ["Churn", "churn", "target"] if c in df.columns),
        df.columns[-1]
    )
    print(f"✅ Target kolom: {target_col}")

    before = len(df)
    df = df.dropna(subset=[target_col])
    after = len(df)
    if before != after:
        print(f"⚠️  Dropped {before - after} baris NaN di kolom '{target_col}'")

    # Encode kolom kategorikal
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # target bertipe integer
    df[target_col] = df[target_col].astype(int)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"📏 Data siap     : {X.shape[0]} baris, {X.shape[1]} fitur")
    print(f"📈 Churn rate    : {y.mean():.2%}")

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def run_experiment(run_name, params):
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name=run_name):

        # Buat model
        model = XGBClassifier(
            **params,
            random_state=42,
            eval_metric="logloss"
        )

        # Handle missing values pada fitur
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp  = imputer.transform(X_test)

        # Training
        model.fit(X_train_imp, y_train)
        y_pred = model.predict(X_test_imp)

        # Hitung metrik
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        # mlflow.log_param() 
        mlflow.log_param("n_estimators",  params["n_estimators"])
        mlflow.log_param("learning_rate", params["learning_rate"])
        mlflow.log_param("max_depth",     params["max_depth"])

        # mlflow.log_metric() 
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall",   recall)

        # mlflow.log_model()
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="ChurnModel-XGBoost"
        )

        # Simpan model lokal di models/
        pickle.dump(model, open(f"{MODELS_DIR}/{run_name}.pkl", "wb"))

        # Print hasil
        run_id = mlflow.active_run().info.run_id
        print(f"\n{'='*55}")
        print(f"  Run Name     : {run_name}")
        print(f"  Run ID       : {run_id}")
        print(f"  n_estimators : {params['n_estimators']}")
        print(f"  learning_rate: {params['learning_rate']}")
        print(f"  max_depth    : {params['max_depth']}")
        print(f"  ─────────────────────────────────────────")
        print(f"  Accuracy     : {acc:.4f}")
        print(f"  F1-Score     : {f1:.4f}")
        print(f"  Recall       : {recall:.4f}")
        print(f"{'='*55}\n")

        return run_id


# 3 run dengan parameter berbeda
if __name__ == "__main__":

    print("\n🚀 Memulai Eksperimen MLflow — Customer Churn Prediction\n")

    # RUN 1 — Baseline
    run_experiment("run_1_baseline", {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 4
    })

    # RUN 2 — Tree lebih dalam, learning rate lebih kecil
    run_experiment("run_2_deep", {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6
    })

    # RUN 3 — Estimator banyak, learning rate sangat kecil
    run_experiment("run_3_aggressive", {
        "n_estimators": 300,
        "learning_rate": 0.01,
        "max_depth": 8
    })

    print("✅ Semua eksperimen selesai!")
    print("👉 Jalankan: mlflow ui --host 0.0.0.0 --port 5000\n")