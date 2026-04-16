# MLOPS-Customer-Churn-Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyek MLOPS untuk memprediksi churn pelanggan pada platform e-commerce menggunakan data perilaku dan transaksi pelanggan, sehingga dapat melakukan tindakan lebih lanjut untuk mempertahankan pelanggan.

## Project Organization

```
├── LICENSE            <- Lisensi open-source yang digunakan pada proyek
├── README.md          <- Dokumentasi utama proyek
├── data
│   ├── interim        <- Data sementara yang telah melalui proses transformasi
│   ├── processed      <- Dataset final yang siap digunakan untuk pemodelan
│   └── raw            <- Data mentah asli yang belum diproses
│
├── models             <- Model yang telah dilatih, hasil prediksi, atau ringkasan model
│
├── notebooks          <- Notebook Jupyter untuk eksplorasi dan eksperimen.
│                         Penamaan menggunakan format:
│                         nomor-versi-inisial-deskripsi
│                         contoh: `1.0-arif-explorasi-data`
│
├── requirements.txt   <- Daftar dependensi Python yang diperlukan untuk menjalankan proyek
│
└── src   <- Source code utama proyek
    │
    ├── __init__.py             <- Menjadikan folder src sebagai modul Python
    │
    ├── config.py               <- Menyimpan konfigurasi dan variabel penting proyek
    │
    ├── dataset.py              <- Script untuk mengambil atau memuat dataset
    │
    ├── features.py             <- Script untuk melakukan feature engineering
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Script untuk melakukan prediksi menggunakan model          
    │   └── train.py            <- Script untuk melatih model machine learning

```
**Cara Menjalankan di Codespaces**
1. Buka halaman repositori GitHub proyek ini.
2. Klik tombol Code.
3. Pilih tab Codespaces.
4. Klik Create Codespace on main.
5. Tunggu hingga lingkungan pengembangan selesai dibuat.

### Setelah Codespaces aktif, semua dependency akan terinstal secara otomatis dan proyek siap digunakan.
--------
### Menjalankan Data Pipeline
1. Proses Pengambilan Data
Gunakan script berikut untuk mengambil dataset dan menyimpannya ke dalam folder data/raw/:
```
python src/data/ingest_data.py
```

2. preprocessing
Tahap ini bertujuan untuk membersihkan data serta menyiapkan dataset agar siap digunakan pada tahap modeling. Hasilnya akan disimpan di data/processed/:
```
python src/data/preprocess.py
```
4. Menjalankan Seluruh Pipeline
Untuk menjalankan kedua proses secara berurutan:
```
python src/data/ingest_data.py && python src/data/preprocess.py
```
--------

## Data Versioning with DVC

### Step:
1. dvc init
2. python ingest_data.py (generate data)
3. dvc add dataset
4. update data (continual learning)
5. dvc add again
6. dvc diff

### Result:
Dataset bertambah secara bertahap

#### Nama: Arif Rahman

#### NIM: 235150201111012

#### Mata Kuliah: Machine Learning Operations (Kelas B) ####
