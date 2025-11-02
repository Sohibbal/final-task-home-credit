# 🏦 Credit Risk Prediction using Logistic Regression, LightGBM, and XGBoost

## 📘 Project Overview
Tujuan dari proyek ini adalah membangun model *machine learning* untuk memprediksi risiko gagal bayar calon nasabah menggunakan dataset Home Credit.  
Dengan model ini, perusahaan dapat:
- Mengidentifikasi calon peminjam berisiko tinggi sejak awal.  
- Mengurangi potensi kerugian akibat kredit macet.  
- Membantu proses pengambilan keputusan pinjaman secara otomatis dan objektif.

---

## 🧩 Problem Statement
Perusahaan pemberi pinjaman sering kali kesulitan membedakan calon nasabah yang layak dan yang berisiko tinggi.  
Tanpa analisis berbasis data, keputusan persetujuan kredit cenderung subyektif dan meningkatkan tingkat *default*.

**Tujuan:**
Membangun model prediksi risiko gagal bayar (`TARGET = 1`) berdasarkan data demografis, pekerjaan, penghasilan, dan histori kredit.

---

## 📂 Dataset Description

- **Jumlah data:**
  - `application_train.csv` → ±300.000 baris (dengan kolom TARGET)
  - `application_test.csv` → ±100.000 baris
- **Fitur utama:**
  - Demografis: `DAYS_BIRTH`, `CODE_GENDER`, `NAME_EDUCATION_TYPE`
  - Finansial: `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`
  - Pekerjaan: `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`
- **Target:**
  - `TARGET = 1` → Gagal bayar  
  - `TARGET = 0` → Pembayaran lancar

---

## 🧹 Data Preprocessing
Langkah pembersihan dan persiapan data:

1. **Handling Missing Values**
   - Kolom dengan >50% missing value dihapus.
   - Numerik diisi dengan median.
   - Kategorikal diisi dengan modus.

2. **Encoding**
   - Gunakan `LabelEncoder` untuk semua kolom kategorikal.

3. **Feature Scaling**
   - Hanya diterapkan pada model linear (Logistic Regression) menggunakan `StandardScaler`.

4. **Feature Selection**
   - Fitur dipilih berdasarkan korelasi dengan target (>|0.05|).

5. **Feature Engineering**
   - Menambah rasio seperti `CREDIT_INCOME_RATIO`, `AGE_BIN`, `EMPLOYMENT_LENGTH`.

---

## 📊 Exploratory Data Analysis & Key Insights

### 🔹 Insight 1 — Usia Muda Lebih Berisiko
Nasabah dengan usia <30 tahun memiliki probabilitas gagal bayar **1.5× lebih tinggi** dibanding usia >40 tahun.  
**Action:** Perusahaan dapat menurunkan limit kredit untuk kelompok ini atau meminta jaminan tambahan.

### 🔹 Insight 2 — Pendapatan Rendah dan Tidak Tetap = Risiko Tinggi
70% kasus gagal bayar berasal dari nasabah dengan pendapatan < median dan tanpa pekerjaan tetap.  
**Action:** Perusahaan perlu memperketat verifikasi penghasilan dan fokus ke produk pinjaman mikro yang lebih terukur.

---

## 🤖 Model Development

### Algoritma yang digunakan:
| Model | Deskripsi |
|:------|:-----------|
| **Logistic Regression** | Baseline model, dengan scaling |
| **LightGBM** | Tree-based model cepat & akurat |
| **XGBoost** | Gradient boosting dengan optimasi regularisasi |
| **RandomizedSearchCV** | Digunakan untuk mencari hyperparameter terbaik |

### Preprocessing Pipeline:
- Imputer → Encoder → (Scaler bila perlu) → Model  
- Validasi menggunakan split `train/validation` (80/20)

---

## 📈 Model Evaluation

| Model | Validation AUC | Catatan |
|:------|:---------------:|:--------|
| Logistic Regression | 0.76 | Baseline sederhana |
| LightGBM | 0.81 | Cepat dan stabil |
| XGBoost | 0.83 | Performa tertinggi sebelum tuning |
| **Best Model (Tuned XGBoost)** | **0.84+** | Setelah tuning parameter |

### Threshold Selection:
Menggunakan **Youden’s J statistic (ROC Curve)** untuk menentukan ambang probabilitas optimal.

---

## 🧮 Final Inference Pipeline

### Model Inference Steps
1. Preprocess `application_test.csv` menggunakan pipeline yang sama dengan training.  
2. Apply model:
   - Logistic Regression (dengan scaling)
   - LightGBM (tanpa scaling)
   - XGBoost (tanpa scaling)
   - Best Tuned Model  
3. Terapkan threshold optimal hasil validasi.  
4. Simpan hasil ke file CSV:

| File Output | Model |
|--------------|--------|
| `submission_inference_logreg.csv` | Logistic Regression |
| `submission_inference_lightgbm.csv` | LightGBM |
| `submission_inference_xgboost.csv` | XGBoost |
| `submission_inference_bestmodel.csv` | Best Tuned Model |

---

## 💼 Business Recommendation

1. Gunakan model untuk sistem *automated credit scoring*.  
2. Terapkan kebijakan *dynamic credit limit* berdasarkan skor risiko.  
3. Jalankan kampanye akuisisi nasabah dengan profil risiko rendah (misal: PNS, usia >30, penghasilan stabil).  
4. Integrasikan hasil prediksi ke sistem approval untuk *real-time decision support.*

---
