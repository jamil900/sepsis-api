# Sepsis Risk Prediction API & Streamlit UI

Proyek ini menyediakan sistem prediksi risiko sepsis berbasis machine learning yang terdiri dari:

* **FastAPI backend** untuk layanan prediksi model
* **Streamlit frontend** sebagai antarmuka pengguna
* Model ensemble tersimpan dalam file artefak `.pkl`
* Notebook pemodelan tersedia di Google Colab

---

## 📁 Struktur Folder

```
.
├── Dockerfile
├── requirements.txt
├── scaler.pkl
├── rf_model.pkl
├── svc_model.pkl
├── nb_model.pkl
├── ensemble_weights.pkl
├── app.py                  # FastAPI backend (endpoint prediksi)
└── src/
    └── streamlit_app.py    # Streamlit frontend UI
```

---

## 🧪 Notebook Pemodelan (Training Model)

Proses eksplorasi data, preprocessing, training model, dan pembuatan ensemble dilakukan di Google Colab:

🔗 https://colab.research.google.com/drive/1X34Fsj9HA3cQZk7O871qBqjnQ5kEmJid?usp=sharing

Notebook ini mencakup:

* Data preprocessing & feature selection
* Training Random Forest, SVC, dan Naive Bayes
* Evaluasi performa model
* Perhitungan bobot ensemble
* Export artefak model `.pkl`

---

## 🚀 Backend API (FastAPI)

API menyediakan endpoint prediksi risiko sepsis.

### 🔗 Dokumentasi API

https://jamils-sepsis-fastapi.hf.space/docs

### Endpoint Prediksi

```
POST /predict
```

### Contoh Request JSON

```json
{
  "heart_rate": 95,
  "respiratory_rate": 22,
  "temperature": 38.1,
  "wbc_count": 12,
  "lactate_level": 2.1,
  "age": 67,
  "num_comorbidities": 2
}
```

### Contoh Response JSON

```json
{
  "sepsis_risk_prediction": 1,
  "risk_probability": 0.78
}
```

Keterangan:

* `sepsis_risk_prediction` → 1 = risiko tinggi, 0 = risiko rendah
* `risk_probability` → probabilitas model terhadap risiko sepsis

---

## 🖥️ Aplikasi Web Streamlit

Aplikasi frontend yang sudah dideploy dapat diakses di:

🔗 https://huggingface.co/spaces/jamils/sepsis_app_detection

Fitur aplikasi:

* Input data vital dan laboratorium pasien
* Mengirim data ke API FastAPI
* Menampilkan hasil prediksi secara visual

---

## ⚙️ Menjalankan Secara Lokal

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Jalankan FastAPI

```
uvicorn app:app --reload --port 8000
```

Docs lokal:

```
http://localhost:8000/docs
```

### 3. Jalankan Streamlit

```
streamlit run src/streamlit_app.py
```

---

## 🧠 Model Ensemble

Model prediksi menggunakan kombinasi:

* Random Forest
* Support Vector Classifier
* Naive Bayes

Prediksi akhir dihitung menggunakan **weighted probability averaging** untuk meningkatkan stabilitas dan performa model.

---

## 🎯 Tujuan Proyek

Proyek ini menunjukkan implementasi end-to-end machine learning deployment meliputi:

* Eksplorasi & training model di notebook
* Penyimpanan artefak model
* Deployment API untuk inference
* Integrasi frontend interaktif
* Arsitektur backend–frontend terpisah

---
