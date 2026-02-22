# Sepsis Risk Prediction System (FastAPI + Streamlit)

Proyek ini merupakan implementasi end-to-end machine learning deployment untuk prediksi risiko sepsis dalam 24 jam menggunakan model ensemble.

Sistem terdiri dari dua komponen terpisah:

* 🔹 **FastAPI Service** → backend inference model
* 🔹 **Streamlit App** → antarmuka pengguna (frontend)
* 🔹 Notebook training tersedia di Google Colab

---

# 📁 Struktur Folder Proyek

```text
.
├── FastApi/
│   ├── Dockerfile
│   ├── app.py
│   ├── ensemble_weights.pkl
│   ├── nb_model.pkl
│   ├── rf_model.pkl
│   ├── svc_model.pkl
│   ├── scaler.pkl
│   └── requirements.txt
│
├── Streamlit/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       └── streamlit_app.py
│
├── README.md
```

---

# 🧪 Notebook Pemodelan

Proses training model, preprocessing, evaluasi, dan pembuatan ensemble dilakukan di Google Colab:

🔗 https://colab.research.google.com/drive/1X34Fsj9HA3cQZk7O871qBqjnQ5kEmJid?usp=sharing

---

# 🚀 Backend API (FastAPI)

API digunakan untuk memproses data pasien dan menghasilkan prediksi risiko sepsis.

### 🔗 Dokumentasi API

https://jamils-sepsis-fastapi.hf.space/docs

### Endpoint Prediksi

```text
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

* `1` → Risiko tinggi sepsis
* `0` → Risiko rendah sepsis
* `risk_probability` → probabilitas model ensemble

---

# 🖥️ Frontend Streamlit

Aplikasi web yang sudah dideploy dapat diakses di:

🔗 https://huggingface.co/spaces/jamils/sepsis_app_detection

Fitur aplikasi:

* Input data vital dan laboratorium pasien
* Mengirim data ke API FastAPI
* Menampilkan hasil prediksi secara visual

---

# ⚙️ Menjalankan Secara Lokal

## Jalankan Backend

```bash
cd FastApi
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Akses dokumentasi lokal:

```
http://localhost:8000/docs
```

---

## Jalankan Frontend

```bash
cd Streamlit
pip install -r requirements.txt
streamlit run src/streamlit_app.py
```

---

# 🧠 Model Ensemble

Sistem menggunakan kombinasi model:

* Random Forest
* Support Vector Classifier
* Naive Bayes

Prediksi akhir dihitung menggunakan **weighted probability averaging**.

---

# 🎯 Tujuan Proyek

Proyek ini dibuat untuk menunjukkan implementasi lengkap machine learning deployment meliputi:

* Eksplorasi & training model di notebook
* Export artefak model
* Deployment API inference
* Integrasi frontend interaktif
* Arsitektur backend–frontend terpisah

---
