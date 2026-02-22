import streamlit as st
import requests

st.set_page_config(page_title="Sepsis Prediction", page_icon="🩺")

st.title("🩺 Sepsis Risk Prediction")
st.write("Masukkan data vital dan lab pasien untuk memprediksi risiko sepsis dalam 24 jam.")

st.divider()

# ===== INPUT FORM =====
col1, col2 = st.columns(2)

with col1:
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=0.0, value=90.0)
    respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=0.0, value=20.0)
    temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)
    wbc_count = st.number_input("WBC Count", min_value=0.0, value=12.0)

with col2:
    lactate_level = st.number_input("Lactate Level", min_value=0.0, value=1.5)
    age = st.number_input("Age", min_value=0, value=45)
    num_comorbidities = st.number_input("Number of Comorbidities", min_value=0, value=0)

st.divider()

# ===== PREDICT BUTTON =====
if st.button("🔎 Predict Sepsis Risk"):

    payload = {
        "heart_rate": heart_rate,
        "respiratory_rate": respiratory_rate,
        "temperature": temperature,
        "wbc_count": wbc_count,
        "lactate_level": lactate_level,
        "age": age,
        "num_comorbidities": num_comorbidities
    }

    try:
        API_URL = "https://jamils-sepsis-fastapi.hf.space/predict"

        with st.spinner("Menghubungi model..."):
            response = requests.post(API_URL, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()

            pred = result["sepsis_risk_prediction"]
            prob = result["risk_probability"]

            st.subheader("🧾 Prediction Result")

            if pred == 1:
                st.error("⚠️ HIGH RISK of Sepsis")
            else:
                st.success("✅ LOW RISK of Sepsis")

            st.metric("Risk Probability", f"{prob:.2%}")

        else:
            st.error(f"API error {response.status_code}")
            st.text(response.text)

    except Exception as e:
        st.error("Tidak bisa terhubung ke API")
        st.text(str(e))
