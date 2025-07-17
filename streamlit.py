import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Customer Churn", layout="centered")

# Judul utama
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📊 Prediksi Customer Churn</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Masukkan data pelanggan dan ketahui apakah pelanggan akan churn atau tidak.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar input
st.sidebar.header("🧾 Masukkan Data Pelanggan")

def input_user():
    Tenure = st.sidebar.slider("🎯 Tenure (bulan)", 0, 50, 10)
    WarehouseToHome = st.sidebar.slider("🚚 Jarak Gudang ke Rumah", 0, 50, 5)
    NumberOfDeviceRegistered = st.sidebar.selectbox("📱 Jumlah Perangkat Terdaftar", [1, 2, 3, 4, 5, 6])
    SatisfactionScore = st.sidebar.selectbox("⭐ Skor Kepuasan", [1, 2, 3, 4, 5])
    NumberOfAddress = st.sidebar.slider("🏠 Jumlah Alamat", 1, 15, 2)
    Complain = st.sidebar.radio("❗ Pernah Komplain?", [0, 1])
    DaySinceLastOrder = st.sidebar.slider("📆 Hari Sejak Order Terakhir", 0, 30, 10)
    CashbackAmount = st.sidebar.number_input("💰 Jumlah Cashback", 0, 500, 10)
    PreferedOrderCat = st.sidebar.selectbox("🛒 Kategori Pesanan Favorit", 
        ["Laptop & Accessory", "Mobile Phone", "Fashion", "Mobile", "Grocery", "Others"])
    MaritalStatus = st.sidebar.radio("💍 Status Pernikahan", ["Married", "Single", 'Divorced'])

    df = pd.DataFrame({
        "Tenure": [Tenure],
        "WarehouseToHome": [WarehouseToHome],
        "NumberOfDeviceRegistered": [NumberOfDeviceRegistered],
        "SatisfactionScore": [SatisfactionScore],
        "NumberOfAddress": [NumberOfAddress],
        "Complain": [Complain],
        "DaySinceLastOrder": [DaySinceLastOrder],
        "CashbackAmount": [CashbackAmount],
        "PreferedOrderCat": [PreferedOrderCat],
        "MaritalStatus": [MaritalStatus]
    })
    return df

# Buat input dari pengguna
df_customer = input_user()

# Tampilkan data yang dimasukkan
st.subheader("📄 Data Pelanggan")
st.dataframe(df_customer, use_container_width=True)

# Load model
model_loaded = pickle.load(open("model_xgb.sav", "rb"))
prediction = model_loaded.predict(df_customer)

# Output prediksi
st.markdown("---")
st.subheader("🔍 Hasil Prediksi")

if prediction[0] == 0:
    st.success("✅ **Pelanggan TIDAK akan churn.**\nPelanggan kemungkinan tetap menggunakan layanan.")
else:
    st.error("⚠️ **Pelanggan AKAN churn.**\nPelanggan berpotensi berhenti menggunakan layanan.")

# Footer
st.markdown("---")
st.markdown("<small><center>Developed with ❤️ using Streamlit</center></small>", unsafe_allow_html=True)
