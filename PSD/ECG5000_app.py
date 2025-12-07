
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model dan scaler
@st.cache_resource
def load_model():
    model = joblib.load('ecg_model.pkl')
    scaler = joblib.load('ecg_scaler.pkl')
    return model, scaler

model, scaler = load_model()

# UI
st.title('🫀 ECG Classification System')
st.markdown('Sistem klasifikasi sinyal ECG untuk mendeteksi anomali jantung')

# Sidebar
st.sidebar.header('Input Data')
upload_file = st.sidebar.file_uploader('Upload ECG data (txt/csv)', type=['txt', 'csv'])

if upload_file is not None:
    # Load data
    if upload_file.name.endswith('.csv'):
        data = pd.read_csv(upload_file, header=None)
    else:
        data = pd.read_csv(upload_file, delim_whitespace=True, header=None)
    
    st.subheader('Data Uploaded')
    st.write(f'Shape: {data.shape}')
    
    # Jika ada kolom label, hapus
    if data.shape[1] == 141:
        X_new = data.iloc[:, 1:].values
    else:
        X_new = data.values
    
    # Prediksi
    X_scaled = scaler.transform(X_new)
    predictions = model.predict(X_scaled)
    
    # Hasil
    st.subheader('Hasil Prediksi')
    result_df = pd.DataFrame({
        'Sample': range(1, len(predictions)+1),
        'Prediction': ['Normal' if p == 1 else 'Abnormal' for p in predictions]
    })
    st.dataframe(result_df)
    
    # Visualisasi
    st.subheader('Visualisasi Sinyal ECG')
    sample_idx = st.slider('Pilih sample', 0, len(X_new)-1, 0)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(X_new[sample_idx], color='blue', linewidth=1.5)
    ax.set_title(f'Sample {sample_idx+1} - {result_df.iloc[sample_idx]["Prediction"]}')
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Summary
    normal_count = sum(predictions == 1)
    abnormal_count = sum(predictions == 0)
    
    col1, col2 = st.columns(2)
    col1.metric('Normal', normal_count, f'{normal_count/len(predictions)*100:.1f}%')
    col2.metric('Abnormal', abnormal_count, f'{abnormal_count/len(predictions)*100:.1f}%')

else:
    st.info('Silakan upload file ECG untuk memulai klasifikasi.')
    
    # Demo dengan sample data
    if st.button('Demo dengan Sample Data'):
        sample_data = np.loadtxt('ECG5000_TEST.txt')[:10, 1:]
        X_scaled = scaler.transform(sample_data)
        predictions = model.predict(X_scaled)
        
        st.subheader('Demo Prediksi (10 Sample)')
        for i, pred in enumerate(predictions):
            status = '✅ Normal' if pred == 1 else '⚠️ Abnormal'
            st.write(f'Sample {i+1}: {status}')
