
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Load model dan scaler
@st.cache_resource
def load_model():
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths to model files
    model_path = os.path.join(current_dir, 'ecg_model.pkl')
    scaler_path = os.path.join(current_dir, 'ecg_scaler.pkl')
    
    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please run the notebook first to generate the model files.")
        return None, None
    
    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found: {scaler_path}")
        st.info("Please run the notebook first to generate the scaler files.")
        return None, None
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, scaler = load_model()

# Check if model loaded successfully
if model is None or scaler is None:
    st.stop()

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
        # Get the directory of this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_data_path = os.path.join(current_dir, 'ECG5000_TEST.txt')
        
        if not os.path.exists(test_data_path):
            st.error(f"Test data file not found: {test_data_path}")
            st.info("Please ensure ECG5000_TEST.txt is in the same directory as this app.")
        else:
            try:
                sample_data = np.loadtxt(test_data_path)[:10, 1:]
                X_scaled = scaler.transform(sample_data)
                predictions = model.predict(X_scaled)
                
                st.subheader('Demo Prediksi (10 Sample)')
                for i, pred in enumerate(predictions):
                    status = '✅ Normal' if pred == 1 else '⚠️ Abnormal'
                    st.write(f'Sample {i+1}: {status}')
            except Exception as e:
                st.error(f"Error loading demo data: {str(e)}")
