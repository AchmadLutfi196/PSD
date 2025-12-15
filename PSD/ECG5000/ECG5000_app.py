
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Definisi 5 kelas ECG
CLASS_NAMES = {
    1: 'Normal',
    2: 'R-on-T PVC',
    3: 'Supraventricular',
    4: 'PVC',
    5: 'Unknown'
}

CLASS_DESCRIPTIONS = {
    1: 'Ritme sinus normal - Detak jantung normal',
    2: 'R-on-T PVC - Kontraksi prematur ventrikel pada gelombang T (berbahaya)',
    3: 'Supraventricular - Denyut ektopik supraventrikular',
    4: 'PVC - Kontraksi prematur ventrikel',
    5: 'Unknown - Denyut fusi atau tidak terklasifikasi'
}

CLASS_COLORS = {
    1: 'green',
    2: 'red',
    3: 'orange',
    4: 'purple',
    5: 'gray'
}

CLASS_EMOJI = {
    1: '✅',
    2: '🚨',
    3: '⚠️',
    4: '⚠️',
    5: '❓'
}

# Helper function untuk mendapatkan info kelas dengan fallback
def get_class_name(p):
    p = int(p)
    return CLASS_NAMES.get(p, f'Unknown ({p})')

def get_class_emoji(p):
    p = int(p)
    return CLASS_EMOJI.get(p, '❓')

def get_class_color(p):
    p = int(p)
    return CLASS_COLORS.get(p, 'gray')

def get_class_description(p):
    p = int(p)
    return CLASS_DESCRIPTIONS.get(p, 'Kelas tidak dikenal')

# Load model dan scaler
@st.cache_resource
def load_model():
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths to model files (bisa di ECG5000 atau di root folder)
    # Cek di folder ECG5000 terlebih dahulu
    model_path = os.path.join(current_dir, 'ECG5000', 'ecg_model.pkl') if not os.path.exists(os.path.join(current_dir, 'ecg_model.pkl')) else os.path.join(current_dir, 'ecg_model.pkl')
    scaler_path = os.path.join(current_dir, 'ECG5000', 'ecg_scaler.pkl') if not os.path.exists(os.path.join(current_dir, 'ecg_scaler.pkl')) else os.path.join(current_dir, 'ecg_scaler.pkl')
    
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
st.markdown('Sistem klasifikasi sinyal ECG dengan 5 kelas aritmia jantung')

# Sidebar
st.sidebar.header('Input Data')

# Tampilkan info kelas di sidebar
st.sidebar.markdown('---')
st.sidebar.subheader('Keterangan Kelas')
for cls_id, cls_name in CLASS_NAMES.items():
    st.sidebar.markdown(f'{get_class_emoji(cls_id)} **{cls_name}**')

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
    
    # Dapatkan probability untuk confidence score
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)
        # Ambil probability untuk kelas yang diprediksi
        confidence = np.max(probabilities, axis=1)
    else:
        confidence = np.ones(len(predictions))  # Jika model tidak support probability
    
    # Hasil - 5 kelas
    st.subheader('Hasil Prediksi')
    result_df = pd.DataFrame({
        'Sample': range(1, len(predictions)+1),
        'Kelas': [int(p) for p in predictions],
        'Prediction': [f'{get_class_emoji(p)} {get_class_name(p)}' for p in predictions],
        'Deskripsi': [get_class_description(p) for p in predictions],
        'Confidence': [f'{c*100:.1f}%' for c in confidence]
    })
    st.dataframe(result_df)
    
    # Visualisasi
    st.subheader('Visualisasi Sinyal ECG')
    sample_idx = st.slider('Pilih sample', 0, len(X_new)-1, 0)
    
    # Tentukan warna berdasarkan prediksi
    pred_class = int(predictions[sample_idx])
    pred_label = get_class_name(pred_class)
    color = get_class_color(pred_class)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(X_new[sample_idx], color=color, linewidth=1.5)
    ax.set_title(f'Sample {sample_idx+1} - {get_class_emoji(pred_class)} {pred_label} (Confidence: {result_df.iloc[sample_idx]["Confidence"]})', 
                 fontweight='bold')
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Deskripsi kelas
    st.info(f'**{pred_label}**: {get_class_description(pred_class)}')
    
    # Summary - 5 kelas
    st.subheader('Ringkasan Klasifikasi')
    cols = st.columns(5)
    for i, (cls_id, cls_name) in enumerate(CLASS_NAMES.items()):
        count = sum(predictions == cls_id)
        percentage = count/len(predictions)*100
        cols[i].metric(
            f'{get_class_emoji(cls_id)} {cls_name}', 
            count, 
            f'{percentage:.1f}%'
        )

else:
    st.info('Silakan upload file ECG untuk memulai klasifikasi.')
    
    # Tampilkan informasi kelas
    st.subheader('Tentang Klasifikasi 5 Kelas ECG')
    for cls_id, cls_name in CLASS_NAMES.items():
        st.markdown(f'**{get_class_emoji(cls_id)} Kelas {cls_id} - {cls_name}**: {get_class_description(cls_id)}')
    
    # Demo dengan sample data
    if st.button('Demo dengan Sample Data'):
        # Get the directory of this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Cek di folder ECG5000 terlebih dahulu
        test_data_path = os.path.join(current_dir, 'ECG5000', 'ECG5000_TEST.txt') if not os.path.exists(os.path.join(current_dir, 'ECG5000_TEST.txt')) else os.path.join(current_dir, 'ECG5000_TEST.txt')
        
        if not os.path.exists(test_data_path):
            st.error(f"Test data file not found: {test_data_path}")
            st.info("Please ensure ECG5000_TEST.txt is in the same directory as this app.")
        else:
            try:
                sample_data = np.loadtxt(test_data_path)[:10, 1:]
                X_scaled = scaler.transform(sample_data)
                predictions = model.predict(X_scaled)
                
                # Dapatkan confidence score
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_scaled)
                    confidence = np.max(probabilities, axis=1)
                else:
                    confidence = np.ones(len(predictions))
                
                st.subheader('Demo Prediksi (10 Sample)')
                
                # Tampilkan dalam format yang lebih baik - 5 kelas
                demo_df = pd.DataFrame({
                    'Sample': range(1, len(predictions)+1),
                    'Kelas': [int(p) for p in predictions],
                    'Prediction': [f'{get_class_emoji(p)} {get_class_name(p)}' for p in predictions],
                    'Confidence': [f'{c*100:.1f}%' for c in confidence]
                })
                st.dataframe(demo_df, use_container_width=True)
                
                # Summary - 5 kelas
                st.markdown('**Ringkasan Demo:**')
                summary_text = []
                for cls_id, cls_name in CLASS_NAMES.items():
                    count = sum(predictions == cls_id)
                    if count > 0:
                        summary_text.append(f'{get_class_emoji(cls_id)} {cls_name}: {count}')
                st.write(' | '.join(summary_text))
            except Exception as e:
                st.error(f"Error loading demo data: {str(e)}")
