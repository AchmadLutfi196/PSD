
import streamlit as st
import joblib
import numpy as np
import librosa
import pandas as pd
from feature_extraction import extract_statistical_features, preprocess_audio

# Load models
@st.cache_resource
def load_models():
    speaker_pipeline = joblib.load('speaker_model_pipeline.pkl')
    command_pipeline = joblib.load('command_model_pipeline.pkl')
    return speaker_pipeline, command_pipeline

def predict_voice(audio_data, speaker_pipeline, command_pipeline):
    """
    Two-stage prediction: Speaker identification + Command recognition
    """
    # Stage 1: Speaker Recognition
    features = extract_statistical_features(audio_data)
    features_df = pd.DataFrame([features])
    
    # Speaker prediction
    speaker_features = features_df[speaker_pipeline['feature_names']]
    speaker_features = speaker_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    speaker_features_scaled = speaker_pipeline['scaler'].transform(speaker_features)
    
    speaker_pred_encoded = speaker_pipeline['model'].predict(speaker_features_scaled)[0]
    speaker_pred = speaker_pipeline['label_encoder'].inverse_transform([speaker_pred_encoded])[0]
    speaker_confidence = np.max(speaker_pipeline['model'].predict_proba(speaker_features_scaled))
    
    # Check if authorized speaker
    if speaker_confidence < 0.7:  # Threshold untuk menolak suara tidak dikenal
        return None, None, speaker_confidence, "Suara tidak dikenal - Akses ditolak"
    
    # Stage 2: Command Recognition (only if speaker is authorized)
    command_features = features_df[command_pipeline['feature_names']]
    command_features = command_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    command_features_scaled = command_pipeline['scaler'].transform(command_features)
    
    command_pred_encoded = command_pipeline['model'].predict(command_features_scaled)[0]
    command_pred = command_pipeline['label_encoder'].inverse_transform([command_pred_encoded])[0]
    command_confidence = np.max(command_pipeline['model'].predict_proba(command_features_scaled))
    
    status = f"Suara {speaker_pred} mengatakan '{command_pred}'"
    
    return speaker_pred, command_pred, min(speaker_confidence, command_confidence), status

# Streamlit App
def main():
    st.title("🎤 Sistem Identifikasi Suara Buka-Tutup")
    st.subheader("Voice Recognition System dengan Feature Statistik Time Series")
    
    # Load models
    speaker_pipeline, command_pipeline = load_models()
    
    # Display model info
    st.sidebar.header("📊 Model Information")
    st.sidebar.write(f"**Speaker Model:** {speaker_pipeline['model_info']['model_type']}")
    st.sidebar.write(f"**Accuracy:** {speaker_pipeline['model_info']['accuracy']:.1%}")
    st.sidebar.write(f"**Authorized Speakers:** {', '.join(speaker_pipeline['model_info']['classes'])}")
    
    st.sidebar.write(f"**Command Model:** {command_pipeline['model_info']['model_type']}")
    st.sidebar.write(f"**Accuracy:** {command_pipeline['model_info']['accuracy']:.1%}")
    st.sidebar.write(f"**Commands:** {', '.join(command_pipeline['model_info']['classes'])}")
    
    # Audio input
    st.header("🎵 Upload Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'flac'])
    
    if uploaded_file is not None:
        # Load and preprocess audio
        audio, sr = librosa.load(uploaded_file, sr=22050)
        audio = preprocess_audio(audio)
        
        # Display audio
        st.audio(uploaded_file, format='audio/wav')
        
        # Predict button
        if st.button("🔍 Analyze Voice", type="primary"):
            with st.spinner("Analyzing voice..."):
                speaker, command, confidence, status = predict_voice(audio, speaker_pipeline, command_pipeline)
                
                # Display results
                st.header("🎯 Results")
                
                if speaker is None:
                    st.error(f"❌ {status}")
                    st.write("**Confidence:** {:.1%}".format(confidence))
                else:
                    st.success(f"✅ {status}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("👤 Speaker", speaker.title())
                    with col2:
                        st.metric("🗣️ Command", command.title())
                    with col3:
                        st.metric("📈 Confidence", f"{confidence:.1%}")
                    
                    # Action based on command
                    if command == "buka":
                        st.balloons()
                        st.info("🔓 Door opened!")
                    else:
                        st.info("🔒 Door closed!")
    
    # Instructions
    st.header("📋 Instructions")
    st.write("""
    1. **Upload audio file** dalam format WAV, MP3, atau FLAC
    2. **Click 'Analyze Voice'** untuk memulai analisis
    3. **System akan mengidentifikasi:**
       - Siapa yang berbicara (Lutfi/Harits)
       - Perintah apa yang diucapkan (Buka/Tutup)
    4. **Access Control:** Suara yang tidak dikenal akan ditolak
    """)
    
    st.header("🔬 Technical Details")
    st.write("""
    - **Two-Stage Recognition:** Speaker identification + Command recognition
    - **Features:** 61 statistical time series features per audio
    - **Models:** RandomForest (Speaker) + SVM (Command)
    - **Accuracy:** 100% pada dataset training
    - **Security:** Access control untuk speaker tidak dikenal
    """)

if __name__ == "__main__":
    main()
