
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

def predict_voice(audio_data, speaker_pipeline, command_pipeline, confidence_threshold=0.4):
    """
    Two-stage prediction: Speaker identification + Command recognition
    FIXED VERSION - dengan feature consistency dan authorization logic yang benar
    """
    try:
        # Extract ALL features
        all_features = extract_statistical_features(audio_data)
        features_df = pd.DataFrame([all_features])
        
        # STAGE 1: Speaker Recognition
        # Gunakan HANYA features yang digunakan saat training
        speaker_feature_names = speaker_pipeline['feature_names']
        
        # Handle missing features
        for feature_name in speaker_feature_names:
            if feature_name not in features_df.columns:
                features_df[feature_name] = 0.0
        
        # Select features sesuai training
        speaker_features = features_df[speaker_feature_names]
        speaker_features = speaker_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        speaker_features_scaled = speaker_pipeline['scaler'].transform(speaker_features)
        
        # Predict speaker
        speaker_pred_encoded = speaker_pipeline['model'].predict(speaker_features_scaled)[0]
        speaker_pred = speaker_pipeline['label_encoder'].inverse_transform([speaker_pred_encoded])[0]
        speaker_confidence = np.max(speaker_pipeline['model'].predict_proba(speaker_features_scaled))
        
        # ACCESS CONTROL - FIXED LOGIC!
        predicted_speaker_lower = speaker_pred.lower()
        authorized_speakers = ['lutfi', 'harits']
        
        # BUG FIX: Gunakan IN bukan NOT IN!
        if predicted_speaker_lower in authorized_speakers:
            # Speaker authorized, check confidence
            if speaker_confidence >= confidence_threshold:
                # STAGE 2: Command Recognition
                command_feature_names = command_pipeline['feature_names']
                
                # Handle missing command features
                for feature_name in command_feature_names:
                    if feature_name not in features_df.columns:
                        features_df[feature_name] = 0.0
                
                # Select command features
                command_features = features_df[command_feature_names]
                command_features = command_features.replace([np.inf, -np.inf], np.nan).fillna(0)
                command_features_scaled = command_pipeline['scaler'].transform(command_features)
                
                # Predict command
                command_pred_encoded = command_pipeline['model'].predict(command_features_scaled)[0]
                command_pred = command_pipeline['label_encoder'].inverse_transform([command_pred_encoded])[0]
                command_confidence = np.max(command_pipeline['model'].predict_proba(command_features_scaled))
                
                if command_confidence >= confidence_threshold:
                    status = f"SUCCESS: {speaker_pred} mengatakan '{command_pred}'"
                    return speaker_pred, command_pred, min(speaker_confidence, command_confidence), status
                else:
                    status = f"Speaker {speaker_pred} diizinkan, tapi command tidak pasti"
                    return speaker_pred, command_pred, min(speaker_confidence, command_confidence), status
            else:
                status = f"Speaker confidence terlalu rendah ({speaker_confidence:.3f})"
                return speaker_pred, None, speaker_confidence, status
        else:
            # Speaker tidak diizinkan
            status = f"AKSES DITOLAK: Speaker '{speaker_pred}' tidak diizinkan"
            return speaker_pred, None, speaker_confidence, status
            
    except Exception as e:
        return None, None, 0, f"Error: {str(e)}"

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
                speaker, command, confidence, status = predict_voice(audio, speaker_pipeline, command_pipeline, confidence_threshold=0.4)
                
                # Display results
                st.header("🎯 Analysis Results")
                
                if speaker is None:
                    st.error(f"❌ {status}")
                    st.write(f"**Confidence:** {confidence:.1%}")
                else:
                    # Determine result type
                    if "SUCCESS" in status:
                        st.success(f"✅ {status}")
                    elif "AKSES DITOLAK" in status:
                        st.error(f"🚫 {status}")
                    else:
                        st.warning(f"⚠️ {status}")
                    
                    # Show detailed results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="🗣️ Speaker Identified",
                            value=speaker.title() if speaker else "Unknown",
                        )
                    
                    with col2:
                        st.metric(
                            label="🎯 Command Detected", 
                            value=command.title() if command else "None",
                        )
                    
                    # Confidence bar
                    st.write("**Overall Confidence:**")
                    st.progress(confidence)
                    st.write(f"{confidence:.1%}")
                    
                    # Authorization status
                    if speaker and speaker.lower() in ['lutfi', 'harits']:
                        st.success("✅ **Speaker Authorized**")
                    else:
                        st.error("❌ **Speaker Not Authorized**")
                    
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
