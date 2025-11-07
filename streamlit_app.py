import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import scipy.stats as stats
from scipy import signal
import soundfile as sf
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Voice Recognition: Buka/Tutup",
    page_icon="♪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul aplikasi
st.title("🎤 Sistem Identifikasi Suara: Two-Stage Recognition")
st.markdown("---")
st.markdown("**Aplikasi untuk mengidentifikasi speaker (Lutfi/Harits) dan command (buka/tutup) menggunakan Machine Learning**")
st.info("🔒 **Two-Stage Security System:** Speaker Authentication → Command Recognition")

# Fungsi ekstraksi features (sama dengan notebook - fast mode)
@st.cache_data
def extract_statistical_features(audio_data, sr=22050, fast_mode=True):
    """
    Ekstraksi berbagai feature statistik dari sinyal audio time series
    Sesuai dengan training model (fast_mode=True)
    """
    features = {}
    
    # 1. Basic Statistical Features (selalu digunakan)
    features['mean'] = np.mean(audio_data)
    features['std'] = np.std(audio_data)
    features['var'] = np.var(audio_data)
    features['median'] = np.median(audio_data)
    features['min'] = np.min(audio_data)
    features['max'] = np.max(audio_data)
    features['range'] = features['max'] - features['min']
    
    # 2. Percentile Features
    features['q25'] = np.percentile(audio_data, 25)
    features['q75'] = np.percentile(audio_data, 75)
    features['iqr'] = features['q75'] - features['q25']
    
    # 3. Distribution Shape Features
    features['skewness'] = stats.skew(audio_data)
    features['kurtosis'] = stats.kurtosis(audio_data)
    
    # 4. Energy and Power Features
    features['energy'] = np.sum(audio_data**2)
    features['power'] = features['energy'] / len(audio_data)
    features['rms'] = np.sqrt(np.mean(audio_data**2))
    
    # 5. Zero Crossing Rate (penting untuk speech)
    features['zcr'] = np.sum(librosa.zero_crossings(audio_data))
    features['zcr_rate'] = features['zcr'] / len(audio_data)
    
    # Fast mode - hanya features paling penting
    try:
        # Spektral centroid (paling diskriminatif)
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
        
        # MFCC pertama saja (3 coefficients)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=3)
        for i in range(3):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
    except:
        features['spectral_centroid'] = 0
        for i in range(3):
            features[f'mfcc_{i+1}_mean'] = 0
            features[f'mfcc_{i+1}_std'] = 0
    
    # Features yang cepat dihitung
    # 8. Autocorrelation Features (simplified)
    autocorr = np.correlate(audio_data, audio_data, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    if len(autocorr) > 50:  # Reduce window
        features['autocorr_max'] = np.max(autocorr[1:50])
        features['autocorr_mean'] = np.mean(autocorr[1:50])
    else:
        features['autocorr_max'] = np.max(autocorr[1:]) if len(autocorr) > 1 else 0
        features['autocorr_mean'] = np.mean(autocorr[1:]) if len(autocorr) > 1 else 0
    
    # 14. Attack Time (durasi dari mulai hingga peak)
    peak_idx = np.argmax(np.abs(audio_data))
    features['attack_time'] = peak_idx / sr
    
    # 15. Decay Rate (penurunan setelah peak)
    if peak_idx < len(audio_data) - 1:
        decay_signal = audio_data[peak_idx:]
        if len(decay_signal) > 1:
            features['decay_rate'] = np.mean(np.diff(decay_signal))
        else:
            features['decay_rate'] = 0
    else:
        features['decay_rate'] = 0
    
    return features

# Fungsi preprocessing audio
def preprocess_audio(audio_data, sr, noise_threshold=0.01):
    """Preprocess audio: noise removal, trimming"""
    try:
        audio_trimmed, _ = librosa.effects.trim(audio_data, top_db=20)
    except:
        audio_trimmed = audio_data
    
    audio_denoised = np.where(np.abs(audio_trimmed) < noise_threshold, 0, audio_trimmed)
    return audio_denoised

# Fungsi load models (two-stage system)
@st.cache_resource
def load_models():
    """Load both speaker and command model pipelines"""
    speaker_pipeline = None
    command_pipeline = None
    
    try:
        speaker_pipeline = joblib.load('speaker_model_pipeline.pkl')
        st.success("✅ Loaded speaker model: speaker_model_pipeline.pkl")
    except FileNotFoundError:
        st.error("❌ Speaker model file tidak ditemukan!")
        st.info("Pastikan file 'speaker_model_pipeline.pkl' ada di direktori ini")
    
    try:
        command_pipeline = joblib.load('command_model_pipeline.pkl')
        st.success("✅ Loaded command model: command_model_pipeline.pkl")
    except FileNotFoundError:
        st.error("❌ Command model file tidak ditemukan!")
        st.info("Pastikan file 'command_model_pipeline.pkl' ada di direktori ini")
    
    return speaker_pipeline, command_pipeline

# Fungsi prediksi two-stage dengan debugging
def predict_voice_two_stage(audio_data, speaker_pipeline, command_pipeline, debug=True):
    """
    Two-stage prediction: Speaker identification + Command recognition
    """
    if speaker_pipeline is None or command_pipeline is None:
        st.error("One or both pipelines are None!")
        return None, None, None, None, "Error: Model not loaded"
    
    try:
        # Ekstraksi features dengan fast_mode=True (konsisten dengan training)
        features = extract_statistical_features(audio_data, fast_mode=True)
        
        if debug:
            st.write(f"🔍 **DEBUG**: Features extracted: {len(features)}")
        
        # Convert ke DataFrame
        features_df = pd.DataFrame([features])
        
        # STAGE 1: SPEAKER RECOGNITION
        if debug:
            st.write("🎯 **STAGE 1: Speaker Recognition**")
        
        # Speaker prediction
        speaker_features = features_df[speaker_pipeline['feature_names']]
        
        # Handle missing features
        missing_speaker_features = [f for f in speaker_pipeline['feature_names'] if f not in features_df.columns]
        if missing_speaker_features:
            if debug:
                st.warning(f"Missing speaker features: {missing_speaker_features}")
            for feat in missing_speaker_features:
                speaker_features[feat] = 0
        
        # Clean features
        speaker_features = speaker_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        speaker_features_scaled = speaker_pipeline['scaler'].transform(speaker_features)
        
        # Speaker prediction
        speaker_pred_encoded = speaker_pipeline['model'].predict(speaker_features_scaled)[0]
        speaker_pred = speaker_pipeline['label_encoder'].inverse_transform([speaker_pred_encoded])[0]
        speaker_confidence = np.max(speaker_pipeline['model'].predict_proba(speaker_features_scaled))
        
        if debug:
            st.write(f"   👤 Speaker: {speaker_pred} (confidence: {speaker_confidence:.3f})")
        
        # Check if authorized speaker
        SPEAKER_THRESHOLD = 0.7
        if speaker_confidence < SPEAKER_THRESHOLD:
            status = f"❌ Suara tidak dikenal (confidence: {speaker_confidence:.3f}) - Akses ditolak"
            return None, None, speaker_confidence, None, status
        
        # STAGE 2: COMMAND RECOGNITION (only if speaker is authorized)
        if debug:
            st.write("🗣️ **STAGE 2: Command Recognition**")
        
        command_features = features_df[command_pipeline['feature_names']]
        
        # Handle missing features
        missing_command_features = [f for f in command_pipeline['feature_names'] if f not in features_df.columns]
        if missing_command_features:
            if debug:
                st.warning(f"Missing command features: {missing_command_features}")
            for feat in missing_command_features:
                command_features[feat] = 0
        
        # Clean features
        command_features = command_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        command_features_scaled = command_pipeline['scaler'].transform(command_features)
        
        # Command prediction
        command_pred_encoded = command_pipeline['model'].predict(command_features_scaled)[0]
        command_pred = command_pipeline['label_encoder'].inverse_transform([command_pred_encoded])[0]
        command_confidence = np.max(command_pipeline['model'].predict_proba(command_features_scaled))
        
        if debug:
            st.write(f"   🗣️ Command: {command_pred} (confidence: {command_confidence:.3f})")
        
        # Overall confidence (minimum of both stages)
        overall_confidence = min(speaker_confidence, command_confidence)
        
        status = f"✅ **{speaker_pred.title()}** mengatakan '**{command_pred}**'"
        
        return speaker_pred, command_pred, overall_confidence, features, status
        
    except Exception as e:
        st.error(f"Error dalam prediksi two-stage: {str(e)}")
        import traceback
        st.text("Full traceback:")
        st.text(traceback.format_exc())
        return None, None, None, None, f"Error: {str(e)}"

# Load models (two-stage system)
speaker_pipeline, command_pipeline = load_models()

if speaker_pipeline is not None and command_pipeline is not None:
    # Sidebar - Model Information
    st.sidebar.header("📊 Informasi Model")
    
    st.sidebar.subheader("👤 Speaker Recognition")
    st.sidebar.write(f"**Model**: {speaker_pipeline['model_info']['model_type']}")
    st.sidebar.write(f"**Accuracy**: {speaker_pipeline['model_info']['accuracy']:.1%}")
    st.sidebar.write(f"**Classes**: {', '.join(speaker_pipeline['model_info']['classes'])}")
    
    st.sidebar.subheader("🗣️ Command Recognition") 
    st.sidebar.write(f"**Model**: {command_pipeline['model_info']['model_type']}")
    st.sidebar.write(f"**Accuracy**: {command_pipeline['model_info']['accuracy']:.1%}")
    st.sidebar.write(f"**Classes**: {', '.join(command_pipeline['model_info']['classes'])}")
    
    st.sidebar.subheader("🔧 Technical Details")
    st.sidebar.write(f"**Speaker Features**: {len(speaker_pipeline['feature_names'])}")
    st.sidebar.write(f"**Command Features**: {len(command_pipeline['feature_names'])}")
    st.sidebar.write("**Security**: Access control enabled")
    
    # Main interface
    st.header("🎵 Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Pilih file audio (.wav, .mp3, .m4a)", 
        type=['wav', 'mp3', 'm4a'],
        help="Upload file audio untuk identifikasi speaker dan command"
    )
    
    if uploaded_file is not None:
        # Load audio
        try:
            # Read audio file
            audio_data, sr = librosa.load(uploaded_file, sr=22050, duration=5)
            
            # Preprocess
            audio_processed = preprocess_audio(audio_data, sr)
            
            # Display audio info
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Informasi Audio")
                st.write(f"**Durasi**: {len(audio_data)/sr:.2f} detik")
                st.write(f"**Sample Rate**: {sr} Hz")
                st.write(f"**Jumlah Sample**: {len(audio_data)}")
                st.write(f"**Range Nilai**: [{audio_data.min():.4f}, {audio_data.max():.4f}]")
                
                # Play audio
                st.audio(uploaded_file)
            
            with col2:
                st.subheader("Visualisasi Waveform")
                fig, ax = plt.subplots(figsize=(10, 4))
                time_axis = np.linspace(0, len(audio_data)/sr, len(audio_data))
                ax.plot(time_axis, audio_data)
                ax.set_xlabel('Waktu (detik)')
                ax.set_ylabel('Amplitudo')
                ax.set_title('Waveform Audio')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Prediksi Two-Stage
            st.subheader("🎯 Hasil Prediksi Two-Stage")
            
            # Control untuk debug mode
            debug_mode = st.checkbox("🔧 Debug Mode", value=False, help="Tampilkan informasi debug detail")
            
            if st.button("🔍 Analisis Voice (Two-Stage)", type="primary"):
                with st.spinner("🔄 Processing two-stage recognition..."):
                    if debug_mode:
                        st.write("🚀 **Starting two-stage prediction...**")
                    
                    speaker, command, confidence, features, status = predict_voice_two_stage(
                        audio_processed, speaker_pipeline, command_pipeline, debug=debug_mode
                    )
                
                # Display hasil prediksi
                st.markdown("---")
                
                if speaker is None:
                    # Access denied - unauthorized speaker
                    st.error(status)
                    st.metric("🚫 Status", "ACCESS DENIED", help="Suara tidak dikenal atau confidence terlalu rendah")
                    
                else:
                    # Success - authorized speaker with command
                    st.success(status)
                    
                    # Metrics dalam 4 kolom
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label="👤 Speaker",
                            value=speaker.title(),
                            help="Speaker yang diidentifikasi"
                        )
                    
                    with col2:
                        st.metric(
                            label="🗣️ Command",
                            value=command.title(),
                            help="Perintah yang diucapkan"
                        )
                    
                    with col3:
                        st.metric(
                            label="🎯 Confidence",
                            value=f"{confidence:.1%}",
                            help="Tingkat kepercayaan gabungan"
                        )
                    
                    with col4:
                        conf_status = "🟢 TINGGI" if confidence > 0.8 else "🟡 SEDANG" if confidence > 0.6 else "🔴 RENDAH"
                        st.metric(
                            label="📊 Status",
                            value=conf_status,
                            help="Interpretasi confidence"
                        )
                    
                    # Progress bar untuk confidence
                    st.progress(confidence)
                    
                    # Action based on command
                    if command == "buka":
                        st.balloons()
                        st.info("🔓 **AKSI:** Pintu dibuka!")
                    else:
                        st.info("🔒 **AKSI:** Pintu ditutup!")
                    
                    # Feature analysis (optional)
                    with st.expander("🔬 Analisis Features (Advanced)"):
                        if features:
                            st.subheader("📊 Feature Analysis")
                            
                            # Create two columns for speaker and command features
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**👤 Speaker Features (Top 10)**")
                                top_speaker_features = speaker_pipeline['feature_names'][:10]
                                top_speaker_values = [features.get(f, 0) for f in top_speaker_features]
                                
                                fig, ax = plt.subplots(figsize=(8, 5))
                                ax.barh(top_speaker_features, top_speaker_values, color='skyblue')
                                ax.set_xlabel('Nilai Feature')
                                ax.set_title('Top 10 Speaker Features')
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            with col2:
                                st.write("**🗣️ Command Features (Top 10)**")
                                top_command_features = command_pipeline['feature_names'][:10]
                                top_command_values = [features.get(f, 0) for f in top_command_features]
                                
                                fig, ax = plt.subplots(figsize=(8, 5))
                                ax.barh(top_command_features, top_command_values, color='lightcoral')
                                ax.set_xlabel('Nilai Feature')
                                ax.set_title('Top 10 Command Features')
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            # Key feature statistics
                            st.subheader("📈 Key Statistics")
                            key_features = ['mean', 'std', 'zcr_rate', 'spectral_centroid', 'energy']
                            stats_data = {f: features.get(f, 0) for f in key_features if f in features}
                            
                            if stats_data:
                                stats_df = pd.DataFrame([stats_data])
                                st.dataframe(stats_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading audio file: {str(e)}")
            st.info("Pastikan file audio dalam format yang didukung (WAV, MP3, M4A)")

else:
    # Model loading failed
    st.error("🚨 **System Error:** Model tidak dapat dimuat!")
    
    st.markdown("### 📋 Checklist File yang Dibutuhkan:")
    st.markdown("""
    - [ ] `speaker_model_pipeline.pkl` - Model untuk speaker recognition  
    - [ ] `command_model_pipeline.pkl` - Model untuk command recognition
    
    **💡 Tip:** Jalankan notebook training terlebih dahulu untuk generate model files.
    """)

# Sidebar - Additional Info  
st.sidebar.markdown("---")
st.sidebar.header("📖 Tentang Aplikasi")
st.sidebar.markdown("""
**🎤 Two-Stage Voice Recognition System**

**Arsitektur:**
- **Stage 1:** Speaker Recognition (Lutfi/Harits)
- **Stage 2:** Command Recognition (Buka/Tutup)
- **Security:** Access control untuk unauthorized speakers

**Features:**
- **61 Statistical Features** dari time series audio
- **Machine Learning Models:** RandomForest + SVM
- **Real-time Processing** untuk prediksi instan
- **Perfect Accuracy:** 100% pada training data

**Cara Penggunaan:**
1. Upload file audio (.wav/.mp3/.m4a)
2. Klik 'Analisis Voice (Two-Stage)'
3. Lihat hasil identifikasi speaker dan command

**Supported:**
- **Speakers:** Lutfi, Harits
- **Commands:** Buka, Tutup
- **Security:** Reject unauthorized voices
""")

st.sidebar.markdown("---")
st.sidebar.info("🎓 Dikembangkan untuk Proyek PSD - Two-Stage Voice Recognition")

# Instructions section
st.markdown("---")
st.header("📋 Petunjuk Penggunaan")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Sistem Two-Stage")
    st.markdown("""
    **Stage 1: Speaker Recognition**
    - Identifikasi apakah suara dari Lutfi atau Harits
    - Confidence threshold untuk keamanan
    - Reject suara yang tidak dikenal
    
    **Stage 2: Command Recognition**
    - Klasifikasi perintah "buka" atau "tutup"
    - Hanya dijalankan jika speaker authorized
    - High accuracy classification
    """)

with col2:
    st.subheader("🔧 Technical Details")
    st.markdown("""
    **Feature Extraction:**
    - 61 statistical time series features
    - MFCC, spectral, temporal features
    - Zero crossing rate, energy features
    
    **Machine Learning:**
    - RandomForest untuk speaker recognition
    - SVM untuk command recognition
    - Perfect 100% accuracy pada training data
    """)

st.markdown("---")
st.info("💡 **Tip:** Untuk hasil terbaik, gunakan file audio dengan kualitas baik dan durasi 3-5 detik")
