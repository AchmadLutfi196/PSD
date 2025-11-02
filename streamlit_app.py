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
st.title("Sistem Identifikasi Suara: Buka/Tutup")
st.markdown("---")
st.markdown("**Aplikasi untuk mengklasifikasi suara 'buka' dan 'tutup' menggunakan Machine Learning**")

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

# Fungsi load model
@st.cache_resource
def load_model():
    """Load model pipeline"""
    try:
        # Try loading the large dataset model first
        model_data = joblib.load('voice_classifier_pipeline_large_400_samples.pkl')
        return model_data
    except FileNotFoundError:
        try:
            # Fallback to original model
            model_data = joblib.load('voice_classifier_pipeline.pkl')
            return model_data
        except FileNotFoundError:
            st.error("Model file tidak ditemukan!")
            st.info("Pastikan file model ada: 'voice_classifier_pipeline_large_400_samples.pkl' atau 'voice_classifier_pipeline.pkl'")
            return None

# Fungsi prediksi
def predict_audio_class(audio_data, pipeline):
    """Prediksi kelas untuk audio baru"""
    if pipeline is None:
        return None, None, None
    
    try:
        # Ekstraksi features
        features = extract_statistical_features(audio_data)
        
        # Convert ke DataFrame dan pilih features yang sama
        features_df = pd.DataFrame([features])
        features_selected = features_df[pipeline['feature_names']]
        
        # Replace inf values
        features_selected = features_selected.replace([np.inf, -np.inf], np.nan)
        features_selected = features_selected.fillna(0)
        
        # Use pipeline to predict (includes scaling automatically)
        prediction_encoded = pipeline['pipeline'].predict(features_selected)[0]
        prediction = pipeline['label_encoder'].inverse_transform([prediction_encoded])[0]
        
        # Probabilitas
        if hasattr(pipeline['pipeline'], 'predict_proba'):
            probabilities = pipeline['pipeline'].predict_proba(features_selected)[0]
            confidence = np.max(probabilities)
        else:
            confidence = 1.0
        
        return prediction, confidence, features
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        return None, None, None

# Load model
pipeline = load_model()

if pipeline is not None:
    # Sidebar - Model Information
    st.sidebar.header("Informasi Model")
    st.sidebar.write(f"**Model Type**: {pipeline['model_info']['name']}")
    st.sidebar.write(f"**Accuracy**: {pipeline['model_info']['accuracy']:.4f}")
    st.sidebar.write(f"**Features**: {pipeline['model_info']['features']}")
    st.sidebar.write(f"**Training Samples**: {pipeline['model_info']['training_samples']}")
    st.sidebar.write(f"**Total Samples**: {pipeline['model_info']['total_samples']}")
    st.sidebar.write(f"**Classes**: {', '.join(pipeline['model_info']['classes'])}")
    
    # Main interface
    st.header("Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Pilih file audio (.wav, .mp3, .m4a)", 
        type=['wav', 'mp3', 'm4a'],
        help="Upload file audio yang ingin diklasifikasi"
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
            
            # Prediksi
            st.subheader("Hasil Prediksi")
            
            if st.button("Klasifikasi Audio", type="primary"):
                with st.spinner("Memproses audio..."):
                    prediction, confidence, features = predict_audio_class(audio_processed, pipeline)
                
                if prediction is not None:
                    # Display hasil prediksi
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Prediksi Kelas",
                            value=prediction.upper(),
                            help="Hasil klasifikasi audio"
                        )
                    
                    with col2:
                        st.metric(
                            label="Confidence Score",
                            value=f"{confidence:.2%}",
                            help="Tingkat kepercayaan model"
                        )
                    
                    with col3:
                        status = "TINGGI" if confidence > 0.8 else "SEDANG" if confidence > 0.6 else "RENDAH"
                        st.metric(
                            label="Status Confidence",
                            value=status,
                            help="Interpretasi tingkat kepercayaan"
                        )
                    
                    # Progress bar untuk confidence
                    st.progress(confidence)
                    
                    # Feature analysis (optional)
                    with st.expander("Analisis Features (Advanced)"):
                        if features:
                            features_df = pd.DataFrame([features]).T
                            features_df.columns = ['Nilai']
                            
                            # Top features
                            st.subheader("Top 10 Features")
                            top_features = pipeline['feature_names'][:10]
                            top_values = [features.get(f, 0) for f in top_features]
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.barh(top_features, top_values)
                            ax.set_xlabel('Nilai Feature')
                            ax.set_title('Top 10 Features yang Digunakan Model')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Feature statistics
                            st.subheader("Statistik Features")
                            selected_features = ['mean', 'std', 'zcr_rate', 'spectral_centroid', 'energy']
                            stats_data = {f: features.get(f, 0) for f in selected_features if f in features}
                            
                            if stats_data:
                                stats_df = pd.DataFrame([stats_data])
                                st.dataframe(stats_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading audio file: {str(e)}")
            st.info("Pastikan file audio dalam format yang didukung (WAV, MP3, M4A)")

# Sidebar - Additional Info
st.sidebar.markdown("---")
st.sidebar.header("Tentang Aplikasi")
st.sidebar.markdown("""
**Voice Recognition System**

Aplikasi ini menggunakan:
- **28 Statistical Features** dari time series audio
- **Machine Learning Models** untuk klasifikasi
- **Real-time Processing** untuk prediksi instan

**Cara Penggunaan:**
1. Upload file audio (.wav/.mp3/.m4a)
2. Klik tombol 'Klasifikasi Audio'
3. Lihat hasil prediksi dan confidence score

**Supported Classes:**
- Buka
- Tutup
""")

st.sidebar.markdown("---")
st.sidebar.info("Dikembangkan untuk Proyek PSD - Voice Classification")
