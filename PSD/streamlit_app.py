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

# Audio recording dependencies
from audio_recorder_streamlit import audio_recorder
import tempfile
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Voice Recognition",
    page_icon="🎤",
    layout="wide"
)

# Judul aplikasi
st.title("Sistem Identifikasi Suara")
st.markdown("Identifikasi speaker (Lutfi/Harits) dan command (buka/tutup)")

@st.cache_data
def extract_statistical_features(audio_data, sr=22050):
    """Ekstraksi feature statistik dari audio"""
    features = {}
    
    # 1. Basic Statistical Features
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
    
    # 5. Zero Crossing Rate
    features['zcr'] = np.sum(librosa.zero_crossings(audio_data))
    features['zcr_rate'] = features['zcr'] / len(audio_data)
    
    # 6. Spectral Features
    try:
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
    except:
        features['spectral_centroid'] = 0
        features['spectral_bandwidth'] = 0
        features['spectral_rolloff'] = 0
    
    # 7. Temporal Features
    try:
        onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sr)
        features['onset_count'] = len(onset_frames)
        tempo = librosa.beat.tempo(y=audio_data, sr=sr)
        features['tempo'] = tempo[0] if len(tempo) > 0 else 0
    except:
        features['onset_count'] = 0
        features['tempo'] = 0
    
    # 8. Autocorrelation Features
    autocorr = np.correlate(audio_data, audio_data, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    if len(autocorr) > 100:
        features['autocorr_max'] = np.max(autocorr[1:100])  # exclude lag 0
        features['autocorr_mean'] = np.mean(autocorr[1:100])
    else:
        features['autocorr_max'] = np.max(autocorr[1:]) if len(autocorr) > 1 else 0
        features['autocorr_mean'] = np.mean(autocorr[1:]) if len(autocorr) > 1 else 0
    
    # 9. Envelope Features
    try:
        envelope = np.abs(signal.hilbert(audio_data))
        features['envelope_mean'] = np.mean(envelope)
        features['envelope_std'] = np.std(envelope)
        features['envelope_max'] = np.max(envelope)
    except:
        features['envelope_mean'] = 0
        features['envelope_std'] = 0
        features['envelope_max'] = 0
    
    # 10. MFCC Statistical Features
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
    except:
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = 0
            features[f'mfcc_{i+1}_std'] = 0
    
    # 11. Chroma Features
    try:
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
    except:
        features['chroma_mean'] = 0
        features['chroma_std'] = 0
    
    # 12. Contrast Features
    try:
        contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        features['contrast_mean'] = np.mean(contrast)
        features['contrast_std'] = np.std(contrast)
    except:
        features['contrast_mean'] = 0
        features['contrast_std'] = 0
    
    # 13. Tonnetz Features
    try:
        tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz)
        features['tonnetz_std'] = np.std(tonnetz)
    except:
        features['tonnetz_mean'] = 0
        features['tonnetz_std'] = 0
    
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

@st.cache_resource
def load_optimized_model():
    """Load model yang sudah diperbaiki"""
    try:
        from voice_recognition_for_streamlit import load_model
        return load_model()
    except ImportError:
        st.error("❌ voice_recognition_for_streamlit.py tidak ditemukan!")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def predict_voice(audio_data, model):
    """Prediksi suara menggunakan model yang sudah diperbaiki"""
    if model is None:
        return None, 0.0, "Error: Model not loaded"
    
    try:
        # Import fungsi dari notebook yang sudah diperbaiki
        from voice_recognition_for_streamlit import streamlit_voice_recognition
        
        # Gunakan fungsi yang sudah fix confidence issue
        result = streamlit_voice_recognition(audio_data, sr=22050)
        
        if result['status'] == 'success':
            speaker = result['speaker']
            confidence = result['confidence']
            status = f"Speaker: **{speaker.title()}** (Confidence: {confidence:.1%})"
            return speaker, confidence, status
        else:
            return None, 0.0, f"Error: {result.get('error_message', 'Unknown error')}"
        
    except ImportError:
        st.error("File voice_recognition_for_streamlit.py tidak ditemukan!")
        return None, 0.0, "Error: Import failed"
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, 0.0, f"Error: {str(e)}"

def process_uploaded_audio(uploaded_file, model):
    """Process uploaded audio file"""
    try:
        audio_data, sr = librosa.load(uploaded_file, sr=22050, duration=5)
        audio_processed = preprocess_audio(audio_data, sr)
        display_audio_analysis(uploaded_file, audio_data, audio_processed, sr, model)
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Fungsi untuk memproses audio yang direkam
def process_recorded_audio(audio_bytes, model):
    """Process recorded audio bytes"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        audio_data, sr = librosa.load(temp_file_path, sr=22050, duration=5)
        os.unlink(temp_file_path)
        
        audio_processed = preprocess_audio(audio_data, sr)
        display_audio_analysis(None, audio_data, audio_processed, sr, model, is_recorded=True)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Coba rekam ulang dengan durasi 2-4 detik")

# Fungsi untuk menampilkan analisis audio
def display_audio_analysis(uploaded_file, audio_data, audio_processed, sr, model, is_recorded=False):
    """Display audio analysis and prediction results"""
    
    # Display audio info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informasi Audio")
        st.write(f"**Durasi**: {len(audio_data)/sr:.2f} detik")
        st.write(f"**Sample Rate**: {sr} Hz")
        st.write(f"**Jumlah Sample**: {len(audio_data)}")
        st.write(f"**Range Nilai**: [{audio_data.min():.4f}, {audio_data.max():.4f}]")
        
        # Play audio
        if is_recorded:
            st.info("Audio dari rekaman langsung")
            # Create audio array for playback
            audio_for_playback = (audio_data * 32767).astype(np.int16)
            st.audio(audio_for_playback, sample_rate=sr)
        else:
            st.audio(uploaded_file)
    
    with col2:
        st.subheader("Visualisasi Waveform")
        fig, ax = plt.subplots(figsize=(10, 4))
        time_axis = np.linspace(0, len(audio_data)/sr, len(audio_data))
        ax.plot(time_axis, audio_data, color='steelblue', linewidth=0.8)
        ax.set_xlabel('Waktu (detik)')
        ax.set_ylabel('Amplitudo')
        ax.set_title('Waveform Audio')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Prediksi Two-Stage
    st.subheader("Hasil Prediksi Two-Stage")
    
    # Control untuk debug mode
    debug_mode = st.checkbox("Debug Mode", value=False, help="Tampilkan informasi debug detail", key=f"debug_{is_recorded}")
    
    if st.button("Analisis Voice (Two-Stage)", type="primary", key=f"analyze_{is_recorded}"):
        with st.spinner("🔄 Processing audio..."):
            speaker, confidence, status = predict_voice(audio_processed, model)
        
        # Display hasil prediksi
        st.markdown("---")
        
        if speaker is None:
            st.error(status)
            st.metric("Status", "ERROR")
        else:
            st.success(status)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Speaker", speaker.title())
            
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
            
            # Progress bar
            st.progress(confidence)

# Load optimized model
model = load_optimized_model()

if model is not None:
    # Sidebar
    st.sidebar.header("Model Info")
    st.sidebar.write("**Model**: RandomForest (Optimized)")
    st.sidebar.write("**Status**: Confidence Fixed")
    st.sidebar.write("**Range**: 65-95%")
    st.sidebar.write("**Security**: Access control enabled")
    
    # Main interface dengan tabs
    st.header("Input Audio")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload File", "Rekam Langsung"])
    
    with tab1:
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader(
            "Pilih file audio (.wav, .mp3, .m4a)", 
            type=['wav', 'mp3', 'm4a'],
            help="Upload file audio untuk identifikasi speaker dan command"
        )
        
        if uploaded_file is not None:
            process_uploaded_audio(uploaded_file, model)
    
    with tab2:
        st.subheader("Rekam Audio Langsung")
        
        # Recording controls
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.info("**Instruksi Rekaman:**")
            st.markdown("""
            1. Klik tombol **"Start Recording"**
            2. Ucapkan **"buka"** atau **"tutup"** dengan jelas
            3. Klik **"Stop Recording"**
            4. Audio akan otomatis dianalisis
            """)
        
        with col2:
            st.warning("**Tips untuk hasil terbaik:**")
            st.markdown("""
            - Rekam di tempat yang tenang
            - Bicara dengan jelas dan normal
            - Durasi ideal: 2-4 detik
            - Jarak mikrofon: 20-30 cm
            """)
        
        # Audio recorder component
        try:
            audio_bytes = audio_recorder(
                text="Klik untuk mulai merekam",
                recording_color="#e8b62c",
                neutral_color="#6aa36f",
                icon_name="microphone",
                icon_size="2x",
                pause_threshold=2.0,
                sample_rate=22050,
                key="audio_recorder"
            )
            
            if audio_bytes:
                st.success("Audio berhasil direkam!")
                
                # Process recorded audio
                process_recorded_audio(audio_bytes, model)
                
        except Exception as e:
            st.error(f"Error dengan audio recorder: {str(e)}")
            st.info("**Solusi alternatif:** Gunakan tab 'Upload File' untuk menganalisis audio yang sudah ada.")
            
            # Fallback manual recording option
            st.markdown("---")
            st.subheader("Alternatif: Upload Rekaman Manual")
            st.markdown("""
            Jika fitur rekam langsung tidak berfungsi:
            1. Rekam suara menggunakan aplikasi lain (Voice Recorder, dll)
            2. Simpan sebagai file .wav atau .mp3
            3. Upload menggunakan tab "Upload File" di atas
            """)
    


else:
    # Model loading failed
    st.error("**System Error:** Model tidak dapat dimuat!")
    
    st.markdown("### Checklist File yang Dibutuhkan:")
    st.markdown("""
    - [ ] `speaker_model_pipeline.pkl` - Model untuk speaker recognition  
    - [ ] `command_model_pipeline.pkl` - Model untuk command recognition
    
    **Tip:** Jalankan notebook training terlebih dahulu untuk generate model files.
    """)

# Sidebar - Additional Info  
st.sidebar.markdown("---")
st.sidebar.header("Tentang Aplikasi")
st.sidebar.markdown("""
**Two-Stage Voice Recognition System**

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
st.sidebar.header("Setup & Dependencies")
st.sidebar.markdown("""
**Untuk fitur rekam langsung:**
```bash
pip install audio-recorder-streamlit
```

**Jika ada masalah microphone:**
- Pastikan browser mengizinkan akses microphone
- Gunakan HTTPS (untuk production)
- Chrome/Firefox direkomendasikan
""")

st.sidebar.markdown("---")
st.sidebar.info("Dikembangkan untuk Proyek PSD - Two-Stage Voice Recognition")

# Instructions section
st.markdown("---")
st.header("Petunjuk Penggunaan")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sistem Two-Stage")
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
    st.subheader("Technical Details")
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
st.info("**Tip:** Untuk hasil terbaik, gunakan file audio dengan kualitas baik dan durasi 3-5 detik")
