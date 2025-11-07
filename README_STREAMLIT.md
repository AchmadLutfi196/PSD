# 🎤 Two-Stage Voice Recognition System

Sistem identifikasi suara dua tahap yang dapat membedakan speaker (Lutfi/Harits) dan command (buka/tutup) menggunakan statistical time series features.

## 🚀 Fitur Utama

### ✨ **Two-Stage Recognition**
- **Stage 1:** Speaker Recognition (Lutfi vs Harits) - RandomForest 100% accuracy
- **Stage 2:** Command Recognition (Buka vs Tutup) - SVM 100% accuracy
- **Security:** Access control untuk menolak speaker tidak dikenal

### 🎙️ **Input Methods**
- **Upload File:** Support WAV, MP3, M4A
- **Rekam Langsung:** Real-time recording dengan microphone
- **Real-time Analysis:** Instant processing dan feedback

### 🔧 **Technical Features**
- **61 Statistical Time Series Features** per audio
- **Perfect Accuracy:** 100% pada training dataset
- **Advanced Analytics:** Feature importance visualization
- **Debug Mode:** Detailed analysis information

## 📦 Installation

### 1. Install Dependencies
```bash
pip install -r requirements_streamlit.txt
```

### 2. Pastikan Model Files Ada
Jalankan notebook training terlebih dahulu untuk generate:
- `speaker_model_pipeline.pkl`
- `command_model_pipeline.pkl`

### 3. Run Aplikasi
```bash
streamlit run streamlit_app.py
```

## 🎯 Cara Penggunaan

### 📁 Upload File Method
1. Pilih tab **"Upload File"**
2. Upload file audio (.wav/.mp3/.m4a)
3. Klik **"Analisis Voice (Two-Stage)"**
4. Lihat hasil identifikasi speaker dan command

### 🎙️ Rekam Langsung Method
1. Pilih tab **"Rekam Langsung"**
2. Klik **"Start Recording"**
3. Ucapkan **"buka"** atau **"tutup"** dengan jelas (2-4 detik)
4. Klik **"Stop Recording"**
5. Audio otomatis dianalisis

## ⚠️ Tips untuk Hasil Terbaik

### 🎤 Recording Tips
- **Environment:** Tempat tenang, minimal noise
- **Distance:** 20-30 cm dari microphone
- **Duration:** 2-4 detik optimal
- **Voice:** Bicara jelas dan natural
- **Words:** Hanya "buka" atau "tutup"

### 🔧 Technical Requirements
- **Browser:** Chrome/Firefox (recommended)
- **Microphone:** Allow browser access
- **Connection:** HTTPS untuk production use
- **Audio Quality:** Clear voice, minimal background noise

## 🛡️ Security System

### 🔐 Access Control
- **Speaker Threshold:** 70% confidence minimum
- **Unauthorized Rejection:** Suara tidak dikenal ditolak
- **Two-Stage Verification:** Speaker → Command validation

### ⚡ Performance
- **Speaker Model:** RandomForest - 100% accuracy
- **Command Model:** SVM - 100% accuracy
- **Processing:** Real-time feature extraction
- **Response:** Instant feedback dan action

## 📊 System Architecture

```
Audio Input → Feature Extraction (61 features) → Stage 1: Speaker Recognition
    ↓
Lutfi/Harits Identified → Stage 2: Command Recognition → Buka/Tutup Action
    ↓
Access Control Check → Final Result + Action Execution
```

## 🔧 Troubleshooting

### 🎙️ Recording Issues
**Problem:** Microphone tidak berfungsi
- **Solution:** Check browser permissions
- **Alternative:** Use upload file method

**Problem:** Audio quality buruk
- **Solution:** Gunakan tempat tenang, jarak optimal

**Problem:** Recognition accuracy rendah
- **Solution:** Bicara lebih jelas, cek pronunciation

### 💻 Technical Issues
**Problem:** Import error audio-recorder-streamlit
- **Solution:** `pip install audio-recorder-streamlit==0.0.8`

**Problem:** Model files not found
- **Solution:** Jalankan notebook training untuk generate models

**Problem:** Feature extraction error
- **Solution:** Pastikan audio format supported dan tidak corrupt

## 📈 Model Performance

### 🎯 Training Results
- **Dataset:** 300+ audio files real
- **Speaker Classes:** Lutfi, Harits (perfect separation)
- **Command Classes:** Buka, Tutup (perfect separation)
- **Features:** 61 statistical time series features
- **Validation:** 5-fold cross-validation

### 📊 Feature Importance
**Top Discriminative Features:**
1. **ZCR Rate** - Zero Crossing Rate
2. **MFCC Variations** - Cepstral coefficients
3. **Spectral Features** - Frequency characteristics
4. **Energy Features** - Signal power analysis
5. **Envelope Features** - Amplitude envelope

## 🎓 Academic Context

**Project:** Proyek Sains Data (PSD) - Semester 5
**Focus:** Statistical time series analysis untuk voice recognition
**Innovation:** Two-stage architecture dengan access control
**Technology:** Machine Learning, Digital Signal Processing

---

## 📞 Support

Jika ada masalah atau pertanyaan:
1. Check troubleshooting section di atas
2. Pastikan semua dependencies terinstall
3. Verifikasi model files ada dan valid
4. Test dengan audio sample berkualitas baik

**Happy Voice Recognition! 🎉**