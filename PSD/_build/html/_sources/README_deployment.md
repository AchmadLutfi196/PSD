# Voice Recognition Deployment with Streamlit

## Deskripsi
Aplikasi web untuk mengklasifikasi suara "buka" dan "tutup" menggunakan machine learning. Aplikasi ini dibuat dengan Streamlit dan menggunakan model yang telah dilatih dengan 300 file audio real.

## Features
- **Real-time Audio Classification**: Upload dan klasifikasi file audio secara instan
- **Interactive Visualization**: Waveform display dan analisis features
- **Confidence Scoring**: Tingkat kepercayaan prediksi model
- **Feature Analysis**: Analisis detail features yang digunakan model
- **Multi-format Support**: Mendukung WAV, MP3, dan M4A files

## Model Performance
- **Training Accuracy**: 100%
- **Validation Accuracy**: 100%
- **Features Used**: 61 statistical time series features
- **Model Type**: Random Forest Classifier

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd PSD
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Ensure Model File Exists
Pastikan file `voice_classifier_pipeline.pkl` ada di direktori yang sama. Jika belum ada, jalankan notebook training terlebih dahulu:
```bash
jupyter notebook voice_open_close_identification.ipynb
```

## Usage

### Local Development
```bash
streamlit run streamlit_app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

### Cloud Deployment

#### Streamlit Cloud
1. Push code ke GitHub repository
2. Login ke [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect repository dan deploy
4. Pastikan `voice_classifier_pipeline.pkl` ter-upload

#### Heroku Deployment
1. Install Heroku CLI
2. Create Procfile:
```
web: sh setup.sh && streamlit run streamlit_app.py
```

3. Create setup.sh:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
```

4. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

## File Structure
```
PSD/
├── streamlit_app.py              # Main Streamlit application
├── voice_classifier_pipeline.pkl # Trained model pipeline
├── voice_open_close_identification.ipynb # Training notebook
├── requirements.txt              # Dependencies
├── README_deployment.md          # This file
└── dataset/                      # Audio dataset (if included)
```

## How to Use the App

1. **Upload Audio File**
   - Click "Browse files" atau drag & drop file audio
   - Supported formats: WAV, MP3, M4A
   - Durasi maksimal: 5 detik (otomatis dipotong)

2. **View Audio Information**
   - Lihat informasi durasi, sample rate, dll
   - Play audio untuk memverifikasi
   - Visualisasi waveform

3. **Classify Audio**
   - Klik tombol "🎯 Klasifikasi Audio"
   - Tunggu proses ekstraksi features
   - Lihat hasil prediksi dan confidence score

4. **Analyze Results**
   - Prediksi kelas: "BUKA" atau "TUTUP"
   - Confidence score dalam persentase
   - Status confidence: TINGGI/SEDANG/RENDAH
   - Optional: Analisis detail features

## Technical Details

### Feature Engineering
Model menggunakan 61 statistical features:
- Basic statistics (mean, std, variance, etc.)
- Zero Crossing Rate (ZCR)
- Spectral features (centroid, bandwidth, rolloff)
- MFCC coefficients and variations
- Temporal features (onset, tempo)
- Autocorrelation features
- Envelope characteristics

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Features**: Top 30 most important features
- **Preprocessing**: StandardScaler normalization
- **Cross-validation**: 5-fold CV with 100% accuracy

## Troubleshooting

### Common Issues

1. **Model file not found**
   ```
   Error: Model file 'voice_classifier_pipeline.pkl' tidak ditemukan!
   ```
   **Solution**: Jalankan notebook training untuk generate model file

2. **Audio loading error**
   ```
   Error loading audio file
   ```
   **Solution**: 
   - Pastikan format file didukung (WAV/MP3/M4A)
   - Cek ukuran file tidak terlalu besar (< 10MB)
   - Pastikan file tidak corrupt

3. **Feature extraction error**
   ```
   Error dalam prediksi
   ```
   **Solution**:
   - Pastikan audio memiliki content (tidak silent)
   - Cek duration minimal 0.1 detik
   - Restart aplikasi

### Performance Tips

1. **Optimize for production**:
   - Use caching (`@st.cache_data`, `@st.cache_resource`)
   - Limit file upload size
   - Add progress indicators for long processes

2. **Memory management**:
   - Clear audio data after processing
   - Limit concurrent users if deploying

## Development

### Local Development Setup
```bash
# Install in development mode
pip install -e .

# Run with auto-reload
streamlit run streamlit_app.py --server.runOnSave true
```

### Testing
```bash
# Test with sample audio files
python -c "
import streamlit_app as app
import librosa

# Load test audio
audio, sr = librosa.load('test_audio.wav')
result = app.predict_audio_class(audio, app.load_model())
print(f'Prediction: {result[0]}, Confidence: {result[1]:.3f}')
"
```

## Contributing
1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License
This project is licensed under the MIT License.

## Contact
- **Developer**: [Your Name]
- **Email**: [Your Email]
- **Project**: PSD Voice Recognition System