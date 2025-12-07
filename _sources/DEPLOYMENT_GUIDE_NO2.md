# 🌬️ Panduan Deployment Aplikasi Prediksi NO2

## 📋 Ringkasan Deployment

Aplikasi Streamlit untuk prediksi konsentrasi NO2 berdasarkan data historis dengan evaluasi standar WHO telah berhasil dibuat.

## 📁 File yang Dibuat

### 1. **streamlit_app_no2.py** - Aplikasi Utama
- **Deskripsi**: Interface Streamlit lengkap untuk prediksi NO2
- **Fitur**:
  - Input manual atau skenario pre-defined
  - Prediksi menggunakan KNeighborsRegressor + MinMaxScaler
  - Evaluasi standar WHO (10 µg/m³ annual, 25 µg/m³ 24-hour)
  - Visualisasi timeline dan perbandingan WHO
  - Prediksi multi-hari eksperimental
  - Interface user-friendly dengan sidebar informatif

### 2. **requirements_no2.txt** - Dependencies
```
streamlit==1.28.2
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
```

### 3. **test_no2_model.py** - Script Testing
- **Deskripsi**: Validasi model sebelum deployment
- **Fungsi**: Test loading model, prediksi dengan 3 skenario berbeda

### 4. **Model Files** (Sudah Ada)
- `knn_model.pkl` - Model KNeighborsRegressor terlatih
- `minmax_scaler.pkl` - Scaler untuk normalisasi data

## 🚀 Cara Menjalankan Aplikasi

### Step 1: Install Dependencies
```bash
pip install -r requirements_no2.txt
```

### Step 2: Test Model (Opsional)
```bash
python test_no2_model.py
```

### Step 3: Jalankan Aplikasi
```bash
streamlit run streamlit_app_no2.py
```

### Alternative dengan Python Module
```bash
python -m streamlit run streamlit_app_no2.py
```

## 📊 Cara Penggunaan Aplikasi

### Input Methods
1. **Manual Input**: 
   - Input nilai NO2(t-2) dan NO2(t-1) dalam mol/m²
   - Range: 0.0 - 0.01 mol/m²
   - Format: 6 decimal places

2. **Skenario Pre-defined**:
   - **Skenario 1 - Rendah**: [0.10, 0.12] mol/m² → Status Sangat Baik
   - **Skenario 2 - Sedang**: [0.00035, 0.00040] mol/m² → Status Perhatian
   - **Skenario 3 - Tinggi**: [0.00080, 0.00090] mol/m² → Status Berbahaya

### Output
- **Prediksi NO2**: Dalam mol/m² dan µg/m³
- **Status WHO**: Sangat Baik ✅ / Perhatian ⚠️ / Berbahaya ❌
- **Visualisasi**: Timeline prediksi dan evaluasi standar WHO
- **Interpretasi**: Penjelasan lengkap kondisi dan rekomendasi

### Fitur Tambahan
- **Prediksi Multi-Hari**: Eksperimental untuk 2 hari ke depan
- **Sidebar Info**: Informasi model dan standar WHO
- **Responsive Design**: Interface yang user-friendly

## 📈 Model Architecture

### Model: KNeighborsRegressor
- **Input Features**: NO2(t-2), NO2(t-1)
- **Preprocessing**: MinMaxScaler normalization
- **Output**: Prediksi NO2(t) dalam mol/m²
- **Conversion**: Factor 46010 untuk konversi ke µg/m³

### WHO Standards
- **Annual Limit**: 10 µg/m³ (Sangat Baik)
- **24-Hour Limit**: 25 µg/m³ (Perhatian)
- **Above 24-Hour**: >25 µg/m³ (Berbahaya)

## 🎯 Contoh Penggunaan

### Skenario Testing:
1. **Input Low**: NO2(t-2)=0.0001, NO2(t-1)=0.00012
   - **Expected**: Status Sangat Baik (< 10 µg/m³)

2. **Input Medium**: NO2(t-2)=0.00035, NO2(t-1)=0.00040
   - **Expected**: Status Perhatian (10-25 µg/m³)

3. **Input High**: NO2(t-2)=0.00080, NO2(t-1)=0.00090
   - **Expected**: Status Berbahaya (> 25 µg/m³)

## 🔧 Troubleshooting

### Error: Model files tidak ditemukan
- **Solusi**: Pastikan `knn_model.pkl` dan `minmax_scaler.pkl` ada di direktori yang sama
- **Check**: Jalankan notebook training untuk generate model files

### Error: Streamlit tidak dikenal
- **Solusi 1**: `pip install streamlit`
- **Solusi 2**: `python -m streamlit run streamlit_app_no2.py`

### Error: Dependencies missing
- **Solusi**: `pip install -r requirements_no2.txt`

## 📝 Catatan Penting

1. **Model Purpose**: Untuk tujuan akademis/penelitian
2. **Prediksi Multi-hari**: Bersifat eksperimental
3. **Konsultasi Expert**: Selalu konsultasi ahli lingkungan untuk keputusan penting
4. **Data Input**: Pastikan input dalam range yang wajar (0.0-0.01 mol/m²)

## 🎉 Status Deployment

✅ **BERHASIL**: Aplikasi Streamlit prediksi NO2 siap digunakan!

**Files Ready**:
- ✅ streamlit_app_no2.py (Main app)
- ✅ requirements_no2.txt (Dependencies)
- ✅ test_no2_model.py (Testing script)
- ✅ knn_model.pkl (Trained model)
- ✅ minmax_scaler.pkl (Scaler)

**Next Steps**: Jalankan `streamlit run streamlit_app_no2.py` untuk mulai menggunakan aplikasi!