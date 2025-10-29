# Perubahan Model: Lag 3 → Lag 5

## Ringkasan Perubahan

Model prediksi NO₂ telah diperbarui dari menggunakan **3 lagged features** menjadi **5 lagged features** untuk meningkatkan akurasi prediksi.

## Detail Perubahan

### Sebelum (Lag 3):
- **Input Features**: 3 fitur
  - NO2(t-3), NO2(t-2), NO2(t-1)
- **Total Samples**: 393 samples
- **Visualisasi**: 3 hari + prediksi (4 bar)

### Sesudah (Lag 5):
- **Input Features**: 5 fitur
  - NO2(t-5), NO2(t-4), NO2(t-3), NO2(t-2), NO2(t-1)
- **Total Samples**: 390 samples
- **Visualisasi**: 5 hari + prediksi (6 bar)

## Dampak Perubahan

### ✅ Keuntungan:
1. **Informasi Historis Lebih Banyak**: Model memiliki akses ke pola 5 hari sebelumnya
2. **Potensi Akurasi Lebih Baik**: Lebih banyak data untuk memprediksi pola time series
3. **Tangkap Pola Lebih Panjang**: Dapat mendeteksi tren yang lebih panjang

### ⚠️ Trade-off:
1. **Samples Berkurang Sedikit**: Dari 393 → 390 samples (pengurangan 3 samples)
   - Alasan: Butuh 5 data sebelumnya, jadi 5 row pertama hilang karena shifting
2. **Dimensi Input Lebih Besar**: Dari 3 → 5 features
   - Model perlu lebih banyak data untuk training yang efektif

## File yang Dimodifikasi

### 1. Judul Notebook
```markdown
# Prediksi Nilai NO2 Satu Hari Kedepan dengan KNN Regression (Lag 5)
```

### 2. LANGKAH 3: Supervised Transformation
```python
# Gunakan 5 lag
n_lags = 5
supervised_df = series_to_supervised(df[["NO2"]], n_in=n_lags, n_out=1)
```

**Output**: 
- Features shape: (390, 5)
- Kolom: NO2(t-5), NO2(t-4), NO2(t-3), NO2(t-2), NO2(t-1), NO2(t)

### 3. LANGKAH 9: Prediksi
```python
# Ambil 5 hari terakhir
last_5_days_original = supervised_df[["NO2(t-5)", "NO2(t-4)", "NO2(t-3)", "NO2(t-2)", "NO2(t-1)"]].iloc[-1].values

# Plot 5 hari terakhir + prediksi
days = ['t-5', 't-4', 't-3', 't-2', 't-1', 'PREDIKSI\n(t)']
```

### 4. Implementasi Deployment
```python
# Data baru sekarang butuh 5 nilai
data_baru = np.array([[0.00012, 0.00015, 0.00018, 0.00020, 0.00022]])
# NO2(t-5), NO2(t-4), NO2(t-3), NO2(t-2), NO2(t-1)
```

### 5. Ringkasan
```python
3. SUPERVISED TRANSFORMATION
   - Lagged features: 5 lag
   - Input features: [NO2(t-5), NO2(t-4), NO2(t-3), NO2(t-2), NO2(t-1)]
   - Target: NO2(t)
   - Total samples: 390
```

## Hasil Eksekusi

### Model Performance (dengan Lag 5):
- **Best K**: 11
- **Training Set**: 312 samples (80%)
- **Test Set**: 78 samples (20%)
- **Threshold**: 0.00003555 mol/m² (75th percentile)
- **R² Score**: ~0.65 (meningkat dengan K=11)

### Prediksi Terakhir:
```
Data 5 hari terakhir (Original):
  - NO2(t-5): 0.00000777 mol/m²
  - NO2(t-4): 0.00000687 mol/m²
  - NO2(t-3): 0.00000596 mol/m²
  - NO2(t-2): 0.00000505 mol/m²
  - NO2(t-1): 0.00000415 mol/m²

Prediksi NO2 hari besok: 0.00001550 mol/m²
Status: AMAN (-56.40% dari threshold)
```

## Cara Menggunakan Model

### 1. Training (sudah dilakukan):
```python
# Data dengan 5 lag features
n_lags = 5
supervised_df = series_to_supervised(df[["NO2"]], n_in=n_lags, n_out=1)

# Split → Normalisasi → Training → Save
# Model tersimpan: knn_model.pkl
# Scaler tersimpan: minmax_scaler.pkl
```

### 2. Deployment (untuk data baru):
```python
import joblib
import numpy as np

# Load model dan scaler
loaded_model = joblib.load('knn_model.pkl')
loaded_scaler = joblib.load('minmax_scaler.pkl')

# Data baru: 5 hari terakhir
data_baru = np.array([[no2_t5, no2_t4, no2_t3, no2_t2, no2_t1]])

# Normalisasi
data_baru_normalized = loaded_scaler.transform(data_baru)

# Prediksi
prediksi_no2 = loaded_model.predict(data_baru_normalized)[0]

# Klasifikasi
threshold = 0.00003555  # atau load dari file
status = 'BERBAHAYA' if prediksi_no2 > threshold else 'AMAN'
```

## Rekomendasi

### ✅ Gunakan Lag 5 jika:
- Data historis tersedia (minimal 5 hari)
- Ingin menangkap pola time series yang lebih panjang
- Dataset cukup besar untuk 5 features

### ⚠️ Pertimbangkan Lag 3 jika:
- Data historis terbatas
- Ingin model lebih sederhana
- Dataset kecil

## Catatan Penting

1. **Data Leakage Prevention**: ✅ 
   - Split dilakukan SEBELUM normalisasi
   - Scaler fit HANYA pada training set

2. **File Tersimpan**: ✅
   - `knn_model.pkl` - Model dengan 5 features
   - `minmax_scaler.pkl` - Scaler untuk 5 features

3. **Konsistensi**: ⚠️
   - Pastikan data baru SELALU memiliki 5 nilai (t-5, t-4, t-3, t-2, t-1)
   - Gunakan scaler yang tersimpan, JANGAN buat scaler baru

## Kesimpulan

Model dengan **Lag 5** memberikan lebih banyak informasi historis untuk prediksi dan berpotensi meningkatkan akurasi. Perubahan ini telah diterapkan ke semua bagian notebook dengan konsisten.

---

**Tanggal Update**: 27 Oktober 2025  
**Status**: ✅ Berhasil diimplementasikan dan diuji
