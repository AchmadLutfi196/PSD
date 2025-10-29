# Standar WHO untuk NO₂ dan Perbandingan dengan Model

## 🏥 Pedoman WHO untuk Nitrogen Dioxide (NO₂)

### Ambang Batas Kesehatan WHO:

World Health Organization (WHO) menetapkan pedoman kualitas udara untuk NO₂:

| Parameter | Nilai Batas | Penjelasan |
|-----------|--------------|------------|
| **Rata-rata Tahunan** | **≤ 10 µg/m³** | Nilai rata-rata konsentrasi NO₂ sepanjang tahun |
| **Rata-rata 24 Jam** | **≤ 25 µg/m³** | Nilai rata-rata konsentrasi NO₂ dalam satu hari |

### Dampak Kesehatan:

Jika konsentrasi NO₂ **melebihi** nilai di atas, maka:
- ⚠️ Kualitas udara berada pada tingkat yang **menimbulkan risiko kesehatan signifikan**
- 🫁 Dapat menyebabkan masalah pernapasan, terutama pada kelompok rentan
- 👶 Anak-anak, lansia, dan penderita asma lebih berisiko

---

## 📊 Perbedaan: Model Prediksi vs Standar WHO

### 1. Unit Pengukuran

| Aspek | Model Kami | WHO |
|-------|------------|-----|
| **Unit** | mol/m² | µg/m³ |
| **Nama** | Vertical Column Density | Konsentrasi Volume |
| **Mengukur** | Total NO₂ dalam kolom atmosfer vertikal | Konsentrasi NO₂ di permukaan tanah |
| **Sumber Data** | Satelit Sentinel-5P | Ground monitoring station |

### 2. Metodologi Threshold

#### Model Prediksi (Proyek Ini):
- **Basis**: 75th percentile dari data training
- **Sifat**: Relatif terhadap data historis lokal (Bangkalan, Madura)
- **Tujuan**: 
  - Deteksi anomali
  - Prediksi pola time series
  - Early warning berdasarkan trend lokal
- **Klasifikasi**:
  - AMAN: NO₂ ≤ 75th percentile
  - BERBAHAYA: NO₂ > 75th percentile

#### Standar WHO:
- **Basis**: Penelitian epidemiologi global
- **Sifat**: Absolut dan universal
- **Tujuan**: Proteksi kesehatan publik
- **Klasifikasi**:
  - AMAN: ≤ 10 µg/m³ (tahunan) atau ≤ 25 µg/m³ (24 jam)
  - BERBAHAYA: > nilai di atas

---

## ⚙️ Mengapa Konversi Tidak Sederhana?

### Perbedaan Fundamental:

```
mol/m² (Satelit)                    µg/m³ (Permukaan)
      ↓                                     ↓
┌─────────────────┐               ┌─────────────────┐
│  Total Kolom    │               │   Konsentrasi   │
│   Vertikal      │   ≠  ≠  ≠    │   di Titik      │
│   Atmosfer      │               │   Spesifik      │
└─────────────────┘               └─────────────────┘
```

### Faktor yang Mempengaruhi Konversi:

1. **Profil Vertikal NO₂**: Distribusi NO₂ di berbagai ketinggian atmosfer
2. **Mixing Layer Height**: Ketinggian lapisan pencampuran atmosfer
3. **Kondisi Meteorologi**: Suhu, tekanan, kelembaban, angin
4. **Waktu Pengukuran**: Satelit vs ground station punya timing berbeda
5. **Resolusi Spasial**: Area coverage satelit vs titik ground station

### Rumus Konversi (Simplified):

```python
# SANGAT SIMPLIFIED - Tidak akurat untuk riset!
# Hanya untuk ilustrasi konsep

def simplified_conversion(column_density_mol_m2):
    """
    Konversi KASAR mol/m² ke µg/m³
    
    Asumsi (TIDAK REALISTIS):
    - Mixing layer height = 1000 m
    - Semua NO₂ ada di mixing layer
    - Distribusi uniform
    """
    
    # Konstanta
    molecular_weight_no2 = 46.0055  # g/mol
    avogadro = 6.022e23  # molecules/mol
    mixing_layer_height = 1000  # meter (ASUMSI!)
    
    # Konversi
    molecules_m2 = column_density_mol_m2 * avogadro
    molecules_m3 = molecules_m2 / mixing_layer_height
    mol_m3 = molecules_m3 / avogadro
    g_m3 = mol_m3 * molecular_weight_no2
    ug_m3 = g_m3 * 1e6
    
    return ug_m3

# ⚠️ WARNING: Ini hanya ilustrasi konsep!
# Konversi RIIL memerlukan:
# 1. Data mixing layer height aktual
# 2. Profil vertikal NO₂
# 3. Model transfer atmosfer
# 4. Validasi dengan ground station
```

---

## 🎯 Penggunaan yang Tepat untuk Setiap Metode

### Model Prediksi (Proyek Ini) Cocok Untuk:

✅ **Monitoring Tren Jangka Panjang**
- Melihat pola musiman
- Trend tahunan NO₂ di wilayah Bangkalan

✅ **Deteksi Anomali**
- Identifikasi hari-hari dengan NO₂ tidak normal
- Early warning berdasarkan data historis

✅ **Prediksi Time Series**
- Forecasting NO₂ hari berikutnya
- Perencanaan berbasis prediksi

✅ **Coverage Area Luas**
- Data satelit mencakup area yang luas
- Tidak perlu banyak sensor fisik

### Standar WHO Cocok Untuk:

✅ **Penilaian Risiko Kesehatan**
- Evaluasi dampak langsung pada kesehatan
- Compliance dengan regulasi

✅ **Policy Making**
- Dasar kebijakan kualitas udara
- Target pengurangan emisi

✅ **Pengukuran Paparan Aktual**
- Exposure assessment untuk populasi
- Studi epidemiologi

✅ **Akurasi Tinggi di Lokasi Spesifik**
- Monitoring zona sensitif (sekolah, rumah sakit)
- Hot spot pollution

---

## 🔄 Integrasi Kedua Pendekatan

### Workflow Ideal:

```
1. Monitoring Satelit (Model Prediksi)
   ├─ Deteksi area dengan nilai tinggi
   ├─ Prediksi trend
   └─ Coverage luas
           ↓
2. Ground Station (WHO Standard)
   ├─ Verifikasi dengan pengukuran permukaan
   ├─ Konversi ke µg/m³
   └─ Penilaian risiko kesehatan
           ↓
3. Aksi & Policy
   ├─ Peringatan kesehatan publik
   ├─ Regulasi sumber emisi
   └─ Mitigasi polusi
```

### Contoh Implementasi:

**Scenario**: Model prediksi mendeteksi anomali

```
🛰️ SATELIT (Model):
   → NO₂ hari ini: 0.00005 mol/m²
   → Threshold: 0.00003555 mol/m²
   → Status: BERBAHAYA (40% di atas threshold lokal)
   → ACTION: Trigger alert!

🏭 GROUND STATION:
   → Ukur konsentrasi permukaan
   → Hasil: 35 µg/m³ (24-hour average)
   → WHO Threshold: 25 µg/m³
   → Status: MELEBIHI STANDAR WHO!

🏥 HEALTH AUTHORITY:
   → Issue public health warning
   → Advise vulnerable groups to stay indoors
   → Investigate pollution sources
   → Activate emergency response protocol
```

---

## 📝 Kesimpulan dan Rekomendasi

### Untuk Proyek Ini (Model Prediksi):

✅ **Valid untuk**:
- Analisis pola dan tren NO₂ di Bangkalan
- Sistem prediksi dan early warning
- Riset dan pengembangan model time series

⚠️ **Keterbatasan**:
- TIDAK menggantikan monitoring permukaan
- TIDAK untuk penilaian risiko kesehatan langsung
- Threshold bersifat RELATIF, bukan absolut

### Rekomendasi Pengembangan:

1. **Jangka Pendek**:
   - Dokumentasikan dengan jelas perbedaan threshold model vs WHO
   - Tambahkan disclaimer di semua output prediksi
   - Fokus pada analisis pola dan tren

2. **Jangka Menengah**:
   - Kerjasama dengan ground monitoring station
   - Validasi prediksi satelit dengan data permukaan
   - Kembangkan model konversi yang lebih akurat

3. **Jangka Panjang**:
   - Integrasikan data satelit + ground station
   - Implementasi sistem peringatan dini terintegrasi
   - Link ke sistem kesehatan publik

### Untuk Stakeholder:

**Jika Anda adalah**:
- 🎓 **Peneliti**: Gunakan model ini untuk analisis tren dan prediksi
- 🏛️ **Pembuat Kebijakan**: Kombinasikan dengan data ground station untuk keputusan
- 👨‍⚕️ **Ahli Kesehatan**: Gunakan standar WHO dengan data permukaan untuk assessment
- 👨‍💼 **Industri**: Gunakan prediksi untuk compliance planning

---

## 📚 Referensi

1. **WHO Air Quality Guidelines (2021)**
   - https://www.who.int/publications/i/item/9789240034228
   - Global Air Quality Guidelines: Particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide

2. **Sentinel-5P TROPOMI NO₂**
   - https://sentinel.esa.int/web/sentinel/missions/sentinel-5p
   - Technical documentation on vertical column density measurements

3. **Konversi Satelit ke Permukaan**
   - Lamsal et al. (2021): "Ground-level nitrogen dioxide concentrations inferred from the satellite-borne Ozone Monitoring Instrument"
   - Duncan et al. (2014): "A space-based, high-resolution view of notable changes in urban NOx pollution around the world"

---

**Tanggal**: 27 Oktober 2025  
**Project**: Prediksi NO₂ Bangkalan dengan KNN Regression  
**Status**: Dokumentasi Standar WHO dan Perbandingan Metodologi
