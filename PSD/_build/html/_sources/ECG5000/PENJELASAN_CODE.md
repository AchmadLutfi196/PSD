# Penjelasan Tahapan Code ECG5000 Classification Project

## Daftar Isi
1. [Business Understanding](#1-business-understanding)
2. [Data Understanding (EDA)](#2-data-understanding-eda)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Pemodelan Data](#4-pemodelan-data)
5. [Evaluasi](#5-evaluasi)
6. [Deployment](#6-deployment)

---

## 1. Business Understanding

### Tujuan
Mengembangkan sistem klasifikasi otomatis untuk mendeteksi anomali dalam sinyal ECG (Elektrokardiogram) guna membantu diagnosis dini penyakit jantung.

### Output yang Diharapkan
- Model machine learning yang dapat mengklasifikasikan sinyal ECG sebagai Normal atau Abnormal
- Akurasi prediksi yang tinggi (>90%)
- Sistem yang dapat di-deploy untuk penggunaan praktis

---

## 2. Data Understanding (EDA)

### 2.1 Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

**Penjelasan:**
- `numpy` & `pandas`: Manipulasi data numerik dan tabel
- `matplotlib` & `seaborn`: Visualisasi data
- `sklearn`: Tools untuk machine learning (preprocessing, modeling, evaluasi)

### 2.2 Load Data
```python
train_data = np.loadtxt('ECG5000_TRAIN.txt')
test_data = np.loadtxt('ECG5000_TEST.txt')
all_data = np.vstack([train_data, test_data])
```

**Penjelasan:**
- Memuat dataset ECG5000 yang terbagi dalam file training dan testing
- `np.vstack()` menggabungkan kedua dataset secara vertikal
- Dataset berisi sinyal ECG dengan 140 time points per sampel

### 2.3 Pemisahan Features dan Labels
```python
y = all_data[:, 0].astype(int)  # Kolom pertama = label
X = all_data[:, 1:]              # Kolom sisanya = features
```

**Penjelasan:**
- Label (y): Kelas ECG (1=Normal, 2-5=Berbagai jenis abnormal)
- Features (X): 140 nilai amplitudo sinyal ECG dalam time series

### 2.4 Analisis Distribusi Label
```python
label_counts = pd.Series(y).value_counts().sort_index()
sns.countplot(x=y, palette='viridis')
```

**Penjelasan:**
- Menghitung jumlah sampel per kelas
- Visualisasi dengan bar chart dan pie chart
- Mengecek apakah data balanced atau imbalanced

### 2.5 Visualisasi Sinyal ECG
```python
# Sample sinyal per kelas
for label in unique_labels:
    idx = np.where(y == label)[0][0]
    plt.plot(X[idx])
```

**Penjelasan:**
- Menampilkan contoh sinyal ECG dari setiap kelas
- Membantu memahami perbedaan visual antar kelas
- Time points (x-axis) vs Amplitude (y-axis)

### 2.6 Rata-rata Sinyal per Kelas
```python
for label in unique_labels:
    class_data = X[y == label]
    mean_signal = class_data.mean(axis=0)
    plt.plot(mean_signal, label=f'Kelas {label}')
```

**Penjelasan:**
- Menghitung rata-rata sinyal untuk setiap kelas
- Mengidentifikasi pola karakteristik tiap kelas
- Membantu memahami perbedaan antara Normal vs Abnormal

### 2.7 Statistik Deskriptif
```python
print(f'Mean amplitude: {class_data.mean():.4f}')
print(f'Std amplitude: {class_data.std():.4f}')
```

**Penjelasan:**
- Menghitung statistik dasar (mean, std, min, max) per kelas
- Mengidentifikasi range nilai dan variabilitas data
- Memastikan tidak ada missing values

---

## 3. Data Preprocessing

### 3.1 Binary Classification
```python
y_binary = (y == 1).astype(int)  # 1 = Normal, 0 = Abnormal
```

**Penjelasan:**
- Mengkonversi multi-class (1-5) menjadi binary (0-1)
- Kelas 1 tetap sebagai Normal (1)
- Kelas 2-5 digabung menjadi Abnormal (0)
- Simplifikasi problem untuk deteksi anomali

### 3.2 Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
```

**Penjelasan:**
- Membagi data: 80% training, 20% testing
- `stratify=y_binary`: Mempertahankan proporsi kelas di train & test
- `random_state=42`: Untuk reproduktibilitas hasil

### 3.3 Standarisasi Data
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Penjelasan:**
- StandardScaler: Mengubah data ke mean=0, std=1
- `fit_transform()` pada training: Belajar parameter & transform
- `transform()` pada testing: Hanya transform (pakai parameter dari training)
- Penting untuk algoritma yang sensitif terhadap skala (SVM, KNN)

---

## 4. Pemodelan Data

### 4.1 Definisi Model
```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}
```

**Penjelasan Model:**

1. **Logistic Regression**: 
   - Model linear untuk klasifikasi binary
   - Cepat, interpretable
   - Baseline model

2. **Decision Tree**: 
   - Model berbasis pohon keputusan
   - Mudah dipahami, dapat handle non-linearity
   - Rentan overfit

3. **Random Forest**: 
   - Ensemble dari banyak decision trees
   - Lebih robust, mengurangi overfit
   - Feature importance tersedia

4. **KNN (K-Nearest Neighbors)**: 
   - Klasifikasi based on similarity dengan neighbors
   - Non-parametric, simple
   - Sensitif terhadap skala data

5. **SVM (Support Vector Machine)**: 
   - Mencari hyperplane optimal untuk pemisah kelas
   - Kernel RBF untuk non-linear boundary
   - Powerful untuk high-dimensional data

6. **Gradient Boosting**: 
   - Ensemble sequentially, fokus pada error sebelumnya
   - Sangat powerful, sering menang di kompetisi
   - Lebih lambat untuk training

### 4.2 Training Model
```python
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
```

**Penjelasan:**
- Loop melalui semua model
- `fit()`: Training model pada data training
- `predict()`: Prediksi pada data testing
- Simpan hasil untuk perbandingan

### 4.3 Perbandingan Model
```python
accuracies = {name: results[name]['accuracy'] for name in models.keys()}
plt.bar(accuracies.keys(), accuracies.values())
```

**Penjelasan:**
- Visualisasi akurasi semua model dalam bar chart
- Memudahkan identifikasi model terbaik
- Menampilkan nilai akurasi di atas setiap bar

---

## 5. Evaluasi

### 5.1 Classification Report
```python
print(classification_report(y_test, y_pred_best, 
                          target_names=['Abnormal', 'Normal']))
```

**Penjelasan Metrics:**
- **Precision**: Dari semua prediksi positif, berapa yang benar?
  - Formula: TP / (TP + FP)
  - Penting untuk minimalisir false alarm

- **Recall (Sensitivity)**: Dari semua actual positif, berapa yang terdeteksi?
  - Formula: TP / (TP + FN)
  - Penting untuk deteksi penyakit (jangan sampai miss)

- **F1-Score**: Harmonic mean dari Precision & Recall
  - Formula: 2 × (Precision × Recall) / (Precision + Recall)
  - Balance antara precision & recall

- **Support**: Jumlah sampel actual per kelas

### 5.2 Confusion Matrix
```python
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

**Penjelasan:**
```
                Predicted
              Abn    Norm
Actual  Abn   TN     FP
        Norm  FN     TP
```

- **True Negative (TN)**: Abnormal diprediksi Abnormal ✓
- **False Positive (FP)**: Abnormal diprediksi Normal ✗
- **False Negative (FN)**: Normal diprediksi Abnormal ✗
- **True Positive (TP)**: Normal diprediksi Normal ✓

### 5.3 Feature Importance (Random Forest)
```python
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
```

**Penjelasan:**
- Menunjukkan time points mana yang paling berpengaruh dalam prediksi
- Nilai tinggi = fitur penting untuk klasifikasi
- Berguna untuk interpretasi dan feature selection

### 5.4 Cross-Validation
```python
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f'{cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})')
```

**Penjelasan:**
- Membagi data training menjadi 5 folds
- Training & testing pada kombinasi berbeda
- Mengukur stabilitas dan generalisasi model
- Mean ± 2×Std memberikan confidence interval 95%

---

## 6. Deployment

### 6.1 Save Model
```python
joblib.dump(best_model, 'ecg_model.pkl')
joblib.dump(scaler, 'ecg_scaler.pkl')
```

**Penjelasan:**
- Menyimpan model dan scaler yang sudah di-training
- Format `.pkl` (pickle) untuk serialisasi object Python
- Model dapat di-load kembali tanpa perlu re-training
- Scaler disimpan untuk preprocessing data baru dengan parameter sama

### 6.2 Streamlit Deployment (Rencana)
Model akan di-deploy menggunakan Streamlit dengan fitur:
- Upload file ECG atau input manual
- Preprocessing otomatis menggunakan scaler tersimpan
- Prediksi real-time
- Visualisasi sinyal dan hasil prediksi
- Interface user-friendly untuk non-technical users

---

## Ringkasan Flow Keseluruhan

```
1. Business Understanding
   ↓
2. Load & Explore Data (EDA)
   ↓
3. Preprocessing
   - Binary classification
   - Train-test split
   - Standardization
   ↓
4. Model Training
   - Train multiple models
   - Compare performance
   ↓
5. Evaluation
   - Select best model
   - Detailed metrics
   - Cross-validation
   ↓
6. Save & Deploy
   - Save model & scaler
   - Streamlit app
```

---

## Tips & Best Practices

### ✓ Yang Sudah Baik:
1. **EDA Komprehensif**: Visualisasi dan analisis menyeluruh
2. **Multiple Models**: Testing berbagai algoritma
3. **Proper Splitting**: Stratified train-test split
4. **Standardization**: Preprocessing yang tepat
5. **Cross-Validation**: Validasi robustness model

### ⚠ Yang Bisa Ditingkatkan:
1. **Hyperparameter Tuning**: Gunakan GridSearchCV/RandomizedSearchCV
2. **Feature Engineering**: Extract statistik (mean, std, peaks, dll)
3. **Imbalanced Data**: Jika ada, gunakan SMOTE atau class_weight
4. **ROC-AUC Curve**: Tambahkan untuk evaluasi lebih detail
5. **Model Ensemble**: Combine beberapa model terbaik
6. **Deep Learning**: Coba LSTM/CNN untuk time series

---

## Interpretasi Hasil

### Ketika Akurasi Tinggi (>90%):
✓ Model berhasil menangkap pola ECG normal vs abnormal
✓ Features (time points) sudah representatif
✓ Dapat dipertimbangkan untuk deployment

### Ketika Recall Rendah untuk Abnormal:
⚠ Banyak kasus abnormal yang terlewat (False Negative)
⚠ Berbahaya dalam konteks medis
⚠ Perlu tuning threshold atau class weights

### Ketika Precision Rendah:
⚠ Banyak false alarm (False Positive)
⚠ Dapat menyebabkan kepanikan tidak perlu
⚠ Perlu penyesuaian decision boundary

---

## Referensi Tambahan

- **Dataset**: ECG5000 from UCR Time Series Archive
- **CRISP-DM**: Cross-Industry Standard Process for Data Mining
- **Scikit-learn Documentation**: https://scikit-learn.org/
- **ECG Interpretation**: Medical literature on ECG analysis

---

**Catatan**: Code ini merupakan implementasi standard pipeline machine learning untuk klasifikasi time series. Dalam aplikasi medis nyata, validasi lebih ketat dan kolaborasi dengan tenaga medis sangat diperlukan.
