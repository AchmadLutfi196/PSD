#!/usr/bin/env python3
"""
Script Testing Model NO2 Prediction
Untuk memastikan model dan scaler dapat dimuat dan berjalan dengan baik
"""

import numpy as np
import joblib
import os

# Konstanta
CONVERSION_FACTOR = 46010  # mol/m² to µg/m³
WHO_ANNUAL = 10  # µg/m³
WHO_24HOUR = 25  # µg/m³

def test_model_loading():
    """Test loading model dan scaler"""
    print("=" * 50)
    print("Testing Model Loading...")
    print("=" * 50)
    
    try:
        # Cek keberadaan file
        if os.path.exists('knn_model.pkl'):
            print("✅ knn_model.pkl ditemukan")
        else:
            print("❌ knn_model.pkl tidak ditemukan")
            return False
            
        if os.path.exists('minmax_scaler.pkl'):
            print("✅ minmax_scaler.pkl ditemukan")
        else:
            print("❌ minmax_scaler.pkl tidak ditemukan")
            return False
        
        # Load model
        model = joblib.load('knn_model.pkl')
        scaler = joblib.load('minmax_scaler.pkl')
        
        print("✅ Model berhasil dimuat")
        print(f"Model type: {type(model)}")
        print(f"Scaler type: {type(scaler)}")
        
        return model, scaler
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None

def test_prediction(model, scaler):
    """Test prediksi dengan sample data"""
    print("\n" + "=" * 50)
    print("Testing Prediction...")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {"name": "Case 1 - Rendah", "t2": 0.0001, "t1": 0.00012},
        {"name": "Case 2 - Sedang", "t2": 0.00035, "t1": 0.00040},
        {"name": "Case 3 - Tinggi", "t2": 0.00080, "t1": 0.00090}
    ]
    
    for case in test_cases:
        try:
            print(f"\n{case['name']}:")
            print(f"  Input NO2(t-2): {case['t2']:.6f} mol/m²")
            print(f"  Input NO2(t-1): {case['t1']:.6f} mol/m²")
            
            # Prediksi
            input_data = np.array([[case['t2'], case['t1']]])
            input_scaled = scaler.transform(input_data)
            prediction_mol = model.predict(input_scaled)[0]
            prediction_ugm3 = prediction_mol * CONVERSION_FACTOR
            
            # Evaluasi WHO
            if prediction_ugm3 <= WHO_ANNUAL:
                status = "SANGAT BAIK ✅"
            elif prediction_ugm3 <= WHO_24HOUR:
                status = "PERHATIAN ⚠️"
            else:
                status = "BERBAHAYA ❌"
            
            print(f"  Prediksi NO2: {prediction_mol:.6f} mol/m²")
            print(f"  Konversi: {prediction_ugm3:.2f} µg/m³")
            print(f"  Status WHO: {status}")
            print("  ✅ Prediksi berhasil")
            
        except Exception as e:
            print(f"  ❌ Error prediksi: {str(e)}")

def main():
    """Main testing function"""
    print("🧪 TESTING MODEL PREDIKSI NO2")
    print(f"Direktori kerja: {os.getcwd()}")
    
    # Test loading
    model, scaler = test_model_loading()
    
    if model is not None and scaler is not None:
        # Test prediction
        test_prediction(model, scaler)
        
        print("\n" + "=" * 50)
        print("✅ SEMUA TEST BERHASIL!")
        print("Model siap untuk deployment Streamlit")
        print("=" * 50)
        
        print("\nUntuk menjalankan Streamlit:")
        print("1. Install streamlit: pip install streamlit")
        print("2. Jalankan: streamlit run streamlit_app_no2.py")
        
    else:
        print("\n" + "=" * 50)
        print("❌ TEST GAGAL!")
        print("Periksa keberadaan file model dan scaler")
        print("=" * 50)

if __name__ == "__main__":
    main()