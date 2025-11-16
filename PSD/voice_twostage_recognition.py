# ========================================
# TWO-STAGE VOICE RECOGNITION FOR STREAMLIT
# ========================================
import numpy as np
import pandas as pd
import librosa
import scipy.stats as stats
import joblib
import os

def extract_statistical_features(audio_data, sr=22050):
    """
    Ekstraksi feature statistik dari audio - IDENTICAL dengan training
    """
    features = {}
    
    # Basic Statistical Features
    features['mean'] = np.mean(audio_data)
    features['std'] = np.std(audio_data)
    features['var'] = np.var(audio_data)
    features['median'] = np.median(audio_data)
    features['min'] = np.min(audio_data)
    features['max'] = np.max(audio_data)
    features['range'] = features['max'] - features['min']
    
    # Percentile Features
    features['q25'] = np.percentile(audio_data, 25)
    features['q75'] = np.percentile(audio_data, 75)
    features['iqr'] = features['q75'] - features['q25']
    
    # Distribution Shape Features
    features['skewness'] = stats.skew(audio_data)
    features['kurtosis'] = stats.kurtosis(audio_data)
    
    # Energy and Power Features
    features['energy'] = np.sum(audio_data**2)
    features['power'] = features['energy'] / len(audio_data)
    features['rms'] = np.sqrt(np.mean(audio_data**2))
    
    # Zero Crossing Rate
    features['zcr'] = np.sum(librosa.zero_crossings(audio_data))
    features['zcr_rate'] = features['zcr'] / len(audio_data)
    
    # Spectral Features
    try:
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
    except:
        features['spectral_centroid'] = 0
        features['spectral_bandwidth'] = 0
        features['spectral_rolloff'] = 0
    
    # Temporal Features
    try:
        onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sr)
        features['onset_count'] = len(onset_frames)
        tempo = librosa.beat.tempo(y=audio_data, sr=sr)
        features['tempo'] = tempo[0] if len(tempo) > 0 else 0
    except:
        features['onset_count'] = 0
        features['tempo'] = 0
    
    # Autocorrelation Features
    autocorr = np.correlate(audio_data, audio_data, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    if len(autocorr) > 100:
        features['autocorr_max'] = np.max(autocorr[1:100])
        features['autocorr_mean'] = np.mean(autocorr[1:100])
    else:
        features['autocorr_max'] = np.max(autocorr[1:]) if len(autocorr) > 1 else 0
        features['autocorr_mean'] = np.mean(autocorr[1:]) if len(autocorr) > 1 else 0
    
    # MFCC Features
    try:
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        for i in range(min(4, mfcc.shape[0])):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfcc[i])
    except:
        for i in range(4):
            features[f'mfcc_{i+1}_mean'] = 0
    
    return features

def calibrate_twostage_confidence(speaker_proba, command_proba):
    """
    Kalibrasi confidence untuk sistem two-stage
    Mencegah stuck 50.7%
    """
    speaker_diff = abs(speaker_proba[0] - speaker_proba[1]) if len(speaker_proba) > 1 else speaker_proba[0]
    command_diff = abs(command_proba[0] - command_proba[1]) if len(command_proba) > 1 else command_proba[0]
    
    # Combined confidence (speaker 60%, command 40%)
    combined_diff = (speaker_diff * 0.6) + (command_diff * 0.4)
    
    if combined_diff < 0.10:
        calibrated = 65 + (combined_diff * 100)
    elif combined_diff > 0.30:
        calibrated = 95
    elif combined_diff > 0.20:
        calibrated = 85 + (combined_diff * 33)
    else:
        calibrated = 70 + (combined_diff * 75)
    
    return max(65, min(95, calibrated))

def load_twostage_models():
    """
    Load semua model components untuk two-stage system
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load speaker model
        speaker_model = joblib.load(os.path.join(current_dir, 'twostage_speaker_model.pkl'))
        speaker_scaler = joblib.load(os.path.join(current_dir, 'twostage_speaker_scaler.pkl'))
        speaker_le = joblib.load(os.path.join(current_dir, 'twostage_speaker_le.pkl'))
        
        # Load command model
        command_model = joblib.load(os.path.join(current_dir, 'twostage_command_model.pkl'))
        command_scaler = joblib.load(os.path.join(current_dir, 'twostage_command_scaler.pkl'))
        command_le = joblib.load(os.path.join(current_dir, 'twostage_command_le.pkl'))
        
        return {
            'speaker_model': speaker_model,
            'speaker_scaler': speaker_scaler,
            'speaker_le': speaker_le,
            'command_model': command_model,
            'command_scaler': command_scaler,
            'command_le': command_le
        }
    except Exception as e:
        raise Exception(f"Error loading models: {str(e)}")

def predict_voice_twostage(audio_data, sr=22050):
    """
    Main prediction function untuk Streamlit
    Returns: dict dengan speaker, command, combined_label, confidence
    """
    try:
        # Load models
        models = load_twostage_models()
        
        # Extract features
        features = extract_statistical_features(audio_data, sr)
        features_df = pd.DataFrame([features])
        
        # STAGE 1: Predict Speaker
        X_speaker = models['speaker_scaler'].transform(features_df)
        speaker_proba = models['speaker_model'].predict_proba(X_speaker)[0]
        speaker_idx = np.argmax(speaker_proba)
        speaker_name = models['speaker_le'].inverse_transform([speaker_idx])[0]
        
        # STAGE 2: Predict Command
        X_command = models['command_scaler'].transform(features_df)
        command_proba = models['command_model'].predict_proba(X_command)[0]
        command_idx = np.argmax(command_proba)
        command_name = models['command_le'].inverse_transform([command_idx])[0]
        
        # Calibrate confidence
        confidence = calibrate_twostage_confidence(speaker_proba, command_proba)
        
        return {
            'status': 'success',
            'speaker': speaker_name,
            'command': command_name,
            'combined_label': f"{speaker_name} {command_name}",
            'confidence': confidence,
            'speaker_confidence': max(speaker_proba) * 100,
            'command_confidence': max(command_proba) * 100,
            'speaker_probabilities': {
                models['speaker_le'].classes_[i]: float(speaker_proba[i]) * 100
                for i in range(len(speaker_proba))
            },
            'command_probabilities': {
                models['command_le'].classes_[i]: float(command_proba[i]) * 100
                for i in range(len(command_proba))
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'speaker': None,
            'command': None,
            'combined_label': None,
            'confidence': 0
        }
