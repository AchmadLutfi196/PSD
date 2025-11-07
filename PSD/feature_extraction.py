
import numpy as np
import librosa
import scipy.stats as stats

def extract_statistical_features(audio, sr=22050):
    """
    Ekstrak feature statistik time series untuk audio
    """
    features = {}
    
    # Basic time domain statistics
    features['mean'] = np.mean(audio)
    features['std'] = np.std(audio)
    features['var'] = np.var(audio)
    features['min'] = np.min(audio)
    features['max'] = np.max(audio)
    features['range'] = features['max'] - features['min']
    features['median'] = np.median(audio)
    features['skewness'] = stats.skew(audio)
    features['kurtosis'] = stats.kurtosis(audio)
    
    # Percentiles
    percentiles = [10, 25, 75, 90]
    for p in percentiles:
        features[f'percentile_{p}'] = np.percentile(audio, p)
    
    # Energy features
    features['energy'] = np.sum(audio**2)
    features['rms'] = np.sqrt(np.mean(audio**2))
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # Spectral features
    stft = np.abs(librosa.stft(audio))
    freqs = librosa.fft_frequencies(sr=sr)
    
    # Spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spec_centroid)
    features['spectral_centroid_std'] = np.std(spec_centroid)
    
    # Spectral rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spec_rolloff)
    features['spectral_rolloff_std'] = np.std(spec_rolloff)
    
    # Spectral bandwidth
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(spec_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spec_bandwidth)
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_std'] = np.std(chroma)
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    features['tempo'] = tempo
    
    return features

def preprocess_audio(audio, target_sr=22050, duration=3.0):
    """
    Preprocessing audio untuk konsistensi
    """
    # Resample jika perlu
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Normalisasi
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # Fixed duration
    target_length = int(target_sr * duration)
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    
    return audio
