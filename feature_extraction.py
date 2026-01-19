"""
Feature extraction for sEMG signals.
Implements time-domain and frequency-domain features from the project specification.
"""

import numpy as np
from scipy import signal, stats
import config


# =============================================================================
# TIME-DOMAIN FEATURES (from project specification)
# =============================================================================

def mean_absolute_value(x):
    """MAV - Average muscle activation level. Eq. (1)"""
    return np.mean(np.abs(x))


def zero_crossing_rate(x, threshold=None):
    """ZCR - Frequency of sign changes. Eq. (2)"""
    if threshold is None:
        threshold = config.ZCR_THRESHOLD
    diff = np.abs(np.diff(x))
    sign_change = x[:-1] * x[1:] < 0
    return np.sum((diff >= threshold) & sign_change)


def waveform_length(x):
    """WL - Cumulative length of waveform. Eq. (3)"""
    return np.sum(np.abs(np.diff(x)))


def slope_sign_changes(x, threshold=None):
    """SSC - Number of slope direction changes. Eq. (4)"""
    if threshold is None:
        threshold = config.SSC_THRESHOLD
    diff1 = x[1:-1] - x[:-2]
    diff2 = x[1:-1] - x[2:]
    return np.sum(diff1 * diff2 >= threshold)


def root_mean_square(x):
    """RMS - Signal energy/intensity. Eq. (5)"""
    return np.sqrt(np.mean(x ** 2))


def variance(x):
    """VAR - Hjorth Activity parameter. Eq. (6)"""
    return np.var(x)


def skewness(x):
    """SKEW - Asymmetry of amplitude distribution. Eq. (7)"""
    return stats.skew(x)


def integrated_emg(x):
    """IEMG - Sum of absolute values."""
    return np.sum(np.abs(x))


def kurtosis(x):
    """KURT - Tailedness of distribution."""
    return stats.kurtosis(x)


def log_detector(x):
    """LOG - Log detector feature."""
    return np.exp(np.mean(np.log(np.abs(x) + 1e-10)))


# =============================================================================
# FREQUENCY-DOMAIN FEATURES
# =============================================================================

def mean_frequency(x, fs=None):
    """MNF - Weighted average frequency. Eq. (11)"""
    if fs is None:
        fs = config.SAMPLING_RATE
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    total = np.sum(psd)
    return np.sum(freqs * psd) / total if total > 0 else 0


def median_frequency(x, fs=None):
    """MDF - Frequency dividing spectrum in half. Eq. (12)"""
    if fs is None:
        fs = config.SAMPLING_RATE
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    cumsum = np.cumsum(psd)
    total = cumsum[-1]
    idx = np.where(cumsum >= total / 2)[0]
    return freqs[idx[0]] if len(idx) > 0 else freqs[-1]


# =============================================================================
# FEATURE EXTRACTION PIPELINE
# =============================================================================

def extract_features_from_channel(x):
    """
    Extract all features from a single channel.
    
    Returns:
        List of 12 feature values
    """
    return [
        mean_absolute_value(x),     # MAV
        zero_crossing_rate(x),      # ZCR
        waveform_length(x),         # WL
        slope_sign_changes(x),      # SSC
        root_mean_square(x),        # RMS
        variance(x),                # VAR
        skewness(x),                # SKEW
        kurtosis(x),                # KURT
        integrated_emg(x),          # IEMG
        log_detector(x),            # LOG
        mean_frequency(x),          # MNF
        median_frequency(x),        # MDF
    ]


FEATURE_NAMES_PER_CHANNEL = ["MAV", "ZCR", "WL", "SSC", "RMS", "VAR", "SKEW", "KURT", "IEMG", "LOG", "MNF", "MDF"]


def extract_features_from_window(window):
    """
    Extract features from a multi-channel window.
    
    Args:
        window: Array (num_channels, window_size)
    
    Returns:
        Feature vector (1D array)
    """
    features = []
    for ch in range(window.shape[0]):
        features.extend(extract_features_from_channel(window[ch]))
    return np.array(features)


def extract_features_batch(windows):
    """
    Extract features from all windows (vectorized for speed).
    
    Args:
        windows: Array (num_windows, num_channels, window_size)
    
    Returns:
        Feature matrix (num_windows, num_features)
    """
    num_windows, num_channels, window_size = windows.shape
    num_features_per_channel = len(FEATURE_NAMES_PER_CHANNEL)
    total_features = num_channels * num_features_per_channel
    
    print(f"Extracting {total_features} features from {num_windows} windows...")
    
    features = np.zeros((num_windows, total_features))
    
    for ch in range(num_channels):
        ch_data = windows[:, ch, :]
        base = ch * num_features_per_channel
        
        # Vectorized time-domain features
        features[:, base + 0] = np.mean(np.abs(ch_data), axis=1)  # MAV
        
        # ZCR
        diff = np.abs(np.diff(ch_data, axis=1))
        sign_change = ch_data[:, :-1] * ch_data[:, 1:] < 0
        features[:, base + 1] = np.sum((diff >= config.ZCR_THRESHOLD) & sign_change, axis=1)
        
        features[:, base + 2] = np.sum(np.abs(np.diff(ch_data, axis=1)), axis=1)  # WL
        
        # SSC
        diff1 = ch_data[:, 1:-1] - ch_data[:, :-2]
        diff2 = ch_data[:, 1:-1] - ch_data[:, 2:]
        features[:, base + 3] = np.sum(diff1 * diff2 >= config.SSC_THRESHOLD, axis=1)
        
        features[:, base + 4] = np.sqrt(np.mean(ch_data ** 2, axis=1))  # RMS
        features[:, base + 5] = np.var(ch_data, axis=1)  # VAR
        features[:, base + 6] = stats.skew(ch_data, axis=1)  # SKEW
        features[:, base + 7] = stats.kurtosis(ch_data, axis=1)  # KURT
        features[:, base + 8] = np.sum(np.abs(ch_data), axis=1)  # IEMG
        features[:, base + 9] = np.exp(np.mean(np.log(np.abs(ch_data) + 1e-10), axis=1))  # LOG
        
        # Frequency features (loop needed for welch)
        for i in range(num_windows):
            freqs, psd = signal.welch(ch_data[i], fs=config.SAMPLING_RATE, nperseg=min(256, window_size))
            total = np.sum(psd)
            features[i, base + 10] = np.sum(freqs * psd) / total if total > 0 else 0  # MNF
            cumsum = np.cumsum(psd)
            idx = np.where(cumsum >= total / 2)[0]
            features[i, base + 11] = freqs[idx[0]] if len(idx) > 0 else freqs[-1]  # MDF
    
    print(f"Feature matrix shape: {features.shape}")
    return features


def get_feature_names(num_channels=8):
    """Get list of all feature names."""
    names = []
    for ch in range(num_channels):
        for feat in FEATURE_NAMES_PER_CHANNEL:
            names.append(f"CH{ch}_{feat}")
    return names


if __name__ == "__main__":
    from data_loader import load_single_recording
    from preprocessing import preprocess_recording
    
    print("Testing feature extraction...")
    data = load_single_recording(0, 0)
    
    if data is not None:
        processed = preprocess_recording(data, segment=False)
        
        # Single channel features
        features = extract_features_from_channel(processed[0])
        print(f"\nFeatures from Channel 0:")
        for name, value in zip(FEATURE_NAMES_PER_CHANNEL, features):
            print(f"  {name}: {value:.4f}")
        
        # Full window features
        windowed = preprocess_recording(data, segment=True)
        feature_matrix = extract_features_batch(windowed[:5])
        print(f"\nFeature matrix (5 windows): {feature_matrix.shape}")
