"""
Data preprocessing for sEMG signals.
Steps: DC removal -> Bandpass filtering -> Normalization -> Windowing
"""

import numpy as np
from scipy import signal
import config


def remove_dc_offset(data):
    """Remove DC offset (mean) from the signal."""
    return data - np.mean(data, axis=-1, keepdims=True)


def bandpass_filter(data, lowcut=20.0, highcut=450.0, order=4):
    """
    Apply Butterworth bandpass filter.
    sEMG useful frequency range: 20-450 Hz.
    """
    nyquist = config.SAMPLING_RATE / 2
    low = max(0.01, lowcut / nyquist)
    high = min(0.99, highcut / nyquist)
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.array([signal.filtfilt(b, a, ch) for ch in data])


def normalize(data, method="z-score"):
    """
    Normalize the signal.
    
    Args:
        data: Signal array
        method: "z-score" or "min-max"
    """
    if method == "z-score":
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        std[std == 0] = 1  # Avoid division by zero
        return (data - mean) / std
    
    elif method == "min-max":
        min_val = np.min(data, axis=-1, keepdims=True)
        max_val = np.max(data, axis=-1, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        return (data - min_val) / range_val
    
    else:
        raise ValueError(f"Unknown method: {method}")


def segment_into_windows(data, window_size=None, overlap=None):
    """
    Divide signal into overlapping windows.
    
    Args:
        data: Signal (channels, samples)
        window_size: Samples per window
        overlap: Overlap fraction (0-1)
    
    Returns:
        Array of shape (num_windows, channels, window_size)
    """
    if window_size is None:
        window_size = config.WINDOW_SIZE
    if overlap is None:
        overlap = config.WINDOW_OVERLAP
    
    step = int(window_size * (1 - overlap))
    num_channels, num_samples = data.shape
    num_windows = (num_samples - window_size) // step + 1
    
    windows = np.zeros((num_windows, num_channels, window_size))
    for i in range(num_windows):
        start = i * step
        windows[i] = data[:, start:start + window_size]
    
    return windows


def preprocess_recording(data, apply_filter=True, apply_normalize=True, segment=False):
    """
    Full preprocessing pipeline for a single recording.
    
    Args:
        data: Raw sEMG (channels, samples)
        apply_filter: Apply bandpass filter
        apply_normalize: Apply normalization
        segment: Divide into windows
    
    Returns:
        Preprocessed data
    """
    processed = data.astype(np.float32)
    
    # Step 1: Remove DC offset
    processed = remove_dc_offset(processed)
    
    # Step 2: Bandpass filter (20-450 Hz)
    if apply_filter:
        processed = bandpass_filter(processed)
    
    # Step 3: Normalize
    if apply_normalize:
        processed = normalize(processed, config.NORMALIZATION_METHOD)
    
    # Step 4: Segment into windows
    if segment:
        processed = segment_into_windows(processed)
    
    return processed


def preprocess_dataset(X, segment=True, apply_normalize=None):
    """
    Preprocess entire dataset.
    
    Args:
        X: Array of recordings (num_recordings, channels, samples)
        segment: Whether to segment into windows
        apply_normalize: Override signal normalization (True/False)
    
    Returns:
        Preprocessed data, windows_per_recording
    """
    if apply_normalize is None:
        apply_normalize = config.SIGNAL_NORMALIZE

    processed_list = []
    
    for i, recording in enumerate(X):
        processed = preprocess_recording(recording, apply_normalize=apply_normalize, segment=segment)
        processed_list.append(processed)
        
        if (i + 1) % 50 == 0:
            print(f"  Preprocessed {i + 1}/{len(X)} recordings")
    
    if segment:
        all_windows = np.concatenate(processed_list, axis=0)
        windows_per_recording = processed_list[0].shape[0]
        return all_windows, windows_per_recording
    else:
        return np.array(processed_list), 1


def expand_labels(y, windows_per_recording):
    """Expand labels to match windowed data."""
    return np.repeat(y, windows_per_recording)


def expand_subject_ids(subject_ids, windows_per_recording):
    """Expand subject IDs to match windowed data."""
    return np.repeat(subject_ids, windows_per_recording)


def expand_recording_ids(recording_ids, windows_per_recording):
    """Expand recording IDs to match windowed data."""
    return np.repeat(recording_ids, windows_per_recording)


if __name__ == "__main__":
    from data_loader import load_single_recording
    
    print("Testing preprocessing...")
    data = load_single_recording(0, 0)
    
    if data is not None:
        print(f"Raw: shape={data.shape}, range=[{data.min():.1f}, {data.max():.1f}]")
        
        processed = preprocess_recording(data, segment=False)
        print(f"Processed: shape={processed.shape}, mean={processed.mean():.4f}, std={processed.std():.4f}")
        
        windowed = preprocess_recording(data, segment=True)
        print(f"Windowed: shape={windowed.shape} (windows, channels, samples)")
