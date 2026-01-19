"""
Data loading utilities for sEMG signals.
Loads EMG Database 1 - 101 subjects, 3 exercises each.
"""

import numpy as np
import os
import config


def load_single_recording(subject_id, exercise_id):
    """
    Load a single sEMG recording.
    
    Args:
        subject_id: Subject identifier (0-100)
        exercise_id: Exercise identifier (0, 1, or 2)
    
    Returns:
        np.ndarray of shape (8, 30720) or None if file not found
    """
    filename = f"Subiect_{subject_id}_{exercise_id}_r.npy"
    filepath = os.path.join(config.DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        return None
    
    data = np.load(filepath, allow_pickle=True).astype(np.float32)
    
    # Truncate to expected length (one file is longer than others)
    if data.shape[1] > config.SAMPLES_PER_RECORDING:
        data = data[:, :config.SAMPLES_PER_RECORDING]
    
    return data


def load_all_data():
    """
    Load entire dataset as arrays.
    
    Returns:
        X: np.ndarray (num_recordings, num_channels, num_samples)
        y: np.ndarray of exercise labels (0, 1, or 2)
        subject_ids: np.ndarray of subject identifiers
    """
    X_list = []
    y_list = []
    subject_ids_list = []
    
    for subject_id in range(config.NUM_SUBJECTS):
        for exercise_id in range(config.NUM_EXERCISES):
            data = load_single_recording(subject_id, exercise_id)
            if data is not None:
                X_list.append(data)
                y_list.append(exercise_id)
                subject_ids_list.append(subject_id)
    
    X = np.array(X_list)
    y = np.array(y_list)
    subject_ids = np.array(subject_ids_list)
    
    print(f"Loaded {X.shape[0]} recordings from {len(np.unique(subject_ids))} subjects")
    print(f"  Shape per recording: {X.shape[1:]} (channels x samples)")
    
    return X, y, subject_ids


def get_flexor_extensor_pairs():
    """
    Get channel pairs for asymmetry analysis (flexor vs extensor).
    
    Returns:
        List of (extensor_channel, flexor_channel) tuples
    """
    # Compare extensor channels with corresponding flexor channels
    pairs = [
        (1, 5),  # Channel 2 (pure extensor) vs Channel 6 (pure flexor)
        (0, 3),  # Channel 1 vs Channel 4
        (2, 4),  # Channel 3 vs Channel 5
        (7, 6),  # Channel 8 vs Channel 7
    ]
    return pairs


def print_dataset_info():
    """Print information about the dataset."""
    files = [f for f in os.listdir(config.DATA_DIR) if f.endswith('.npy')]
    
    print("=" * 60)
    print("EMG Database 1 Information")
    print("=" * 60)
    print(f"Total files: {len(files)}")
    print(f"Subjects: 0 - {config.NUM_SUBJECTS - 1}")
    print(f"Exercises: 0, 1, 2 (three flexion types)")
    print(f"Channels: {config.NUM_CHANNELS} (8-sensor bracelet)")
    print(f"Samples per recording: {config.SAMPLES_PER_RECORDING}")
    print(f"Duration: ~{config.SAMPLES_PER_RECORDING / config.SAMPLING_RATE:.0f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    print_dataset_info()
