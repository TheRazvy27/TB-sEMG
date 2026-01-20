"""
Configuration settings for the sEMG Analysis Project.
"""

import os

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "sEmg_databases")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# DATASET PARAMETERS
# =============================================================================
NUM_SUBJECTS = 101          # Subjects 0-100
NUM_EXERCISES = 3           # Exercises 0, 1, 2 (three flexion exercises)
NUM_CHANNELS = 8            # 8 sEMG channels on the bracelet
SAMPLING_RATE = 512         # Hz
SAMPLES_PER_RECORDING = 30720  # ~60 seconds at 512 Hz

# Sensor placement on forearm bracelet (clockwise order)
# Channel 2 (index 1) = pure extensor, Channel 6 (index 5) = pure flexor
EXTENSOR_CHANNELS = [0, 1, 2, 7]  # Channels 1, 2, 3, 8 (outer arm)
FLEXOR_CHANNELS = [3, 4, 5, 6]    # Channels 4, 5, 6, 7 (inner arm)

# =============================================================================
# PREPROCESSING PARAMETERS
# =============================================================================
WINDOW_SIZE = 512           # Samples per window (~1 second)
WINDOW_OVERLAP = 0.25       # 25% overlap (less windows = faster, less overfitting)
NORMALIZATION_METHOD = "z-score"  # Options: "z-score", "min-max"

# Feature extraction thresholds (for noise reduction)
ZCR_THRESHOLD = 0.01
SSC_THRESHOLD = 0.01

# =============================================================================
# DATA SPLITTING (Train / Validation / Test)
# =============================================================================
# Split by SUBJECT to avoid data leakage
# All windows from one subject go to the same set
TRAIN_RATIO = 0.70   # 70 subjects for training
VAL_RATIO = 0.15     # 15 subjects for validation
TEST_RATIO = 0.15    # 16 subjects for testing

# Split search iterations for group-stratified split
SPLIT_MAX_ITER = 10000

# Use signal normalization before windowing (set False for variant B)
SIGNAL_NORMALIZE = True

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42

# =============================================================================
# MODEL PARAMETERS (Random Forest)
# =============================================================================
# These settings are more regularized to reduce overfitting.
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 8
RF_MIN_SAMPLES_SPLIT = 50
RF_MIN_SAMPLES_LEAF = 20
RF_MAX_FEATURES = 0.25
RF_MAX_SAMPLES = 0.6
RF_BOOTSTRAP = True

# =============================================================================
# FINAL TRAINING
# =============================================================================
FINAL_RETRAIN_ON_TRAIN_VAL = True

# =============================================================================
# TUNING
# =============================================================================
TUNING_CV_SPLITS = 5
TUNING_METRIC = "f1"  # Options: "f1", "balanced_accuracy"

# =============================================================================
# ASYMMETRY CLASSIFICATION THRESHOLDS (K_As percentage)
# =============================================================================
# Separate thresholds for RMS and Skewness because:
# - RMS is always positive, K_As typically 0-100%
# - Skewness can be positive or negative, K_As can exceed 100%
# Thresholds calibrated based on actual data distribution (percentiles)
ASYMMETRY_THRESHOLDS_RMS = {
    "healthy": 35,      # K_As < 35% (below 25th percentile)
    "mild": 45,         # 35% <= K_As < 45% (around median)
    "moderate": 55,     # 45% <= K_As < 55% (above 75th percentile)
    "high": 100         # K_As >= 55%
}

ASYMMETRY_THRESHOLDS_SKEW = {
    "healthy": 75,      # K_As < 75% (below 25th percentile)
    "mild": 95,         # 75% <= K_As < 95% (around median)
    "moderate": 115,    # 95% <= K_As < 115% (above 75th percentile)
    "high": 200         # K_As >= 115%
}
