"""
sEMG Analysis Pipeline - Main Script
=====================================
Automatic Movement Classification + Muscle Asymmetry Analysis

Dataset: EMG Database 1 (101 subjects, 3 exercises)
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import config
from data_loader import load_all_data, print_dataset_info, load_single_recording
from preprocessing import (
    preprocess_dataset,
    preprocess_recording,
    expand_labels,
    expand_subject_ids,
    expand_recording_ids,
)
from feature_extraction import extract_features_batch
from asymmetry_analysis import analyze_all_subjects, compute_statistics
from model import (
    scale_features,
    train_random_forest,
    evaluate_model,
    save_model,
    cross_validate_by_subject,
    stratified_group_split,
    assert_disjoint_arrays,
    set_seed,
)
from visualization import (
    plot_raw_signal, plot_spectrogram,
    plot_confusion_matrix, plot_asymmetry_distribution
)


def print_class_distribution(name, labels):
    """Print label distribution for a split."""
    values, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    dist = {int(v): int(c) for v, c in zip(values, counts)}
    print(f"{name} class distribution: {dist} (total={total})")


def print_subject_class_distribution(name, labels, subjects):
    """Print subject counts per class for a split."""
    classes = np.unique(labels)
    dist = {}
    for c in classes:
        dist[int(c)] = int(len(np.unique(subjects[labels == c])))
    print(f"{name} subjects per class: {dist} (total={len(np.unique(subjects))})")


def main():
    """
    Main pipeline for sEMG analysis project.
    
    Steps:
    1. Load and explore data
    2. Preprocess (filter, normalize, window)
    3. Extract features
    4. Split data (by subject)
    5. Train classifier
    6. Evaluate
    7. Asymmetry analysis
    """
    
    print("\n" + "=" * 60)
    print(" sEMG MOVEMENT CLASSIFICATION PROJECT")
    print("=" * 60)

    set_seed(config.SEED)
    print(f"Seed: {config.SEED}")
    
    # =========================================================================
    # STEP 1: DATA EXPLORATION
    # =========================================================================
    print("\n[STEP 1] Data Exploration")
    print("-" * 40)
    
    print_dataset_info()
    
    # Visualize sample recordings (one per exercise)
    for ex_id in range(3):
        data = load_single_recording(0, ex_id)
        if data is not None:
            plot_raw_signal(data, channel=0,
                title=f"Subject 0 - Exercise {ex_id}",
                save_path=os.path.join(config.OUTPUT_DIR, f"raw_signal_ex{ex_id}.png"))
    
    # Spectrogram (frequency content over time)
    data = load_single_recording(0, 0)
    processed = preprocess_recording(data, segment=False)
    plot_spectrogram(processed, channel=0, title="Spectrogram (Channel 1)",
        save_path=os.path.join(config.OUTPUT_DIR, "spectrogram.png"))
    
    # =========================================================================
    # STEP 2: PREPROCESSING
    # =========================================================================
    print("\n[STEP 2] Preprocessing")
    print("-" * 40)
    
    # Load all data
    X, y, subject_ids, recording_ids = load_all_data(return_recording_ids=True)
    
    # Preprocess with windowing
    print("\nApplying preprocessing pipeline:")
    print("  1. Remove DC offset")
    print("  2. Bandpass filter (20-450 Hz)")
    if config.SIGNAL_NORMALIZE:
        print("  3. Z-score normalization")
    else:
        print("  3. Signal normalization: DISABLED")
    print("  4. Window segmentation")

    X_windows, windows_per_recording = preprocess_dataset(
        X, segment=True, apply_normalize=config.SIGNAL_NORMALIZE
    )
    y_windows = expand_labels(y, windows_per_recording)
    subject_windows = expand_subject_ids(subject_ids, windows_per_recording)
    recording_windows = expand_recording_ids(recording_ids, windows_per_recording)
    
    print(f"\nResult: {X_windows.shape[0]} windows of shape {X_windows.shape[1:]}")
    
    # =========================================================================
    # STEP 3: FEATURE EXTRACTION
    # =========================================================================
    print("\n[STEP 3] Feature Extraction")
    print("-" * 40)
    
    features = extract_features_batch(X_windows)
    num_feats_per_channel = features.shape[1] // config.NUM_CHANNELS
    print(f"\nFeatures per window: {features.shape[1]} ({num_feats_per_channel} features x 8 channels)")

    if not (len(y_windows) == len(subject_windows) == len(recording_windows) == features.shape[0]):
        raise ValueError("Metadata mismatch: window labels/subjects/recordings not aligned with features.")
    
    # =========================================================================
    # STEP 4: DATA SPLITTING
    # =========================================================================
    print("\n[STEP 4] Data Splitting")
    print("-" * 40)
    
    print("Splitting by SUBJECT to avoid data leakage:")
    print("  - Training set: 70% of subjects (for learning)")
    print("  - Validation set: 15% of subjects (for tuning)")
    print("  - Test set: 15% of subjects (for final evaluation)")

    train_subjects, val_subjects, test_subjects = stratified_group_split(
        y_windows,
        subject_windows,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=config.SEED,
        max_iter=config.SPLIT_MAX_ITER,
    )

    train_mask = np.isin(subject_windows, train_subjects)
    val_mask = np.isin(subject_windows, val_subjects)
    test_mask = np.isin(subject_windows, test_subjects)

    X_train, y_train = features[train_mask], y_windows[train_mask]
    X_val, y_val = features[val_mask], y_windows[val_mask]
    X_test, y_test = features[test_mask], y_windows[test_mask]

    train_subjects = np.unique(subject_windows[train_mask])
    val_subjects = np.unique(subject_windows[val_mask])
    test_subjects = np.unique(subject_windows[test_mask])

    train_recordings = np.unique(recording_windows[train_mask])
    val_recordings = np.unique(recording_windows[val_mask])
    test_recordings = np.unique(recording_windows[test_mask])

    assert_disjoint_arrays("subject_ids", [train_subjects, val_subjects, test_subjects])
    assert_disjoint_arrays("recording_ids", [train_recordings, val_recordings, test_recordings])

    print(f"\nData split by subject (stratified):")
    print(f"  Training:   {len(train_subjects)} subjects, {len(X_train)} samples")
    print(f"  Validation: {len(val_subjects)} subjects, {len(X_val)} samples")
    print(f"  Testing:    {len(test_subjects)} subjects, {len(X_test)} samples")
    print_class_distribution("Train", y_train)
    print_class_distribution("Val", y_val)
    print_class_distribution("Test", y_test)
    print_subject_class_distribution("Train", y_train, subject_windows[train_mask])
    print_subject_class_distribution("Val", y_val, subject_windows[val_mask])
    print_subject_class_distribution("Test", y_test, subject_windows[test_mask])

    # Scale features (fit only on TRAIN)
    print("\nScaling features (StandardScaler) [fit on TRAIN only]...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )
    
    # =========================================================================
    # STEP 5: CROSS-VALIDATION (Model Robustness Assessment)
    # =========================================================================
    print("\n[STEP 5] Cross-Validation")
    print("-" * 40)
    
    print("Assessing model robustness with 5-fold cross-validation...")
    print("(Each fold uses different subjects for validation, train split only)")
    
    cv_results = cross_validate_by_subject(
        X_train, y_train, subject_windows[train_mask], n_splits=5
    )
    
    # =========================================================================
    # STEP 6: MODEL TRAINING + VALIDATION
    # =========================================================================
    print("\n[STEP 6] Model Training (Train) + Validation")
    print("-" * 40)
    
    print("Training model on TRAIN split:")
    print(f"  - {config.RF_N_ESTIMATORS} trees")
    print(f"  - max_depth={config.RF_MAX_DEPTH}")
    print(f"  - min_samples_split={config.RF_MIN_SAMPLES_SPLIT}")
    print(f"  - min_samples_leaf={config.RF_MIN_SAMPLES_LEAF}")
    print(f"  - max_features={config.RF_MAX_FEATURES}")
    print(f"  - max_samples={config.RF_MAX_SAMPLES}")
    print("  - Balanced class weights")
    print("  - All CPU cores")
    
    model = train_random_forest(X_train_scaled, y_train)
    
    val_results = evaluate_model(model, X_val_scaled, y_val)
    
    print("\n" + "=" * 40)
    print(" VALIDATION RESULTS")
    print("=" * 40)
    print(f"  Accuracy:          {val_results['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {val_results['balanced_accuracy']:.4f}")
    print(f"  Precision:         {val_results['precision']:.4f}")
    print(f"  Recall:            {val_results['recall']:.4f}")
    print(f"  F1 Score:          {val_results['f1']:.4f}")
    print("\nClassification Report (Val):")
    print(val_results['report'])
    # Confusion matrix for validation is omitted to keep a single report matrix.
    
    # =========================================================================
    # STEP 7: FINAL RETRAIN + TEST EVALUATION (ONCE)
    # =========================================================================
    print("\n[STEP 7] Final Retrain + Test Evaluation")
    print("-" * 40)
    
    if config.FINAL_RETRAIN_ON_TRAIN_VAL:
        print("Retraining on TRAIN+VAL, then evaluating on TEST once...")
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        scaler_final = StandardScaler()
        X_train_full_scaled = scaler_final.fit_transform(X_train_full)
        X_test_scaled_final = scaler_final.transform(X_test)
        
        final_model = train_random_forest(X_train_full_scaled, y_train_full)
        test_results = evaluate_model(final_model, X_test_scaled_final, y_test)
        save_model(final_model, scaler_final, "emg_classifier")
    else:
        print("Evaluating TEST with model trained on TRAIN only (no retrain).")
        test_results = evaluate_model(model, X_test_scaled, y_test)
        save_model(model, scaler, "emg_classifier")
    
    print("\n" + "=" * 40)
    print(" TEST RESULTS")
    print("=" * 40)
    print(f"  Accuracy:          {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.1f}%)")
    print(f"  Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")
    print(f"  Precision:         {test_results['precision']:.4f}")
    print(f"  Recall:            {test_results['recall']:.4f}")
    print(f"  F1 Score:          {test_results['f1']:.4f}")
    print("\nClassification Report (Test):")
    print(test_results['report'])
    plot_confusion_matrix(test_results['confusion_matrix'],
        save_path=os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"))
    
    # =========================================================================
    # STEP 8: ASYMMETRY ANALYSIS
    # =========================================================================
    print("\n[STEP 8] Muscle Asymmetry Analysis")
    print("-" * 40)
    
    print("Analyzing flexor vs extensor asymmetry...")
    print("  Comparing channel pairs using RMS and Skewness")
    print("  Formula: K_As = |X1 - X2| x 100% / X1")
    
    all_asymmetry = analyze_all_subjects()
    stats = compute_statistics(all_asymmetry)
    
    print("\n" + "=" * 40)
    print(" ASYMMETRY ANALYSIS RESULTS")
    print("=" * 40)
    print(f"  Total measurements: {stats['total_measurements']}")
    print(f"\n  RMS-based Asymmetry:")
    print(f"    Mean: {stats['rms']['mean']:.1f}%")
    print(f"    Std:  {stats['rms']['std']:.1f}%")
    print(f"    Classes: {stats['rms']['class_distribution']}")
    print(f"\n  Skewness-based Asymmetry:")
    print(f"    Mean: {stats['skewness']['mean']:.1f}%")
    print(f"    Std:  {stats['skewness']['std']:.1f}%")
    print(f"    Classes: {stats['skewness']['class_distribution']}")
    
    plot_asymmetry_distribution(all_asymmetry,
        save_path=os.path.join(config.OUTPUT_DIR, "asymmetry_distribution.png"))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to: {config.OUTPUT_DIR}/")
    for f in [
        "raw_signal_ex0.png",
        "raw_signal_ex1.png",
        "raw_signal_ex2.png",
        "spectrogram.png",
        "confusion_matrix.png",
        "asymmetry_distribution.png",
    ]:
        if os.path.exists(os.path.join(config.OUTPUT_DIR, f)):
            print(f"  - {f}")
    
    print(f"\nModel saved to: {config.MODELS_DIR}/")
    
    print("\n" + "=" * 60)
    print(" FINAL SUMMARY")
    print("=" * 60)
    print(f"\n  Movement Classification:")
    print(f"    Cross-validation: {cv_results['accuracy']['mean']*100:.1f}% +/- {cv_results['accuracy']['std']*100:.1f}%")
    print(f"    Test set accuracy: {test_results['accuracy']*100:.1f}%")
    print(f"    Test set F1 score: {test_results['f1']*100:.1f}%")
    print(f"\n  Asymmetry Analysis:")
    print(f"    Mean RMS Asymmetry: {stats['rms']['mean']:.1f}%")
    print(f"    Mean Skewness Asymmetry: {stats['skewness']['mean']:.1f}%")


if __name__ == "__main__":
    main()
