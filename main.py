"""
sEMG Analysis Pipeline - Main Script
=====================================
Automatic Movement Classification + Muscle Asymmetry Analysis

Dataset: EMG Database 1 (101 subjects, 3 exercises)
"""

import os
import numpy as np
import config
from data_loader import load_all_data, print_dataset_info, load_single_recording
from preprocessing import preprocess_dataset, preprocess_recording, expand_labels, expand_subject_ids
from feature_extraction import extract_features_batch, get_feature_names
from asymmetry_analysis import analyze_all_subjects, compute_statistics
from model import (
    split_by_subject, scale_features, train_random_forest, 
    evaluate_model, get_feature_importance, save_model,
    cross_validate_by_subject
)
from visualization import (
    plot_raw_signal, plot_spectrogram,
    plot_confusion_matrix, plot_asymmetry_distribution
)


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
    X, y, subject_ids = load_all_data()
    
    # Preprocess with windowing
    print("\nApplying preprocessing pipeline:")
    print("  1. Remove DC offset")
    print("  2. Bandpass filter (20-450 Hz)")
    print("  3. Z-score normalization")
    print("  4. Window segmentation")
    
    X_windows, windows_per_recording = preprocess_dataset(X, segment=True)
    y_windows = expand_labels(y, windows_per_recording)
    subject_windows = expand_subject_ids(subject_ids, windows_per_recording)
    
    print(f"\nResult: {X_windows.shape[0]} windows of shape {X_windows.shape[1:]}")
    
    # =========================================================================
    # STEP 3: FEATURE EXTRACTION
    # =========================================================================
    print("\n[STEP 3] Feature Extraction")
    print("-" * 40)
    
    print("Extracting features:")
    print("  Time-domain: MAV, ZCR, WL, SSC, RMS, VAR, SKEW, KURT, IEMG, LOG")
    print("  Frequency-domain: MNF, MDF")
    
    features = extract_features_batch(X_windows)
    feature_names = get_feature_names()
    
    print(f"\nFeatures per window: {features.shape[1]} (12 features × 8 channels)")
    
    # =========================================================================
    # STEP 4: DATA SPLITTING
    # =========================================================================
    print("\n[STEP 4] Data Splitting")
    print("-" * 40)
    
    print("Splitting by SUBJECT to avoid data leakage:")
    print("  - Training set: 70% of subjects (for learning)")
    print("  - Validation set: 15% of subjects (for tuning)")
    print("  - Test set: 15% of subjects (for final evaluation)")
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_by_subject(
        features, y_windows, subject_windows
    )
    
    # Combine train+val for sklearn (no separate validation needed)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    # Scale features
    print("\nScaling features (StandardScaler)...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train_full, X_val, X_test
    )
    
    # =========================================================================
    # STEP 5: CROSS-VALIDATION (Model Robustness Assessment)
    # =========================================================================
    print("\n[STEP 5] Cross-Validation")
    print("-" * 40)
    
    print("Assessing model robustness with 5-fold cross-validation...")
    print("(Each fold uses different subjects for validation)")
    
    cv_results = cross_validate_by_subject(features, y_windows, subject_windows, n_splits=5)
    
    # =========================================================================
    # STEP 6: MODEL TRAINING (Final Model)
    # =========================================================================
    print("\n[STEP 6] Final Model Training")
    print("-" * 40)
    
    print("Training final model on full training set:")
    print("  - 200 trees")
    print("  - Balanced class weights")
    print("  - All CPU cores")
    
    model = train_random_forest(X_train_scaled, y_train_full, n_estimators=200)
    
    # Feature importance (print only, no plot)
    importance = get_feature_importance(model, feature_names)
    print("\nTop 10 most important features:")
    for i, (name, score) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1}. {name}: {score:.4f}")
    
    # Save model
    save_model(model, scaler, "emg_classifier")
    
    # =========================================================================
    # STEP 7: MODEL EVALUATION (Test Set)
    # =========================================================================
    print("\n[STEP 7] Model Evaluation")
    print("-" * 40)
    
    results = evaluate_model(model, X_test_scaled, y_test)
    
    print("\n" + "=" * 40)
    print(" CLASSIFICATION RESULTS (Test Set)")
    print("=" * 40)
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    print("\nClassification Report:")
    print(results['report'])
    
    # Confusion matrix plot
    plot_confusion_matrix(results['confusion_matrix'],
        save_path=os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"))
    
    # =========================================================================
    # STEP 8: ASYMMETRY ANALYSIS
    # =========================================================================
    print("\n[STEP 8] Muscle Asymmetry Analysis")
    print("-" * 40)
    
    print("Analyzing flexor vs extensor asymmetry...")
    print("  Comparing channel pairs using RMS and Skewness")
    print("  Formula: K_As = |X1 - X2| × 100% / X1")
    
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
    for f in sorted(os.listdir(config.OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f"  - {f}")
    
    print(f"\nModel saved to: {config.MODELS_DIR}/")
    
    print("\n" + "=" * 60)
    print(" FINAL SUMMARY")
    print("=" * 60)
    print(f"\n  Movement Classification:")
    print(f"    Cross-validation: {cv_results['accuracy']['mean']*100:.1f}% ± {cv_results['accuracy']['std']*100:.1f}%")
    print(f"    Test set accuracy: {results['accuracy']*100:.1f}%")
    print(f"    Test set F1 score: {results['f1']*100:.1f}%")
    print(f"\n  Asymmetry Analysis:")
    print(f"    Mean RMS Asymmetry: {stats['rms']['mean']:.1f}%")
    print(f"    Mean Skewness Asymmetry: {stats['skewness']['mean']:.1f}%")


if __name__ == "__main__":
    main()
