"""
Classification model for sEMG movement recognition.
Uses Random Forest (scikit-learn) - best for sEMG feature classification.
"""

import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, balanced_accuracy_score
)
import joblib
import os
import config


def set_seed(seed=None):
    """Set RNG seeds for reproducibility."""
    if seed is None:
        seed = config.SEED
    random.seed(seed)
    np.random.seed(seed)


def build_random_forest(**overrides):
    """Create a RandomForestClassifier using config defaults."""
    params = {
        "n_estimators": config.RF_N_ESTIMATORS,
        "max_depth": config.RF_MAX_DEPTH,
        "min_samples_split": config.RF_MIN_SAMPLES_SPLIT,
        "min_samples_leaf": config.RF_MIN_SAMPLES_LEAF,
        "max_features": config.RF_MAX_FEATURES,
        "max_samples": config.RF_MAX_SAMPLES,
        "bootstrap": config.RF_BOOTSTRAP,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": config.SEED,
    }
    params.update({k: v for k, v in overrides.items() if v is not None})
    return RandomForestClassifier(**params)


def split_by_subject(X, y, subject_ids):
    """
    Split data by SUBJECT to avoid data leakage.
    
    Why: If we split randomly, adjacent windows from the same recording
    could end up in both train and test sets (nearly identical data).
    By splitting by subject, we ensure the model is tested on completely
    unseen subjects.
    
    Args:
        X: Feature matrix (num_samples, num_features)
        y: Labels (num_samples,)
        subject_ids: Subject identifier for each sample
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Get unique subjects and shuffle
    unique_subjects = np.unique(subject_ids)
    np.random.seed(config.SEED)  # For reproducibility
    np.random.shuffle(unique_subjects)
    
    # Calculate split indices
    n_subjects = len(unique_subjects)
    n_train = int(n_subjects * config.TRAIN_RATIO)
    n_val = int(n_subjects * config.VAL_RATIO)
    
    # Assign subjects to splits
    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train:n_train + n_val]
    test_subjects = unique_subjects[n_train + n_val:]
    
    # Create masks
    train_mask = np.isin(subject_ids, train_subjects)
    val_mask = np.isin(subject_ids, val_subjects)
    test_mask = np.isin(subject_ids, test_subjects)
    
    # Split data
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\nData split by subject:")
    print(f"  Training:   {len(train_subjects)} subjects, {len(X_train)} samples")
    print(f"  Validation: {len(val_subjects)} subjects, {len(X_val)} samples")
    print(f"  Testing:    {len(test_subjects)} subjects, {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """
    Standardize features (zero mean, unit variance).
    Fit only on training data to avoid data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_random_forest(X_train, y_train, **overrides):
    """
    Train a Random Forest classifier.
    
    Why Random Forest:
    - Works well with tabular/feature data
    - Robust to overfitting
    - Provides feature importance
    - Fast training, no GPU needed
    """
    model = build_random_forest(**overrides)
    print(f"\nTraining Random Forest ({model.n_estimators} trees)...")
    
    model.fit(X_train, y_train)
    print("Training complete!")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and return metrics.
    """
    y_pred = model.predict(X_test)
    
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "f1": f1_score(y_test, y_pred, average='macro'),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
        "y_pred": y_pred,
    }
    
    return results


def get_feature_importance(model, feature_names):
    """
    Get feature importance from Random Forest.
    Shows which features are most useful for classification.
    """
    importance = model.feature_importances_
    importance_dict = dict(zip(feature_names, importance))
    # Sort by importance (descending)
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def cross_validate_by_subject(X, y, subject_ids, n_splits=5):
    """
    Perform cross-validation with subject-based splits.
    
    This ensures that all windows from the same subject are in the same fold,
    preventing data leakage between training and validation.
    
    Args:
        X: Feature matrix (num_samples, num_features)
        y: Labels (num_samples,)
        subject_ids: Subject identifier for each sample
        n_splits: Number of CV folds (default 5)
    
    Returns:
        Dictionary with mean and std of each metric across folds
    """
    set_seed(config.SEED)
    print(f"\nPerforming {n_splits}-fold cross-validation (split by subject)...")
    print("Scaling performed per-fold (fit on fold train).")
    
    # Use StratifiedGroupKFold to:
    # 1. Keep all samples from same subject in same fold (GroupKFold)
    # 2. Try to balance class distribution across folds (Stratified)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    
    # Scale features for each fold
    scaler = StandardScaler()
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=subject_ids)):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale (fit on train only)
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train model
        model = build_random_forest()
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val_scaled)
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='macro')
        rec = recall_score(y_val, y_pred, average='macro')
        f1 = f1_score(y_val, y_pred, average='macro')
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        
        # Count subjects in this fold
        n_train_subjects = len(np.unique(subject_ids[train_idx]))
        n_val_subjects = len(np.unique(subject_ids[val_idx]))
        print(f"  Fold {fold+1}: Acc={acc:.3f}, F1={f1:.3f} "
              f"(train: {n_train_subjects} subj, val: {n_val_subjects} subj)")
    
    results = {
        "accuracy": {"mean": np.mean(accuracies), "std": np.std(accuracies)},
        "precision": {"mean": np.mean(precisions), "std": np.std(precisions)},
        "recall": {"mean": np.mean(recalls), "std": np.std(recalls)},
        "f1": {"mean": np.mean(f1_scores), "std": np.std(f1_scores)},
    }
    
    print(f"\nCross-validation results ({n_splits} folds):")
    print(f"  Accuracy:  {results['accuracy']['mean']:.4f} +/- {results['accuracy']['std']:.4f}")
    print(f"  Precision: {results['precision']['mean']:.4f} +/- {results['precision']['std']:.4f}")
    print(f"  Recall:    {results['recall']['mean']:.4f} +/- {results['recall']['std']:.4f}")
    print(f"  F1 Score:  {results['f1']['mean']:.4f} +/- {results['f1']['std']:.4f}")
    
    return results


def stratified_group_split(y, groups, train_ratio, val_ratio, test_ratio, seed=None, max_iter=500):
    """
    Approximate stratified split by groups, minimizing class distribution drift.
    """
    if seed is None:
        seed = config.SEED
    rng = np.random.RandomState(seed)

    unique_groups = np.unique(groups)
    if len(unique_groups) == 0:
        raise ValueError("No groups provided for split.")

    n_groups = len(unique_groups)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)
    n_test = n_groups - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("Invalid split ratios for number of groups.")

    classes = np.unique(y)
    total_counts = np.zeros(len(classes), dtype=np.float64)
    group_counts = {}
    for g in unique_groups:
        mask = groups == g
        counts = np.array([np.sum(y[mask] == c) for c in classes], dtype=np.float64)
        group_counts[g] = counts
        total_counts += counts

    total_dist = total_counts / np.sum(total_counts)

    best_score = None
    best_split = None

    for _ in range(max_iter):
        perm = rng.permutation(unique_groups)
        train_groups = perm[:n_train]
        val_groups = perm[n_train:n_train + n_val]
        test_groups = perm[n_train + n_val:]

        def dist_for(gs):
            counts = np.sum([group_counts[g] for g in gs], axis=0)
            return counts / np.sum(counts)

        train_dist = dist_for(train_groups)
        val_dist = dist_for(val_groups)
        test_dist = dist_for(test_groups)

        score = (
            np.sum(np.abs(train_dist - total_dist)) +
            np.sum(np.abs(val_dist - total_dist)) +
            np.sum(np.abs(test_dist - total_dist))
        )

        if best_score is None or score < best_score:
            best_score = score
            best_split = (train_groups, val_groups, test_groups)

    return best_split


def assert_disjoint_arrays(name, arrays):
    """Raise if arrays share any common elements."""
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            inter = np.intersect1d(arrays[i], arrays[j])
            if inter.size > 0:
                raise ValueError(f"{name} overlap between splits: {inter[:5]}")


def save_model(model, scaler, name="emg_classifier"):
    """Save trained model and scaler."""
    model_path = os.path.join(config.MODELS_DIR, f"{name}.joblib")
    scaler_path = os.path.join(config.MODELS_DIR, f"{name}_scaler.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved to {model_path}")


def load_model(name="emg_classifier"):
    """Load trained model and scaler."""
    model_path = os.path.join(config.MODELS_DIR, f"{name}.joblib")
    scaler_path = os.path.join(config.MODELS_DIR, f"{name}_scaler.joblib")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler


if __name__ == "__main__":
    print("Model module loaded successfully.")
    print("Use train_random_forest() to create a classifier.")
