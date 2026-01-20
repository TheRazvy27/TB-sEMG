"""
Train and evaluate an RBF SVM on a subset of subjects.
This keeps runtime reasonable while testing feasibility.
"""

import argparse
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import config
from data_loader import load_all_data
from preprocessing import preprocess_dataset, expand_labels, expand_subject_ids
from feature_extraction import extract_features_batch
from model import (
    stratified_group_split,
    assert_disjoint_arrays,
    scale_features,
    evaluate_model,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="RBF SVM on a subject subset")
    parser.add_argument("--num-subjects", type=int, default=30,
                        help="Number of subjects to keep (default: 30)")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed for subject sampling (default: 42)")
    parser.add_argument("--c", type=float, default=10.0,
                        help="SVM C parameter (default: 10.0)")
    parser.add_argument("--gamma", type=str, default="scale",
                        help="SVM gamma parameter (default: scale)")
    parser.add_argument("--cv", action="store_true",
                        help="Run subject-based cross-validation")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    set_seed(args.seed)

    print("\n" + "=" * 60)
    print(" sEMG RBF SVM - SUBSET EXPERIMENT")
    print("=" * 60)
    print(f"Using {args.num_subjects} subjects (seed={args.seed})")

    # Load full dataset
    X, y, subject_ids = load_all_data()

    # Sample subjects
    unique_subjects = np.unique(subject_ids)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(unique_subjects)
    keep_subjects = unique_subjects[:args.num_subjects]
    mask = np.isin(subject_ids, keep_subjects)

    X = X[mask]
    y = y[mask]
    subject_ids = subject_ids[mask]

    print(f"Subset recordings: {X.shape[0]} from {len(np.unique(subject_ids))} subjects")

    # Preprocess and window
    X_windows, windows_per_recording = preprocess_dataset(X, segment=True)
    y_windows = expand_labels(y, windows_per_recording)
    subject_windows = expand_subject_ids(subject_ids, windows_per_recording)

    print(f"Windows: {X_windows.shape[0]} of shape {X_windows.shape[1:]}")

    # Feature extraction
    features = extract_features_batch(X_windows)

    # Optional cross-validation
    if args.cv:
        print(f"\nRunning {args.n_splits}-fold CV (split by subject)...")
        cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        scaler = StandardScaler()

        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(features, y_windows, groups=subject_windows)):
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = y_windows[train_idx], y_windows[val_idx]

            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = SVC(kernel="rbf", C=args.c, gamma=args.gamma)
            model.fit(X_train_scaled, y_train)

            fold_results = evaluate_model(model, X_val_scaled, y_val)
            accuracies.append(fold_results["accuracy"])
            precisions.append(fold_results["precision"])
            recalls.append(fold_results["recall"])
            f1_scores.append(fold_results["f1"])

            n_train_subjects = len(np.unique(subject_windows[train_idx]))
            n_val_subjects = len(np.unique(subject_windows[val_idx]))
            print(f"  Fold {fold + 1}: Acc={fold_results['accuracy']:.3f}, "
                  f"F1={fold_results['f1']:.3f} (train: {n_train_subjects} subj, "
                  f"val: {n_val_subjects} subj)")

        print("\nCross-validation results:")
        print(f"  Accuracy:  {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
        print(f"  Precision: {np.mean(precisions):.4f} +/- {np.std(precisions):.4f}")
        print(f"  Recall:    {np.mean(recalls):.4f} +/- {np.std(recalls):.4f}")
        print(f"  F1 Score:  {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")

    # Split by subject with stratification (subset only)
    train_subjects, val_subjects, test_subjects = stratified_group_split(
        y_windows,
        subject_windows,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=args.seed,
        max_iter=200,
    )

    train_mask = np.isin(subject_windows, train_subjects)
    val_mask = np.isin(subject_windows, val_subjects)
    test_mask = np.isin(subject_windows, test_subjects)

    train_subjects = np.unique(subject_windows[train_mask])
    val_subjects = np.unique(subject_windows[val_mask])
    test_subjects = np.unique(subject_windows[test_mask])
    assert_disjoint_arrays("subject_ids", [train_subjects, val_subjects, test_subjects])

    X_train, y_train = features[train_mask], y_windows[train_mask]
    X_val, y_val = features[val_mask], y_windows[val_mask]
    X_test, y_test = features[test_mask], y_windows[test_mask]

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )

    # Train RBF SVM
    print("\nTraining SVM (RBF kernel)...")
    model = SVC(kernel="rbf", C=args.c, gamma=args.gamma)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    results = evaluate_model(model, X_test_scaled, y_test)

    elapsed = time.time() - start
    print("\n" + "=" * 40)
    print(" SVM RBF RESULTS (Subset)")
    print("=" * 40)
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    print(f"\nElapsed: {elapsed:.1f} sec")


if __name__ == "__main__":
    main()
