"""
Random Forest tuning using Group-Stratified CV on TRAIN only.
VAL is used only as a sanity check; TEST is evaluated once at the end.
"""

import itertools
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import config
from data_loader import load_all_data
from preprocessing import (
    preprocess_dataset,
    expand_labels,
    expand_subject_ids,
    expand_recording_ids,
)
from feature_extraction import extract_features_batch
from model import (
    stratified_group_split,
    assert_disjoint_arrays,
    set_seed,
    build_random_forest,
    evaluate_model,
)


def run_cv(X, y, groups, params, n_splits):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    scores = []
    for train_idx, val_idx in cv.split(X, y, groups=groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = build_random_forest(**params)
        model.fit(X_train_scaled, y_train)
        results = evaluate_model(model, X_val_scaled, y_val)
        scores.append(results)

    mean_metrics = {
        "accuracy": float(np.mean([s["accuracy"] for s in scores])),
        "balanced_accuracy": float(np.mean([s["balanced_accuracy"] for s in scores])),
        "f1": float(np.mean([s["f1"] for s in scores])),
    }
    return mean_metrics


def main():
    set_seed(config.SEED)
    print(f"Seed: {config.SEED}")

    X, y, subject_ids, recording_ids = load_all_data(return_recording_ids=True)
    X_windows, windows_per_recording = preprocess_dataset(
        X, segment=True, apply_normalize=config.SIGNAL_NORMALIZE
    )
    y_windows = expand_labels(y, windows_per_recording)
    subject_windows = expand_subject_ids(subject_ids, windows_per_recording)
    recording_windows = expand_recording_ids(recording_ids, windows_per_recording)

    features = extract_features_batch(X_windows)

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

    train_subjects = np.unique(subject_windows[train_mask])
    val_subjects = np.unique(subject_windows[val_mask])
    test_subjects = np.unique(subject_windows[test_mask])

    train_recordings = np.unique(recording_windows[train_mask])
    val_recordings = np.unique(recording_windows[val_mask])
    test_recordings = np.unique(recording_windows[test_mask])

    assert_disjoint_arrays("subject_ids", [train_subjects, val_subjects, test_subjects])
    assert_disjoint_arrays("recording_ids", [train_recordings, val_recordings, test_recordings])

    X_train, y_train = features[train_mask], y_windows[train_mask]
    X_val, y_val = features[val_mask], y_windows[val_mask]
    X_test, y_test = features[test_mask], y_windows[test_mask]

    print("Scaling performed per-fold (fit on fold train).")

    # Focused grid for runtime; refine later if needed.
    param_grid = [
        {"max_depth": 10, "min_samples_leaf": 10, "min_samples_split": 20, "max_features": 0.3, "n_estimators": 50},
        {"max_depth": 12, "min_samples_leaf": 10, "min_samples_split": 20, "max_features": 0.3, "n_estimators": 50},
    ]

    best = None
    for idx, params in enumerate(param_grid, 1):
        mean_metrics = run_cv(X_train, y_train, subject_windows[train_mask], params, config.TUNING_CV_SPLITS)

        key_metric = mean_metrics[config.TUNING_METRIC]
        if best is None or key_metric > best["score"]:
            best = {
                "score": key_metric,
                "params": params,
                "metrics": mean_metrics,
            }

        print(
            "params="
            f"{params['max_depth']},{params['min_samples_leaf']},{params['min_samples_split']},"
            f"{params['max_features']},{params['n_estimators']} | "
            f"cv_acc={mean_metrics['accuracy']:.3f}, "
            f"cv_bal_acc={mean_metrics['balanced_accuracy']:.3f}, "
            f"cv_f1={mean_metrics['f1']:.3f}"
        )
        print(f"Progress: {idx}/{len(param_grid)}")

    print("\nBest params (CV on TRAIN only):")
    print(best["params"])
    print(
        f"Best CV metrics: acc={best['metrics']['accuracy']:.3f}, "
        f"bal_acc={best['metrics']['balanced_accuracy']:.3f}, "
        f"f1={best['metrics']['f1']:.3f}"
    )

    # Sanity check on VAL (single split)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = build_random_forest(**best["params"])
    model.fit(X_train_scaled, y_train)
    val_results = evaluate_model(model, X_val_scaled, y_val)

    print("\nValidation (sanity check, not for selection):")
    print(
        f"val_acc={val_results['accuracy']:.3f}, "
        f"val_bal_acc={val_results['balanced_accuracy']:.3f}, "
        f"val_f1={val_results['f1']:.3f}"
    )

    if config.FINAL_RETRAIN_ON_TRAIN_VAL:
        print("\nRetrain on TRAIN+VAL, evaluate TEST once...")
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])

        scaler_final = StandardScaler()
        X_train_full_scaled = scaler_final.fit_transform(X_train_full)
        X_test_scaled = scaler_final.transform(X_test)

        final_model = build_random_forest(**best["params"])
        final_model.fit(X_train_full_scaled, y_train_full)
        test_results = evaluate_model(final_model, X_test_scaled, y_test)

        print(
            f"test_acc={test_results['accuracy']:.3f}, "
            f"test_bal_acc={test_results['balanced_accuracy']:.3f}, "
            f"test_f1={test_results['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
