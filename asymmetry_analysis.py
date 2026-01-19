"""
Muscle Asymmetry Analysis.
Calculates K_As (asymmetry coefficient) using RMS and Skewness.
Compares flexor vs extensor muscle groups.
"""

import numpy as np
from scipy import stats
import config
from data_loader import load_single_recording, get_flexor_extensor_pairs
from preprocessing import remove_dc_offset, bandpass_filter


def calculate_rms(x):
    """Calculate Root Mean Square."""
    return np.sqrt(np.mean(x ** 2))


def calculate_skewness(x):
    """Calculate Skewness."""
    return stats.skew(x)


def calculate_asymmetry_coefficient(x1, x2):
    """
    Calculate asymmetry coefficient K_As.
    
    K_As = |X1 - X2| * 100% / X1
    where X1 is the larger value.
    
    Args:
        x1, x2: RMS or Skewness values from two channels
    
    Returns:
        K_As as percentage
    """
    # Ensure X1 is the larger value
    if abs(x2) > abs(x1):
        x1, x2 = x2, x1
    
    if abs(x1) < 1e-10:
        return 0.0
    
    return (abs(x1 - x2) * 100) / abs(x1)


def classify_asymmetry(k_as):
    """
    Classify asymmetry level based on K_As value.
    
    Returns:
        (class_number, class_name)
    """
    if k_as < config.ASYMMETRY_THRESHOLDS["healthy"]:
        return 1, "Healthy"
    elif k_as < config.ASYMMETRY_THRESHOLDS["mild"]:
        return 2, "Mild Asymmetry"
    elif k_as < config.ASYMMETRY_THRESHOLDS["moderate"]:
        return 3, "Moderate Asymmetry"
    else:
        return 4, "High Asymmetry"


def analyze_recording(data):
    """
    Analyze asymmetry between channel pairs in a recording.
    
    Args:
        data: sEMG recording (channels, samples) - should be filtered but NOT normalized
    
    Returns:
        List of results for each channel pair
    """
    pairs = get_flexor_extensor_pairs()
    results = []
    
    for ext_ch, flex_ch in pairs:
        # Calculate RMS and Skewness for each channel
        rms_ext = calculate_rms(data[ext_ch])
        rms_flex = calculate_rms(data[flex_ch])
        skew_ext = calculate_skewness(data[ext_ch])
        skew_flex = calculate_skewness(data[flex_ch])
        
        # Calculate asymmetry coefficients
        k_as_rms = calculate_asymmetry_coefficient(rms_ext, rms_flex)
        k_as_skew = calculate_asymmetry_coefficient(skew_ext, skew_flex)
        
        # Classify
        rms_class, rms_label = classify_asymmetry(k_as_rms)
        skew_class, skew_label = classify_asymmetry(k_as_skew)
        
        results.append({
            "channels": (ext_ch + 1, flex_ch + 1),  # 1-indexed for display
            "rms_ext": rms_ext,
            "rms_flex": rms_flex,
            "skew_ext": skew_ext,
            "skew_flex": skew_flex,
            "k_as_rms": k_as_rms,
            "k_as_skew": k_as_skew,
            "rms_class": rms_class,
            "rms_label": rms_label,
            "skew_class": skew_class,
            "skew_label": skew_label,
        })
    
    return results


def analyze_subject(subject_id):
    """
    Analyze asymmetry for all exercises of a subject.
    
    Note: Uses filtered (NOT normalized) data to preserve amplitude differences.
    """
    results = {}
    
    for exercise_id in range(config.NUM_EXERCISES):
        data = load_single_recording(subject_id, exercise_id)
        if data is None:
            continue
        
        # Filter but DON'T normalize (need actual amplitudes for asymmetry)
        data = data.astype(np.float32)
        data = remove_dc_offset(data)
        data = bandpass_filter(data)
        
        results[exercise_id] = analyze_recording(data)
    
    return results


def analyze_all_subjects():
    """Analyze asymmetry for all subjects."""
    all_results = {}
    
    for subject_id in range(config.NUM_SUBJECTS):
        results = analyze_subject(subject_id)
        if results:
            all_results[subject_id] = results
        
        if (subject_id + 1) % 20 == 0:
            print(f"  Analyzed {subject_id + 1}/{config.NUM_SUBJECTS} subjects")
    
    return all_results


def compute_statistics(all_results):
    """Compute population-level statistics."""
    rms_values = []
    skew_values = []
    rms_classes = []
    skew_classes = []
    
    for subject_data in all_results.values():
        for exercise_results in subject_data.values():
            for r in exercise_results:
                rms_values.append(r["k_as_rms"])
                skew_values.append(r["k_as_skew"])
                rms_classes.append(r["rms_class"])
                skew_classes.append(r["skew_class"])
    
    rms_values = np.array(rms_values)
    skew_values = np.array(skew_values)
    
    return {
        "total_measurements": len(rms_values),
        "rms": {
            "mean": np.mean(rms_values),
            "std": np.std(rms_values),
            "median": np.median(rms_values),
            "class_distribution": dict(zip(*np.unique(rms_classes, return_counts=True))),
        },
        "skewness": {
            "mean": np.mean(skew_values),
            "std": np.std(skew_values),
            "median": np.median(skew_values),
            "class_distribution": dict(zip(*np.unique(skew_classes, return_counts=True))),
        },
    }


if __name__ == "__main__":
    print("Testing Asymmetry Analysis...")
    
    # Analyze one subject
    results = analyze_subject(0)
    
    print("\nSubject 0 Results:")
    for ex_id, ex_results in results.items():
        print(f"\n  Exercise {ex_id}:")
        for r in ex_results:
            ch = r["channels"]
            print(f"    Channels {ch[0]} vs {ch[1]}:")
            print(f"      RMS: {r['rms_ext']:.2f} vs {r['rms_flex']:.2f} -> K_As={r['k_as_rms']:.1f}% ({r['rms_label']})")
            print(f"      Skew: {r['skew_ext']:.2f} vs {r['skew_flex']:.2f} -> K_As={r['k_as_skew']:.1f}% ({r['skew_label']})")
