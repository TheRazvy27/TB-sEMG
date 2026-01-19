"""
Visualization functions for sEMG analysis.
Uses matplotlib only (no display needed - saves to files).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (saves to file)
import matplotlib.pyplot as plt
from scipy import signal
import os
import config


def plot_raw_signal(data, channel=0, title="Raw sEMG Signal", save_path=None):
    """Plot raw sEMG signal from one channel."""
    time = np.arange(data.shape[1]) / config.SAMPLING_RATE
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, data[channel], linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_spectrogram(data, channel=0, title="Spectrogram", save_path=None):
    """Plot spectrogram of sEMG signal."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    f, t, Sxx = signal.spectrogram(data[channel], fs=config.SAMPLING_RATE, nperseg=256, noverlap=128)
    
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    ax.set_ylim(0, 250)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    im = ax.imshow(cm_norm, cmap='Blues')
    
    # Labels
    labels = ['Exercise 0', 'Exercise 1', 'Exercise 2']
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Add text
    for i in range(3):
        for j in range(3):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm_norm[i,j]:.1%}\n(n={cm[i,j]})',
                   ha='center', va='center', color=color, fontsize=10)
    
    plt.colorbar(im, ax=ax, label='Proportion')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_asymmetry_distribution(all_results, save_path=None):
    """Plot K_As distribution for RMS and Skewness."""
    rms_values = []
    skew_values = []
    
    for subject_data in all_results.values():
        for exercise_results in subject_data.values():
            for r in exercise_results:
                rms_values.append(r["k_as_rms"])
                skew_values.append(r["k_as_skew"])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # RMS histogram
    axes[0].hist(rms_values, bins=25, color='#3498db', edgecolor='white', alpha=0.7)
    axes[0].axvline(10, color='green', linestyle='--', label='Healthy (<10%)')
    axes[0].axvline(20, color='orange', linestyle='--', label='Mild (<20%)')
    axes[0].axvline(35, color='red', linestyle='--', label='Moderate (<35%)')
    axes[0].set_xlabel('K_As (RMS) %')
    axes[0].set_ylabel('Count')
    axes[0].set_title('RMS Asymmetry Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Skewness histogram
    axes[1].hist(skew_values, bins=25, color='#e74c3c', edgecolor='white', alpha=0.7)
    axes[1].axvline(10, color='green', linestyle='--')
    axes[1].axvline(20, color='orange', linestyle='--')
    axes[1].axvline(35, color='red', linestyle='--')
    axes[1].set_xlabel('K_As (Skewness) %')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Skewness Asymmetry Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("Visualization module loaded.")
    print("All plots are saved to files (no display window).")
