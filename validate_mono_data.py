#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Validation Script for Monoenergetic Data (105-106 eV)
# This script validates and analyzes the available data in the 105-106 eV range

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from src.preprocessing import DataRescaler

def analyze_file(file_path):
    """Analyze a single HDF5 file and return energy statistics."""
    with h5py.File(file_path, 'r') as f:
        # Get available datasets
        datasets = list(f.keys())
        print(f"\nAnalyzing {os.path.basename(file_path)}")
        print(f"Available datasets: {datasets}")
        
        # Get energy data
        energy0 = f['energy0'][:]
        energy1 = np.array([]) if 'energy1' not in f else f['energy1'][:]
        
        # Find indices for target energy range
        indices0 = np.where((energy0 >= 105) & (energy0 <= 106))[0]
        indices1 = np.where((energy1 >= 105) & (energy1 <= 106))[0] if len(energy1) > 0 else np.array([])
        
        # Combine indices
        indices = np.unique(np.concatenate([indices0, indices1])) if len(indices1) > 0 else indices0
        
        # Calculate statistics
        stats = {
            'total_samples': len(energy0) + len(energy1),
            'target_samples': len(indices),
            'energy0_samples': len(indices0),
            'energy1_samples': len(indices1),
            'energy0_range': (np.min(energy0), np.max(energy0)),
            'energy1_range': (np.min(energy1), np.max(energy1)) if len(energy1) > 0 else (0, 0)
        }
        
        return stats, energy0, energy1

def plot_energy_distribution(train_stats, test_stats, train_energy0, train_energy1, test_energy0, test_energy1):
    """Create visualizations for energy distributions."""
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # Full energy distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(train_energy0, bins=50, alpha=0.5, label='Train (energy0)', color='skyblue')
    if len(train_energy1) > 0:
        ax1.hist(train_energy1, bins=50, alpha=0.5, label='Train (energy1)', color='lightblue')
    ax1.hist(test_energy0, bins=50, alpha=0.5, label='Test (energy0)', color='lightgreen')
    ax1.set_title('Full Energy Distribution')
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Zoomed view around target range
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(train_energy0, bins=50, alpha=0.5, label='Train (energy0)', color='skyblue')
    if len(train_energy1) > 0:
        ax2.hist(train_energy1, bins=50, alpha=0.5, label='Train (energy1)', color='lightblue')
    ax2.hist(test_energy0, bins=50, alpha=0.5, label='Test (energy0)', color='lightgreen')
    ax2.axvspan(105, 106, color='red', alpha=0.2, label='Target Range')
    ax2.set_xlim(104, 107)
    ax2.set_title('Zoomed View (104-107 eV)')
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Sample counts in target range
    ax3 = fig.add_subplot(gs[1, 0])
    labels = ['Train (energy0)']
    counts = [train_stats['energy0_samples']]
    colors = ['skyblue']
    if len(train_energy1) > 0:
        labels.append('Train (energy1)')
        counts.append(train_stats['energy1_samples'])
        colors.append('lightblue')
    labels.append('Test (energy0)')
    counts.append(test_stats['energy0_samples'])
    colors.append('lightgreen')
    
    bars = ax3.bar(labels, counts, color=colors)
    ax3.set_title('Samples in 105-106 eV Range')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # Percentages in target range
    ax4 = fig.add_subplot(gs[1, 1])
    percentages = [train_stats['energy0_samples'] / train_stats['total_samples'] * 100]
    if len(train_energy1) > 0:
        percentages.append(train_stats['energy1_samples'] / train_stats['total_samples'] * 100)
    percentages.append(test_stats['energy0_samples'] / test_stats['total_samples'] * 100)
    
    bars = ax4.bar(labels, percentages, color=colors)
    ax4.set_title('Percentage in 105-106 eV Range')
    ax4.set_ylabel('Percentage (%)')
    ax4.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('energy_distribution_analysis.png')
    plt.close()

def main():
    # Create output directory
    os.makedirs('mono_energy_validation', exist_ok=True)
    
    # Analyze files
    train_stats, train_energy0, train_energy1 = analyze_file('./data/training/train.h5')
    test_stats, test_energy0, test_energy1 = analyze_file('./data/test/test.h5')
    
    # Print statistics
    print("\nTraining Statistics:")
    print(f"Total samples: {train_stats['total_samples']}")
    print(f"Samples in energy0 (105-106 eV): {train_stats['energy0_samples']}")
    print(f"Samples in energy1 (105-106 eV): {train_stats['energy1_samples']}")
    print(f"Total unique samples in target range: {train_stats['target_samples']}")
    print(f"Energy0 range: {train_stats['energy0_range'][0]:.2f} - {train_stats['energy0_range'][1]:.2f} eV")
    print(f"Energy1 range: {train_stats['energy1_range'][0]:.2f} - {train_stats['energy1_range'][1]:.2f} eV")
    
    print("\nTest Statistics:")
    print(f"Total samples: {test_stats['total_samples']}")
    print(f"Samples in energy0 (105-106 eV): {test_stats['energy0_samples']}")
    print(f"Total unique samples in target range: {test_stats['target_samples']}")
    print(f"Energy0 range: {test_stats['energy0_range'][0]:.2f} - {test_stats['energy0_range'][1]:.2f} eV")
    
    # Create visualizations
    plot_energy_distribution(train_stats, test_stats, train_energy0, train_energy1, test_energy0, test_energy1)
    
    print("\nResults saved to ./mono_energy_validation/")
    print("Recommended energy range for normalization:")
    print("e_min=105, e_max=106")

if __name__ == "__main__":
    main() 