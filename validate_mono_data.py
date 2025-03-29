#!/usr/bin/env python
# coding: utf-8

# Validation Script for Monoenergetic Data (105-106 eV)
# This script validates and analyzes the available data in the 105-106 eV range

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_file(filename, label):
    """Analyze a single HDF5 file and return energy statistics."""
    print(f"\nAnalyzing {label} ({filename})...")
    try:
        with h5py.File(filename, 'r') as f:
            # Print dataset names
            print(f"Available datasets in {label}:")
            for key in f.keys():
                print(f"  - {key}")
            
            # Get energy values
            if label == 'train':
                # Combine energy0 and energy1 for training data
                energies = np.concatenate([f['energy0'][:], f['energy1'][:]])
            else:
                energies = f['energy0'][:]
                
            min_energy = np.min(energies)
            max_energy = np.max(energies)
            mean_energy = np.mean(energies)
            std_energy = np.std(energies)
            
            # Count samples in target range
            target_indices = np.where((energies >= 105) & (energies <= 106))[0]
            num_target = len(target_indices)
            
            print(f"\nEnergy statistics for {label}:")
            print(f"  Total samples: {len(energies)}")
            print(f"  Min energy: {min_energy:.2f} eV")
            print(f"  Max energy: {max_energy:.2f} eV")
            print(f"  Mean energy: {mean_energy:.2f} eV")
            print(f"  Std energy: {std_energy:.2f} eV")
            print(f"  Samples in 105-106 eV range: {num_target}")
            print(f"  Percentage in range: {(num_target/len(energies)*100):.2f}%")
            
            return energies, num_target
            
    except Exception as e:
        print(f"Error analyzing {filename}: {str(e)}")
        return None, 0

def plot_energy_distribution(energies_dict, output_dir):
    """Plot energy distribution for all files."""
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # Full range distribution
    ax1 = fig.add_subplot(gs[0, 0])
    for label, energies in energies_dict.items():
        if energies is not None:
            ax1.hist(energies, bins=np.linspace(70, 220, 75), alpha=0.5, 
                    label=label, color='skyblue' if label == 'train' else 'lightgreen')
    ax1.axvline(x=105, color='r', linestyle='--', alpha=0.5)
    ax1.axvline(x=106, color='r', linestyle='--', alpha=0.5)
    ax1.set_title('Energy Distribution Across All Datasets')
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True)
    
    # Zoomed view around target range
    ax2 = fig.add_subplot(gs[0, 1])
    for label, energies in energies_dict.items():
        if energies is not None:
            ax2.hist(energies, bins=np.linspace(100, 110, 40), alpha=0.5,
                    label=label, color='skyblue' if label == 'train' else 'lightgreen')
    ax2.axvspan(105, 106, color='red', alpha=0.2)
    ax2.set_title('Energy Distribution Around 105-106 eV Range')
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True)
    
    # Sample counts
    ax3 = fig.add_subplot(gs[1, 0])
    counts = []
    labels = []
    colors = []
    for label, energies in energies_dict.items():
        if energies is not None:
            mask = (energies >= 105) & (energies <= 106)
            counts.append(np.sum(mask))
            labels.append(label)
            colors.append('skyblue' if label == 'train' else 'lightgreen')
    
    bars = ax3.bar(labels, counts, color=colors)
    ax3.set_title('Sample Count in 105-106 eV Range')
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Number of Samples')
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # Percentages
    ax4 = fig.add_subplot(gs[1, 1])
    percentages = []
    for label, energies in energies_dict.items():
        if energies is not None:
            mask = (energies >= 105) & (energies <= 106)
            percentages.append(np.sum(mask) / len(energies) * 100)
    
    bars = ax4.bar(labels, percentages, color=colors)
    ax4.set_title('Percentage of Samples in 105-106 eV Range')
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Percentage (%)')
    # Add percentage labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/energy_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create output directory
    output_dir = './mono_energy_validation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze files
    energies_dict = {}
    total_target_samples = 0
    
    # Analyze training file
    train_energies, train_target = analyze_file('./data/training/train.h5', 'train')
    energies_dict['train'] = train_energies
    total_target_samples += train_target
    
    # Analyze test file
    test_energies, test_target = analyze_file('./data/test/test.h5', 'test')
    energies_dict['test'] = test_energies
    total_target_samples += test_target
    
    print(f"\nTotal samples in 105-106 eV range:")
    print(f"Training: {train_target}")
    print(f"Test: {test_target}")
    
    # Create visualizations
    plot_energy_distribution(energies_dict, output_dir)
    
    if total_target_samples == 0:
        print("\nWarning: No samples found in the specified energy range!")
        print("You may need to adjust the energy range or check the data files.")
    else:
        print("\nValidation completed successfully!")
        print(f"Found {total_target_samples} total samples in the target energy range.")
        print(f"Results saved to {output_dir}/")
        print("Recommended energy range for normalization: e_min=105, e_max=106")

if __name__ == "__main__":
    main() 