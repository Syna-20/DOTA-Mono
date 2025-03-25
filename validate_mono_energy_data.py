#!/usr/bin/env python
# coding: utf-8

# Validation Script for Monoenergetic Data (105-106 eV)
# This script validates and analyzes the available data in the 105-106 eV range

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

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
            energies = f['energy0'][:]
            min_energy = np.min(energies)
            max_energy = np.max(energies)
            mean_energy = np.mean(energies)
            std_energy = np.std(energies)
            
            # Count samples in target range
            target_indices = np.where((energies >= 105) & (energies <= 106))[0]
            num_target = len(target_indices)
            
            print(f"\nEnergy statistics for {label}:")
            print(f"  Min energy: {min_energy:.2f} eV")
            print(f"  Max energy: {max_energy:.2f} eV")
            print(f"  Mean energy: {mean_energy:.2f} eV")
            print(f"  Std energy: {std_energy:.2f} eV")
            print(f"  Samples in 105-106 eV range: {num_target}")
            
            return energies, num_target
            
    except Exception as e:
        print(f"Error analyzing {filename}: {str(e)}")
        return None, 0

def plot_energy_distribution(energies_dict, output_dir):
    """Plot energy distribution for all files."""
    plt.figure(figsize=(12, 6))
    
    for label, energies in energies_dict.items():
        if energies is not None:
            plt.hist(energies, bins=50, alpha=0.5, label=label)
    
    plt.axvline(x=105, color='r', linestyle='--', label='Target Range Start')
    plt.axvline(x=106, color='r', linestyle='--', label='Target Range End')
    plt.title('Energy Distribution Across Datasets')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(f'{output_dir}/energy_distribution.png')
    plt.close()

def plot_target_range_samples(energies_dict, output_dir):
    """Plot detailed view of samples in target energy range."""
    plt.figure(figsize=(12, 6))
    
    for label, energies in energies_dict.items():
        if energies is not None:
            mask = (energies >= 105) & (energies <= 106)
            plt.hist(energies[mask], bins=20, alpha=0.5, label=label)
    
    plt.title('Samples in 105-106 eV Range')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(f'{output_dir}/target_range_samples.png')
    plt.close()

def main():
    # Create output directory
    os.makedirs('./mono_energy_validation', exist_ok=True)
    
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
    
    print(f"\nTotal training samples in range: {train_target}")
    print(f"Total test samples in range: {test_target}")
    
    # Create visualizations
    plot_energy_distribution(energies_dict, './mono_energy_validation')
    plot_target_range_samples(energies_dict, './mono_energy_validation')
    
    if total_target_samples == 0:
        print("\nWarning: No training samples found in the specified energy range!")
        print("You may need to adjust the energy range or check the data files.")
    else:
        print("\nValidation completed successfully!")
        print(f"Found {total_target_samples} samples in the target energy range.")
        print("Results saved to ./mono_energy_validation/")
        print("Recommended energy range for normalization: e_min=105, e_max=106")

if __name__ == "__main__":
    main() 