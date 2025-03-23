#!/usr/bin/env python
# coding: utf-8

# Validation Script for Monoenergetic Data (105-106 eV)
# This script validates and analyzes the available data in the 105-106 eV range

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

# Create output directory for validation results
os.makedirs('mono_energy_validation', exist_ok=True)

# Define data file paths
data_paths = {
    'train_part1': './data/training/train_part1.h5',
    'train_part2': './data/training/train_part2.h5',
    'test': './data/test/test.h5'
}

# Energy range to filter
min_energy = 105
max_energy = 106

# Function to analyze energy distribution
def analyze_energy_distribution(filename, key='energy0'):
    """
    Analyze the distribution of energies in an HDF5 file.
    Returns statistics and indices within the specified range.
    """
    try:
        with h5py.File(filename, 'r') as f:
            if key not in f:
                print(f"Warning: Key '{key}' not found in {filename}")
                return None, []
            
            energies = f[key][:]
            
            # Basic statistics
            stats = {
                'min': np.min(energies),
                'max': np.max(energies),
                'mean': np.mean(energies),
                'median': np.median(energies),
                'std': np.std(energies),
                'total_samples': len(energies)
            }
            
            # Find indices within range
            indices = np.where((energies >= min_energy) & (energies <= max_energy))[0]
            
            return stats, indices
    except Exception as e:
        print(f"Error analyzing {filename}: {e}")
        return None, []

# Analyze all data files
results = {}
total_training_samples = 0
total_test_samples = 0

for name, path in data_paths.items():
    print(f"\nAnalyzing {name} ({path})...")
    stats, indices = analyze_energy_distribution(path)
    
    if stats:
        results[name] = {
            'stats': stats,
            'indices': indices,
            'count_in_range': len(indices),
            'percentage_in_range': (len(indices) / stats['total_samples']) * 100
        }
        
        print(f"Energy range: {stats['min']:.2f} - {stats['max']:.2f} eV")
        print(f"Mean energy: {stats['mean']:.2f} eV, Std: {stats['std']:.2f} eV")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Samples in {min_energy}-{max_energy} eV range: {len(indices)} ({results[name]['percentage_in_range']:.2f}%)")
        
        if 'train' in name:
            total_training_samples += len(indices)
        elif name == 'test':
            total_test_samples = len(indices)

print(f"\nTotal training samples in range: {total_training_samples}")
print(f"Total test samples in range: {total_test_samples}")

# Create visualizations
plt.figure(figsize=(15, 10))

# Plot 1: Histogram of all energy distributions
plt.subplot(2, 2, 1)
for name, result in results.items():
    if result['stats']:
        with h5py.File(data_paths[name], 'r') as f:
            energies = f['energy0'][:]
            plt.hist(energies, bins=50, alpha=0.5, label=name)

plt.axvspan(min_energy, max_energy, color='red', alpha=0.2)
plt.axvline(min_energy, color='red', linestyle='--')
plt.axvline(max_energy, color='red', linestyle='--')
plt.title('Energy Distribution Across All Datasets')
plt.xlabel('Energy (eV)')
plt.ylabel('Count')
plt.legend()

# Plot 2: Zoom in on the 105-106 eV range
plt.subplot(2, 2, 2)
for name, result in results.items():
    if result['stats']:
        with h5py.File(data_paths[name], 'r') as f:
            energies = f['energy0'][:]
            mask = (energies >= min_energy - 5) & (energies <= max_energy + 5)
            plt.hist(energies[mask], bins=50, alpha=0.5, label=name)

plt.axvspan(min_energy, max_energy, color='red', alpha=0.2)
plt.axvline(min_energy, color='red', linestyle='--')
plt.axvline(max_energy, color='red', linestyle='--')
plt.title(f'Energy Distribution Around {min_energy}-{max_energy} eV Range')
plt.xlabel('Energy (eV)')
plt.ylabel('Count')
plt.legend()

# Plot 3: Bar chart of sample counts
plt.subplot(2, 2, 3)
names = []
counts = []
percentages = []

for name, result in results.items():
    names.append(name)
    counts.append(result['count_in_range'])
    percentages.append(result['percentage_in_range'])

x = np.arange(len(names))
plt.bar(x, counts)
plt.xticks(x, names)
plt.title(f'Sample Count in {min_energy}-{max_energy} eV Range')
plt.xlabel('Dataset')
plt.ylabel('Number of Samples')

for i, count in enumerate(counts):
    plt.text(i, count + 5, str(count), ha='center')

# Plot 4: Bar chart of percentages
plt.subplot(2, 2, 4)
plt.bar(x, percentages)
plt.xticks(x, names)
plt.title(f'Percentage of Samples in {min_energy}-{max_energy} eV Range')
plt.xlabel('Dataset')
plt.ylabel('Percentage (%)')

for i, pct in enumerate(percentages):
    plt.text(i, pct + 0.5, f"{pct:.2f}%", ha='center')

plt.tight_layout()
plt.savefig('./mono_energy_validation/energy_distribution_analysis.png')
plt.close()

# Save detailed results
if total_training_samples > 0:
    print("\nData appears valid for monoenergetic training!")
    
    # Inspect a sample within the range
    for name, result in results.items():
        if len(result['indices']) > 0:
            sample_idx = result['indices'][0]
            print(f"\nInspecting sample with ID {sample_idx} from {name}:")
            
            with h5py.File(data_paths[name], 'r') as f:
                energy = f['energy0'][sample_idx]
                geometry = np.transpose(f['geometry'][:,:,:,sample_idx])
                dose = np.transpose(f['dose0'][:,:,:,sample_idx])
                
                # Get the shapes
                geometry_shape = geometry.shape
                dose_shape = dose.shape
                
                print(f"Energy: {energy:.2f} eV")
                print(f"Geometry shape: {geometry_shape}")
                print(f"Dose shape: {dose_shape}")
                
                # Use the middle slice for each dimension
                mid_x = geometry_shape[1] // 2
                
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 2, 1)
                plt.imshow(geometry[:,mid_x,:], cmap='gray')
                plt.title(f'Geometry (Sample {sample_idx}, Energy {energy:.2f} eV)')
                plt.colorbar()
                
                plt.subplot(1, 2, 2)
                plt.imshow(dose[:,mid_x,:], cmap='jet')
                plt.title(f'Dose (Sample {sample_idx}, Energy {energy:.2f} eV)')
                plt.colorbar()
                
                plt.tight_layout()
                plt.savefig(f'./mono_energy_validation/sample_{sample_idx}_from_{name}.png')
                plt.close()
                
                # Only do one sample for brevity
                break
else:
    print("\nWarning: No training samples found in the specified energy range!")
    print("You may need to adjust the energy range or check the data files.")

print("\nValidation completed. Results saved to ./mono_energy_validation/")
print(f"Recommended energy range for normalization: e_min={min_energy}, e_max={max_energy}") 