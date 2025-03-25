import h5py
import numpy as np
import os

def find_energy_indices(filename, energy_range):
    """Find indices of samples within specified energy range."""
    with h5py.File(filename, 'r') as f:
        energies = f['energy0'][:]
        indices = np.where((energies >= energy_range[0]) & (energies <= energy_range[1]))[0]
    return indices

def calculate_minmax(filename, indices, dataset_name):
    """Calculate min and max values for a dataset."""
    with h5py.File(filename, 'r') as f:
        data = f[dataset_name][indices]
        min_val = np.min(data)
        max_val = np.max(data)
    return min_val, max_val

def process_split_file(filename, energy_range, dataset_name):
    """Process a split data file and return min/max values."""
    try:
        indices = find_energy_indices(filename, energy_range)
        if len(indices) > 0:
            return calculate_minmax(filename, indices, dataset_name)
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
    return None, None

def main():
    # Define energy range
    energy_range = (105, 106)
    
    # File paths
    train_files = [
        './data/training/train_part1.h5',
        './data/training/train_part2.h5'
    ]
    test_file = './data/test/test.h5'
    
    # Initialize min/max values
    geom_min = float('inf')
    geom_max = float('-inf')
    dose_min = float('inf')
    dose_max = float('-inf')
    
    # Process training files
    for train_file in train_files:
        if os.path.exists(train_file):
            print(f"Processing {train_file}...")
            gmin, gmax = process_split_file(train_file, energy_range, 'geometry')
            dmin, dmax = process_split_file(train_file, energy_range, 'dose')
            
            if gmin is not None and gmax is not None:
                geom_min = min(geom_min, gmin)
                geom_max = max(geom_max, gmax)
            if dmin is not None and dmax is not None:
                dose_min = min(dose_min, dmin)
                dose_max = max(dose_max, dmax)
    
    # Process test file
    if os.path.exists(test_file):
        print(f"Processing {test_file}...")
        gmin, gmax = process_split_file(test_file, energy_range, 'geometry')
        dmin, dmax = process_split_file(test_file, energy_range, 'dose')
        
        if gmin is not None and gmax is not None:
            geom_min = min(geom_min, gmin)
            geom_max = max(geom_max, gmax)
        if dmin is not None and dmax is not None:
            dose_min = min(dose_min, dmin)
            dose_max = max(dose_max, dmax)
    
    # Save minmax values
    if geom_min != float('inf') and geom_max != float('-inf'):
        np.savetxt('minmax_geometry.txt', np.array([geom_min, geom_max]))
        print(f"Geometry min: {geom_min:.6f}, max: {geom_max:.6f}")
    
    if dose_min != float('inf') and dose_max != float('-inf'):
        np.savetxt('minmax_dose.txt', np.array([dose_min, dose_max]))
        print(f"Dose min: {dose_min:.6f}, max: {dose_max:.6f}")

if __name__ == "__main__":
    main() 