import h5py
import numpy as np
import os

def find_energy_indices(filename, energy_range):
    """Find indices of samples within specified energy range."""
    try:
        with h5py.File(filename, 'r') as f:
            energies = f['energy0'][:]
            indices = np.where((energies >= energy_range[0]) & (energies <= energy_range[1]))[0]
            print(f"Found {len(indices)} samples in {filename} within energy range {energy_range}")
            return indices
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")
        return np.array([])

def get_geometry_minmax(filename, indices):
    """Get min and max values for geometry."""
    try:
        with h5py.File(filename, 'r') as f:
            # Geometry shape is (25, 25, 150, N)
            geometry = f['geometry'][..., indices]  # Get all slices for selected indices
            min_val = np.min(geometry)
            max_val = np.max(geometry)
            return min_val, max_val
    except Exception as e:
        print(f"Error reading geometry from {filename}: {str(e)}")
        return None, None

def get_dose_minmax(filename, indices):
    """Get min and max values for dose (combining dose0 and dose1 if available)."""
    try:
        with h5py.File(filename, 'r') as f:
            # Dose shape is (25, 25, 150, N)
            dose0 = f['dose0'][..., indices]  # Get all slices for selected indices
            min_val = np.min(dose0)
            max_val = np.max(dose0)
            
            if 'dose1' in f:
                dose1 = f['dose1'][..., indices]
                min_val = min(min_val, np.min(dose1))
                max_val = max(max_val, np.max(dose1))
            
            return min_val, max_val
    except Exception as e:
        print(f"Error reading dose from {filename}: {str(e)}")
        return None, None

def print_dataset_names(filename):
    """Print all dataset names in the HDF5 file."""
    try:
        with h5py.File(filename, 'r') as f:
            print(f"\nDatasets in {filename}:")
            f.visit(lambda name: print(name))
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")

def main():
    # Define energy range for mono-energy
    energy_range = (105, 106)
    
    # File paths
    train_file = './data/training/train.h5'
    test_file = './data/test/test.h5'
    
    # Print dataset names from training file
    print_dataset_names(train_file)
    
    # Initialize arrays for storing min/max values
    geom_mins = []
    geom_maxs = []
    dose_mins = []
    dose_maxs = []
    
    # Process training file
    total_samples = 0
    if os.path.exists(train_file):
        indices = find_energy_indices(train_file, energy_range)
        total_samples += len(indices)
        
        if len(indices) > 0:
            # Get geometry min/max
            gmin, gmax = get_geometry_minmax(train_file, indices)
            if gmin is not None:
                geom_mins.append(gmin)
                geom_maxs.append(gmax)
                print(f"Geometry range for {train_file}: {gmin:.6f} to {gmax:.6f}")
            
            # Get dose min/max
            dmin, dmax = get_dose_minmax(train_file, indices)
            if dmin is not None:
                dose_mins.append(dmin)
                dose_maxs.append(dmax)
                print(f"Dose range for {train_file}: {dmin:.6f} to {dmax:.6f}")
    
    # Process test file
    if os.path.exists(test_file):
        indices = find_energy_indices(test_file, energy_range)
        total_samples += len(indices)
        
        if len(indices) > 0:
            # Get geometry min/max
            gmin, gmax = get_geometry_minmax(test_file, indices)
            if gmin is not None:
                geom_mins.append(gmin)
                geom_maxs.append(gmax)
                print(f"Geometry range for {test_file}: {gmin:.6f} to {gmax:.6f}")
            
            # Get dose min/max
            dmin, dmax = get_dose_minmax(test_file, indices)
            if dmin is not None:
                dose_mins.append(dmin)
                dose_maxs.append(dmax)
                print(f"Dose range for {test_file}: {dmin:.6f} to {dmax:.6f}")
    
    # Calculate and save overall min/max
    if geom_mins and geom_maxs:
        geom_min = min(geom_mins)
        geom_max = max(geom_maxs)
        print(f"\nOverall geometry min: {geom_min:.6f}, max: {geom_max:.6f}")
        np.savetxt('mono_minmax_geometry.txt', [geom_min, geom_max])
    
    if dose_mins and dose_maxs:
        dose_min = min(dose_mins)
        dose_max = max(dose_maxs)
        print(f"Overall dose min: {dose_min:.6f}, max: {dose_max:.6f}")
        np.savetxt('mono_minmax_dose.txt', [dose_min, dose_max])
    
    print(f"\nTotal samples in energy range {energy_range}: {total_samples}")

if __name__ == "__main__":
    main() 