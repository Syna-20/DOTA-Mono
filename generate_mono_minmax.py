#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np
import os

def find_energy_indices(filename, energy_range, energy_dataset='energy0'):
    """Find indices of samples within specified energy range for a specific energy dataset."""
    try:
        with h5py.File(filename, 'r') as f:
            if energy_dataset not in f:
                print(f"Dataset {energy_dataset} not found in {filename}")
                return np.array([])
                
            energies = f[energy_dataset][:]
            indices = np.where((energies >= energy_range[0]) & (energies <= energy_range[1]))[0]
            print(f"Found {len(indices)} samples in {filename}/{energy_dataset} within energy range {energy_range}")
            return indices
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")
        return np.array([])

def get_geometry_minmax(filename, indices, energy_part='energy0'):
    """Get min and max values for geometry."""
    try:
        with h5py.File(filename, 'r') as f:
            geometry = f['geometry'][..., indices]
            min_val = np.min(geometry)
            max_val = np.max(geometry)
            return min_val, max_val
    except Exception as e:
        print(f"Error reading geometry from {filename}: {str(e)}")
        return None, None

def get_dose_minmax(filename, indices, energy_part='energy0'):
    """Get min and max values for dose."""
    try:
        with h5py.File(filename, 'r') as f:
            # Use dose0 for energy0, dose1 for energy1
            dose_dataset = 'dose0' if energy_part == 'energy0' else 'dose1'
            if dose_dataset not in f:
                print(f"Dataset {dose_dataset} not found in {filename}")
                return None, None
                
            dose = f[dose_dataset][..., indices]
            min_val = np.min(dose)
            max_val = np.max(dose)
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
    
    # Create output directory if it doesn't exist
    os.makedirs('./mono_minmax', exist_ok=True)
    
    # File paths
    train_file = './data/training/train.h5'
    test_file = './data/test/test.h5'
    
    # Print dataset names from files
    print_dataset_names(train_file)
    print_dataset_names(test_file)
    
    # Initialize arrays for storing min/max values
    geom_mins = []
    geom_maxs = []
    dose_mins = []
    dose_maxs = []
    
    # Process training file - energy0
    total_samples = 0
    if os.path.exists(train_file):
        # Process energy0
        indices = find_energy_indices(train_file, energy_range, 'energy0')
        total_samples += len(indices)
        
        if len(indices) > 0:
            # Get geometry min/max
            gmin, gmax = get_geometry_minmax(train_file, indices, 'energy0')
            if gmin is not None:
                geom_mins.append(gmin)
                geom_maxs.append(gmax)
                print(f"Geometry range for {train_file}/energy0: {gmin:.6f} to {gmax:.6f}")
            
            # Get dose min/max
            dmin, dmax = get_dose_minmax(train_file, indices, 'energy0')
            if dmin is not None:
                dose_mins.append(dmin)
                dose_maxs.append(dmax)
                print(f"Dose range for {train_file}/energy0: {dmin:.6f} to {dmax:.6f}")
        
        # Process energy1 if it exists
        with h5py.File(train_file, 'r') as f:
            if 'energy1' in f:
                indices = find_energy_indices(train_file, energy_range, 'energy1')
                total_samples += len(indices)
                
                if len(indices) > 0:
                    # Get geometry min/max
                    gmin, gmax = get_geometry_minmax(train_file, indices, 'energy1')
                    if gmin is not None:
                        geom_mins.append(gmin)
                        geom_maxs.append(gmax)
                        print(f"Geometry range for {train_file}/energy1: {gmin:.6f} to {gmax:.6f}")
                    
                    # Get dose min/max
                    dmin, dmax = get_dose_minmax(train_file, indices, 'energy1')
                    if dmin is not None:
                        dose_mins.append(dmin)
                        dose_maxs.append(dmax)
                        print(f"Dose range for {train_file}/energy1: {dmin:.6f} to {dmax:.6f}")
    
    # Process test file
    if os.path.exists(test_file):
        indices = find_energy_indices(test_file, energy_range, 'energy0')
        total_samples += len(indices)
        
        if len(indices) > 0:
            # Get geometry min/max
            gmin, gmax = get_geometry_minmax(test_file, indices, 'energy0')
            if gmin is not None:
                geom_mins.append(gmin)
                geom_maxs.append(gmax)
                print(f"Geometry range for {test_file}: {gmin:.6f} to {gmax:.6f}")
            
            # Get dose min/max
            dmin, dmax = get_dose_minmax(test_file, indices, 'energy0')
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
        
        # Also save as NPZ for compatibility
        np.savez('./mono_minmax/mono_minmax.npz',
                 geo_min=geom_min,
                 geo_max=geom_max,
                 dose_min=min(dose_mins) if dose_mins else 0,
                 dose_max=max(dose_maxs) if dose_maxs else 1)
    
    if dose_mins and dose_maxs:
        dose_min = min(dose_mins)
        dose_max = max(dose_maxs)
        print(f"Overall dose min: {dose_min:.6f}, max: {dose_max:.6f}")
        np.savetxt('mono_minmax_dose.txt', [dose_min, dose_max])
    
    print(f"\nTotal samples in energy range {energy_range}: {total_samples}")

if __name__ == '__main__':
    main() 