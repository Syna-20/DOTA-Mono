import numpy as np
import pandas as pd
import h5py
from pathlib import Path

def print_hdf5_structure(file_path):
    """Print the structure of HDF5 file"""
    with h5py.File(file_path, 'r') as f:
        print("\nHDF5 file structure:")
        print("Keys:", list(f.keys()))
        for key in f.keys():
            try:
                print(f"\n{key}:")
                print(f"Shape: {f[key].shape}")
                print(f"Type: {f[key].dtype}")
            except:
                print(f"{key} is a group with keys:", list(f[key].keys()))

def extract_mono_energy_data(data_path, energy_range=(105, 106), chunk_size=1000):
    """
    Extract data for a specific energy range and create new listIDs
    
    Parameters:
    -----------
    data_path : str or Path
        Path to the data directory
    energy_range : tuple
        (min_energy, max_energy) in eV
    chunk_size : int
        Number of samples to process at once
    """
    data_path = Path(data_path)
    h5_path = data_path / 'data/training/train.h5'
    
    # First, find all indices where energy is in the desired range
    print("Finding indices in energy range...")
    with h5py.File(h5_path, 'r') as f:
        energy0 = f['energy0'][:]
        mask = (energy0 >= energy_range[0]) & (energy0 < energy_range[1])
        indices = np.where(mask)[0]
    
    if len(indices) == 0:
        print(f"No data found in energy range {energy_range[0]}-{energy_range[1]} eV")
        return None
    
    print(f"Found {len(indices)} samples in energy range {energy_range[0]}-{energy_range[1]} eV")
    
    # Create output file
    output_path = data_path / f'mono_energy_{energy_range[0]}_{energy_range[1]}ev.h5'
    
    # Process data in chunks
    with h5py.File(h5_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        # Create datasets in output file with appropriate shapes
        n_samples = len(indices)
        f_out.create_dataset('energy', shape=(n_samples,), dtype=f_in['energy0'].dtype)
        f_out.create_dataset('listID', shape=(n_samples,), dtype=np.int64)
        f_out.create_dataset('dose0', shape=(25, 25, 150, n_samples), dtype=f_in['dose0'].dtype)
        f_out.create_dataset('dose1', shape=(25, 25, 150, n_samples), dtype=f_in['dose1'].dtype)
        
        # Get geometry shape from input file
        geom_shape = f_in['geometry'].shape
        if len(geom_shape) > 1:
            # If geometry has multiple dimensions, keep all but the last and append n_samples
            out_geom_shape = (*geom_shape[:-1], n_samples)
        else:
            # If geometry is 1D, keep it as is
            out_geom_shape = geom_shape
        f_out.create_dataset('geometry', shape=out_geom_shape, dtype=f_in['geometry'].dtype)
        
        # Process data in chunks
        for i in range(0, len(indices), chunk_size):
            chunk_indices = indices[i:i + chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(len(indices) + chunk_size - 1)//chunk_size}")
            
            # Copy data for this chunk
            f_out['energy'][i:i + len(chunk_indices)] = f_in['energy0'][chunk_indices]
            f_out['listID'][i:i + len(chunk_indices)] = chunk_indices
            f_out['dose0'][..., i:i + len(chunk_indices)] = f_in['dose0'][..., chunk_indices]
            f_out['dose1'][..., i:i + len(chunk_indices)] = f_in['dose1'][..., chunk_indices]
            
            if len(geom_shape) > 1:
                f_out['geometry'][..., i:i + len(chunk_indices)] = f_in['geometry'][..., chunk_indices]
            else:
                f_out['geometry'][:] = f_in['geometry'][:]
    
    print(f"\nExtraction complete!")
    print(f"Data saved to: {output_path}")
    print(f"Total samples extracted: {len(indices)}")
    
    return output_path

if __name__ == "__main__":
    # Use the current directory as the data path
    data_path = Path(".")
    filtered_data = extract_mono_energy_data(data_path) 