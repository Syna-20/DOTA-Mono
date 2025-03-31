#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
from pathlib import Path

def extract_mono_energetic_data(input_file, output_file, energy_range=(105, 106), chunk_size=50):
    """Extract mono-energetic data from input file and save to output file."""
    print(f"Extracting data from {input_file}")
    print(f"Energy range: {energy_range[0]}-{energy_range[1]} eV")
    
    with h5py.File(input_file, 'r') as f_in:
        # Get energy data
        energy0 = f_in['energy0'][:]
        energy1 = f_in['energy1'][:] if 'energy1' in f_in else None
        
        # Find indices for target energy range
        indices0 = np.where((energy0 >= energy_range[0]) & (energy0 <= energy_range[1]))[0]
        indices1 = np.where((energy1 >= energy_range[0]) & (energy1 <= energy_range[1]))[0] if energy1 is not None else np.array([])
        
        # Combine indices
        indices = np.unique(np.concatenate([indices0, indices1])) if len(indices1) > 0 else indices0
        n_samples = len(indices)
        
        print(f"Found {n_samples} samples in target energy range")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create new HDF5 file with extracted data
        with h5py.File(output_file, 'w') as f_out:
            # Create datasets with chunking for efficient I/O
            dose_chunks = (25, 25, 1, min(chunk_size, n_samples))  # Chunk along z and sample dimensions
            dose_shape = (25, 25, 150, n_samples)
            
            f_out.create_dataset('dose0', shape=dose_shape, dtype=f_in['dose0'].dtype, chunks=dose_chunks)
            if 'dose1' in f_in:
                f_out.create_dataset('dose1', shape=dose_shape, dtype=f_in['dose1'].dtype, chunks=dose_chunks)
            
            # Copy energy data (small enough to do in one go)
            f_out.create_dataset('energy0', data=energy0[indices])
            if energy1 is not None:
                f_out.create_dataset('energy1', data=energy1[indices])
            
            # Process dose data in chunks along the z-dimension and sample dimension
            total_chunks = ((150 + dose_chunks[2] - 1) // dose_chunks[2]) * ((n_samples + dose_chunks[3] - 1) // dose_chunks[3])
            chunk_count = 0
            
            for z_start in range(0, 150, dose_chunks[2]):
                z_end = min(z_start + dose_chunks[2], 150)
                
                for s_start in range(0, n_samples, dose_chunks[3]):
                    s_end = min(s_start + dose_chunks[3], n_samples)
                    chunk_indices = indices[s_start:s_end]
                    
                    chunk_count += 1
                    print(f"Processing chunk {chunk_count}/{total_chunks} "
                          f"(z={z_start}-{z_end}, samples={s_start}-{s_end})")
                    
                    # Copy dose0 chunk
                    f_out['dose0'][:, :, z_start:z_end, s_start:s_end] = \
                        f_in['dose0'][:, :, z_start:z_end, chunk_indices]
                    
                    # Copy dose1 chunk if it exists
                    if 'dose1' in f_in:
                        f_out['dose1'][:, :, z_start:z_end, s_start:s_end] = \
                            f_in['dose1'][:, :, z_start:z_end, chunk_indices]
            
            # Add metadata
            f_out.attrs['energy_range'] = energy_range
            f_out.attrs['total_samples'] = n_samples
            f_out.attrs['source_file'] = os.path.basename(input_file)
            
            print(f"Saved {n_samples} samples to {output_file}")

def main():
    # Create output directory
    output_dir = './mono_energy_validation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract training data
    train_input = './data/training/train.h5'
    train_output = os.path.join(output_dir, 'mono_train.h5')
    extract_mono_energetic_data(train_input, train_output)
    
    # Extract test data
    test_input = './data/test/test.h5'
    test_output = os.path.join(output_dir, 'mono_test.h5')
    extract_mono_energetic_data(test_input, test_output)
    
    print("\nMono-energetic data extraction complete!")
    print(f"Training data saved to: {train_output}")
    print(f"Test data saved to: {test_output}")

if __name__ == "__main__":
    main() 