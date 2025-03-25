#!/usr/bin/env python
# coding: utf-8

# Evaluation of Monoenergetic Transformer Model (105-106 eV)
# This script evaluates the DoTA model trained on 105-106 eV energy range

import h5py
import numpy as np
import json
import time
import sys
import os
import matplotlib.pyplot as plt
from tensorflow.config import list_physical_devices
from dota_energies import dota_energies

sys.path.append('./src')
from preprocessing import DataRescaler
from evaluation import infer
from plot import plot_slice, plot_beam

print("Available GPUs:", list_physical_devices('GPU'))

# Create output directory for evaluation results
os.makedirs('mono_energy_eval', exist_ok=True)

# Load model and data hyperparameters
with open("./hyperparam.json", "r") as hfile:
    param = json.load(hfile)

# Prepare input data paths
path = "./data/training/"
path_test = "./data/test/"
path_mono_weights = "./weights/weights_mono_energy.ckpt"
filename_test = path_test + "test.h5"

def find_energy_indices(filename, energy_range):
    """Find indices of samples within specified energy range."""
    with h5py.File(filename, 'r') as f:
        energies = f['energy0'][:]
        indices = np.where((energies >= energy_range[0]) & (energies <= energy_range[1]))[0]
        print(f"Found {len(indices)} samples in {filename} within energy range {energy_range}")
        return indices

def load_data(filename, indices, scale):
    """Load data for specified indices."""
    with h5py.File(filename, 'r') as f:
        geometry = f['geometry'][..., indices]
        dose = f['dose0'][..., indices]
        
        # Normalize data
        geometry = (geometry - scale['geom_min']) / (scale['geom_max'] - scale['geom_min'])
        dose = (dose - scale['dose_min']) / (scale['dose_max'] - scale['dose_min'])
        
        return geometry, dose

def denormalize_data(data, scale):
    """Denormalize data using the scale parameters."""
    return data * (scale['dose_max'] - scale['dose_min']) + scale['dose_min']

def evaluate_model(model, test_geometry, test_dose, scale, output_dir):
    """Evaluate model on test data and save results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions
    predictions = model.predict(test_geometry)
    
    # Denormalize predictions and ground truth
    predictions = denormalize_data(predictions, scale)
    test_dose = denormalize_data(test_dose, scale)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - test_dose))
    mse = np.mean((predictions - test_dose) ** 2)
    
    print(f"\nEvaluation Results:")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Mean Squared Error: {mse:.6f}")
    
    # Save metrics
    np.savez(f'{output_dir}/metrics.npz', mae=mae, mse=mse)
    
    # Visualize results for a few samples
    num_samples = min(5, predictions.shape[-1])
    for i in range(num_samples):
        plt.figure(figsize=(15, 5))
        
        # Plot input geometry
        plt.subplot(1, 3, 1)
        plt.imshow(test_geometry[..., 75, i], cmap='gray')
        plt.title('Input Geometry')
        plt.colorbar()
        
        # Plot predicted dose
        plt.subplot(1, 3, 2)
        plt.imshow(predictions[..., 75, i], cmap='viridis')
        plt.title('Predicted Dose')
        plt.colorbar()
        
        # Plot ground truth dose
        plt.subplot(1, 3, 3)
        plt.imshow(test_dose[..., 75, i], cmap='viridis')
        plt.title('Ground Truth Dose')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sample_{i}.png')
        plt.close()
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    errors = predictions - test_dose
    plt.hist(errors.flatten(), bins=50, density=True)
    plt.title('Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.savefig(f'{output_dir}/error_distribution.png')
    plt.close()

def main():
    # Create output directory
    output_dir = './mono_energy_eval'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load hyperparameters
    with open('./hyperparam.json', 'r') as f:
        hyperparams = json.load(f)
    
    # Load normalization constants
    scale = {
        'geom_min': float(np.loadtxt('mono_minmax_geometry.txt')[0]),
        'geom_max': float(np.loadtxt('mono_minmax_geometry.txt')[1]),
        'dose_min': float(np.loadtxt('mono_minmax_dose.txt')[0]),
        'dose_max': float(np.loadtxt('mono_minmax_dose.txt')[1])
    }
    
    # Define energy range
    energy_range = (105, 106)
    
    # Load test data
    test_file = './data/test/test.h5'
    test_indices = find_energy_indices(test_file, energy_range)
    test_geometry, test_dose = load_data(test_file, test_indices, scale)
    
    # Create and load model
    transformer = dota_energies(
        num_layers=hyperparams['num_layers'],
        d_model=hyperparams['d_model'],
        num_heads=hyperparams['num_heads'],
        dff=hyperparams['dff'],
        input_vocab_size=hyperparams['input_vocab_size'],
        target_vocab_size=hyperparams['target_vocab_size'],
        maximum_position_encoding=hyperparams['maximum_position_encoding'],
        dropout_rate=hyperparams['dropout_rate']
    )
    
    # Load weights
    try:
        transformer.load_weights('./weights/weights_mono_energy.ckpt')
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        return
    
    # Evaluate model
    evaluate_model(transformer, test_geometry, test_dose, scale, output_dir)
    
    print(f"\nEvaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()

# Optionally use KV caching for comparison
print("\nRunning comparison with KV caching...")

# Run inference with KV caching
transformer.enable_kv_cache(True)

# Evaluate on a few samples
kv_inference_times = []
for i in range(min(5, len(test_indices))):
    start_time = time.time()
    infer(transformer, test_indices[i], filename_test, scale)
    inference_time = time.time() - start_time
    kv_inference_times.append(inference_time)
    transformer.reset_kv_cache()  # Important to reset cache between samples

# Compare inference times
avg_kv_time = np.mean(kv_inference_times)
speedup = results['avg_inference_time'] / avg_kv_time if avg_kv_time > 0 else 0

print(f"Average inference time with KV caching: {avg_kv_time:.4f}s")
print(f"Speedup with KV caching: {speedup:.2f}x")

# Disable KV caching when done
transformer.enable_kv_cache(False) 