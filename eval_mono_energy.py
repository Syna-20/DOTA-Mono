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

sys.path.append('./src')
from models import dota_energies
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

# Function to find indices within energy range
def find_energy_indices(filename, energy_key='energy0', min_energy=105, max_energy=106):
    with h5py.File(filename, 'r') as fh:
        energies = fh[energy_key][:]
        indices = np.where((energies >= min_energy) & (energies <= max_energy))[0]
        print(f"Found {len(indices)} samples in range {min_energy}-{max_energy} eV in {filename}")
        return indices.tolist()

# Get test indices in the 105-106 eV range
test_indices = find_energy_indices(filename_test)
if not test_indices:
    raise ValueError("No test samples found in the 105-106 eV range!")

# Load normalization constants
scaler = DataRescaler(path, filename=path + "train_part1.h5")
scaler.load(inputs=True, outputs=True)
scale = {
    "y_min": scaler.y_min, 
    "y_max": scaler.y_max,
    "x_min": scaler.x_min, 
    "x_max": scaler.x_max,
    "e_min": 105,  # Override energy min for targeted range
    "e_max": 106   # Override energy max for targeted range
}

# Define and load the transformer model
transformer = dota_energies(
    num_tokens=param["num_tokens"],
    input_shape=param["data_shape"],
    projection_dim=param["projection_dim"],
    num_heads=param["num_heads"],
    num_transformers=param["num_transformers"], 
    kernel_size=param["kernel_size"],
    causal=True
)
transformer.summary()

# Load the monoenergetic model weights
try:
    transformer.load_weights(path_mono_weights)
    print(f"Successfully loaded weights from {path_mono_weights}")
except:
    print(f"Warning: Could not load weights from {path_mono_weights}")
    print("Using randomly initialized weights - results will be poor!")

# Function to evaluate the model
def evaluate_mono_model(model, testIDs, filename, scale):
    """
    Evaluate the monoenergetic model on test samples.
    """
    mae_values = []
    mse_values = []
    inference_times = []
    
    print(f"Evaluating model on {len(testIDs)} test samples...")
    
    # Evaluate on a subset of test samples for visualization
    vis_samples = min(5, len(testIDs))
    for i in range(vis_samples):
        # Infer and measure time
        start_time = time.time()
        inputs, prediction, ground_truth = infer(model, testIDs[i], filename, scale)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Calculate metrics
        mae = np.mean(np.abs(prediction - ground_truth))
        mse = np.mean(np.square(prediction - ground_truth))
        mae_values.append(mae)
        mse_values.append(mse)
        
        print(f"Sample {i+1}/{vis_samples} - MAE: {mae:.6f}, MSE: {mse:.6f}, Time: {inference_time:.4f}s")
        
        # Get the middle slice for visualization
        mid_slice = inputs.shape[2] // 2
        
        # Create visualizations
        plt.figure(figsize=(15, 5))
        
        # Plot input geometry
        plt.subplot(1, 3, 1)
        plt.imshow(inputs[:, :, mid_slice], cmap='gray')
        plt.title(f'Input Geometry (Middle Slice {mid_slice})')
        plt.colorbar()
        
        # Plot prediction
        plt.subplot(1, 3, 2)
        plt.imshow(prediction[:, :, mid_slice], cmap='jet')
        plt.title(f'Predicted Dose (Middle Slice {mid_slice})')
        plt.colorbar()
        
        # Plot ground truth
        plt.subplot(1, 3, 3)
        plt.imshow(ground_truth[:, :, mid_slice], cmap='jet')
        plt.title(f'Ground Truth Dose (Middle Slice {mid_slice})')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(f'./mono_energy_eval/sample_{testIDs[i]}_comparison.png')
        plt.close()
        
        # Additional visualizations using specialized plotting functions
        try:
            # Plot beam view
            plot_beam(ground_truth, prediction, title=f"Sample {testIDs[i]}: Ground Truth vs Prediction")
            plt.savefig(f'./mono_energy_eval/sample_{testIDs[i]}_beam.png')
            plt.close()
            
            # Plot slice comparison
            plot_slice(ground_truth, prediction, title=f"Sample {testIDs[i]}: Slice Comparison")
            plt.savefig(f'./mono_energy_eval/sample_{testIDs[i]}_slice.png')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create specialized visualizations: {e}")
    
    # Evaluate on all test samples for metrics
    for i, test_id in enumerate(testIDs):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{len(testIDs)}...")
        
        _, prediction, ground_truth = infer(model, test_id, filename, scale)
        
        mae = np.mean(np.abs(prediction - ground_truth))
        mse = np.mean(np.square(prediction - ground_truth))
        mae_values.append(mae)
        mse_values.append(mse)
    
    # Calculate overall metrics
    avg_mae = np.mean(mae_values)
    avg_mse = np.mean(mse_values)
    avg_inference_time = np.mean(inference_times)
    
    print("\nEvaluation Results:")
    print(f"Average MAE: {avg_mae:.6f}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average Inference Time: {avg_inference_time:.4f}s")
    
    # Save results
    results = {
        'mae': mae_values,
        'mse': mse_values,
        'inference_times': inference_times,
        'avg_mae': avg_mae,
        'avg_mse': avg_mse,
        'avg_inference_time': avg_inference_time
    }
    
    np.savez('./mono_energy_eval/evaluation_results.npz', **results)
    
    # Create summary plots
    plt.figure(figsize=(15, 5))
    
    # Plot MAE distribution
    plt.subplot(1, 3, 1)
    plt.hist(mae_values, bins=20)
    plt.axvline(avg_mae, color='r', linestyle='dashed', linewidth=2)
    plt.title(f'MAE Distribution (Avg: {avg_mae:.6f})')
    plt.xlabel('MAE')
    plt.ylabel('Count')
    
    # Plot MSE distribution
    plt.subplot(1, 3, 2)
    plt.hist(mse_values, bins=20)
    plt.axvline(avg_mse, color='r', linestyle='dashed', linewidth=2)
    plt.title(f'MSE Distribution (Avg: {avg_mse:.6f})')
    plt.xlabel('MSE')
    plt.ylabel('Count')
    
    # Plot inference time
    plt.subplot(1, 3, 3)
    plt.hist(inference_times, bins=10)
    plt.axvline(avg_inference_time, color='r', linestyle='dashed', linewidth=2)
    plt.title(f'Inference Time (Avg: {avg_inference_time:.4f}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('./mono_energy_eval/metrics_summary.png')
    plt.close()
    
    return results

# Run the evaluation
results = evaluate_mono_model(transformer, test_indices, filename_test, scale)

print("\nEvaluation completed!")
print("Results and visualizations saved to ./mono_energy_eval/")

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