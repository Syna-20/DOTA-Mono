#!/usr/bin/env python
# coding: utf-8

# KV Caching Analysis for DoTA
# This script analyzes the performance of KV caching and generates visualizations

import h5py
import numpy as np
import random
import math
import json
import time
import sys
import os
sys.path.append('./src')
from models import dota_energies
from preprocessing import DataRescaler
from generators import DataGenerator
import matplotlib.pyplot as plt
from tensorflow.config import list_physical_devices
print(list_physical_devices('GPU'))

# Create output directory for visualizations
os.makedirs('kv_cache_analysis', exist_ok=True)

# Load model and data hyperparameters
with open("./hyperparam.json", "r") as hfile:
    param = json.load(hfile)

# Prepare input data
path = "./data/training/"
path_test = "./data/test/"
path_weights = "./weights/weights.ckpt"
filename = path + "train.h5"
filename_test = path_test + "test.h5"
with h5py.File(filename_test, 'r') as fh:
    testIDs = [*range(fh['geometry'].shape[-1])]

# Load normalization constants
scaler = DataRescaler(path, filename=filename)
scaler.load(inputs=True, outputs=True)
scale = {"y_min":scaler.y_min, "y_max":scaler.y_max,
        "x_min":scaler.x_min, "x_max":scaler.x_max,
        "e_min":70, "e_max":220}

# Define and load the transformer
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

# Load weights from checkpoint
transformer.load_weights(path_weights)

def infer_with_kv_cache(model, ID, filename, scale, ikey='geometry', okey='dose', cutoff=0.5, use_cache=False):
    """
    Get model prediction from test sample ID with optional KV caching.
    """
    # Load test sample input and ground truth
    with h5py.File(filename, 'r') as fh:
        geometry = np.expand_dims(np.transpose(fh[ikey][:-1,:-1,:,ID]), axis=(0,-1))
        inputs = (geometry - scale['x_min']) / (scale['x_max'] - scale['x_min'])
        ground_truth = np.transpose(fh[okey+'0'][:-1,:-1,:,ID])
        energies = (fh['energy'+'0'][ID] - scale['e_min']) / (scale['e_max'] - scale['e_min'])

    # Enable or disable KV caching
    if use_cache:
        model.enable_kv_cache(True)
    else:
        model.enable_kv_cache(False)
        
    # Predict dose distribution
    start_time = time.time()
    prediction = model.predict([inputs, np.expand_dims(energies, -1)])
    end_time = time.time()
    
    # Reset cache after prediction
    if use_cache:
        model.reset_kv_cache()
        
    prediction = prediction * (scale['y_max']-scale['y_min']) + scale['y_min']
    prediction[prediction<(cutoff/100)*scale['y_max']] = 0

    return np.squeeze(geometry), np.squeeze(prediction), np.squeeze(ground_truth), end_time - start_time

def benchmark_kv_cache(model, test_ids, filename, scale, num_samples=10):
    """
    Benchmark the performance improvement from KV caching.
    """
    # Select a subset of test IDs for benchmarking
    if num_samples < len(test_ids):
        benchmark_ids = random.sample(test_ids, num_samples)
    else:
        benchmark_ids = test_ids
    
    # Run inference with and without KV caching
    times_without_cache = []
    times_with_cache = []
    
    print("Running inference without KV caching...")
    for i, test_id in enumerate(benchmark_ids):
        _, _, _, inference_time = infer_with_kv_cache(model, test_id, filename, scale, use_cache=False)
        times_without_cache.append(inference_time)
        print(f"Sample {i+1}/{len(benchmark_ids)}: {inference_time:.4f} seconds")
    
    print("\nRunning inference with KV caching...")
    for i, test_id in enumerate(benchmark_ids):
        _, _, _, inference_time = infer_with_kv_cache(model, test_id, filename, scale, use_cache=True)
        times_with_cache.append(inference_time)
        print(f"Sample {i+1}/{len(benchmark_ids)}: {inference_time:.4f} seconds")
    
    # Calculate statistics
    avg_time_without_cache = np.mean(times_without_cache)
    avg_time_with_cache = np.mean(times_with_cache)
    speedup = avg_time_without_cache / avg_time_with_cache
    
    print("\n=== KV Caching Performance Results ===")
    print(f"Average inference time without KV caching: {avg_time_without_cache:.4f} seconds")
    print(f"Average inference time with KV caching: {avg_time_with_cache:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(['Without KV Cache', 'With KV Cache'], [avg_time_without_cache, avg_time_with_cache])
    plt.ylabel('Inference Time (seconds)')
    plt.title('KV Caching Performance Comparison')
    plt.savefig('./kv_cache_analysis/performance_comparison.png')
    
    # Plot individual sample times
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(benchmark_ids) + 1), times_without_cache, 'b-', label='Without KV Cache')
    plt.plot(range(1, len(benchmark_ids) + 1), times_with_cache, 'r-', label='With KV Cache')
    plt.xlabel('Sample Number')
    plt.ylabel('Inference Time (seconds)')
    plt.title('Inference Time Comparison per Sample')
    plt.legend()
    plt.grid(True)
    plt.savefig('./kv_cache_analysis/per_sample_comparison.png')
    
    return times_without_cache, times_with_cache, benchmark_ids

def generate_visualizations(model, test_ids, filename, scale, num_samples=5):
    """
    Generate visualizations of the model output with and without KV caching.
    """
    # Select a subset of test IDs for visualization
    if num_samples < len(test_ids):
        viz_ids = random.sample(test_ids, num_samples)
    else:
        viz_ids = test_ids
    
    for i, test_id in enumerate(viz_ids):
        print(f"Generating visualizations for sample {i+1}/{len(viz_ids)} (ID: {test_id})...")
        
        # Get predictions with and without KV caching
        geometry_no_cache, prediction_no_cache, ground_truth, _ = infer_with_kv_cache(
            model, test_id, filename, scale, use_cache=False
        )
        
        geometry_with_cache, prediction_with_cache, _, _ = infer_with_kv_cache(
            model, test_id, filename, scale, use_cache=True
        )
        
        # Verify outputs are identical
        max_diff = np.max(np.abs(prediction_no_cache - prediction_with_cache))
        print(f"  Maximum difference between outputs: {max_diff:.8f}")
        
        # Generate slice visualizations
        plt.figure(figsize=(15, 5))
        
        # Ground truth
        plt.subplot(1, 3, 1)
        plt.imshow(ground_truth[:, :, ground_truth.shape[2]//2], cmap='jet')
        plt.title('Ground Truth')
        plt.colorbar()
        
        # Without KV cache
        plt.subplot(1, 3, 2)
        plt.imshow(prediction_no_cache[:, :, prediction_no_cache.shape[2]//2], cmap='jet')
        plt.title('Prediction (No KV Cache)')
        plt.colorbar()
        
        # With KV cache
        plt.subplot(1, 3, 3)
        plt.imshow(prediction_with_cache[:, :, prediction_with_cache.shape[2]//2], cmap='jet')
        plt.title('Prediction (With KV Cache)')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(f'./kv_cache_analysis/slice_comparison_{test_id}.png')
        plt.close()
        
        # Generate beam visualizations
        plt.figure(figsize=(15, 10))
        
        # Ground truth
        plt.subplot(2, 2, 1)
        plt.imshow(np.max(ground_truth, axis=2), cmap='jet')
        plt.title('Ground Truth (Max Projection)')
        plt.colorbar()
        
        # Without KV cache
        plt.subplot(2, 2, 2)
        plt.imshow(np.max(prediction_no_cache, axis=2), cmap='jet')
        plt.title('Prediction (No KV Cache)')
        plt.colorbar()
        
        # With KV cache
        plt.subplot(2, 2, 3)
        plt.imshow(np.max(prediction_with_cache, axis=2), cmap='jet')
        plt.title('Prediction (With KV Cache)')
        plt.colorbar()
        
        # Difference
        plt.subplot(2, 2, 4)
        diff = np.abs(prediction_no_cache - prediction_with_cache)
        plt.imshow(np.max(diff, axis=2), cmap='hot')
        plt.title(f'Difference (Max: {np.max(diff):.8f})')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(f'./kv_cache_analysis/beam_comparison_{test_id}.png')
        plt.close()

def analyze_larger_batch_sizes(model, test_ids, filename, scale, batch_sizes=[1, 2, 4, 8]):
    """
    Analyze the performance improvement with different batch sizes.
    """
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n=== Testing batch size: {batch_size} ===")
        
        # Ensure we have enough samples
        num_batches = 5
        total_samples = batch_size * num_batches
        if total_samples > len(test_ids):
            total_samples = len(test_ids)
            num_batches = total_samples // batch_size
        
        batch_ids = test_ids[:total_samples]
        
        # Run without KV cache
        times_without_cache = []
        for i in range(num_batches):
            batch = batch_ids[i*batch_size:(i+1)*batch_size]
            start_time = time.time()
            
            # Disable KV caching
            model.enable_kv_cache(False)
            
            # Process each sample in the batch
            for test_id in batch:
                with h5py.File(filename, 'r') as fh:
                    geometry = np.expand_dims(np.transpose(fh['geometry'][:-1,:-1,:,test_id]), axis=(0,-1))
                    inputs = (geometry - scale['x_min']) / (scale['x_max'] - scale['x_min'])
                    energies = (fh['energy'+'0'][test_id] - scale['e_min']) / (scale['e_max'] - scale['e_min'])
                
                model.predict([inputs, np.expand_dims(energies, -1)])
            
            end_time = time.time()
            times_without_cache.append((end_time - start_time) / batch_size)  # Time per sample
        
        # Run with KV cache
        times_with_cache = []
        for i in range(num_batches):
            batch = batch_ids[i*batch_size:(i+1)*batch_size]
            start_time = time.time()
            
            # Enable KV caching
            model.enable_kv_cache(True)
            
            # Process each sample in the batch
            for test_id in batch:
                with h5py.File(filename, 'r') as fh:
                    geometry = np.expand_dims(np.transpose(fh['geometry'][:-1,:-1,:,test_id]), axis=(0,-1))
                    inputs = (geometry - scale['x_min']) / (scale['x_max'] - scale['x_min'])
                    energies = (fh['energy'+'0'][test_id] - scale['e_min']) / (scale['e_max'] - scale['e_min'])
                
                model.predict([inputs, np.expand_dims(energies, -1)])
                model.reset_kv_cache()  # Reset cache between samples
            
            end_time = time.time()
            times_with_cache.append((end_time - start_time) / batch_size)  # Time per sample
        
        # Calculate statistics
        avg_time_without_cache = np.mean(times_without_cache)
        avg_time_with_cache = np.mean(times_with_cache)
        speedup = avg_time_without_cache / avg_time_with_cache
        
        print(f"Average time without KV cache: {avg_time_without_cache:.4f} seconds per sample")
        print(f"Average time with KV cache: {avg_time_with_cache:.4f} seconds per sample")
        print(f"Speedup: {speedup:.2f}x")
        
        results[batch_size] = {
            'without_cache': avg_time_without_cache,
            'with_cache': avg_time_with_cache,
            'speedup': speedup
        }
    
    # Plot batch size results
    plt.figure(figsize=(12, 6))
    batch_sizes_list = list(results.keys())
    speedups = [results[bs]['speedup'] for bs in batch_sizes_list]
    
    plt.bar(range(len(batch_sizes_list)), speedups, tick_label=[f'Batch {bs}' for bs in batch_sizes_list])
    plt.ylabel('Speedup Factor (x)')
    plt.title('KV Caching Speedup for Different Batch Sizes')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add speedup values on top of bars
    for i, v in enumerate(speedups):
        plt.text(i, v + 0.05, f'{v:.2f}x', ha='center')
    
    plt.savefig('./kv_cache_analysis/batch_size_speedup.png')
    
    return results

# Run the benchmarks
print("\n=== Starting KV Caching Benchmarks ===")
times_without_cache, times_with_cache, benchmark_ids = benchmark_kv_cache(transformer, testIDs, filename_test, scale, num_samples=20)

# Generate visualizations
print("\n=== Generating Visualizations ===")
generate_visualizations(transformer, testIDs, filename_test, scale, num_samples=5)

# Analyze batch sizes
print("\n=== Analyzing Batch Size Impact ===")
batch_results = analyze_larger_batch_sizes(transformer, testIDs, filename_test, scale, batch_sizes=[1, 2, 4, 8])

# Save all results
np.savez('./kv_cache_analysis/benchmark_results.npz', 
         times_without_cache=times_without_cache, 
         times_with_cache=times_with_cache,
         benchmark_ids=benchmark_ids,
         batch_results=batch_results)

print("\nAnalysis complete. Results saved to kv_cache_analysis/ directory.") 