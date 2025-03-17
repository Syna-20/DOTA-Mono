#!/usr/bin/env python
# coding: utf-8

# Transformer Dose Calculation with KV Caching
# This script evaluates the performance improvement from KV caching

import h5py
import numpy as np
import random
import math
import json
import time
import sys
sys.path.append('./src')
from models import dota_energies
from preprocessing import DataRescaler
from generators import DataGenerator
from evaluation import infer, from_file
from plot import plot_slice, plot_beam
import matplotlib.pyplot as plt
from tensorflow.config import list_physical_devices
print(list_physical_devices('GPU'))

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
    plt.savefig('./kv_cache_performance.png')
    plt.show()
    
    return times_without_cache, times_with_cache

def verify_output_correctness(model, test_id, filename, scale):
    """
    Verify that the outputs with and without KV caching are identical.
    """
    # Run inference without KV caching
    _, pred_without_cache, ground_truth, _ = infer_with_kv_cache(model, test_id, filename, scale, use_cache=False)
    
    # Run inference with KV caching
    _, pred_with_cache, _, _ = infer_with_kv_cache(model, test_id, filename, scale, use_cache=True)
    
    # Compare outputs
    is_identical = np.allclose(pred_without_cache, pred_with_cache, rtol=1e-5, atol=1e-5)
    max_diff = np.max(np.abs(pred_without_cache - pred_with_cache))
    
    print("\n=== Output Correctness Verification ===")
    print(f"Outputs are identical: {is_identical}")
    print(f"Maximum absolute difference: {max_diff:.8f}")
    
    return is_identical, max_diff

# Run the benchmarks
print("\n=== Starting KV Caching Benchmarks ===")
# Verify output correctness first
verify_output_correctness(transformer, testIDs[0], filename_test, scale)

# Run performance benchmark
times_without_cache, times_with_cache = benchmark_kv_cache(transformer, testIDs, filename_test, scale, num_samples=20)

# Save results
np.savez('./kv_cache_benchmark_results.npz', 
         times_without_cache=times_without_cache, 
         times_with_cache=times_with_cache)

print("\nBenchmark results saved to kv_cache_benchmark_results.npz")
print("KV caching performance plot saved to kv_cache_performance.png") 