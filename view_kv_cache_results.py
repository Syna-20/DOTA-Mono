#!/usr/bin/env python
# coding: utf-8

# Script to view KV caching analysis results
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def display_performance_comparison():
    """Display the performance comparison bar chart."""
    img = mpimg.imread('./kv_cache_analysis/performance_comparison.png')
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Performance Comparison')
    plt.show()

def display_per_sample_comparison():
    """Display the per-sample comparison line chart."""
    img = mpimg.imread('./kv_cache_analysis/per_sample_comparison.png')
    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Per-Sample Comparison')
    plt.show()

def display_batch_size_comparison():
    """Display the batch size comparison bar chart."""
    img = mpimg.imread('./kv_cache_analysis/batch_size_speedup.png')
    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Batch Size Comparison')
    plt.show()

def display_slice_comparisons():
    """Display the slice comparisons for all samples."""
    slice_files = [f for f in os.listdir('./kv_cache_analysis') if f.startswith('slice_comparison_')]
    
    for slice_file in slice_files:
        sample_id = slice_file.split('_')[-1].split('.')[0]
        img = mpimg.imread(f'./kv_cache_analysis/{slice_file}')
        plt.figure(figsize=(15, 5))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Slice Comparison - Sample {sample_id}')
        plt.show()

def display_beam_comparisons():
    """Display the beam comparisons for all samples."""
    beam_files = [f for f in os.listdir('./kv_cache_analysis') if f.startswith('beam_comparison_')]
    
    for beam_file in beam_files:
        sample_id = beam_file.split('_')[-1].split('.')[0]
        img = mpimg.imread(f'./kv_cache_analysis/{beam_file}')
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Beam Comparison - Sample {sample_id}')
        plt.show()

def load_benchmark_results():
    """Load and display the benchmark results."""
    results = np.load('./kv_cache_analysis/benchmark_results.npz', allow_pickle=True)
    
    times_without_cache = results['times_without_cache']
    times_with_cache = results['times_with_cache']
    
    avg_without_cache = np.mean(times_without_cache)
    avg_with_cache = np.mean(times_with_cache)
    speedup = avg_without_cache / avg_with_cache
    
    print("=== Benchmark Results ===")
    print(f"Average time without KV cache: {avg_without_cache:.4f} seconds")
    print(f"Average time with KV cache: {avg_with_cache:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    print()
    
    # First sample speedup
    first_without_cache = times_without_cache[0]
    first_with_cache = times_with_cache[0]
    first_speedup = first_without_cache / first_with_cache
    
    print(f"First sample without KV cache: {first_without_cache:.4f} seconds")
    print(f"First sample with KV cache: {first_with_cache:.4f} seconds")
    print(f"First sample speedup: {first_speedup:.2f}x")
    print()
    
    # Subsequent samples
    subsequent_without_cache = np.mean(times_without_cache[1:])
    subsequent_with_cache = np.mean(times_with_cache[1:])
    subsequent_speedup = subsequent_without_cache / subsequent_with_cache
    
    print(f"Subsequent samples without KV cache: {subsequent_without_cache:.4f} seconds")
    print(f"Subsequent samples with KV cache: {subsequent_with_cache:.4f} seconds")
    print(f"Subsequent samples speedup: {subsequent_speedup:.2f}x")

def main():
    """Main function to display all results."""
    print("KV Caching Analysis Results")
    print("==========================")
    print()
    
    # Load benchmark results
    load_benchmark_results()
    print("\nPress Enter to continue to visualizations...")
    input()
    
    # Display performance comparisons
    print("\nDisplaying performance comparison...")
    display_performance_comparison()
    
    print("\nDisplaying per-sample comparison...")
    display_per_sample_comparison()
    
    print("\nDisplaying batch size comparison...")
    display_batch_size_comparison()
    
    # Ask if user wants to see all visualizations
    print("\nDo you want to see all slice and beam comparisons? (y/n)")
    response = input().lower()
    
    if response == 'y':
        print("\nDisplaying slice comparisons...")
        display_slice_comparisons()
        
        print("\nDisplaying beam comparisons...")
        display_beam_comparisons()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 