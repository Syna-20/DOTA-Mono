#!/usr/bin/env python
# coding: utf-8

# Monoenergetic DoTA Model Pipeline Runner
# This script runs the complete pipeline for the monoenergetic model:
# 1. Validates the data
# 2. Trains the model
# 3. Evaluates the model

import os
import sys
import time
import subprocess
import argparse

def run_script(script_name, description, timeout=None):
    """Run a Python script and handle errors with optional timeout"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                               check=True, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               universal_newlines=True,
                               timeout=timeout)
        
        print(result.stdout)
        
        if result.stderr:
            print("WARNINGS/ERRORS:")
            print(result.stderr)
            
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f} seconds")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Script {script_name} timed out after {timeout} seconds")
        print("Terminating process...")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to run {script_name}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        elapsed = time.time() - start_time
        print(f"\nFailed after {elapsed:.2f} seconds")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run the monoenergetic DoTA model pipeline')
    parser.add_argument('--skip-validation', action='store_true', 
                        help='Skip the data validation step')
    parser.add_argument('--skip-training', action='store_true', 
                        help='Skip the model training step')
    parser.add_argument('--skip-evaluation', action='store_true', 
                        help='Skip the model evaluation step')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Timeout in seconds for each script (default: 600 seconds/10 minutes)')
    args = parser.parse_args()
    
    print("\nMonoenergetic DoTA Model Pipeline")
    print("=================================\n")
    print(f"Timeout set to {args.timeout} seconds per script")
    
    pipeline_start = time.time()
    success = True
    
    # Step 1: Validate data
    if not args.skip_validation:
        success = run_script('validate_mono_energy_data.py', 'Data Validation', timeout=args.timeout)
        if not success:
            print("WARNING: Data validation failed, but continuing with pipeline")
    else:
        print("Skipping data validation step")
    
    # Step 2: Train model
    if success and not args.skip_training:
        success = run_script('train_mono_energy.py', 'Model Training', timeout=args.timeout)
        if not success:
            print("ERROR: Model training failed, stopping pipeline")
            return
    else:
        print("Skipping model training step")
    
    # Step 3: Evaluate model
    if success and not args.skip_evaluation:
        success = run_script('eval_mono_energy.py', 'Model Evaluation', timeout=args.timeout)
        if not success:
            print("ERROR: Model evaluation failed")
    else:
        print("Skipping model evaluation step")
    
    # Report total time
    pipeline_time = time.time() - pipeline_start
    print(f"\nTotal pipeline execution time: {pipeline_time:.2f} seconds ({pipeline_time/60:.2f} minutes)")
    
    # Create summary report
    if success:
        print("\nGenerating summary report...")
        
        with open('mono_energy_summary.txt', 'w') as f:
            f.write("Monoenergetic DoTA Model Pipeline Summary\n")
            f.write("=========================================\n\n")
            
            f.write(f"Execution completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total execution time: {pipeline_time:.2f} seconds ({pipeline_time/60:.2f} minutes)\n\n")
            
            f.write("Results Locations:\n")
            f.write("- Validation results: ./mono_energy_validation/\n")
            f.write("- Training results: ./mono_energy_training/\n")
            f.write("- Model weights: ./weights/weights_mono_energy.ckpt\n")
            f.write("- Evaluation results: ./mono_energy_eval/\n\n")
            
            f.write("Next Steps:\n")
            f.write("1. Review the validation results to verify data quality\n")
            f.write("2. Check training history plots for convergence\n")
            f.write("3. Analyze evaluation metrics and visualizations\n")
            f.write("4. Compare performance with general DoTA model\n")
        
        print(f"Summary report saved to mono_energy_summary.txt")
    
    print("\nPipeline execution completed.")

if __name__ == "__main__":
    main() 