# Monoenergetic DoTA Model Training and Evaluation

This directory contains scripts for training and evaluating a specialized DoTA (Dose Transformers for Accelerated dose calculations) model trained exclusively on monoenergetic proton beams in the 105-106 eV energy range.

## Overview

The monoenergetic DoTA model is a variant of the standard DoTA model that focuses on a narrow energy range of 105-106 eV, enabling more precise dose calculations for this specific energy window. This specialization can provide several benefits:

1. Better predictive accuracy for the targeted energy range
2. Faster inference times due to reduced model complexity
3. Specialized clinical applications for treatments requiring this energy range

## Files

- `validate_mono_energy_data.py`: Analyzes the available data in the 105-106 eV range and validates the dataset
- `train_mono_energy.py`: Trains the DoTA model using only samples with energy in the 105-106 eV range
- `eval_mono_energy.py`: Evaluates the trained monoenergetic model on test data

## Data Selection

The scripts automatically filter the training and test datasets to include only samples with proton beam energies between 105 and 106 eV. This is done by:

1. Loading the complete dataset
2. Identifying sample indices where the energy falls within the target range
3. Creating filtered training and validation sets using only these indices

## Training Process

The training process for the monoenergetic model follows these steps:

1. Load and filter data from HDF5 files based on energy constraints
2. Initialize data generators for training and validation sets
3. Create a transformer model with the standard DoTA architecture
4. Train the model using the Adam optimizer
5. Apply early stopping to prevent overfitting
6. Save training history and best model weights

## Evaluation

The evaluation script assesses the model's performance by:

1. Loading the trained monoenergetic model
2. Evaluating on test samples within the 105-106 eV range
3. Calculating Mean Absolute Error (MAE) and Mean Squared Error (MSE)
4. Generating visualizations comparing predictions to ground truth
5. Optionally benchmarking inference time with and without KV caching

## Usage

1. **Validate data availability**:
   ```
   python validate_mono_energy_data.py
   ```

2. **Train the monoenergetic model**:
   ```
   python train_mono_energy.py
   ```

3. **Evaluate the trained model**:
   ```
   python eval_mono_energy.py
   ```

## Results

Training results are saved to the `./mono_energy_training/` directory, including:
- Training and validation loss history
- Trained model weights in `./weights/weights_mono_energy.ckpt`

Evaluation results are saved to the `./mono_energy_eval/` directory, including:
- Performance metrics (MAE, MSE)
- Visualizations of model predictions
- Inference time benchmarks

## Extending to Other Energy Ranges

To train and evaluate models for different energy ranges:

1. Modify the `min_energy` and `max_energy` variables in each script
2. Adjust the model name and output directories as needed
3. Run the validation, training, and evaluation scripts with the new parameters

## KV Caching Support

The evaluation script includes optional support for KV caching to accelerate inference. This feature can be enabled or disabled in the evaluation script as needed.

## Requirements

The scripts require the same dependencies as the standard DoTA model:
- TensorFlow 2.x
- NumPy
- h5py
- matplotlib
- json 