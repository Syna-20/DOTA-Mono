# Training DoTA Model with KV Caching

This README explains how to train the DoTA (Dose calculation via Transformers) model with KV caching enabled.

## Overview

Key-Value (KV) caching is an optimization technique that improves inference speed by storing and reusing attention scores. While KV caching is primarily an inference optimization, training the model with KV caching periodically enabled helps ensure the model works effectively with caching during inference.

## Prerequisites

- Python 3.7+
- TensorFlow 2.4.1
- TensorFlow Addons 0.13.0
- NumPy, Matplotlib, h5py

## Training Process

The training script (`train_with_kv_cache.py`) implements a custom training loop that:

1. Periodically enables KV caching during training (every 5th batch)
2. Alternates between using KV cache and not using it during validation
3. Saves the best model weights based on validation loss
4. Generates training history plots

This approach helps the model learn to work effectively with both cached and non-cached modes.

## How to Run

1. Ensure your virtual environment is activated:
   ```
   conda activate kv
   ```
   or
   ```
   source kv/bin/activate
   ```

2. Run the training script:
   ```
   python train_with_kv_cache.py
   ```

3. Monitor the training progress. The script will display:
   - Loss and MAE for each batch
   - Average metrics for each epoch
   - Validation metrics
   - Best model notifications

4. After training completes, the script will:
   - Save the best model weights to `./weights/weights_kv_cache.ckpt`
   - Generate training history plots in `./kv_cache_training/training_history.png`
   - Save training history data to `./kv_cache_training/training_history.npz`
   - Evaluate the final model with and without KV caching

## Training Parameters

The training parameters are loaded from `hyperparam.json`. Key parameters include:

- `batch_size`: Number of samples per batch
- `epochs`: Maximum number of training epochs
- `num_tokens`, `projection_dim`, `num_heads`, etc.: Model architecture parameters

## Output Files

- `./weights/weights_kv_cache.ckpt`: Best model weights
- `./kv_cache_training/training_history.png`: Training history plots
- `./kv_cache_training/training_history.npz`: Training history data

## Notes on KV Caching

- KV caching increases memory usage but reduces computation time during inference
- The model is trained to work with both cached and non-cached modes
- For inference, KV caching is most effective for single-sample inference
- Always reset the cache between samples during inference

## Performance Comparison

After training, you can run the KV caching analysis script to compare performance:

```
python kv_cache_analysis.py
```

This will generate performance comparisons and visualizations in the `kv_cache_analysis/` directory.

## Viewing Results

To view the training results and performance analysis:

```
python view_kv_cache_results.py
```

This interactive script will display:
- Benchmark results
- Performance comparison charts
- Batch size impact analysis
- Output visualizations 