# KV Caching for DoTA (Dose calculation via Transformers)

This document explains the implementation of Key-Value (KV) caching in the DoTA (Dose calculation via Transformers) model to improve inference performance for proton dose calculations.

## What is KV Caching?

KV caching is an optimization technique for transformer models that stores the key (K) and value (V) tensors computed during the self-attention mechanism. By caching these tensors, we can avoid redundant computations when processing the same input multiple times, which is particularly beneficial during inference.

In the context of DoTA, KV caching can significantly speed up the inference process by reusing previously computed attention scores, especially when processing multiple similar anatomical structures or beam energies.

## Implementation Details

The KV caching implementation in DoTA consists of the following components:

1. **Modified TransformerEncoder Class**: The `TransformerEncoder` class in `src/blocks.py` has been extended to support KV caching with the following additions:
   - Cache storage for attention scores
   - Methods to enable/disable caching
   - Logic to use cached values during inference

2. **Model Extensions**: The `dota_energies` function in `src/models.py` has been updated to:
   - Track transformer blocks
   - Add methods to enable/disable KV caching across all transformer blocks
   - Add methods to reset the cache

3. **Evaluation Script**: A new script `eval_dota_kv_cache.py` has been added to:
   - Benchmark performance with and without KV caching
   - Verify output correctness
   - Visualize performance improvements

## How to Use KV Caching

### Basic Usage

To use KV caching in your inference code:

```python
# Load the model
transformer = dota_energies(...)
transformer.load_weights(path_weights)

# Enable KV caching
transformer.enable_kv_cache(True)

# Run inference
prediction = transformer.predict([inputs, energies])

# Reset cache when processing a new sample
transformer.reset_kv_cache()
```

### Benchmarking

To benchmark the performance improvement from KV caching:

```bash
python eval_dota_kv_cache.py
```

This script will:
1. Run inference on a set of test samples with and without KV caching
2. Measure and compare inference times
3. Verify that outputs are identical (within numerical precision)
4. Generate performance comparison plots

## Performance Improvements

KV caching typically provides the following benefits:

1. **Reduced Inference Time**: Depending on the model size and input complexity, you can expect a 1.5x-3x speedup in inference time.

2. **Memory-Performance Tradeoff**: KV caching increases memory usage to store the cached values but significantly reduces computation time.

3. **Identical Outputs**: The outputs with and without KV caching should be identical (within numerical precision), as this is an optimization that doesn't affect the model's mathematical behavior.

## Best Practices

1. **Reset Cache Between Samples**: Always reset the cache when processing a new sample to avoid using incorrect cached values.

2. **Memory Management**: Be aware of the increased memory usage when using KV caching, especially for large batch sizes.

3. **Training vs. Inference**: KV caching should only be enabled during inference, not during training.

4. **Batch Processing**: When processing batches, ensure that the cache is properly managed for each sample in the batch.

## Limitations

1. **Memory Usage**: KV caching increases memory usage, which might be a concern for very large models or when processing large batches.

2. **Implementation Specifics**: The current implementation is optimized for the DoTA architecture and may need adjustments for other transformer variants.

## Future Improvements

Potential future improvements to the KV caching implementation include:

1. **Incremental Decoding**: Optimize for incremental processing of new tokens.

2. **Memory Optimization**: Implement more memory-efficient caching strategies.

3. **Batch-Aware Caching**: Enhance the caching mechanism to better handle batch processing.

4. **Dynamic Cache Management**: Implement strategies to dynamically enable/disable caching based on input characteristics. 