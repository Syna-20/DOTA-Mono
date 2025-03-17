# DoTA with KV Caching

DoTA (Dose calculation via Transformers) is a transformer-based model for proton dose calculations. This repository includes an implementation of Key-Value (KV) caching to improve inference speed.

## Overview

The DoTA model uses a transformer architecture to predict dose distributions for proton therapy. KV caching is an optimization technique that improves inference speed by storing and reusing attention scores.

## Repository Structure

- `src/`: Source code for the DoTA model
- `data/`: Training and test data (not included in the repository)
- `weights/`: Model weights (not included in the repository)
- `kv_cache_analysis.py`: Script to analyze KV caching performance
- `train_with_kv_cache.py`: Script to train the model with KV caching
- `view_kv_cache_results.py`: Script to view KV caching analysis results
- `KV_CACHE_TRAINING_README.md`: Detailed instructions for training with KV caching
- `KV_CACHING_REPORT.md`: Comprehensive report on KV caching performance

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd dota
   ```

2. Create a virtual environment:
   ```
   conda create -n kv python=3.7
   conda activate kv
   ```
   or
   ```
   python -m venv kv
   source kv/bin/activate  # On Windows: kv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install tensorflow==2.4.1
   pip install tensorflow-addons==0.13.0
   pip install numpy matplotlib h5py
   ```

## Usage

### Training with KV Caching

```
python train_with_kv_cache.py
```

See `KV_CACHE_TRAINING_README.md` for detailed instructions.

### Analyzing KV Caching Performance

```
python kv_cache_analysis.py
```

### Viewing Results

```
python view_kv_cache_results.py
```

## KV Caching Performance

KV caching provides significant performance improvements:

- Overall Speedup: 1.27x faster inference time
- First Sample Speedup: 3.43x faster
- Output Correctness: 100% identical outputs

See `KV_CACHING_REPORT.md` for a comprehensive analysis.

## License

[MIT License](LICENSE)

## Acknowledgments

This project builds upon the original DoTA model for proton dose calculations.
