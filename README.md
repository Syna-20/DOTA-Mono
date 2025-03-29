# DOTA-Mono: Mono-energetic Dose Calculation using Transformer Architecture

This repository contains the implementation of DOTA-Mono, a transformer-based model for calculating dose distributions in mono-energetic scenarios (105-106 eV range).

## System Requirements

### Hardware
- GPU: NVIDIA Tesla V100-SXM2-32GB (or equivalent)
- CUDA Version: 12.8
- Driver Version: 570.86.15 or higher

### Software Dependencies
- Python 3.8+
- TensorFlow 2.x
- CUDA Toolkit 12.8
- cuDNN compatible with CUDA 12.8

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Syna-20/DOTA-Mono.git
cd DOTA-Mono
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
DOTA-Mono/
├── data/
│   └── training/
│       └── train.h5
├── src/
│   ├── generators.py
│   ├── models.py
│   └── preprocessing.py
├── weights/
│   └── ckpt/
├── hyperparam.json
├── train_mono_dota.py
├── validate_mono_data.py
└── generate_mono_minmax.py
```

## Usage

### 1. Data Validation
First, validate the mono-energetic data in the 105-106 eV range:
```bash
python validate_mono_data.py
```
This will create visualizations and statistics in the `mono_energy_validation/` directory.

### 2. Generate Min-Max Files
Generate normalization files for the mono-energetic data:
```bash
python generate_mono_minmax.py
```
This will create min-max files in the `mono_minmax/` directory.

### 3. Train the Model
Train the DOTA-Mono model:
```bash
python train_mono_dota.py
```
The script will:
- Load mono-energetic samples (105-106 eV)
- Split data into training (90%) and validation (10%) sets
- Train the model for 30 epochs
- Save checkpoints and training history

## Model Architecture

The model uses a transformer architecture with the following key components:
- Multi-head attention mechanism
- Position-wise feed-forward networks
- Layer normalization
- Residual connections

## Training Parameters

- Batch size: 8
- Number of epochs: 30
- Learning rate: 0.001
- Weight decay: 0.0001
- Energy range: 105-106 eV

## Output Files

- Model checkpoints: `weights/ckpt/mono_weights.ckpt`
- Final weights: `weights/mono_weights.ckpt`
- Training history: `weights/mono_training_history.npz`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:
```
@misc{dota-mono2024,
  author = {Syna},
  title = {DOTA-Mono: Mono-energetic Dose Calculation using Transformer Architecture},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Syna-20/DOTA-Mono}
}
```
