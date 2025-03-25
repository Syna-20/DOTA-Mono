#!/usr/bin/env python
# coding: utf-8

# Monoenergetic Transformer Dose Calculation (105-106 eV)
# This script trains the DoTA model using only samples with energy in the 105-106 eV range

import h5py
import json
import random
import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow_addons.optimizers import LAMB
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.config import list_physical_devices
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

sys.path.append('./src')
from models import dota_energies
from preprocessing import DataRescaler
from generators import DataGenerator

print("TensorFlow version:", tf.__version__)
print("Available GPUs:", list_physical_devices('GPU'))

# Create output directory for training results
os.makedirs('mono_energy_training', exist_ok=True)

# Training parameters
batch_size = 8
num_epochs = 5  # Reduced from 30 to 5 for faster testing
learning_rate = 0.001
weight_decay = 0.0001

print(f"Training with {num_epochs} epochs for faster testing")

# Load model and data hyperparameters
with open('./hyperparam.json', 'r') as hfile:
    param = json.load(hfile)
    
# Load data files
path = './data/training/'
path_test = './data/test/'
path_mono_weights = './weights/weights_mono_energy.ckpt'
filename_train_part1 = path + 'train_part1.h5'
filename_train_part2 = path + 'train_part2.h5'
filename_test = path_test + 'test.h5'

# Filter IDs based on energy (105-106 eV)
print("Extracting samples with energy between 105-106 eV...")

def find_energy_indices(filename, energy_range):
    """Find indices of samples within specified energy range."""
    with h5py.File(filename, 'r') as f:
        energies = f['energy0'][:]
        indices = np.where((energies >= energy_range[0]) & (energies <= energy_range[1]))[0]
        print(f"Found {len(indices)} samples in {filename} within energy range {energy_range}")
        return indices

def load_data(filename, indices, scale):
    """Load data for specified indices."""
    with h5py.File(filename, 'r') as f:
        geometry = f['geometry'][..., indices]
        dose = f['dose0'][..., indices]
        
        # Normalize data
        geometry = (geometry - scale['geom_min']) / (scale['geom_max'] - scale['geom_min'])
        dose = (dose - scale['dose_min']) / (scale['dose_max'] - scale['dose_min'])
        
        return geometry, dose

def create_data_generator(geometry, dose, batch_size=8):
    """Create a data generator for training."""
    num_samples = geometry.shape[-1]
    indices = np.arange(num_samples)
    
    while True:
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield geometry[..., batch_indices], dose[..., batch_indices]

# Get indices from each file
try:
    train_indices = find_energy_indices('./data/training/train.h5', (105, 106))
    print(f"Training samples in range: {len(train_indices)}")
except:
    print(f"Warning: Could not process ./data/training/train.h5")
    train_indices = np.array([])
    
try:
    test_indices = find_energy_indices('./data/test/test.h5', (105, 106))
    print(f"Test samples in range: {len(test_indices)}")
except:
    print(f"Warning: Could not process ./data/test/test.h5")
    test_indices = np.array([])

if len(train_indices) == 0:
    raise ValueError("No training samples found in the specified energy range!")

# Use all training indices
all_train_indices = train_indices

print(f"Total training samples in energy range 105-106 eV: {len(all_train_indices)}")

# Split into training and validation
random.seed(333)
random.shuffle(all_train_indices)
train_split = 0.90
trainIDs = all_train_indices[:int(round(train_split*len(all_train_indices)))]
valIDs = all_train_indices[int(round(train_split*len(all_train_indices))):]
    
print(f"Training samples: {len(trainIDs)}")
print(f"Validation samples: {len(valIDs)}")

# Calculate or load normalization constants
scaler = DataRescaler(path, filename=filename_train_part1)
scaler.load(inputs=True, outputs=True)
scale = {
    'x_min': float(np.loadtxt('mono_minmax_geometry.txt')[0]),
    'x_max': float(np.loadtxt('mono_minmax_geometry.txt')[1]),
    'y_min': float(np.loadtxt('mono_minmax_dose.txt')[0]),
    'y_max': float(np.loadtxt('mono_minmax_dose.txt')[1]),
    'e_min': 105,  # Override energy min for targeted range
    'e_max': 106   # Override energy max for targeted range
}

# The file to use for training
training_file = './data/training/train.h5'

# Initialize generators
train_gen = DataGenerator(trainIDs, batch_size, training_file, scale, num_energies=1)
val_gen = DataGenerator(valIDs, batch_size, training_file, scale, num_energies=1)

# Define and build the transformer model
transformer = dota_energies(
    num_tokens=param['num_tokens'],
    input_shape=param['data_shape'],
    projection_dim=param['projection_dim'],
    num_heads=param['num_heads'],
    num_transformers=param['num_transformers'],
    kernel_size=param['kernel_size'],
    causal=True
)
transformer.summary()

# Define learning rate schedule
def lr_scheduler(epoch, lr):
    if epoch < 2:  # Adjusted for fewer epochs
        return learning_rate
    elif epoch < 3:  # Adjusted for fewer epochs
        return learning_rate * 0.1
    else:
        return learning_rate * 0.01

# Define callbacks
callbacks = [
    ModelCheckpoint(
        filepath=path_mono_weights,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    ),
    LearningRateScheduler(lr_scheduler, verbose=1),
    EarlyStopping(  # Added early stopping
        monitor='val_loss',
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
]

# Create the weights directory if it doesn't exist
os.makedirs(os.path.dirname(path_mono_weights), exist_ok=True)

# Compile the model
optimizer = LAMB(
    learning_rate=learning_rate, 
    weight_decay_rate=weight_decay
)

transformer.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)

# Train the model
history = transformer.fit(
    train_gen,
    validation_data=val_gen,
    epochs=num_epochs,
    callbacks=callbacks,
    verbose=1
)

# Save training history
np.savez('./mono_energy_training/training_history.npz', 
         loss=history.history['loss'], 
         val_loss=history.history['val_loss'],
         mae=history.history['mae'],
         val_mae=history.history['val_mae'])

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig('./mono_energy_training/training_history.png')
plt.close()

print("Training completed! Results saved to ./mono_energy_training/")
print(f"Best weights saved to {path_mono_weights}") 