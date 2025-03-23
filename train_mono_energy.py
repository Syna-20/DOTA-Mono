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

# Function to find indices within energy range
def find_energy_indices(filename, energy_key='energy0', min_energy=105, max_energy=106):
    with h5py.File(filename, 'r') as fh:
        energies = fh[energy_key][:]
        indices = np.where((energies >= min_energy) & (energies <= max_energy))[0]
        print(f"Found {len(indices)} samples in range {min_energy}-{max_energy} eV in {filename}")
        return indices.tolist()

# Get indices from each file
try:
    train_part1_indices = find_energy_indices(filename_train_part1)
except:
    print(f"Warning: Could not process {filename_train_part1}")
    train_part1_indices = []
    
try:
    train_part2_indices = find_energy_indices(filename_train_part2)
except:
    print(f"Warning: Could not process {filename_train_part2}")
    train_part2_indices = []
    
try:
    test_indices = find_energy_indices(filename_test)
    print(f"Test samples in range: {len(test_indices)}")
except:
    print(f"Warning: Could not process {filename_test}")
    test_indices = []

# Combine all training indices
all_train_indices = train_part1_indices + train_part2_indices
if not all_train_indices:
    raise ValueError("No training samples found in the specified energy range!")

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
    'y_min': scaler.y_min, 
    'y_max': scaler.y_max,
    'x_min': scaler.x_min, 
    'x_max': scaler.x_max,
    'e_min': 105,  # Override energy min for targeted range
    'e_max': 106   # Override energy max for targeted range
}

# The file to use for training (choose the one with most samples)
training_file = filename_train_part1 if len(train_part1_indices) >= len(train_part2_indices) else filename_train_part2

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
import matplotlib.pyplot as plt

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