#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import h5py
from src.models import dota_energies
from src.generators import DataGenerator
from src.preprocessing import DataRescaler

# Load hyperparameters
with open('./hyperparam.json', 'r') as f:
    hyperparam = json.load(f)

# Load data files
train_file = './data/training/train.h5'
test_file = './data/training/test.h5'

# Set checkpoint path
checkpoint_path = './weights/ckpt/mono_weights.ckpt'

# Find indices for mono-energetic samples in both energy0 and energy1
with h5py.File(train_file, 'r') as f:
    energy0 = f['energy0'][:]
    energy1 = f['energy1'][:]
    
    # Find indices for energy0
    indices0 = np.where((energy0 >= 105) & (energy0 <= 106))[0]
    print(f"Found {len(indices0)} samples in energy0 in range (105, 106)")
    
    # Find indices for energy1
    indices1 = np.where((energy1 >= 105) & (energy1 <= 106))[0]
    print(f"Found {len(indices1)} samples in energy1 in range (105, 106)")
    
    # Combine indices
    indices = np.unique(np.concatenate([indices0, indices1]))
    print(f"Total unique samples: {len(indices)}")

# Split into training and validation sets
np.random.shuffle(indices)
train_size = int(0.9 * len(indices))
trainIDs = indices[:train_size]
valIDs = indices[train_size:]

print(f"Training samples: {len(trainIDs)}")
print(f"Validation samples: {len(valIDs)}")

# Calculate normalization constants
rescaler = DataRescaler('./data/training/', filename=train_file)
rescaler.fit(inputs=True, outputs=True)
scale = {
    'x_min': rescaler.x_min,
    'x_max': rescaler.x_max,
    'y_min': rescaler.y_min,
    'y_max': rescaler.y_max,
    'e_min': 105,
    'e_max': 106
}

# Initialize data generators
batch_size = 8
train_gen = DataGenerator(trainIDs, batch_size, train_file, scale, num_energies=2)
val_gen = DataGenerator(valIDs, batch_size, train_file, scale, num_energies=2)

# Define and compile model
model = dota_energies(hyperparam)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Define callbacks
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
lr_scheduler = LearningRateScheduler(lambda epoch: 0.001 * (0.1 ** (epoch // 10)))

# Train model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[checkpoint, lr_scheduler]
)

# Save final weights and history
model.save_weights('./weights/mono_weights.ckpt')
np.savez('./weights/mono_training_history.npz', history=history.history) 