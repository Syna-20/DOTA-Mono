#!/usr/bin/env python
# coding: utf-8

# Transformer Dose Calculation for Mono-energetic Data
## Import libraries and define auxiliary functions
import h5py
import json
import random
import sys
sys.path.append('./src')
import numpy as np
from generators import DataGenerator
from models import dota_energies
from preprocessing import DataRescaler
from tensorflow_addons.optimizers import LAMB
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.config import list_physical_devices
print(list_physical_devices('GPU'))

## Define hyperparameters
# Training parameters
batch_size = 8
num_epochs = 30
learning_rate = 0.001
weight_decay = 0.0001

# Load model and data hyperparameters
with open('./hyperparam.json', 'r') as hfile:
    param = json.load(hfile)
    
# Load data files
path = './data/training/'
path_ckpt = './weights/ckpt/mono_weights.ckpt'
filename = path + 'train.h5'

# Get indices for mono-energetic samples (105-106 eV)
with h5py.File(filename, 'r') as fh:
    energy0 = fh['energy0'][:]
    listIDs = np.where((energy0 >= 105) & (energy0 <= 106))[0].tolist()
print(f"Found {len(listIDs)} samples in energy range 105-106 eV")

# Training, validation split
train_split = 0.90
random.seed(333)
random.shuffle(listIDs)
trainIDs = listIDs[:int(round(train_split*len(listIDs)))]
valIDs = listIDs[int(round(train_split*len(listIDs))):]
print(f"Training samples: {len(trainIDs)}, Validation samples: {len(valIDs)}")
    
# Calculate or load normalization constants
scaler = DataRescaler(path, filename=filename)
scaler.load(inputs=True, outputs=True)
scale = {'y_min':scaler.y_min, 'y_max':scaler.y_max,
        'x_min':scaler.x_min, 'x_max':scaler.x_max,
        'e_min':105, 'e_max':106}  # Set energy range to 105-106 eV

print("\nNormalization ranges:")
print(f"Geometry: [{scale['x_min']}, {scale['x_max']}]")
print(f"Dose: [{scale['y_min']}, {scale['y_max']}]")
print(f"Energy: [{scale['e_min']}, {scale['e_max']}]")

# Initialize generators
train_gen = DataGenerator(trainIDs, batch_size, filename, scale, num_energies=1)  # Only use energy0
val_gen = DataGenerator(valIDs, batch_size, filename, scale, num_energies=1)

## Define and train the transformer
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

# Compile the model
optimizer = LAMB(learning_rate=learning_rate, weight_decay_rate=weight_decay)
transformer.compile(optimizer=optimizer, loss='mse', metrics=[])

# Callbacks
# Save best model at the end of the epoch
checkpoint = ModelCheckpoint(
    filepath=path_ckpt,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min')

# Learning rate scheduler. Manually reduce the learning rate
sel_epochs = [4,8,12,16,20,24,28]
lr_scheduler = LearningRateScheduler(
    lambda epoch, lr: lr*0.5 if epoch in sel_epochs else lr,
    verbose=1)

# Train the model
optimizer.learning_rate.assign(learning_rate)
history = transformer.fit(
    x=train_gen,
    validation_data=val_gen,
    epochs=num_epochs,
    verbose=1,
    callbacks=[checkpoint, lr_scheduler]
)

# Save last weights
path_last = './weights/mono_weights.ckpt'
transformer.save_weights(path_last)

# Save training history
history_file = './weights/mono_training_history.npz'
np.savez(history_file,
         loss=history.history['loss'],
         val_loss=history.history['val_loss']) 