#!/usr/bin/env python
# coding: utf-8

# Training DoTA Model with KV Caching
# This script trains the DoTA model with KV caching enabled

import h5py
import numpy as np
import json
import time
import sys
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
sys.path.append('./src')
from models import dota_energies
from preprocessing import DataRescaler
from generators import DataGenerator
import matplotlib.pyplot as plt
from tensorflow.config import list_physical_devices

print("TensorFlow version:", tf.__version__)
print("Available GPUs:", list_physical_devices('GPU'))

# Create output directory for training results
os.makedirs('kv_cache_training', exist_ok=True)

# Load model and data hyperparameters
with open("./hyperparam.json", "r") as hfile:
    param = json.load(hfile)

# Prepare input data paths
path = "./data/training/"
path_test = "./data/test/"
path_weights = "./weights/weights.ckpt"  # Original weights
path_kv_weights = "./weights/weights_kv_cache.ckpt"  # New weights with KV caching
filename = path + "train.h5"
filename_test = path_test + "test.h5"

# Load normalization constants
scaler = DataRescaler(path, filename=filename)
scaler.load(inputs=True, outputs=True)
scale = {"y_min": scaler.y_min, "y_max": scaler.y_max,
         "x_min": scaler.x_min, "x_max": scaler.x_max,
         "e_min": 70, "e_max": 220}

# Define and create the transformer model
transformer = dota_energies(
    num_tokens=param["num_tokens"],
    input_shape=param["data_shape"],
    projection_dim=param["projection_dim"],
    num_heads=param["num_heads"],
    num_transformers=param["num_transformers"],
    kernel_size=param["kernel_size"],
    causal=True
)
transformer.summary()

# Load pre-trained weights if they exist
try:
    transformer.load_weights(path_weights)
    print(f"Loaded weights from {path_weights}")
except:
    print("No pre-trained weights found. Starting from scratch.")

# Create data generators
train_generator = DataGenerator(
    path=path,
    filename=filename,
    batch_size=param["batch_size"],
    shuffle=True,
    scale=scale
)

validation_generator = DataGenerator(
    path=path_test,
    filename=filename_test,
    batch_size=param["batch_size"],
    shuffle=False,
    scale=scale
)

# Define callbacks
callbacks = [
    ModelCheckpoint(
        filepath=path_kv_weights,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
]

# Custom training loop with KV caching
def train_with_kv_cache(model, train_gen, val_gen, epochs=10, initial_epoch=0, callbacks=None):
    """
    Custom training loop that periodically enables KV caching during training.
    This helps the model learn to work effectively with KV caching.
    """
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )
    
    # Initialize training history
    history = {
        'loss': [],
        'mae': [],
        'val_loss': [],
        'val_mae': []
    }
    
    # Get total number of batches
    train_steps = len(train_gen)
    val_steps = len(val_gen)
    
    best_val_loss = float('inf')
    
    for epoch in range(initial_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        start_time = time.time()
        
        # Training phase
        train_loss = 0
        train_mae = 0
        
        for batch_idx in range(train_steps):
            # Get batch data
            X, y = train_gen[batch_idx]
            
            # Enable KV caching for some batches (e.g., every 5th batch)
            # This helps the model learn to work with both cached and non-cached modes
            use_kv_cache = (batch_idx % 5 == 0)
            
            if use_kv_cache:
                model.enable_kv_cache(True)
            
            # Train on batch
            metrics = model.train_on_batch(X, y)
            
            if use_kv_cache:
                model.reset_kv_cache()
                model.enable_kv_cache(False)
            
            # Update metrics
            train_loss += metrics[0]
            train_mae += metrics[1]
            
            # Print progress
            progress = (batch_idx + 1) / train_steps * 100
            print(f"\rTraining: {progress:.1f}% - loss: {metrics[0]:.4f} - mae: {metrics[1]:.4f}", end="")
        
        # Calculate average metrics
        train_loss /= train_steps
        train_mae /= train_steps
        
        # Validation phase
        val_loss = 0
        val_mae = 0
        
        for batch_idx in range(val_steps):
            # Get batch data
            X, y = val_gen[batch_idx]
            
            # Alternate between using KV cache and not using it
            use_kv_cache = (batch_idx % 2 == 0)
            
            if use_kv_cache:
                model.enable_kv_cache(True)
            
            # Evaluate on batch
            metrics = model.test_on_batch(X, y)
            
            if use_kv_cache:
                model.reset_kv_cache()
                model.enable_kv_cache(False)
            
            # Update metrics
            val_loss += metrics[0]
            val_mae += metrics[1]
            
            # Print progress
            progress = (batch_idx + 1) / val_steps * 100
            print(f"\rValidation: {progress:.1f}% - val_loss: {metrics[0]:.4f} - val_mae: {metrics[1]:.4f}", end="")
        
        # Calculate average metrics
        val_loss /= val_steps
        val_mae /= val_steps
        
        # Update history
        history['loss'].append(train_loss)
        history['mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        # Print epoch summary
        time_taken = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{epochs} - {time_taken:.1f}s - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best model! Saving weights to {path_kv_weights}")
            model.save_weights(path_kv_weights)
        
        # Call callbacks
        if callbacks:
            for callback in callbacks:
                if isinstance(callback, ModelCheckpoint):
                    callback.on_epoch_end(epoch, logs={'val_loss': val_loss})
                elif isinstance(callback, ReduceLROnPlateau):
                    callback.on_epoch_end(epoch, logs={'val_loss': val_loss})
                elif isinstance(callback, EarlyStopping):
                    callback.on_epoch_end(epoch, logs={'val_loss': val_loss})
                    if callback.stopped_epoch == epoch:
                        print("Early stopping triggered!")
                        return history
    
    return history

# Train the model with KV caching
print("\n=== Starting Training with KV Caching ===")
history = train_with_kv_cache(
    model=transformer,
    train_gen=train_generator,
    val_gen=validation_generator,
    epochs=param.get("epochs", 50),
    callbacks=callbacks
)

# Plot training history
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history['mae'], label='Training MAE')
plt.plot(history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('./kv_cache_training/training_history.png')

# Save training history
np.savez('./kv_cache_training/training_history.npz', 
         loss=history['loss'], 
         mae=history['mae'],
         val_loss=history['val_loss'],
         val_mae=history['val_mae'])

print("\nTraining complete. Results saved to kv_cache_training/ directory.")

# Evaluate the final model
print("\n=== Evaluating Final Model ===")

# Evaluate without KV caching
transformer.enable_kv_cache(False)
metrics_no_cache = transformer.evaluate(validation_generator, verbose=1)
print(f"Evaluation without KV caching - Loss: {metrics_no_cache[0]:.4f}, MAE: {metrics_no_cache[1]:.4f}")

# Evaluate with KV caching
transformer.enable_kv_cache(True)
metrics_with_cache = transformer.evaluate(validation_generator, verbose=1)
print(f"Evaluation with KV caching - Loss: {metrics_with_cache[0]:.4f}, MAE: {metrics_with_cache[1]:.4f}")

print("\nEvaluation complete. The model has been trained to work with KV caching.") 