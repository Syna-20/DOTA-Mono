# -*- coding: utf-8 -*-
# Models of the data-driven dose calculator.
# COPYRIGHT: TU Delft, Netherlands. 2021.
import numpy as np
from tensorflow.keras import Sequential, layers, Model
from blocks import ConvEncoder, ConvDecoder, TransformerEncoder

def dota_energies(num_tokens, input_shape, projection_dim,
    num_heads, num_transformers, kernel_size, dropout_rate=0.2,
    causal=True):
    """ Creates the transformer model for dose calculation using multiple
    energies and patients."""

    # Input CT values
    inputs = layers.Input((num_tokens-1, *input_shape))

    # Input energies
    energies = layers.Input((1))

    # Encode inputs + positional embedding
    tokens = ConvEncoder(
        num_tokens,
        projection_dim,
        kernel_size=kernel_size, 
    )(inputs, energies)
    
    # Stack transformer encoders
    transformer_blocks = []
    for i in range(num_transformers):
        # Transformer encoder blocks.
        transformer_block = TransformerEncoder(
            num_heads, 
            num_tokens, 
            projection_dim,
            causal=causal,
            dropout_rate=dropout_rate,
        )
        transformer_blocks.append(transformer_block)
        tokens = transformer_block(tokens)

    # Decode and upsample
    outputs = ConvDecoder(
        projection_dim, 
        kernel_size=kernel_size,
    )(tokens)

    model = Model(inputs=[inputs, energies], outputs=outputs)
    
    # Add methods to enable/disable KV caching
    def enable_kv_cache(self, enable=True):
        """Enable or disable KV caching for all transformer blocks."""
        for block in transformer_blocks:
            block.enable_caching(enable)
        return self
    
    def reset_kv_cache(self):
        """Reset KV cache for all transformer blocks."""
        for block in transformer_blocks:
            block.reset_cache()
        return self
    
    # Add methods to the model
    model.enable_kv_cache = enable_kv_cache.__get__(model)
    model.reset_kv_cache = reset_kv_cache.__get__(model)
    
    return model
