#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OliTunes Real Model Architectures
=================================

This module defines TensorFlow model architectures for guitar
tablature position prediction, including LSTM and Transformer models.
"""

import os
import logging
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, Input, Concatenate, 
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D,
    Reshape, Bidirectional, Add, Embedding
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("real_models")

# Guitar constants
STRINGS = 6
FRETS = 24


def create_lstm_model(input_dim, context_window, features_per_note):
    """
    Create an LSTM model for sequence-based position prediction.
    
    Args:
        input_dim: Total input dimensions
        context_window: Number of previous notes to use as context
        features_per_note: Number of features per note in the input
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=(input_dim,))
    
    # Reshape for LSTM
    x = Reshape((context_window, features_per_note))(inputs)
    
    # LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.2)(x)
    
    # Output layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(STRINGS + FRETS, activation='linear')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"Created LSTM model with {context_window} context window")
    return model


def create_transformer_model(input_dim, context_window, features_per_note):
    """
    Create a transformer-based model with attention mechanisms.
    
    Args:
        input_dim: Total input dimensions
        context_window: Number of previous notes to use as context
        features_per_note: Number of features per note in the input
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=(input_dim,))
    
    # Reshape for transformer
    x = Reshape((context_window, features_per_note))(inputs)
    
    # Positional encoding (learning position embeddings)
    position_embedding = Embedding(input_dim=context_window, output_dim=features_per_note)(
        tf.range(start=0, limit=context_window, delta=1)
    )
    
    # Add positional embeddings
    x = x + position_embedding
    
    # Transformer encoder block 1
    x = LayerNormalization(epsilon=1e-6)(x)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=features_per_note)(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed forward network
    forward_output = Dense(128, activation='relu')(x)
    forward_output = Dense(features_per_note)(forward_output)
    x = Add()([x, forward_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Transformer encoder block 2
    attention_output = MultiHeadAttention(num_heads=4, key_dim=features_per_note)(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed forward network
    forward_output = Dense(128, activation='relu')(x)
    forward_output = Dense(features_per_note)(forward_output)
    x = Add()([x, forward_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Global attention pooling
    x = GlobalAveragePooling1D()(x)
    
    # Output layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(STRINGS + FRETS, activation='linear')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"Created transformer-based model with {context_window} context window")
    return model


def create_hybrid_model(input_dim, context_window, features_per_note):
    """
    Create a hybrid model combining LSTM and transformer approaches.
    
    Args:
        input_dim: Total input dimensions
        context_window: Number of previous notes to use as context
        features_per_note: Number of features per note in the input
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=(input_dim,))
    
    # Reshape for sequence processing
    x = Reshape((context_window, features_per_note))(inputs)
    
    # LSTM path
    lstm = Bidirectional(LSTM(64, return_sequences=True))(x)
    lstm = Dropout(0.2)(lstm)
    
    # Transformer path
    norm_x = LayerNormalization(epsilon=1e-6)(x)
    attention = MultiHeadAttention(num_heads=4, key_dim=features_per_note)(norm_x, norm_x)
    attention = LayerNormalization(epsilon=1e-6)(attention)
    
    # Combine paths
    combined = Concatenate()([lstm, attention])
    combined = Bidirectional(LSTM(64))(combined)
    combined = Dropout(0.2)(combined)
    
    # Output layers
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(STRINGS + FRETS, activation='linear')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"Created hybrid model with {context_window} context window")
    return model
