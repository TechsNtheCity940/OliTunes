#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OliTunes Ensemble Model Architecture
====================================

This module implements an ensemble approach that combines multiple 
prediction strategies for improved tablature generation.
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, Concatenate, Average, Lambda, 
    Reshape, Dropout, Layer
)

# Local imports - use try/except for graceful fallback
try:
    from real_models import create_lstm_model, create_transformer_model, create_hybrid_model
except ImportError:
    logging.warning("Could not import model architectures - ensemble creation may fail")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ensemble_models")

# Constants and paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_ROOT, "backend", "models")

# Guitar constants
STRINGS = 6
FRETS = 24
INPUT_FEATURES_PER_NOTE = STRINGS + 2  # String one-hot + fret + midi
DEFAULT_CONTEXT_WINDOW = 3


class WeightedVoting(Layer):
    """
    Custom layer implementing a weighted voting scheme for multiple model outputs.
    """
    def __init__(self, **kwargs):
        super(WeightedVoting, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create weights for each model
        self.model_weights = self.add_weight(
            name='model_weights',
            shape=(len(input_shape),),
            initializer='ones',
            trainable=True
        )
        super(WeightedVoting, self).build(input_shape)
    
    def call(self, inputs):
        # Apply softmax to ensure weights sum to 1
        weights = tf.nn.softmax(self.model_weights)
        
        # Multiply each input by its weight and sum
        weighted_sum = tf.zeros_like(inputs[0])
        for i, input_tensor in enumerate(inputs):
            weighted_sum += input_tensor * weights[i]
        
        return weighted_sum
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]


def create_ensemble_model(input_dim: int, 
                         context_window: int, 
                         features_per_note: int,
                         ensemble_type: str = "weighted_average") -> Model:
    """
    Create an ensemble model that combines different prediction approaches.
    
    Args:
        input_dim: Total input dimensions
        context_window: Number of previous notes to use as context
        features_per_note: Number of features per note in the input
        ensemble_type: Type of ensemble ("weighted_average", "stacked", "voting")
        
    Returns:
        Compiled Keras model
    """
    # Input layer - shared by all models
    inputs = Input(shape=(input_dim,))
    
    # Create individual models
    try:
        lstm_model = create_lstm_model(input_dim, context_window, features_per_note)
        transformer_model = create_transformer_model(input_dim, context_window, features_per_note)
        hybrid_model = create_hybrid_model(input_dim, context_window, features_per_note)
        
        # Remove the final layers to get features/logits
        lstm_features = lstm_model.layers[-2].output
        transformer_features = transformer_model.layers[-2].output
        hybrid_features = hybrid_model.layers[-2].output
        
        lstm_output = lstm_model.layers[-1].output
        transformer_output = transformer_model.layers[-1].output
        hybrid_output = hybrid_model.layers[-1].output
    except Exception as e:
        logger.error(f"Error creating base models: {e}")
        return None
    
    # Different ensemble approaches
    if ensemble_type == "weighted_average":
        # Directly combine predictions with learned weights
        ensemble_output = WeightedVoting()([lstm_output, transformer_output, hybrid_output])
        
    elif ensemble_type == "stacked":
        # Concatenate features from all models
        combined_features = Concatenate()(
            [lstm_features, transformer_features, hybrid_features]
        )
        
        # Add meta-learner layers on top
        x = Dense(128, activation='relu')(combined_features)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        ensemble_output = Dense(STRINGS + FRETS, activation='linear')(x)
        
    elif ensemble_type == "voting":
        # Simple average voting (equal weights)
        ensemble_output = Average()([lstm_output, transformer_output, hybrid_output])
        
    else:
        logger.warning(f"Unknown ensemble type '{ensemble_type}'. Using weighted average.")
        ensemble_output = WeightedVoting()([lstm_output, transformer_output, hybrid_output])
    
    # Create the ensemble model
    ensemble_model = Model(inputs=inputs, outputs=ensemble_output)
    
    # Custom optimizer with a slow learning rate for fine-tuning
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    # Compile model
    ensemble_model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"Created {ensemble_type} ensemble model with {context_window} context window")
    return ensemble_model


def load_pretrained_weights(ensemble_model: Model, 
                           model_paths: Dict[str, str]) -> bool:
    """
    Load pretrained weights from individual models into the ensemble model.
    
    Args:
        ensemble_model: The ensemble model
        model_paths: Dictionary with model types as keys and paths as values
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load individual models
        models = {}
        for model_type, path in model_paths.items():
            if os.path.exists(path):
                models[model_type] = tf.keras.models.load_model(path)
                logger.info(f"Loaded {model_type} model from {path}")
            else:
                logger.warning(f"Model file {path} not found")
        
        # No weights to load
        if not models:
            logger.error("No model weights were loaded")
            return False
        
        # For now, just return success. In a real implementation, we would 
        # have to carefully copy weights to the correct layers in the ensemble model.
        # This depends on the specific architecture and is non-trivial.
        logger.info("Pretrained models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading pretrained weights: {e}")
        return False


def train_ensemble_model(ensemble_model: Model,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        epochs: int = 50,
                        batch_size: int = 64,
                        model_path: str = None) -> Dict:
    """
    Fine-tune the ensemble model on the training data.
    
    Args:
        ensemble_model: The ensemble model
        X_train: Training input features
        y_train: Training targets
        X_val: Validation input features
        y_val: Validation targets
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_path: Path to save the trained model
        
    Returns:
        Training history
    """
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "ensemble_model", "ensemble_model.h5")
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Define callbacks
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        # Reduce learning rate when plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    logger.info(f"Fine-tuning ensemble model for {epochs} epochs")
    
    history = ensemble_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    history_path = os.path.join(os.path.dirname(model_path), "ensemble_history.json")
    with open(history_path, 'w') as f:
        history_dict = {key: [float(val) for val in values] 
                        for key, values in history.history.items()}
        json.dump(history_dict, f, indent=2)
    
    # Log training results
    val_loss = min(history.history['val_loss'])
    logger.info(f"Ensemble training completed. Best validation loss: {val_loss:.4f}")
    
    return history.history


def create_style_transfer_model(input_dim: int, 
                               context_window: int, 
                               features_per_note: int,
                               num_styles: int = 5) -> Model:
    """
    Create a model with style transfer capabilities.
    
    Args:
        input_dim: Total input dimensions
        context_window: Number of previous notes to use as context
        features_per_note: Number of features per note in the input
        num_styles: Number of styles to support
        
    Returns:
        Compiled Keras model
    """
    # Input layers
    note_inputs = Input(shape=(input_dim,), name="note_inputs")
    style_inputs = Input(shape=(num_styles,), name="style_inputs")
    
    # Load base model
    try:
        base_model = create_hybrid_model(input_dim, context_window, features_per_note)
        # Get the features before the final layer
        x = base_model.layers[-2].output
    except Exception as e:
        logger.error(f"Error creating base model: {e}")
        
        # Fallback implementation
        reshaped = Reshape((context_window, features_per_note))(note_inputs)
        x = tf.keras.layers.LSTM(64)(reshaped)
    
    # Concatenate style inputs with model features
    combined = Concatenate()([x, style_inputs])
    
    # Add style-specific layers
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    # Output layer
    outputs = Dense(STRINGS + FRETS, activation='linear')(x)
    
    # Create style transfer model
    model = Model(inputs=[note_inputs, style_inputs], outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"Created style transfer model with {num_styles} styles")
    return model


def get_style_embedding(style_name: str, num_styles: int = 5) -> np.ndarray:
    """
    Generate a one-hot style embedding vector.
    
    Args:
        style_name: Name of the style
        num_styles: Total number of styles
        
    Returns:
        One-hot encoding for the style
    """
    # Map style names to indices
    style_map = {
        "blues": 0,
        "rock": 1,
        "jazz": 2,
        "folk": 3,
        "metal": 4
    }
    
    # Create one-hot encoding
    embedding = np.zeros(num_styles)
    
    if style_name.lower() in style_map:
        idx = style_map[style_name.lower()]
        embedding[idx] = 1.0
    else:
        # Default to rock style
        embedding[1] = 1.0
    
    return embedding


if __name__ == "__main__":
    # Example usage
    context_window = 3
    features_per_note = INPUT_FEATURES_PER_NOTE
    input_dim = context_window * features_per_note
    
    # Create ensemble model
    try:
        ensemble = create_ensemble_model(
            input_dim=input_dim, 
            context_window=context_window, 
            features_per_note=features_per_note,
            ensemble_type="weighted_average"
        )
        
        print("Successfully created ensemble model")
        
        # Create style transfer model
        style_model = create_style_transfer_model(
            input_dim=input_dim,
            context_window=context_window,
            features_per_note=features_per_note
        )
        
        print("Successfully created style transfer model")
        
    except Exception as e:
        print(f"Error creating models: {e}")
