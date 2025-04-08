#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OliTunes Position Model Trainer
===============================

This module trains the guitar fretboard position prediction model using
a combination of real tablature data (downloaded from Ultimate Guitar)
and synthetic training data.

The model is designed to predict optimal fretboard positions for notes
based on musical context, hand constraints, and playing techniques.
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split

# Import our tablature processing modules
from tab_processor import TrainingDataGenerator, TRAINING_DATA_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, "synthetic_data")
LOG_DIR = os.path.join(DATA_DIR, "logs")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Guitar constants
STRINGS = 6
FRETS = 24
INPUT_FEATURES_PER_NOTE = STRINGS + 2  # String one-hot + fret + midi
MAX_CONTEXT_WINDOW = 5


class PositionModelTrainer:
    """
    Trains the guitar position prediction model.
    """
    def __init__(self, model_name: str = "position_model"):
        """
        Initialize the model trainer.
        
        Args:
            model_name: Name for the model
        """
        self.model_name = model_name
        self.model = None
        
        # Create timestamp for this training session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_path = os.path.join(MODEL_DIR, f"{model_name}_{self.timestamp}.h5")
        self.log_dir = os.path.join(LOG_DIR, f"{model_name}_{self.timestamp}")
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger.info(f"Position model trainer initialized. Model: {model_name}")

    def load_training_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training data from file.
        
        Args:
            data_path: Path to training data file
            
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            data = np.load(data_path)
            X = data['X']
            y = data['y']
            
            logger.info(f"Loaded training data: {X.shape[0]} examples")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            return np.array([]), np.array([])

    def create_lstm_model(self, context_window: int = 3) -> Model:
        """
        Create an LSTM model for sequence-based position prediction.
        
        Args:
            context_window: Number of previous notes to use as context
            
        Returns:
            Compiled Keras model
        """
        # Calculate input shape
        input_dim = context_window * INPUT_FEATURES_PER_NOTE
        
        # Create a sequential model with LSTM layers
        model = Sequential([
            # Reshape input for LSTM
            tf.keras.layers.Reshape((context_window, INPUT_FEATURES_PER_NOTE), 
                                   input_shape=(input_dim,)),
            
            # LSTM layers
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            
            # Output layers
            Dense(32, activation='relu'),
            Dense(STRINGS + FRETS, activation='linear')  # String + fret prediction
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Created LSTM model with {context_window} context window")
        model.summary(print_fn=logger.info)
        
        return model

    def create_attention_model(self, context_window: int = 3) -> Model:
        """
        Create a transformer-based model with attention mechanisms.
        
        Args:
            context_window: Number of previous notes to use as context
            
        Returns:
            Compiled Keras model
        """
        # Calculate input shape
        input_dim = context_window * INPUT_FEATURES_PER_NOTE
        
        # Input layer
        inputs = Input(shape=(input_dim,))
        
        # Reshape for transformer
        x = tf.keras.layers.Reshape((context_window, INPUT_FEATURES_PER_NOTE))(inputs)
        
        # Transformer encoder block
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=INPUT_FEATURES_PER_NOTE)(x, x)
        x = tf.keras.layers.Add()([x, attention_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed forward network
        forward_output = tf.keras.layers.Dense(64, activation='relu')(x)
        forward_output = tf.keras.layers.Dense(INPUT_FEATURES_PER_NOTE)(forward_output)
        x = tf.keras.layers.Add()([x, forward_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Global attention pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layers
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(STRINGS + FRETS, activation='linear')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Created attention-based model with {context_window} context window")
        model.summary(print_fn=logger.info)
        
        return model

    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   model_type: str = 'lstm',
                   context_window: int = 3,
                   epochs: int = 50,
                   batch_size: int = 32,
                   validation_split: float = 0.2) -> Model:
        """
        Train the position prediction model.
        
        Args:
            X: Input feature array
            y: Target array
            model_type: Type of model to train ('lstm' or 'attention')
            context_window: Size of context window
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data to use for validation
            
        Returns:
            Trained model
        """
        if X.size == 0 or y.size == 0:
            logger.error("Cannot train model with empty data")
            return None
        
        # Create model based on type
        if model_type == 'attention':
            model = self.create_attention_model(context_window)
        else:
            model = self.create_lstm_model(context_window)
        
        # Define callbacks
        callbacks = [
            # Save best model
            ModelCheckpoint(
                self.model_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                mode='min',
                verbose=1
            ),
            # TensorBoard logging
            TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42)
        
        logger.info(f"Training set: {X_train.shape[0]} examples")
        logger.info(f"Validation set: {X_val.shape[0]} examples")
        
        # Train the model
        logger.info(f"Starting model training ({epochs} epochs, batch size {batch_size})")
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=2
        )
        
        # Save training history
        history_path = os.path.join(self.log_dir, 'history.json')
        with open(history_path, 'w') as f:
            history_dict = {}
            for key, values in history.history.items():
                history_dict[key] = [float(v) for v in values]
            json.dump(history_dict, f, indent=2)
        
        logger.info(f"Model training completed. Best model saved to {self.model_path}")
        
        # Load the best model
        self.model = load_model(self.model_path)
        return self.model

    def evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate the trained model.
        
        Args:
            X: Test feature array
            y: Test target array
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            logger.error("No model to evaluate")
            return {}
            
        if X.size == 0 or y.size == 0:
            logger.error("Cannot evaluate model with empty data")
            return {}
        
        logger.info(f"Evaluating model on {X.shape[0]} examples")
        
        # Evaluate the model
        loss, mae = self.model.evaluate(X, y, verbose=0)
        
        # Calculate additional metrics
        y_pred = self.model.predict(X)
        
        # Split predictions into string and fret
        string_actual = y[:, 0].astype(int)
        fret_actual = y[:, 1].astype(int)
        
        string_pred = np.argmax(y_pred[:, :STRINGS], axis=1)
        fret_pred = np.argmax(y_pred[:, STRINGS:], axis=1)
        
        # Calculate accuracy
        string_accuracy = np.mean(string_pred == string_actual)
        fret_accuracy = np.mean(np.abs(fret_pred - fret_actual) <= 2)  # Within 2 frets
        
        # Log results
        metrics = {
            'loss': float(loss),
            'mae': float(mae),
            'string_accuracy': float(string_accuracy),
            'fret_accuracy': float(fret_accuracy),
            'examples': int(X.shape[0])
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        # Save metrics
        metrics_path = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics

    def save_model_info(self, metrics: Dict, context_window: int, model_type: str) -> None:
        """
        Save model information and metadata.
        
        Args:
            metrics: Evaluation metrics
            context_window: Context window size
            model_type: Type of model
        """
        info = {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'path': self.model_path,
            'type': model_type,
            'context_window': context_window,
            'metrics': metrics,
            'parameters': {
                'strings': STRINGS,
                'frets': FRETS,
                'features_per_note': INPUT_FEATURES_PER_NOTE
            }
        }
        
        # Save model info
        info_path = os.path.join(MODEL_DIR, f"{self.model_name}_{self.timestamp}_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        # Also save as latest model info
        latest_path = os.path.join(MODEL_DIR, f"{self.model_name}_latest_info.json")
        with open(latest_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Model info saved to {info_path}")


def main():
    """Main entry point for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the OliTunes Position Prediction Model")
    parser.add_argument('--data', default=None, 
                      help='Path to training data (will use latest if not specified)')
    parser.add_argument('--model-type', choices=['lstm', 'attention'], default='lstm',
                      help='Type of model to train')
    parser.add_argument('--context', type=int, default=3, 
                      help='Context window size (previous notes)')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--model-name', default='position_model',
                      help='Base name for the model')
    args = parser.parse_args()
    
    try:
        # Find training data if not specified
        data_path = args.data
        if data_path is None:
            # Use the latest training data file
            training_files = [f for f in os.listdir(TRAINING_DATA_DIR) 
                           if f.endswith('.npz')]
            if training_files:
                training_files.sort(reverse=True)  # Sort by name (should include timestamp)
                data_path = os.path.join(TRAINING_DATA_DIR, training_files[0])
                logger.info(f"Using latest training data: {data_path}")
            else:
                logger.error("No training data found")
                return
        
        # Create trainer
        trainer = PositionModelTrainer(model_name=args.model_name)
        
        # Load data
        X, y = trainer.load_training_data(data_path)
        if X.size == 0 or y.size == 0:
            logger.error("Failed to load valid training data")
            return
        
        # Train model
        model = trainer.train_model(
            X, y,
            model_type=args.model_type,
            context_window=args.context,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        if model is None:
            logger.error("Model training failed")
            return
        
        # Evaluate model
        metrics = trainer.evaluate_model(X, y)
        
        # Save model info
        trainer.save_model_info(metrics, args.context, args.model_type)
        
        print(f"\nModel training completed successfully!")
        print(f"Model saved to: {trainer.model_path}")
        print(f"Metrics: {metrics}")
        
    except Exception as e:
        logger.exception("Error in model training")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
