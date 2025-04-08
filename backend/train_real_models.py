#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OliTunes Real Model Trainer
===========================

This script trains actual neural network models for guitar tablature generation
using both real downloaded tabs and synthetic data. It implements three model
architectures: LSTM, Transformer, and Hybrid.
"""

import os
import json
import logging
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard, 
    ReduceLROnPlateau, CSVLogger
)

# Local imports
from tab_data_pipeline import TabDataPipeline
from real_models import create_lstm_model, create_transformer_model, create_hybrid_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants and paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "backend", "models")
LOG_DIR = os.path.join(DATA_DIR, "logs")
MODEL_CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")

# Guitar constants
STRINGS = 6
FRETS = 24
INPUT_FEATURES_PER_NOTE = STRINGS + 2  # String one-hot + fret + midi
MAX_CONTEXT_WINDOW = 5

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)


class RealModelTrainer:
    """
    Trains actual neural network models for guitar position prediction.
    """
    def __init__(self, 
                model_type: str = "lstm", 
                model_name: str = None,
                context_window: int = 3):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model to train ('lstm', 'transformer', or 'hybrid')
            model_name: Name for the model (defaults to model_type)
            context_window: Number of previous notes to use as context
        """
        self.model_type = model_type.lower()
        self.model_name = model_name or f"{model_type}_model"
        self.context_window = context_window
        self.model = None
        self.history = None
        
        # Validate model type
        valid_types = ["lstm", "transformer", "hybrid"]
        if self.model_type not in valid_types:
            logger.warning(f"Invalid model type '{model_type}'. Using lstm instead.")
            self.model_type = "lstm"
        
        # Create timestamp for this training session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define paths
        self.model_dir = os.path.join(MODELS_DIR, f"{self.model_name}")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.model_path = os.path.join(self.model_dir, f"{self.model_name}_{self.timestamp}.h5")
        self.json_path = os.path.join(self.model_dir, f"{self.model_name}_{self.timestamp}.json")
        self.log_dir = os.path.join(LOG_DIR, f"{self.model_name}_{self.timestamp}")
        self.history_path = os.path.join(self.model_dir, f"history_{self.timestamp}.json")
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger.info(f"Initialized {self.model_type} model trainer with context window {context_window}")
        logger.info(f"Model will be saved to {self.model_path}")
    
    def _create_model(self) -> tf.keras.Model:
        """
        Create the neural network model based on model_type.
        
        Returns:
            TensorFlow Keras model
        """
        # Calculate input dimension
        input_dim = self.context_window * INPUT_FEATURES_PER_NOTE
        
        # Create model based on type
        if self.model_type == "lstm":
            model = create_lstm_model(
                input_dim=input_dim,
                context_window=self.context_window,
                features_per_note=INPUT_FEATURES_PER_NOTE
            )
        elif self.model_type == "transformer":
            model = create_transformer_model(
                input_dim=input_dim,
                context_window=self.context_window,
                features_per_note=INPUT_FEATURES_PER_NOTE
            )
        elif self.model_type == "hybrid":
            model = create_hybrid_model(
                input_dim=input_dim,
                context_window=self.context_window,
                features_per_note=INPUT_FEATURES_PER_NOTE
            )
        else:
            logger.warning(f"Unsupported model type '{self.model_type}'. Using LSTM model.")
            model = create_lstm_model(
                input_dim=input_dim,
                context_window=self.context_window,
                features_per_note=INPUT_FEATURES_PER_NOTE
            )
        
        return model
    
    def train(self, 
             epochs: int = 100, 
             batch_size: int = 64,
             patience: int = 10,
             validation_split: float = 0.2) -> Dict:
        """
        Train the model on the data from the pipeline.
        
        Args:
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            patience: Patience for early stopping
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history
        """
        # Initialize data pipeline
        pipeline = TabDataPipeline(
            context_window=self.context_window,
            validation_split=validation_split
        )
        
        # Load and prepare data
        X_train, X_val, y_train, y_val = pipeline.load_training_data()
        
        if len(X_train) == 0 or len(y_train) == 0:
            logger.error("No training data available")
            return {}
        
        # Create model
        self.model = self._create_model()
        
        # Define callbacks
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=self.model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                verbose=1,
                restore_best_weights=True
            ),
            # TensorBoard logs
            TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,
                write_graph=True
            ),
            # Reduce learning rate when plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-5,
                verbose=1
            ),
            # CSV Logger
            CSVLogger(
                os.path.join(self.log_dir, 'training_log.csv')
            )
        ]
        
        # Train model
        logger.info(f"Starting {self.model_type} model training for up to {epochs} epochs")
        logger.info(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        
        # Save training history
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Log training results
        val_loss = min(history.history['val_loss'])
        logger.info(f"Training completed. Best validation loss: {val_loss:.4f}")
        
        # Save model architecture
        self._save_model_details()
        
        return self.history
    
    def _save_model_details(self):
        """Save model architecture and configuration details."""
        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open(self.json_path, 'w') as f:
            f.write(model_json)
        
        # Create a user-friendly model info file
        model_info = {
            "model_type": self.model_type,
            "name": self.model_name,
            "context_window": self.context_window,
            "input_features": INPUT_FEATURES_PER_NOTE,
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "training_metrics": {
                "final_loss": self.history["loss"][-1] if self.history else None,
                "final_val_loss": self.history["val_loss"][-1] if self.history else None,
                "best_val_loss": min(self.history["val_loss"]) if self.history else None,
                "epochs_trained": len(self.history["loss"]) if self.history else 0
            },
            "file_paths": {
                "model_h5": self.model_path,
                "model_json": self.json_path,
                "history": self.history_path
            }
        }
        
        info_path = os.path.join(self.model_dir, f"info_{self.timestamp}.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Saved model details to {info_path}")
        
        # Create a compatibility file for the tablature generator
        compat_info = {
            "model_type": self.model_type,
            "real_model": True,
            "input_dim": self.context_window * INPUT_FEATURES_PER_NOTE,
            "output_dim": STRINGS + FRETS,
            "context_window": self.context_window,
            "features_per_note": INPUT_FEATURES_PER_NOTE,
            "model_path": os.path.relpath(self.model_path, PROJECT_ROOT),
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        compat_path = os.path.join(self.model_dir, "model_config.json")
        with open(compat_path, 'w') as f:
            json.dump(compat_info, f, indent=2)
        
        logger.info(f"Saved compatibility configuration to {compat_path}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test input features
            y_test: Test targets
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            logger.error("No model to evaluate")
            return {}
        
        logger.info(f"Evaluating {self.model_type} model on test data")
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = float(results[i])
            logger.info(f"{metric_name}: {results[i]:.4f}")
        
        return metrics
    
    def load_model(self, model_path: str):
        """
        Load a previously trained model.
        
        Args:
            model_path: Path to the model file (.h5)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Train real neural network models for guitar tablature")
    
    parser.add_argument("--model_type", type=str, default="lstm",
                        choices=["lstm", "transformer", "hybrid"],
                        help="Type of model to train")
    
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name for the model (defaults to model_type)")
    
    parser.add_argument("--context_window", type=int, default=3,
                        help="Number of previous notes to use as context")
    
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of training epochs")
    
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience for early stopping")
    
    args = parser.parse_args()
    
    # Create and train model
    trainer = RealModelTrainer(
        model_type=args.model_type,
        model_name=args.model_name,
        context_window=args.context_window
    )
    
    # Train the model
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience
    )


if __name__ == "__main__":
    main()
