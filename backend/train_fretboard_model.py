#!/usr/bin/env python
"""
Fretboard Position Model Training Script

This script trains a neural network to predict optimal fretboard positions
for guitar tablature generation based on musical context.

The model is trained on real guitar tab data extracted from Ultimate Guitar
and supplemented with synthetic data generated according to music theory principles.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_trainer")

# Constants
TAB_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "tab_data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_X_PATH = os.path.join(TAB_DATA_DIR, "X_tab_positions.npy")
DATA_Y_PATH = os.path.join(TAB_DATA_DIR, "y_tab_positions.npy")
MODEL_PATH = os.path.join(MODELS_DIR, "fretboard_position_model")
MODEL_JSON_PATH = os.path.join(MODELS_DIR, "fretboard_position_model.json")
MODEL_WEIGHTS_PATH = os.path.join(MODELS_DIR, "fretboard_position_model_weights.h5")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")

# Ensure directories exist
os.makedirs(TAB_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class FretboardModelTrainer:
    """Trains a neural network model for fretboard position prediction"""
    
    def __init__(self, input_dim=176, output_dim=150, use_lstm=False):
        """
        Initialize the trainer
        
        Args:
            input_dim: Input dimension (MIDI note, prev position, string, style)
            output_dim: Output dimension (string/fret combinations)
            use_lstm: Whether to use LSTM layers for sequence processing
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_lstm = use_lstm
        self.model = None
        self.history = None
    
    def load_data(self):
        """
        Load the training data from prepared numpy files
        
        Returns:
            Dictionary with training and validation data splits
        """
        # Check if data files exist
        if not os.path.exists(DATA_X_PATH) or not os.path.exists(DATA_Y_PATH):
            logger.error("Training data files not found. Please run tab_data_processor.py first.")
            return None
        
        try:
            # Load the data
            X = np.load(DATA_X_PATH)
            y = np.load(DATA_Y_PATH)
            
            logger.info(f"Loaded {len(X)} training examples")
            
            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            return {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val
            }
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None
    
    def prepare_lstm_data(self, data):
        """
        Reshape data for LSTM model (adding sequence dimension)
        
        Args:
            data: Dictionary with training and validation data
            
        Returns:
            Dictionary with reshaped data
        """
        if not self.use_lstm:
            return data
        
        try:
            # Reshape for LSTM: (n_samples, 1, n_features)
            data['X_train'] = data['X_train'].reshape(data['X_train'].shape[0], 1, self.input_dim)
            data['X_val'] = data['X_val'].reshape(data['X_val'].shape[0], 1, self.input_dim)
            
            logger.info("Data reshaped for LSTM")
            return data
            
        except Exception as e:
            logger.error(f"Error reshaping data for LSTM: {e}")
            return None
    
    def create_model(self):
        """
        Create the neural network model architecture
        
        Returns:
            Compiled model
        """
        try:
            model = Sequential()
            
            if self.use_lstm:
                # LSTM-based model
                model.add(LSTM(256, input_shape=(1, self.input_dim), return_sequences=True))
                model.add(Dropout(0.2))
                model.add(BatchNormalization())
                
                model.add(LSTM(128, return_sequences=False))
                model.add(Dropout(0.2))
                model.add(BatchNormalization())
                
                model.add(Dense(128, activation='relu'))
                model.add(Dropout(0.2))
                
            else:
                # Feedforward model
                model.add(Dense(256, input_dim=self.input_dim, activation='relu'))
                model.add(Dropout(0.3))
                model.add(BatchNormalization())
                
                model.add(Dense(128, activation='relu'))
                model.add(Dropout(0.2))
                model.add(BatchNormalization())
                
                model.add(Dense(128, activation='relu'))
                model.add(Dropout(0.2))
            
            # Output layer
            model.add(Dense(self.output_dim, activation='softmax'))
            
            # Compile the model
            model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
            
            self.model = model
            logger.info(f"Created {'LSTM' if self.use_lstm else 'feedforward'} model")
            model.summary(print_fn=logger.info)
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return None
    
    def train_model(self, data, epochs=50, batch_size=32):
        """
        Train the neural network model
        
        Args:
            data: Dictionary with training and validation data
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if self.model is None:
            logger.error("Model not created. Call create_model first.")
            return None
        
        try:
            # Callbacks for training
            callbacks = [
                # Early stopping to prevent overfitting
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                # Learning rate reduction when plateau
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                ),
                # Save checkpoints
                ModelCheckpoint(
                    filepath=os.path.join(MODELS_DIR, "checkpoint_{epoch}.h5"),
                    save_best_only=True,
                    monitor='val_accuracy',
                    mode='max',
                    save_weights_only=True,
                    verbose=1
                )
            ]
            
            # Train the model
            logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
            
            history = self.model.fit(
                data['X_train'], data['y_train'],
                validation_data=(data['X_val'], data['y_val']),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=2
            )
            
            self.history = history.history
            logger.info("Training completed")
            
            # Show final metrics
            val_loss = history.history['val_loss'][-1]
            val_acc = history.history['val_accuracy'][-1]
            logger.info(f"Final validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
            
            return history.history
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
    
    def save_model(self):
        """
        Save the trained model architecture and weights
        
        Returns:
            Tuple of saved paths (json_path, weights_path)
        """
        if self.model is None:
            logger.error("No model to save")
            return None
        
        try:
            # Save model architecture as JSON
            model_json = self.model.to_json()
            with open(MODEL_JSON_PATH, 'w') as json_file:
                json_file.write(model_json)
            logger.info(f"Model architecture saved to {MODEL_JSON_PATH}")
            
            # Save model weights
            self.model.save_weights(MODEL_WEIGHTS_PATH)
            logger.info(f"Model weights saved to {MODEL_WEIGHTS_PATH}")
            
            # Save the complete model if TF supports it in this environment
            try:
                self.model.save(MODEL_PATH)
                logger.info(f"Complete model saved to {MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Could not save complete model: {e}")
                logger.warning("This is fine, we'll load from JSON and weights instead")
            
            # Save training history
            if self.history:
                # Convert numpy values to Python types for JSON serialization
                serializable_history = {}
                for key, values in self.history.items():
                    serializable_history[key] = [float(val) for val in values]
                
                with open(HISTORY_PATH, 'w') as f:
                    json.dump(serializable_history, f)
                logger.info(f"Training history saved to {HISTORY_PATH}")
            
            return (MODEL_JSON_PATH, MODEL_WEIGHTS_PATH)
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None
    
    def plot_training_history(self):
        """
        Plot training metrics and save figures
        
        Returns:
            Paths to saved figures
        """
        if not self.history:
            logger.error("No training history to plot")
            return None
        
        try:
            # Loss plot
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(self.history['loss'], label='Training Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Accuracy plot
            plt.subplot(1, 2, 2)
            plt.plot(self.history['accuracy'], label='Training Accuracy')
            plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Save figure
            plt.tight_layout()
            history_plot_path = os.path.join(MODELS_DIR, "training_history.png")
            plt.savefig(history_plot_path)
            logger.info(f"Training history plot saved to {history_plot_path}")
            
            return history_plot_path
            
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")
            return None
    
    def test_model(self, data=None):
        """
        Test the model on some examples and print predictions
        
        Args:
            data: Optional test data to use
            
        Returns:
            Dictionary with test metrics
        """
        if self.model is None:
            logger.error("No model to test")
            return None
        
        try:
            # If no specific test data provided, use validation data
            if data is None:
                test_data = self.load_data()
                if test_data is None:
                    return None
                
                X_test, y_test = test_data['X_val'][:5], test_data['y_val'][:5]
            else:
                X_test, y_test = data['X_test'], data['y_test']
            
            # Reshape for LSTM if needed
            if self.use_lstm:
                X_test = X_test.reshape(X_test.shape[0], 1, self.input_dim)
            
            # Make predictions
            predictions = self.model.predict(X_test)
            
            # Evaluate performance
            metrics = self.model.evaluate(
                X_test, y_test,
                verbose=0
            )
            
            logger.info("Model test results:")
            metric_names = self.model.metrics_names
            for name, value in zip(metric_names, metrics):
                logger.info(f"{name}: {value:.4f}")
            
            # Show some example predictions
            for i in range(min(3, len(predictions))):
                # Get the predicted position (string and fret)
                pred_idx = np.argmax(predictions[i])
                pred_string = pred_idx // 25
                pred_fret = pred_idx % 25
                
                # Get the actual position
                true_idx = np.argmax(y_test[i])
                true_string = true_idx // 25
                true_fret = true_idx % 25
                
                logger.info(f"Example {i+1}:")
                logger.info(f"  Predicted: String {pred_string+1}, Fret {pred_fret}")
                logger.info(f"  Actual: String {true_string+1}, Fret {true_fret}")
            
            return {
                'metrics': {name: float(value) for name, value in zip(metric_names, metrics)},
                'examples': [
                    {
                        'predicted': {'string': int(np.argmax(p) // 25), 'fret': int(np.argmax(p) % 25)},
                        'actual': {'string': int(np.argmax(y) // 25), 'fret': int(np.argmax(y) % 25)}
                    }
                    for p, y in zip(predictions[:5], y_test[:5])
                ]
            }
            
        except Exception as e:
            logger.error(f"Error testing model: {e}")
            return None

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Train the fretboard position prediction model")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lstm', action='store_true', help='Use LSTM model architecture')
    args = parser.parse_args()
    
    logger.info("Starting model training")
    
    # Check if data is available
    if not os.path.exists(DATA_X_PATH) or not os.path.exists(DATA_Y_PATH):
        logger.error("Training data not found. Please run the tab data processor first.")
        print("\n❌ Error: Training data not found!")
        print("  Please run the tab data processor first with:")
        print("  python tab_data_processor.py")
        sys.exit(1)
    
    # Create trainer with desired architecture
    trainer = FretboardModelTrainer(use_lstm=args.lstm)
    
    # Load the data
    data = trainer.load_data()
    if data is None:
        print("\n❌ Error: Could not load training data!")
        sys.exit(1)
    
    # Prepare data for LSTM if needed
    if args.lstm:
        data = trainer.prepare_lstm_data(data)
        if data is None:
            print("\n❌ Error: Could not prepare data for LSTM!")
            sys.exit(1)
    
    # Create the model
    model = trainer.create_model()
    if model is None:
        print("\n❌ Error: Could not create model!")
        sys.exit(1)
    
    # Train the model
    history = trainer.train_model(data, epochs=args.epochs, batch_size=args.batch_size)
    if history is None:
        print("\n❌ Error: Model training failed!")
        sys.exit(1)
    
    # Save the model
    saved_paths = trainer.save_model()
    if saved_paths is None:
        print("\n❌ Error: Could not save model!")
        sys.exit(1)
    
    # Plot training history
    plot_path = trainer.plot_training_history()
    
    # Test the model
    test_results = trainer.test_model()
    
    print("\n✅ Model training complete!")
    print(f"  Model architecture saved to: {os.path.basename(MODEL_JSON_PATH)}")
    print(f"  Model weights saved to: {os.path.basename(MODEL_WEIGHTS_PATH)}")
    
    if test_results and 'metrics' in test_results:
        print(f"  Test accuracy: {test_results['metrics'].get('accuracy', 0):.2%}")
    
    print("\nNext steps:")
    print("  1. Use the model in the tablature generator by updating the model paths")
    print("  2. Collect more tab data to improve the model")
    print("  3. Experiment with different model architectures")
    
    logger.info("Model training process complete")

if __name__ == "__main__":
    main()
