import os
import json
import numpy as np
import logging
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import pickle

class FretboardPositionPredictor:
    """
    Predicts optimal guitar fretboard positions based on musical context.
    Uses a neural network model to determine the most appropriate string/fret combinations.
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize the fretboard position predictor.
        
        Args:
            model_dir (str, optional): Directory containing model files. 
                                      If None, uses the current directory.
        """
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
            
        self.model_dir = model_dir
        self.model = None
        self.metadata = None
        self.load_model()
        
    def load_model(self):
        """
        Load the model architecture, weights, and metadata.
        """
        try:
            # Check if model is ready
            model_ready_path = os.path.join(self.model_dir, 'model_ready')
            if not os.path.exists(model_ready_path):
                logging.warning(f"Fretboard position model not ready (missing {model_ready_path})")
                return False
                
            # Load model architecture
            architecture_path = os.path.join(self.model_dir, 'model_architecture.json')
            if not os.path.exists(architecture_path):
                logging.warning(f"Model architecture file not found: {architecture_path}")
                return False
                
            with open(architecture_path, 'r') as f:
                model_json = f.read()
                
            self.model = model_from_json(model_json)
            
            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Try to load weights from pickle file first
            weights_pkl_path = os.path.join(self.model_dir, 'model_weights.pkl')
            if os.path.exists(weights_pkl_path):
                try:
                    with open(weights_pkl_path, 'rb') as f:
                        weights = pickle.load(f)
                    self.model.set_weights(weights)
                    logging.info(f"Loaded fretboard position model weights from {weights_pkl_path}")
                    
                    # Load metadata
                    metadata_path = os.path.join(self.model_dir, 'metadata.json')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            self.metadata = json.load(f)
                        logging.info(f"Loaded fretboard position model metadata")
                    else:
                        logging.warning(f"Model metadata file not found: {metadata_path}")
                    
                    return True
                except Exception as e:
                    logging.error(f"Error loading weights from pickle: {str(e)}")
            
            # Fall back to other weight formats
            weights_path = os.path.join(self.model_dir, 'model_weights')
            weights_h5_path = os.path.join(self.model_dir, 'model_weights.h5')
            
            if os.path.exists(weights_path):
                self.model.load_weights(weights_path)
                logging.info(f"Loaded fretboard position model weights from {weights_path}")
            elif os.path.exists(weights_h5_path):
                self.model.load_weights(weights_h5_path)
                logging.info(f"Loaded fretboard position model weights from {weights_h5_path}")
            else:
                logging.warning(f"Model weights file not found: {weights_pkl_path}, {weights_path}, or {weights_h5_path}")
                return False
                
            # Load metadata
            metadata_path = os.path.join(self.model_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logging.info(f"Loaded fretboard position model metadata")
            else:
                logging.warning(f"Model metadata file not found: {metadata_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading fretboard position model: {str(e)}")
            self.model = None
            return False
    
    def prepare_input_features(self, midi_note, previous_position=0, string=0, key_context=0, style=0):
        """
        Prepare input features for model prediction.
        
        Args:
            midi_note (int): MIDI note number (0-127)
            previous_position (int): Previous fret position (0-24)
            string (int): Guitar string number (0-5)
            key_context (int): Musical key context (0-11)
            style (int): Playing style (0-4)
            
        Returns:
            numpy.ndarray: Prepared input features
        """
        # One-hot encode MIDI note (128 values)
        midi_one_hot = np.zeros(128)
        if 0 <= midi_note < 128:
            midi_one_hot[midi_note] = 1
            
        # One-hot encode previous position (25 values)
        prev_pos_one_hot = np.zeros(25)
        if 0 <= previous_position < 25:
            prev_pos_one_hot[previous_position] = 1
            
        # One-hot encode string (6 values)
        string_one_hot = np.zeros(6)
        if 0 <= string < 6:
            string_one_hot[string] = 1
            
        # One-hot encode key context (12 values)
        key_one_hot = np.zeros(12)
        if 0 <= key_context < 12:
            key_one_hot[key_context] = 1
            
        # One-hot encode style (5 values)
        style_one_hot = np.zeros(5)
        if 0 <= style < 5:
            style_one_hot[style] = 1
            
        # Combine all features
        features = np.concatenate([
            midi_one_hot,
            prev_pos_one_hot,
            string_one_hot,
            key_one_hot,
            style_one_hot
        ])
        
        return features.reshape(1, -1)  # Reshape for model input
    
    def predict_position(self, midi_note, previous_position=0, string=0, key_context=0, style=0):
        """
        Predict the optimal fretboard position for a given note and context.
        
        Args:
            midi_note (int): MIDI note number (0-127)
            previous_position (int): Previous fret position (0-24)
            string (int): Guitar string number (0-5)
            key_context (int): Musical key context (0-11)
            style (int): Playing style (0-4)
            
        Returns:
            tuple: (string, fret) representing the optimal position
                  Returns (0, 0) if prediction fails
        """
        if self.model is None:
            logging.warning("Cannot predict position: model not loaded")
            return 0, 0
            
        try:
            # Prepare input features
            input_features = self.prepare_input_features(
                midi_note, previous_position, string, key_context, style
            )
            
            # Make prediction
            prediction = self.model.predict(input_features, verbose=0)[0]
            
            # Convert prediction to string/fret combination
            # Assuming output is a 150-element array (6 strings × 25 frets)
            max_index = np.argmax(prediction)
            predicted_string = max_index // 25
            predicted_fret = max_index % 25
            
            return predicted_string, predicted_fret
            
        except Exception as e:
            logging.error(f"Error predicting position: {str(e)}")
            return 0, 0
    
    def predict_positions_for_notes(self, midi_notes, key_context=0, style=0):
        """
        Predict optimal positions for a sequence of notes.
        
        Args:
            midi_notes (list): List of MIDI note numbers
            key_context (int): Musical key context (0-11)
            style (int): Playing style (0-4)
            
        Returns:
            list: List of (string, fret) tuples
        """
        if self.model is None:
            logging.warning("Cannot predict positions: model not loaded")
            return [(0, 0)] * len(midi_notes)
            
        positions = []
        prev_position = 0
        prev_string = 0
        
        for note in midi_notes:
            string, fret = self.predict_position(
                note, prev_position, prev_string, key_context, style
            )
            positions.append((string, fret))
            prev_position = fret
            prev_string = string
            
        return positions
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Train the fretboard position model with new data.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        if self.model is None:
            logging.error("Cannot train: model not loaded")
            return False
            
        try:
            # Train the model
            self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2
            )
            
            # Save the weights using pickle
            weights = self.model.get_weights()
            weights_path = os.path.join(self.model_dir, 'model_weights.pkl')
            with open(weights_path, 'wb') as f:
                pickle.dump(weights, f)
            logging.info(f"Model weights saved to {weights_path}")
            
            # Create model_ready file if it doesn't exist
            model_ready_path = os.path.join(self.model_dir, 'model_ready')
            with open(model_ready_path, 'w') as f:
                f.write('1')
                
            return True
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            return False
