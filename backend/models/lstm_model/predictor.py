"""
LSTM Tablature Refinement
"""
import os
import json
import numpy as np
import tensorflow as tf
from typing import List

class LSTMPredictor:
    """Refines raw tab predictions using LSTM"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'model_config.json')
            
        with open(config_path) as f:
            self.config = json.load(f)
            
        self.model = tf.keras.models.load_model(self.config['model_path'])
    
    def predict(self, raw_predictions: np.ndarray) -> np.ndarray:
        """Refine raw TabCNN predictions"""
        # Reshape for LSTM (samples, timesteps, features)
        input_data = raw_predictions.reshape(
            -1, 
            self.config['context_window'], 
            self.config['features_per_note']
        )
        
        # Get refined predictions
        return self.model.predict(input_data)
