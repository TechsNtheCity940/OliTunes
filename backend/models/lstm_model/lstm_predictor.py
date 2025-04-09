"""
LSTM Tablature Refinement
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Reshape
from tensorflow.keras.models import Model
from typing import List

class LSTMPredictor:
    """Refines raw tab predictions using LSTM"""
    
    def __init__(self, config_path: str = None, input_shape=(None, 6, 21)):
        # Try to load from config if provided
        if config_path is not None:
            try:
                with open(config_path) as f:
                    self.config = json.load(f)
                self.model = tf.keras.models.load_model(self.config['model_path'])
                return
            except Exception as e:
                print(f"Could not load model from config: {str(e)}")
                print("Falling back to default model")
        
        # Build default model if no config or loading failed
        self.model = self._build_model(input_shape)
        # Try to load weights if they exist
        weights_path = os.path.join(os.path.dirname(__file__), 'lstm_weights.h5')
        if os.path.exists(weights_path):
            try:
                self.model.load_weights(weights_path)
                print(f"Loaded weights from {weights_path}")
            except Exception as e:
                print(f"Could not load weights: {str(e)}")
    
    def _build_model(self, input_shape):
        """Build the LSTM model for tablature refinement"""
        # Handle different input shapes more robustly
        if len(input_shape) == 3:  # (timesteps, 6, 21)
            # For 3D input (timesteps, strings, frets)
            inputs = Input(shape=input_shape)
            # Reshape to (timesteps, features)
            reshaped = Reshape((input_shape[0], input_shape[1] * input_shape[2]))(inputs)
        elif len(input_shape) == 2:  # (timesteps, 6*21)
            # For already flattened features
            inputs = Input(shape=input_shape)
            reshaped = inputs
        else:
            # Default case - assume we need to flatten the last dimensions
            inputs = Input(shape=input_shape)
            # Get total features by multiplying all dimensions except the first (timesteps)
            total_features = np.prod(input_shape[1:])
            reshaped = Reshape((input_shape[0], total_features))(inputs)
            
        # LSTM layers
        x = Bidirectional(LSTM(128, return_sequences=True))(reshaped)
        x = LSTM(64, return_sequences=True)(x)
        outputs = Dense(6 * 21, activation='softmax')(x)  # Predict fret probabilities
        
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def predict(self, raw_tab: np.ndarray) -> np.ndarray:
        """Refine raw TabCNN predictions with enhanced playability constraints"""
        # Reshape for LSTM: (batch, timesteps, features)
        raw_tab_reshaped = raw_tab.reshape(1, raw_tab.shape[0], -1)
        refined = self.model.predict(raw_tab_reshaped)
        refined = refined.reshape(raw_tab.shape)
        
        # Enforce playability (post-process)
        for t in range(1, refined.shape[0]):
            for s in range(6):
                curr_fret = np.argmax(refined[t, s])
                prev_fret = np.argmax(refined[t-1, s])
                if abs(curr_fret - prev_fret) > 4:  # Limit large jumps
                    refined[t, s] = refined[t-1, s]  # Use previous fret
        return refined
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the LSTM model with tablature data"""
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        weights_path = os.path.join(os.path.dirname(__file__), 'lstm_weights.h5')
        self.model.save_weights(weights_path)
        print(f"Model weights saved to {weights_path}")
