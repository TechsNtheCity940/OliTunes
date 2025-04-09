import os
import sys
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import json
from pathlib import Path
import re
import random
import pickle

# Add parent directory to path to import position_predictor
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.fretboard_position_model.position_predictor import FretboardPositionPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FretboardPositionModelTrainer:
    """Trainer for the fretboard position prediction model."""
    
    def __init__(self, model_dir=None, tab_data_dir=None):
        """
        Initialize the trainer.
        
        Args:
            model_dir (str): Directory for model files
            tab_data_dir (str): Directory containing tablature data
        """
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        
        if tab_data_dir is None:
            # Default to tab_data directory
            tab_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), 'data', 'tab_data', 'tab_files')
        
        self.model_dir = model_dir
        self.tab_data_dir = tab_data_dir
        
        # Initialize the position predictor to access its model
        self.position_predictor = FretboardPositionPredictor(model_dir=model_dir)
        
        # Load metadata
        self.metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Create model
        self._create_model()
        
        # Define string tunings (E2, A2, D3, G3, B3, E4)
        self.string_midi_values = [40, 45, 50, 55, 59, 64]
    
    def _create_model(self):
        """Create a new model."""
        input_shape = self.metadata['input_shape'][0]
        output_shape = self.metadata['output_shape'][0]
        
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(output_shape, activation='softmax')
        ])
        
        # Save architecture
        model_json = self.model.to_json()
        architecture_path = os.path.join(self.model_dir, 'model_architecture.json')
        with open(architecture_path, 'w') as f:
            f.write(model_json)
        logger.info(f"Created and saved model architecture to {architecture_path}")
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def parse_tab_file(self, file_path):
        """
        Parse a tablature file to extract note positions.
        
        Args:
            file_path (str): Path to the tab file
            
        Returns:
            list: List of (string, fret, midi_note) tuples
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []
        
        # Extract tab sections (lines with |----|)
        tab_sections = re.findall(r'((?:[EeAaDdGgBb][-|][-\d\w\s|/\\\.]+[\r\n]*){1,6})', content)
        if not tab_sections:
            return []
        
        notes = []
        
        for section in tab_sections:
            # Split into lines
            lines = section.strip().split('\n')
            
            # Filter to only include tab lines
            tab_lines = [line for line in lines if re.match(r'^[EeAaDdGgBb][-|]', line)]
            
            # Skip if we don't have enough strings
            if len(tab_lines) < 4:  # At least 4 strings to be a valid tab
                continue
            
            # Determine string numbers
            string_numbers = []
            for line in tab_lines:
                if line.startswith('E') or line.startswith('e'):
                    if len(string_numbers) == 0:  # First E string encountered is the high E (string 0)
                        string_numbers.append(0)
                    else:  # Second E string is the low E (string 5)
                        string_numbers.append(5)
                elif line.startswith('B') or line.startswith('b'):
                    string_numbers.append(1)
                elif line.startswith('G') or line.startswith('g'):
                    string_numbers.append(2)
                elif line.startswith('D') or line.startswith('d'):
                    string_numbers.append(3)
                elif line.startswith('A') or line.startswith('a'):
                    string_numbers.append(4)
            
            # Skip if string numbers couldn't be determined
            if not string_numbers or len(string_numbers) != len(tab_lines):
                continue
            
            # Extract fret numbers
            for i, line in enumerate(tab_lines):
                if i >= len(string_numbers):
                    continue
                    
                string = string_numbers[i]
                
                # Find all fret numbers in the line
                frets = re.findall(r'(\d+)', line[2:])  # Skip the first two chars (e.g., "E|")
                
                for fret in frets:
                    try:
                        fret_num = int(fret)
                        if 0 <= fret_num <= 24:  # Valid fret range
                            # Calculate MIDI note
                            midi_note = self.string_midi_values[string] + fret_num
                            notes.append((string, fret_num, midi_note))
                    except ValueError:
                        continue
        
        return notes
    
    def load_data(self, max_files=None):
        """
        Load tablature data from tab files.
        
        Args:
            max_files (int, optional): Maximum number of files to load
            
        Returns:
            tuple: (X_train, y_train) training data
        """
        logger.info(f"Loading data from {self.tab_data_dir}")
        
        # Find all tab files
        tab_files = []
        for root, _, files in os.walk(self.tab_data_dir):
            for file in files:
                if file.endswith('.txt'):
                    tab_files.append(os.path.join(root, file))
        
        if not tab_files:
            raise FileNotFoundError(f"No tab files found in {self.tab_data_dir}")
        
        logger.info(f"Found {len(tab_files)} tab files")
        
        if max_files is not None:
            tab_files = tab_files[:max_files]
            logger.info(f"Using {len(tab_files)} files for training")
        
        # Extract notes from tab files
        all_notes = []
        for tab_file in tab_files:
            logger.info(f"Parsing {tab_file}")
            notes = self.parse_tab_file(tab_file)
            all_notes.extend(notes)
            logger.info(f"Extracted {len(notes)} notes from {tab_file}")
        
        if not all_notes:
            raise ValueError("No notes could be extracted from the tab files")
        
        logger.info(f"Extracted a total of {len(all_notes)} notes from all files")
        
        # Prepare training data
        return self._prepare_training_data(all_notes)
    
    def _prepare_training_data(self, notes):
        """
        Prepare training data for the fretboard position model.
        
        Args:
            notes (list): List of (string, fret, midi_note) tuples
            
        Returns:
            tuple: (X_train, y_train) formatted for the model
        """
        logger.info("Preparing training data")
        
        if not notes:
            raise ValueError("No notes available for training")
        
        # Initialize training data arrays
        input_shape = self.metadata['input_shape'][0]
        output_shape = self.metadata['output_shape'][0]
        
        X_train_list = []
        y_train_list = []
        
        # Group notes by their position in the music (to simulate sequences)
        # We'll use a sliding window approach to create context
        window_size = 5
        for i in range(len(notes) - window_size + 1):
            window = notes[i:i+window_size]
            
            # Use the middle note as the target
            target_idx = window_size // 2
            string, fret, midi_note = window[target_idx]
            
            # Use previous notes for context
            if target_idx > 0:
                prev_string, prev_fret, _ = window[target_idx - 1]
            else:
                prev_string, prev_fret = 0, 0
            
            # Create random key and style contexts
            key_context = random.randint(0, 11)
            style = random.randint(0, 4)
            
            # Prepare input features
            features = self.position_predictor.prepare_input_features(
                midi_note, prev_fret, prev_string, key_context, style
            )
            
            # Create one-hot output (string and fret combination)
            output = np.zeros(output_shape)
            output_index = string * 25 + fret
            output[output_index] = 1
            
            X_train_list.append(features[0])
            y_train_list.append(output)
        
        # Add more training examples with variations
        for _ in range(3):  # Create 3 variations for each note
            for string, fret, midi_note in notes:
                # Randomize context
                prev_string = random.randint(0, 5)
                prev_fret = random.randint(0, 24)
                key_context = random.randint(0, 11)
                style = random.randint(0, 4)
                
                # Prepare input features
                features = self.position_predictor.prepare_input_features(
                    midi_note, prev_fret, prev_string, key_context, style
                )
                
                # Create one-hot output (string and fret combination)
                output = np.zeros(output_shape)
                output_index = string * 25 + fret
                output[output_index] = 1
                
                X_train_list.append(features[0])
                y_train_list.append(output)
        
        if not X_train_list:
            raise ValueError("No training examples could be created from the data")
        
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        
        logger.info(f"Created {len(X_train)} training examples")
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        
        return X_train, y_train
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the fretboard position model.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        logger.info("Training fretboard position model")
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks
        )
        
        # Save the model weights using pickle
        weights = self.model.get_weights()
        weights_path = os.path.join(self.model_dir, 'model_weights.pkl')
        with open(weights_path, 'wb') as f:
            pickle.dump(weights, f)
        logger.info(f"Model weights saved to {weights_path}")
        
        # Create model_ready file
        with open(os.path.join(self.model_dir, 'model_ready'), 'w') as f:
            f.write('1')
        
        logger.info("Training complete")
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model.
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating model")
        
        # Evaluate the model
        results = self.model.evaluate(X_test, y_test)
        
        metrics = {
            'loss': results[0],
            'accuracy': results[1]
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

if __name__ == "__main__":
    # Create trainer
    trainer = FretboardPositionModelTrainer()
    
    # Load data
    X_train, y_train = trainer.load_data(max_files=None)  # Use all available files
    
    # Split data for training and testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train model
    trainer.train(X_train, y_train, epochs=50, batch_size=32)
    
    # Evaluate model
    trainer.evaluate(X_test, y_test)
