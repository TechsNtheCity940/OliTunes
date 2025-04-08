#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OliTunes Tab Data Pipeline
==========================

This module creates a robust data pipeline for processing both synthetic data
and real downloaded tabs for training machine learning models.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Local imports
from tab_data_processor import TabDataProcessor
from generate_synthetic_data import SyntheticTabGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tab_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants and paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TAB_DATA_DIR = os.path.join(DATA_DIR, "tab_data")
TAB_FILES_DIR = os.path.join(TAB_DATA_DIR, "tab_files")
PROCESSED_DATA_DIR = os.path.join(TAB_DATA_DIR, "processed")
TRAINING_DATA_DIR = os.path.join(TAB_DATA_DIR, "training")

# Alternative paths (for when tabs are in backend directory)
BACKEND_DATA_DIR = os.path.join(PROJECT_ROOT, "backend", "data")
BACKEND_TAB_DATA_DIR = os.path.join(BACKEND_DATA_DIR, "tab_data")
BACKEND_TAB_FILES_DIR = os.path.join(BACKEND_TAB_DATA_DIR, "tab_files")

# Guitar constants
STRINGS = 6
FRETS = 24
INPUT_FEATURES_PER_NOTE = STRINGS + 2  # String one-hot + fret + midi
MAX_CONTEXT_WINDOW = 5

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)


class TabDataPipeline:
    """
    Data pipeline for processing guitar tablature data and preparing it for model training.
    Combines both synthetic data and real-world tabs.
    """
    
    def __init__(self, context_window: int = 3, validation_split: float = 0.2):
        """
        Initialize the data pipeline.
        
        Args:
            context_window: Number of previous notes to use as context
            validation_split: Fraction of data to use for validation
        """
        self.context_window = context_window
        self.validation_split = validation_split
        self.tab_processor = TabDataProcessor()
        self.synthetic_generator = SyntheticTabGenerator()
        
        # Data tracking
        self.real_tabs_count = 0
        self.synthetic_samples_count = 0
        self.total_training_samples = 0
        
        logger.info(f"Initialized Tab Data Pipeline with context window {context_window}")
    
    def process_real_tabs(self, tab_dir: str = None) -> int:
        """
        Process all tab files in the specified directory.
        
        Args:
            tab_dir: Directory containing tab files (defaults to TAB_FILES_DIR)
            
        Returns:
            Number of processed tabs
        """
        if tab_dir is None:
            # Check both possible tab file locations
            if os.path.exists(BACKEND_TAB_FILES_DIR) and os.listdir(BACKEND_TAB_FILES_DIR):
                tab_dir = BACKEND_TAB_FILES_DIR
                logger.info(f"Using tab files from backend directory: {tab_dir}")
            elif os.path.exists(TAB_FILES_DIR) and os.listdir(TAB_FILES_DIR):
                tab_dir = TAB_FILES_DIR
                logger.info(f"Using tab files from data directory: {tab_dir}")
            else:
                logger.error(f"No tab files found in either {BACKEND_TAB_FILES_DIR} or {TAB_FILES_DIR}")
                return 0
        
        if not os.path.exists(tab_dir):
            logger.error(f"Tab directory {tab_dir} does not exist")
            return 0
        
        # Find all .txt files in the directory
        tab_files = [os.path.join(tab_dir, f) for f in os.listdir(tab_dir) 
                    if f.endswith('.txt')]
        
        if not tab_files:
            logger.warning(f"No tab files found in {tab_dir}")
            return 0
        
        logger.info(f"Processing {len(tab_files)} tab files from {tab_dir}")
        
        # Process each tab file
        tabs_processed = 0
        total_positions = 0
        
        for tab_file in tqdm(tab_files, desc="Processing tab files"):
            try:
                # Extract metadata from filename
                filename = os.path.basename(tab_file)
                parts = filename.split('_', 1)
                
                metadata = {
                    "artist": parts[0] if len(parts) > 1 else "Unknown",
                    "song": parts[1].rsplit('.', 1)[0] if len(parts) > 1 else filename.rsplit('.', 1)[0],
                    "genre": self._extract_genre_from_metadata(parts[0] if len(parts) > 1 else "Unknown"),
                    "rating": 5.0  # Default rating
                }
                
                # Extract positions from tab file
                positions = self.tab_processor.parse_tab_file(tab_file, metadata)
                total_positions += positions
                
                if positions > 0:
                    tabs_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing tab file {tab_file}: {e}")
        
        logger.info(f"Successfully processed {tabs_processed} tab files with {total_positions} positions")
        self.real_tabs_count = tabs_processed
        
        return tabs_processed
    
    def _extract_genre_from_metadata(self, artist: str) -> str:
        """
        Attempt to extract genre information from artist name.
        
        Args:
            artist: Artist name
            
        Returns:
            Estimated genre
        """
        # This is a simplistic approach - could be improved with a lookup table
        artist = artist.lower()
        
        if any(metal_band in artist for metal_band in ['metallica', 'killswitch', 'all that remains']):
            return "metal"
        elif any(rock_band in artist for rock_band in ['paramore', 'incubus']):
            return "rock"
        elif any(country_artist in artist for country_artist in ['johnny cash', 'morgan wallen']):
            return "country"
        elif any(pop_artist in artist for pop_artist in ['ed sheeran']):
            return "pop"
        elif any(blues_artist in artist for blues_artist in ['john mayer']):
            return "blues"
        
        return "rock"  # Default genre
    
    def generate_synthetic_data(self, num_samples: int = 5000, styles: List[str] = None) -> int:
        """
        Generate synthetic training data based on music theory principles.
        
        Args:
            num_samples: Number of samples to generate
            styles: List of styles to generate data for
            
        Returns:
            Number of samples generated
        """
        if styles is None:
            styles = ["blues", "rock", "jazz", "folk", "metal", "country", "pop"]
        
        logger.info(f"Generating {num_samples} synthetic samples across {len(styles)} styles")
        
        # Generate data using the synthetic generator
        samples_generated = self.synthetic_generator.generate_dataset(
            output_path=os.path.join(PROCESSED_DATA_DIR, "synthetic_positions.csv"),
            num_samples=num_samples,
            styles=styles
        )
        
        self.synthetic_samples_count = samples_generated
        logger.info(f"Generated {samples_generated} synthetic samples")
        
        return samples_generated
    
    def create_training_data(self, 
                           include_real: bool = True, 
                           include_synthetic: bool = True,
                           synthetic_ratio: float = 0.5,
                           total_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training datasets combining real and synthetic data.
        
        Args:
            include_real: Whether to include real tab data
            include_synthetic: Whether to include synthetic data
            synthetic_ratio: Ratio of synthetic to real data (if both included)
            total_samples: Target number of total samples
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        real_data = None
        synthetic_data = None
        
        # Process real tabs data if needed
        if include_real:
            real_data_path = os.path.join(PROCESSED_DATA_DIR, "real_positions.csv")
            
            if not os.path.exists(real_data_path) or os.path.getsize(real_data_path) == 0:
                logger.info("Real tab positions file not found, processing tab files...")
                self.process_real_tabs()
                self.tab_processor.save_positions(real_data_path)
            
            try:
                real_data = pd.read_csv(real_data_path)
                logger.info(f"Loaded {len(real_data)} real tab positions")
            except Exception as e:
                logger.error(f"Error loading real tab data: {e}")
                real_data = None
        
        # Process synthetic data if needed
        if include_synthetic:
            synthetic_data_path = os.path.join(PROCESSED_DATA_DIR, "synthetic_positions.csv")
            
            if not os.path.exists(synthetic_data_path) or os.path.getsize(synthetic_data_path) == 0:
                logger.info("Synthetic positions file not found, generating data...")
                synthetic_samples = total_samples if not include_real else int(total_samples * synthetic_ratio)
                self.generate_synthetic_data(num_samples=synthetic_samples)
            
            try:
                synthetic_data = pd.read_csv(synthetic_data_path)
                logger.info(f"Loaded {len(synthetic_data)} synthetic tab positions")
            except Exception as e:
                logger.error(f"Error loading synthetic data: {e}")
                synthetic_data = None
        
        # Combine datasets based on the synthetic ratio
        if real_data is not None and synthetic_data is not None:
            # Calculate number of samples to include
            real_samples = int(total_samples * (1 - synthetic_ratio))
            synth_samples = int(total_samples * synthetic_ratio)
            
            # Sample from both datasets
            if len(real_data) > real_samples:
                real_data = real_data.sample(real_samples)
            
            if len(synthetic_data) > synth_samples:
                synthetic_data = synthetic_data.sample(synth_samples)
            
            # Combine datasets
            combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
            logger.info(f"Combined dataset has {len(combined_data)} samples")
            
        elif real_data is not None:
            combined_data = real_data
            logger.info(f"Using only real data with {len(combined_data)} samples")
            
        elif synthetic_data is not None:
            combined_data = synthetic_data
            logger.info(f"Using only synthetic data with {len(combined_data)} samples")
            
        else:
            logger.error("No data available for training")
            return np.array([]), np.array([])
        
        # Convert to features and targets
        X, y = self._convert_to_features(combined_data)
        self.total_training_samples = len(X)
        
        # Save the processed data
        output_path = os.path.join(TRAINING_DATA_DIR, f"training_data_cw{self.context_window}")
        np.savez(output_path, X=X, y=y)
        logger.info(f"Saved training data with {len(X)} samples to {output_path}.npz")
        
        return X, y
    
    def _convert_to_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert position data to input features and targets for the model.
        
        Args:
            data: DataFrame with position data
            
        Returns:
            X (input features) and y (target values) as numpy arrays
        """
        logger.info("Converting position data to training features...")
        
        # Sort data by song and position
        data = data.sort_values(['song', 'position_in_tab'])
        
        # Get unique songs
        songs = data['song'].unique()
        logger.info(f"Processing {len(songs)} unique songs")
        
        # Create empty lists for features and targets
        X = []
        y = []
        
        # Track a few examples for debugging
        debug_samples = 0
        
        # Process each song to maintain context
        for song in tqdm(songs, desc="Processing songs"):
            song_data = data[data['song'] == song]
            
            # Get positions sorted by position in tab
            sorted_positions = song_data.sort_values('position_in_tab')
            
            # Skip songs with too few positions
            if len(sorted_positions) < self.context_window + 1:
                continue
                
            try:
                # Process each position with its context
                for i in range(len(sorted_positions) - self.context_window):
                    # Get context window positions
                    context = sorted_positions.iloc[i:i+self.context_window]
                    # Get next position (target)
                    target = sorted_positions.iloc[i+self.context_window]
                    
                    # Create input features
                    features = []
                    for _, pos in context.iterrows():
                        # For each position in context, create one-hot for string
                        string_features = [0] * STRINGS
                        
                        # Make sure we have an integer for the string index
                        string_idx = int(pos['string'])
                        if 0 <= string_idx < STRINGS:
                            string_features[string_idx] = 1
                        else:
                            logger.warning(f"Invalid string index: {string_idx}, using 0")
                            string_features[0] = 1  # Default to first string
                        
                        # Add string one-hot encoding
                        features.extend(string_features)
                        
                        # Add fret (normalized)
                        fret = int(pos['fret'])
                        features.append(fret / FRETS)
                        
                        # Add MIDI note (normalized)
                        midi = int(pos['midi_note'])
                        features.append(midi / 127.0)
                    
                    # Create target vector (one-hot for string, normalized fret)
                    target_string_idx = int(target['string'])
                    target_fret = int(target['fret'])
                    
                    # String one-hot
                    target_string = [0] * STRINGS
                    if 0 <= target_string_idx < STRINGS:
                        target_string[target_string_idx] = 1
                    else:
                        logger.warning(f"Invalid target string: {target_string_idx}, using 0")
                        target_string[0] = 1  # Default to first string
                    
                    # Create target vector by concatenating string one-hot and fret features
                    target_vector = np.zeros(STRINGS + FRETS)
                    
                    # First part is string one-hot
                    for i in range(STRINGS):
                        if i < len(target_string):  # Add this check for safety
                            target_vector[i] = target_string[i]
                    
                    # Second part is fret one-hot
                    if 0 <= target_fret < FRETS:
                        target_vector[STRINGS + target_fret] = 1
                    
                    # Log a few examples for debugging
                    if debug_samples < 3:
                        logger.debug(f"Feature vector length: {len(features)}")
                        logger.debug(f"Target vector length: {len(target_vector)}")
                        logger.debug(f"Target: String {target_string_idx}, Fret {target_fret}")
                        debug_samples += 1
                    
                    X.append(features)
                    y.append(target_vector)
            except Exception as e:
                logger.error(f"Error processing song {song}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        if not X:
            logger.error("No valid examples were generated")
            return np.array([]), np.array([])
        
        # Convert to numpy arrays
        try:
            X_array = np.array(X, dtype=np.float32)
            y_array = np.array(y, dtype=np.float32)
            
            logger.info(f"Created dataset with {len(X_array)} examples")
            logger.info(f"Input shape: {X_array.shape}, Output shape: {y_array.shape}")
            
            return X_array, y_array
        except Exception as e:
            logger.error(f"Error converting to numpy arrays: {e}")
            return np.array([]), np.array([])
    
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load or create training data and split into train/validation sets.
        
        Returns:
            X_train, X_val, y_train, y_val
        """
        # Check for existing processed data
        training_file = os.path.join(TRAINING_DATA_DIR, f"training_data_cw{self.context_window}.npz")
        
        if os.path.exists(training_file):
            logger.info(f"Loading existing training data from {training_file}")
            
            try:
                data = np.load(training_file)
                X = data['X']
                y = data['y']
                
                # Split into train/validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.validation_split, random_state=42
                )
                
                logger.info(f"Loaded training data: {X.shape[0]} samples")
                logger.info(f"Training set: {X_train.shape[0]} samples, Validation set: {X_val.shape[0]} samples")
                
                return X_train, X_val, y_train, y_val
                
            except Exception as e:
                logger.error(f"Error loading training data: {e}")
                # Continue to create new dataset
        
        logger.info(f"Training data not found at {training_file}, creating new dataset")
        
        # Create the training data
        X, y = self.create_training_data()
        
        if len(X) == 0 or len(y) == 0:
            logger.error("Failed to create training data")
            raise ValueError("No training data could be created")
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42
        )
        
        logger.info(f"Created new training data: {X.shape[0]} samples")
        logger.info(f"Training set: {X_train.shape[0]} samples, Validation set: {X_val.shape[0]} samples")
        
        return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    # Example usage
    pipeline = TabDataPipeline(context_window=3)
    X_train, X_val, y_train, y_val = pipeline.load_training_data()
    print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
