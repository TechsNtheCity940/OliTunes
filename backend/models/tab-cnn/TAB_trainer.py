import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import time
from datetime import datetime
from tensorflow.keras import mixed_precision
import argparse
import json
import glob
from scipy.ndimage import gaussian_filter1d
import traceback

# Get the backend directory path
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(script_dir, "../.."))

# GPU configuration function
def configure_gpu():
    """Configure TensorFlow to use GPU if available, with memory growth enabled."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
            return True
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPU found. Using CPU.")
    return False

# Configure GPU at module import time
has_gpu = configure_gpu()

# Load training data from batch directories
def load_training_data(data_path=None, use_synthetic=True, use_ultimate_guitar=False, batch_size=32, validation_split=0.2):
    """
    Load training data from the specified path.
    
    Args:
        data_path: Path to the training data directory
        use_synthetic: Whether to use the synthetic data (batch_0 to batch_19)
        use_ultimate_guitar: Whether to use the Ultimate Guitar tab data
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        
    Returns:
        X_train, y_train, X_val, y_val: Training and validation data
    """
    if data_path is None:
        data_path = os.path.join(backend_dir, "data", "guitarset_training_data")
    
    X_all = []
    y_all = []
    
    # Load synthetic data if requested
    if use_synthetic:
        print("Loading synthetic data...")
        batch_dirs = [os.path.join(data_path, f"batch_{i}") for i in range(20)]
        for batch_dir in batch_dirs:
            if os.path.exists(batch_dir):
                print(f"Loading from {batch_dir}")
                X_files = sorted([f for f in os.listdir(batch_dir) if f.endswith('_x.npy')])
                y_files = sorted([f for f in os.listdir(batch_dir) if f.endswith('_y.npy')])
                
                for x_file, y_file in zip(X_files, y_files):
                    try:
                        x = np.load(os.path.join(batch_dir, x_file))
                        y = np.load(os.path.join(batch_dir, y_file))
                        
                        # Ensure data has the correct shape (time_steps, features)
                        if len(x.shape) == 2:
                            X_all.append(x)
                            y_all.append(y)
                    except Exception as e:
                        print(f"Error loading {x_file} or {y_file}: {e}")
    
    # Load Ultimate Guitar tab data if requested
    if use_ultimate_guitar:
        ug_dir = os.path.join(data_path, "ultimate_guitar")
        if os.path.exists(ug_dir):
            print(f"Loading Ultimate Guitar tab data from {ug_dir}")
            X_files = sorted([f for f in os.listdir(ug_dir) if f.endswith('_x.npy')])
            y_files = sorted([f for f in os.listdir(ug_dir) if f.endswith('_y.npy')])
            
            for x_file, y_file in zip(X_files, y_files):
                try:
                    x = np.load(os.path.join(ug_dir, x_file))
                    y = np.load(os.path.join(ug_dir, y_file))
                    
                    # Ensure data has the correct shape (time_steps, features)
                    if len(x.shape) == 2:
                        X_all.append(x)
                        y_all.append(y)
                except Exception as e:
                    print(f"Error loading {x_file} or {y_file}: {e}")
    
    # Check if we have any data
    if len(X_all) == 0:
        print("No training data found. Generating synthetic data...")
        X_all, y_all = generate_synthetic_time_series_data(num_samples=100)
    
    # Convert to numpy arrays
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    # Print shapes for debugging
    print(f"Loaded data shapes: X: {X_all.shape}, y: {y_all.shape}")
    
    # Ensure the data has the correct shape for Conv1D layers
    # Conv1D expects (batch_size, time_steps, features)
    if len(X_all.shape) == 3:
        # Data is already in the correct shape
        pass
    elif len(X_all.shape) == 2:
        # Reshape to add a time dimension if it's missing
        # Assume the first dimension is the batch size
        # and the second dimension is the features
        X_all = X_all.reshape(X_all.shape[0], 1, X_all.shape[1])
        y_all = y_all.reshape(y_all.shape[0], 1, y_all.shape[1])
    
    # Split into training and validation sets
    split_idx = int(len(X_all) * (1 - validation_split))
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]
    
    print(f"Training data shapes: X: {X_train.shape}, y: {y_train.shape}")
    print(f"Validation data shapes: X: {X_val.shape}, y: {y_val.shape}")
    
    return X_train, y_train, X_val, y_val

def generate_synthetic_time_series_data(num_samples=1000, time_steps=50, mel_bins=128, fret_features=150):
    """
    Generate synthetic time-series data matching the actual training data structure
    
    Args:
        num_samples: Number of samples to generate
        time_steps: Number of time steps in each sample
        mel_bins: Number of mel bins in the input features
        fret_features: Number of fret features in the output (6 strings x 25 frets = 150)
        
    Returns:
        X, y: Input features and target positions
    """
    print(f"Generating {num_samples} synthetic training samples...")
    
    # Initialize arrays
    X = np.zeros((num_samples, time_steps, mel_bins))
    y = np.zeros((num_samples, time_steps, fret_features))
    
    # Base frequency for each string (in Hz)
    string_base_freq = {
        0: 329.63,  # E4 (high E)
        1: 246.94,  # B3
        2: 196.00,  # G3
        3: 146.83,  # D3
        4: 110.00,  # A2
        5: 82.41    # E2 (low E)
    }
    
    # Generate data for each sample
    for i in range(num_samples):
        # Create a random "melody" with 3-10 notes
        num_notes = np.random.randint(3, 11)
        note_times = sorted(np.random.choice(range(time_steps), size=num_notes, replace=False))
        
        # Choose a random string for this sample (weighted toward middle strings)
        string_weights = [0.15, 0.2, 0.25, 0.25, 0.1, 0.05]  # Weights for strings 0-5
        primary_string = np.random.choice(6, p=string_weights)
        
        # Generate a sequence of frets that would be physically playable
        # Start with a random position between 0-12
        current_fret = np.random.randint(0, 13)
        
        # For each note time, generate a note
        for t_idx, t in enumerate(note_times):
            # Add some randomness to fret movement (but keep it physically plausible)
            # Most movements are small, occasionally larger jumps
            if t_idx > 0:
                if np.random.random() < 0.7:  # 70% chance of small movement
                    fret_change = np.random.choice([-2, -1, 0, 1, 2], p=[0.1, 0.2, 0.4, 0.2, 0.1])
                else:  # 30% chance of larger jump
                    fret_change = np.random.choice([-5, -4, -3, 3, 4, 5], p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.1])
                
                current_fret += fret_change
                # Keep fret in valid range
                current_fret = max(0, min(24, current_fret))
            
            # Set the fret position in the target array
            y[i, t, primary_string * 25 + current_fret] = 1
            
            # Occasionally add a note on an adjacent string
            if np.random.random() < 0.3:  # 30% chance
                # Choose an adjacent string
                adjacent_strings = []
                if primary_string > 0:
                    adjacent_strings.append(primary_string - 1)
                if primary_string < 5:
                    adjacent_strings.append(primary_string + 1)
                
                if adjacent_strings:
                    secondary_string = np.random.choice(adjacent_strings)
                    # Choose a nearby fret
                    secondary_fret = current_fret + np.random.choice([-2, -1, 0, 1, 2])
                    secondary_fret = max(0, min(24, secondary_fret))
                    
                    # Add the note
                    y[i, t, secondary_string * 25 + secondary_fret] = 1
        
        # Generate corresponding mel spectrogram features
        # Add some background noise
        X[i] = np.random.rand(time_steps, mel_bins) * 0.05
        
        # For each time step, add energy in frequency bands corresponding to active notes
        for t in range(time_steps):
            # Find active positions at this time step
            active_positions = np.where(y[i, t] > 0)[0]
            
            for pos in active_positions:
                # Determine which string and fret
                string_number = pos // 25
                fret = pos % 25
                
                if string_number in string_base_freq:
                    # Calculate the frequency of the note (base frequency * 2^(fret/12))
                    freq = string_base_freq[string_number] * (2 ** (fret / 12))
                    
                    # Convert frequency to mel bin index (approximate)
                    # Mel scale formula: m = 2595 * log10(1 + f/700)
                    mel = 2595 * np.log10(1 + freq / 700)
                    
                    # Scale to our mel_bins range (0 to mel_bins-1)
                    max_mel = 2595 * np.log10(1 + 20000 / 700)  # Max frequency ~20kHz
                    bin_index = int((mel / max_mel) * mel_bins)
                    
                    # Ensure bin_index is within range
                    bin_index = max(0, min(bin_index, mel_bins - 1))
                    
                    # Add energy in a band around this frequency (with harmonics)
                    X[i, t, bin_index] = 1.0
                    
                    # Add harmonics
                    for h in range(1, 5):
                        harmonic_idx = int(min(mel_bins - 1, (freq * h) / 20))
                        X[i, t, harmonic_idx] = 1.0 / h
        
        # Apply some temporal smoothing to simulate real audio
        for t in range(1, time_steps):
            X[i, t] = 0.7 * X[i, t] + 0.3 * X[i, t-1]
    
    # Normalize the input features
    for i in range(num_samples):
        X[i] = (X[i] - X[i].min()) / (X[i].max() - X[i].min() + 1e-8)
    
    print(f"Generated synthetic data shapes: X: {X.shape}, y: {y.shape}")
    return X, y

def generate_synthetic_data(num_samples=1000, time_steps=50, mel_bins=128, output_dir=None):
    """
    Generate synthetic training data for the TabCNN model.
    
    Args:
        num_samples: Number of synthetic samples to generate
        time_steps: Number of time steps in each sample
        mel_bins: Number of mel frequency bins in each sample
        output_dir: Directory to save the synthetic data
        
    Returns:
        X: Input features (num_samples, time_steps, mel_bins)
        y: Target tablature (num_samples, time_steps, 6*25)
    """
    print(f"Generating {num_samples} synthetic samples...")
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                 'data', 'guitarset_training_data', 'synthetic')
    
    os.makedirs(output_dir, exist_ok=True)
    
    X = np.zeros((num_samples, time_steps, mel_bins))
    y = np.zeros((num_samples, time_steps, 6 * 25))
    
    # Common guitar chord shapes and patterns
    common_chords = [
        # Open chords
        [0, 2, 2, 1, 0, 0],  # E major
        [3, 2, 0, 0, 0, 3],  # G major
        [0, 0, 0, 2, 3, 2],  # A major
        [0, 1, 2, 2, 0, 0],  # E minor
        [1, 3, 3, 2, 1, 1],  # F major
        [3, 3, 0, 0, 2, 3],  # G minor
        # Barre chords
        [1, 3, 3, 2, 1, 1],  # F major barre
        [2, 4, 4, 3, 2, 2],  # F# major barre
        [3, 5, 5, 4, 3, 3],  # G major barre
        [1, 3, 3, 1, 1, 1],  # F minor barre
        # Power chords
        [1, 3, 3, -1, -1, -1],  # F5
        [3, 5, 5, -1, -1, -1],  # G5
        [5, 7, 7, -1, -1, -1],  # A5
    ]
    
    # Common scale patterns
    common_scales = [
        # Major scale patterns
        [0, 2, 4, 5, 7, 9, 11, 12],  # C major
        [2, 4, 6, 7, 9, 11, 13, 14],  # D major
        # Minor scale patterns
        [0, 2, 3, 5, 7, 8, 10, 12],  # A minor
        [2, 4, 5, 7, 9, 10, 12, 14],  # B minor
    ]
    
    # Common melodic patterns
    common_melodic_patterns = [
        # Simple ascending/descending runs
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],  # Ascending on high E
        [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],  # Ascending on B
        [(0, 4), (0, 3), (0, 2), (0, 1), (0, 0)],  # Descending on high E
        # Pentatonic patterns
        [(0, 0), (0, 3), (1, 0), (1, 3), (2, 0)],  # E minor pentatonic
        [(0, 5), (0, 8), (1, 5), (1, 8), (2, 5)],  # A minor pentatonic
    ]
    
    for i in range(num_samples):
        # Create a random melody or chord progression
        pattern_type = np.random.choice(['chord', 'scale', 'melody', 'mixed'])
        
        # Initialize tab for this sample
        tab = np.zeros((time_steps, 6, 25))
        
        if pattern_type == 'chord':
            # Generate a chord progression
            num_chords = np.random.randint(3, 8)
            chord_indices = np.random.choice(len(common_chords), num_chords)
            chord_durations = np.random.randint(2, 10, size=num_chords)
            
            current_step = 0
            for c_idx, duration in zip(chord_indices, chord_durations):
                if current_step >= time_steps:
                    break
                    
                chord = common_chords[c_idx]
                end_step = min(current_step + duration, time_steps)
                
                for t in range(current_step, end_step):
                    for string, fret in enumerate(chord):
                        if fret >= 0:  # -1 means don't play this string
                            tab[t, string, fret] = 1
                
                current_step = end_step
                
        elif pattern_type == 'scale':
            # Generate a scale run
            scale_idx = np.random.choice(len(common_scales))
            scale = common_scales[scale_idx]
            
            # Choose a starting position
            start_fret = np.random.randint(0, 12)
            start_string = np.random.randint(0, 4)  # Start on the top 4 strings
            
            # Generate a scale run
            current_step = 0
            direction = np.random.choice(['ascending', 'descending', 'mixed'])
            
            if direction == 'ascending':
                notes = scale
            elif direction == 'descending':
                notes = scale[::-1]
            else:
                # Mix ascending and descending
                notes = scale + scale[::-1]
            
            for note in notes:
                if current_step >= time_steps:
                    break
                
                fret = (start_fret + note) % 25
                string = max(0, min(5, start_string + np.random.randint(-1, 2)))
                
                tab[current_step, string, fret] = 1
                current_step += np.random.randint(1, 3)  # Random duration
                
        elif pattern_type == 'melody':
            # Generate a melodic pattern
            pattern_idx = np.random.choice(len(common_melodic_patterns))
            pattern = common_melodic_patterns[pattern_idx]
            
            # Choose a starting position
            start_fret = np.random.randint(0, 15)
            
            # Apply the pattern multiple times with variations
            current_step = 0
            repetitions = np.random.randint(2, 5)
            
            for _ in range(repetitions):
                if current_step >= time_steps:
                    break
                
                for string, fret_offset in pattern:
                    if current_step >= time_steps:
                        break
                    
                    fret = min(24, start_fret + fret_offset)
                    tab[current_step, string, fret] = 1
                    current_step += 1
                
                # Add a small variation to the starting position
                start_fret = max(0, min(20, start_fret + np.random.randint(-2, 3)))
                
        else:  # mixed
            # Combine chord and melody
            # Start with a chord
            chord_idx = np.random.choice(len(common_chords))
            chord = common_chords[chord_idx]
            
            # Add the chord at various positions
            chord_positions = np.random.choice(time_steps, size=np.random.randint(3, 8), replace=False)
            chord_positions.sort()
            
            for pos in chord_positions:
                if pos < time_steps:
                    for string, fret in enumerate(chord):
                        if fret >= 0:
                            tab[pos, string, fret] = 1
            
            # Add a melody on top
            melody_string = np.random.randint(0, 3)  # Top 3 strings for melody
            current_step = 0
            
            # Choose a scale to use for the melody
            scale_idx = np.random.choice(len(common_scales))
            scale = common_scales[scale_idx]
            start_fret = np.random.randint(0, 12)
            
            # Generate a simple melody
            num_notes = np.random.randint(10, 20)
            for _ in range(num_notes):
                if current_step >= time_steps:
                    break
                
                # Skip positions where we already have a chord
                while current_step in chord_positions and current_step < time_steps:
                    current_step += 1
                
                if current_step >= time_steps:
                    break
                
                note_idx = np.random.choice(len(scale))
                fret = (start_fret + scale[note_idx]) % 25
                
                tab[current_step, melody_string, fret] = 1
                current_step += np.random.randint(1, 4)  # Random duration
        
        # Convert 3D tab to 2D target format
        y_sample = tab.reshape(time_steps, -1)
        y[i] = y_sample
        
        # Generate corresponding input features (simplified mel spectrogram)
        X_sample = np.zeros((time_steps, mel_bins))
        
        for t in range(time_steps):
            # Find which strings and frets are played at this time step
            for string in range(6):
                for fret in range(25):
                    if tab[t, string, fret] > 0:
                        # Calculate the fundamental frequency for this string/fret
                        # E2=82.41Hz, A2=110Hz, D3=146.8Hz, G3=196Hz, B3=246.9Hz, E4=329.6Hz
                        base_freqs = [329.6, 246.9, 196.0, 146.8, 110.0, 82.41]
                        freq = base_freqs[string] * (2 ** (fret / 12))
                        
                        # Convert frequency to mel bin index (simplified)
                        # Mel scale formula: m = 2595 * log10(1 + f/700)
                        mel = 2595 * np.log10(1 + freq / 700)
                        
                        # Scale to our mel_bins range (0 to mel_bins-1)
                        max_mel = 2595 * np.log10(1 + 20000 / 700)  # Max frequency ~20kHz
                        bin_index = int((mel / max_mel) * mel_bins)
                        
                        # Ensure bin_index is within range
                        bin_index = max(0, min(bin_index, mel_bins - 1))
                        
                        # Add energy at this frequency and harmonics
                        X_sample[t, bin_index] = 1.0
                        
                        # Add harmonics
                        for h in range(1, 5):
                            harmonic_idx = int(min(mel_bins - 1, (freq * h) / 20))
                            X_sample[t, harmonic_idx] = 1.0 / h
        
        # Add some noise and blur to make it more realistic
        X_sample += np.random.normal(0, 0.01, X_sample.shape)
        X_sample = gaussian_filter1d(X_sample, sigma=1, axis=1)
        
        # Normalize
        if X_sample.max() > 0:
            X_sample /= X_sample.max()
        
        X[i] = X_sample
        
        # Save every 100 samples to avoid memory issues
        if (i + 1) % 100 == 0 or i == num_samples - 1:
            batch_num = i // 100
            batch_file = os.path.join(output_dir, f'synthetic_data_batch_{batch_num}.npz')
            start_idx = batch_num * 100
            end_idx = min(start_idx + 100, num_samples)
            
            np.savez(batch_file, 
                     X=X[start_idx:end_idx], 
                     y=y[start_idx:end_idx])
            
            print(f"Saved batch {batch_num} to {batch_file}")
    
    return X, y

def build_tabcnn_model(input_shape, output_shape, model_type='enhanced'):
    """
    Build the TabCNN model.
    
    Args:
        input_shape: Shape of the input data (time_steps, mel_bins)
        output_shape: Shape of the output data (time_steps, 6*25)
        model_type: Type of model architecture to use
        
    Returns:
        model: Compiled Keras model
    """
    print(f"Building model with input_shape={input_shape}, output_shape={output_shape}")
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    if model_type == 'basic':
        # Basic model with convolutional layers
        x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
        x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        
        # Upsample back to original time steps
        x = tf.keras.layers.UpSampling1D(2)(x)
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling1D(2)(x)
        x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(output_shape[-1], activation='sigmoid')(x)
    
    elif model_type == 'enhanced':
        # Enhanced model with residual connections and more layers
        # Initial convolutional layers
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # First residual block
        residual = x
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, residual])
        
        # Second residual block with increased filters
        residual = tf.keras.layers.Conv1D(128, 1, padding='same')(x)
        x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, residual])
        
        # LSTM for temporal modeling
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output layers
        x = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        outputs = tf.keras.layers.Dense(output_shape[-1], activation='sigmoid')(x)
    
    elif model_type == 'attention':
        # Model with attention mechanism using Keras layers properly
        # First, apply convolutional layers to extract features
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Apply self-attention using MultiHeadAttention layer
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=32
        )(x, x)
        
        # Add skip connection
        x = tf.keras.layers.Add()([x, attention_output])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Add bidirectional LSTM for temporal modeling
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(output_shape[-1], activation='sigmoid')(x)
        
    elif model_type == 'advanced':
        # Advanced model with multiple attention layers and deeper architecture
        
        # Initial feature extraction
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # First attention block with proper query/key/value dimensions
        residual1 = x
        attention1 = tf.keras.layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=32,
            value_dim=32,
            dropout=0.1
        )(x, x, x, use_causal_mask=False)
        x = tf.keras.layers.Add()([x, attention1])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Feed-forward network
        ffn1 = tf.keras.layers.Dense(256, activation='relu')(x)
        ffn1 = tf.keras.layers.Dropout(0.1)(ffn1)
        ffn1 = tf.keras.layers.Dense(128, activation='relu')(ffn1)
        x = tf.keras.layers.Add()([x, ffn1])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Second attention block
        residual2 = x
        attention2 = tf.keras.layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=32,
            value_dim=32,
            dropout=0.1
        )(x, x, x, use_causal_mask=False)
        x = tf.keras.layers.Add()([x, attention2])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Feed-forward network
        ffn2 = tf.keras.layers.Dense(256, activation='relu')(x)
        ffn2 = tf.keras.layers.Dropout(0.1)(ffn2)
        ffn2 = tf.keras.layers.Dense(128, activation='relu')(ffn2)
        x = tf.keras.layers.Add()([x, ffn2])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Add bidirectional LSTM for temporal modeling
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, 
                                                              recurrent_dropout=0.1, 
                                                              dropout=0.1))(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Add a second LSTM layer
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True,
                                                              recurrent_dropout=0.1,
                                                              dropout=0.1))(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Fretboard-aware output layer
        # Process each string with a separate Conv1D
        string_outputs = []
        
        # Create a dense layer to map to 6*25 features first
        x = tf.keras.layers.Dense(6*25)(x)
        
        # Reshape to (batch, time_steps, 6, 25) to process each string separately
        reshape_layer = tf.keras.layers.Reshape((input_shape[0], 6, 25))(x)
        
        for i in range(6):
            # Extract this string's data
            string_data = tf.keras.layers.Lambda(lambda x: x[:, :, i, :])(reshape_layer)
            
            # Apply string-specific processing
            string_conv = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(string_data)
            string_conv = tf.keras.layers.BatchNormalization()(string_conv)
            string_output = tf.keras.layers.Dense(25, activation='sigmoid')(string_conv)
            string_outputs.append(string_output)
        
        # Concatenate string outputs
        concat_strings = tf.keras.layers.Concatenate(axis=-1)(string_outputs)
        
        # Final output
        outputs = tf.keras.layers.Reshape((input_shape[0], 6*25))(concat_strings)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model with custom metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss='binary_crossentropy',
        metrics=[
            CustomAccuracy(),
            CustomPrecision(),
            CustomRecall(),
            CustomF1Score(),
            CustomAUC()
        ]
    )
    
    model.summary()
    
    return model

class CustomAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='custom_accuracy', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten all dimensions except the last one
        y_true_flat = tf.reshape(y_true, [-1, y_true.shape[-1]])
        y_pred_flat = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        
        # Apply threshold to get binary predictions
        y_pred_binary = tf.cast(y_pred_flat > self.threshold, tf.float32)
        
        # Calculate metrics
        self.true_positives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_flat, 1), tf.equal(y_pred_binary, 1)), tf.float32)))
        self.false_positives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_flat, 0), tf.equal(y_pred_binary, 1)), tf.float32)))
        self.true_negatives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_flat, 0), tf.equal(y_pred_binary, 0)), tf.float32)))
        self.false_negatives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_flat, 1), tf.equal(y_pred_binary, 0)), tf.float32)))
    
    def result(self):
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / (total + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_negatives.assign(0)

class CustomPrecision(tf.keras.metrics.Metric):
    def __init__(self, name='custom_precision', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten all dimensions except the last one
        y_true_flat = tf.reshape(y_true, [-1, y_true.shape[-1]])
        y_pred_flat = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        
        # Apply threshold to get binary predictions
        y_pred_binary = tf.cast(y_pred_flat > self.threshold, tf.float32)
        
        # Calculate metrics
        self.true_positives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_flat, 1), tf.equal(y_pred_binary, 1)), tf.float32)))
        self.false_positives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_flat, 0), tf.equal(y_pred_binary, 1)), tf.float32)))
    
    def result(self):
        return self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)

class CustomRecall(tf.keras.metrics.Metric):
    def __init__(self, name='custom_recall', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten all dimensions except the last one
        y_true_flat = tf.reshape(y_true, [-1, y_true.shape[-1]])
        y_pred_flat = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        
        # Apply threshold to get binary predictions
        y_pred_binary = tf.cast(y_pred_flat > self.threshold, tf.float32)
        
        # Calculate metrics
        self.true_positives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_flat, 1), tf.equal(y_pred_binary, 1)), tf.float32)))
        self.false_negatives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_flat, 1), tf.equal(y_pred_binary, 0)), tf.float32)))
    
    def result(self):
        return self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.true_positives.assign(0)
        self.false_negatives.assign(0)

class CustomF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='custom_f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = CustomPrecision(threshold=threshold)
        self.recall = CustomRecall(threshold=threshold)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

class CustomAUC(tf.keras.metrics.AUC):
    def __init__(self, name='custom_auc', **kwargs):
        super().__init__(name=name, **kwargs)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten all dimensions except the last one
        y_true_flat = tf.reshape(y_true, [-1, y_true.shape[-1]])
        y_pred_flat = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        
        return super().update_state(y_true_flat, y_pred_flat, sample_weight)

# Training script
def train_tabcnn(X_train, y_train, X_val, y_val, input_shape, output_shape, model_type='enhanced', batch_size=32, epochs=50, verbose=1):
    """
    Train the TabCNN model.
    
    Args:
        X_train: Training input features
        y_train: Training target tablature
        X_val: Validation input features
        y_val: Validation target tablature
        input_shape: Shape of the input data (time_steps, mel_bins)
        output_shape: Shape of the output data (time_steps, 6*25)
        model_type: Type of model architecture to use
        batch_size: Batch size for training
        epochs: Number of epochs to train for
        verbose: Verbosity level for training
        
    Returns:
        model: Trained model
        history: Training history
    """
    print(f"Training TabCNN model with {model_type} architecture...")
    print(f"Input shape: {input_shape}, Output shape: {output_shape}")
    
    # Build the model
    model = build_tabcnn_model(input_shape, output_shape, model_type)
    
    # Create a timestamp-based directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(backend_dir, "models", "tab-cnn", "trained_models", f"tabcnn_{model_type}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model architecture diagram
    try:
        tf.keras.utils.plot_model(
            model, 
            to_file=os.path.join(model_dir, 'model_architecture.png'),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True
        )
        print(f"Model architecture diagram saved to {os.path.join(model_dir, 'model_architecture.png')}")
    except Exception as e:
        print(f"Could not save model architecture diagram: {e}")
    
    # Set up TensorBoard callback
    tensorboard_dir = os.path.join(model_dir, "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch=0
    )
    
    # Set up model checkpoint callback
    checkpoint_path = os.path.join(model_dir, "checkpoints", "model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    # Set up CSV logger callback
    csv_path = os.path.join(model_dir, "training_log.csv")
    csv_logger = tf.keras.callbacks.CSVLogger(
        csv_path,
        separator=',',
        append=True
    )
    
    # Set up early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Set up learning rate scheduler
    def lr_schedule(epoch, lr):
        if epoch < 10:
            return float(lr)
        else:
            return float(lr * tf.math.exp(-0.1))
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
    
    # Custom callback for detailed progress reporting
    class DetailedProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print(f"\nEpoch {epoch+1}/{epochs}")
            self.epoch_start_time = time.time()
            
        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start_time
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            
            # Print metrics
            metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            print(f"Metrics: {metrics_str}")
            
            # Estimate remaining time
            remaining_epochs = epochs - (epoch + 1)
            estimated_remaining_time = remaining_epochs * epoch_time
            hours, remainder = divmod(estimated_remaining_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Estimated time remaining: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            # Save current model state
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                intermediate_model_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.h5")
                self.model.save(intermediate_model_path)
                print(f"Saved intermediate model to {intermediate_model_path}")
    
    # Combine all callbacks
    callbacks = [
        tensorboard_callback,
        checkpoint_callback,
        csv_logger,
        early_stopping,
        lr_scheduler,
        DetailedProgressCallback()
    ]
    
    # Train the model
    try:
        print("\nStarting model training...")
        print(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples")
        print(f"Using batch size: {batch_size}, epochs: {epochs}")
        print(f"Model metrics: {', '.join([m.name if hasattr(m, 'name') else str(m) for m in model.metrics])}")
        print("\nUse 'python visualize_model_performance.py --log_dir {} --use_tensorboard' to monitor training in real-time".format(model_dir))
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Save the final model
        model_path = os.path.join(model_dir, "final_model.h5")
        model.save(model_path)
        print(f"\nTraining completed. Final model saved to {model_path}")
        
        # Save training history
        history_path = os.path.join(model_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
        print(f"Training history saved to {history_path}")
        
        # Generate performance visualizations
        try:
            import matplotlib.pyplot as plt
            
            # Plot training & validation accuracy
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(history.history['custom_accuracy'])
            plt.plot(history.history['val_custom_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='lower right')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot training & validation loss
            plt.subplot(2, 2, 2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot precision
            plt.subplot(2, 2, 3)
            plt.plot(history.history['custom_precision'])
            plt.plot(history.history['val_custom_precision'])
            plt.title('Model Precision')
            plt.ylabel('Precision')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='lower right')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot recall
            plt.subplot(2, 2, 4)
            plt.plot(history.history['custom_recall'])
            plt.plot(history.history['val_custom_recall'])
            plt.title('Model Recall')
            plt.ylabel('Recall')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='lower right')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(model_dir, "training_performance.png")
            plt.savefig(plot_path)
            print(f"Performance visualization saved to {plot_path}")
            plt.close()
            
        except Exception as e:
            print(f"Could not generate performance visualizations: {e}")
        
        return history, model_dir
        
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        return None, model_dir

def main():
    """Main function to parse arguments and train the model"""
    parser = argparse.ArgumentParser(description='Train TabCNN model for guitar tablature prediction')
    
    # Data options
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the training data directory')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic data for training')
    parser.add_argument('--generate_synthetic', action='store_true',
                        help='Generate synthetic data before training')
    parser.add_argument('--synthetic_samples', type=int, default=1000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--use_ultimate_guitar', action='store_true',
                        help='Use Ultimate Guitar tab data for training')
    parser.add_argument('--process_ultimate_guitar', action='store_true',
                        help='Process Ultimate Guitar tabs before training')
    
    # Model options
    parser.add_argument('--model_type', type=str, default='enhanced',
                        choices=['simple', 'enhanced', 'experimental', 'attention', 'advanced'],
                        help='Type of model to build')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)')
    parser.add_argument('--visualize', action='store_true',
                        help='Launch visualization tools after training starts')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process Ultimate Guitar tabs if requested
    if args.process_ultimate_guitar:
        print("Processing Ultimate Guitar tabs...")
        from parse_ultimate_guitar_tabs import process_all_tab_files
        process_all_tab_files()
    
    # Generate synthetic data if requested
    if args.generate_synthetic:
        print(f"Generating {args.synthetic_samples} synthetic samples...")
        X, y = generate_synthetic_data(num_samples=args.synthetic_samples)
        
        # Save the synthetic data
        synthetic_dir = os.path.join(backend_dir, "data", "guitarset_training_data", "synthetic")
        os.makedirs(synthetic_dir, exist_ok=True)
        
        for i in range(len(X)):
            np.save(os.path.join(synthetic_dir, f"x_{i}.npy"), X[i])
            np.save(os.path.join(synthetic_dir, f"y_{i}.npy"), y[i])
        
        print(f"Saved {len(X)} synthetic samples to {synthetic_dir}")
    
    # Load training data
    print("Loading training data...")
    X_train, y_train, X_val, y_val = load_training_data(
        data_path=args.data_path,
        use_synthetic=args.use_synthetic or args.generate_synthetic,
        use_ultimate_guitar=args.use_ultimate_guitar,
        batch_size=args.batch_size
    )
    
    # Print shapes for debugging
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    
    # Ensure data has the correct shape for the model
    # Conv1D expects (batch_size, time_steps, features)
    if len(X_train.shape) < 3:
        print(f"Error: Input data must have 3 dimensions, but has shape {X_train.shape}")
        print("Reshaping data to add time dimension...")
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        y_val = y_val.reshape(y_val.shape[0], 1, y_val.shape[1])
        print(f"Reshaped X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Train the model
    history, model_dir = train_tabcnn(
        X_train, y_train, X_val, y_val,
        input_shape=X_train.shape[1:],
        output_shape=y_train.shape[1:],
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.verbose
    )
    
    if history is None:
        sys.exit(1)
    
    # Launch visualization if requested
    if args.visualize and model_dir:
        try:
            import subprocess
            import threading
            
            def launch_visualizer():
                subprocess.Popen([
                    'python', 
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualize_model_performance.py'),
                    '--log_dir', model_dir
                ])
            
            # Launch in a separate thread to not block the main process
            vis_thread = threading.Thread(target=launch_visualizer)
            vis_thread.daemon = True
            vis_thread.start()
            print(f"Launched visualization tool for {model_dir}")
        except Exception as e:
            print(f"Failed to launch visualization: {e}")
    
    print("Training completed successfully.")
    return 0

if __name__ == "__main__":
    main()