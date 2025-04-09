"""
Simple preprocessing script for GuitarSet data that doesn't depend on UnifiedTabProcessor.
This script extracts spectrograms and tablature data from GuitarSet for training TabCNN and LSTM models.
"""
import os
import numpy as np
import librosa
import jams
import tensorflow as tf
from datetime import datetime

# GPU configuration function
def configure_gpu():
    """Configure TensorFlow to use GPU if available, with memory growth enabled."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU acceleration enabled. Found {len(gpus)} GPU(s).")
            return True
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    print("No GPU found. Using CPU for preprocessing (this will be slower).")
    return False

# Configure GPU at module import time
has_gpu = configure_gpu()

def process_guitarset(guitarset_dir: str, annotations_dir: str, output_dir: str, hop_length: int = 512, timesteps: int = 50):
    """Convert GuitarSet to spectrograms and tablature sequences."""
    os.makedirs(output_dir, exist_ok=True)
    
    X_spec, X_lstm, y_tab = [], [], []  # Spectrograms, LSTM inputs, and tab sequences
    
    print(f"Processing GuitarSet data from {guitarset_dir}...")
    print(f"Using annotations from {annotations_dir}...")
    file_count = 0
    processed_count = 0
    
    # Get all available JAMS files
    jams_files = [f for f in os.listdir(annotations_dir) if f.endswith('.jams')]
    print(f"Found {len(jams_files)} JAMS files in annotations directory")
    
    # Create a mapping from audio file pattern to JAMS file
    jams_mapping = {}
    for jams_file in jams_files:
        # Extract the base part without _comp.jams or _solo.jams
        base_part = jams_file.rsplit('_', 1)[0]
        jams_mapping[base_part] = jams_file
    
    for root, _, files in os.walk(guitarset_dir):
        for file in files:
            if file.endswith('.wav'):
                try:
                    audio_path = os.path.join(root, file)
                    # Extract the base filename without extension and without the last part (hex, mic, mix, etc.)
                    base_filename = os.path.splitext(os.path.basename(file))[0]
                    
                    # Determine if it's a comp or solo recording
                    if "_comp_" in base_filename:
                        recording_type = "comp"
                        base_part = base_filename.split("_comp_")[0]
                    elif "_solo_" in base_filename:
                        recording_type = "solo"
                        base_part = base_filename.split("_solo_")[0]
                    else:
                        print(f"Unknown recording type for {base_filename}, skipping...")
                        continue
                    
                    # Look for corresponding JAMS file in annotations directory
                    jams_filename = f"{base_part}_{recording_type}.jams"
                    jams_path = os.path.join(annotations_dir, jams_filename)
                    
                    if not os.path.exists(jams_path):
                        print(f"JAMS file not found for {base_filename} (looking for {jams_filename}), skipping...")
                        continue
                    
                    print(f"Processing file {file_count+1}: {file} with annotation {jams_filename}")
                    
                    # Load audio and compute spectrogram
                    y, sr = librosa.load(audio_path, sr=22050, mono=True)
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
                    S_db = librosa.power_to_db(S, ref=np.max)
                    
                    # Load JAMS annotation
                    jam = jams.load(jams_path)
                    
                    # Extract tablature from JAMS
                    tab_matrix = np.zeros((timesteps, 6, 21))  # 6 strings, 21 frets (0-20)
                    
                    # Process annotations for each string
                    for string_idx in range(6):
                        try:
                            # Get the annotation for this string
                            string_annot = jam.annotations[string_idx]
                            
                            # Process each note
                            for note in string_annot:
                                # Extract time and fret information
                                time_start = note.time
                                duration = note.duration
                                # The fret value might be in different formats depending on the JAMS file structure
                                # Try different approaches to extract the fret number
                                try:
                                    if isinstance(note.value, dict) and 'fret' in note.value:
                                        fret = int(note.value['fret'])
                                    elif isinstance(note.value, dict) and 'pitch' in note.value:
                                        # Convert MIDI pitch to fret (simplified)
                                        # This is a placeholder - proper conversion would depend on tuning
                                        midi_pitch = note.value['pitch']
                                        # Standard tuning MIDI values for open strings: E2(40), A2(45), D3(50), G3(55), B3(59), E4(64)
                                        open_string_midi = [40, 45, 50, 55, 59, 64]
                                        fret = midi_pitch - open_string_midi[string_idx]
                                    else:
                                        fret = int(note.value)
                                except (TypeError, ValueError):
                                    print(f"Could not parse fret value: {note.value}, skipping this note")
                                    continue
                                
                                # Ensure fret is within valid range
                                if fret < 0 or fret > 20:
                                    continue
                                
                                # Convert time to frame index
                                frame_start = int(time_start * sr / hop_length)
                                frame_end = int((time_start + duration) * sr / hop_length)
                                
                                # Map to our timesteps
                                for t in range(timesteps):
                                    # Simple mapping of frames to timesteps
                                    frame = int(t * S_db.shape[1] / timesteps)
                                    if frame_start <= frame < frame_end:
                                        tab_matrix[t, string_idx, fret] = 1.0
                        except Exception as e:
                            print(f"Error processing string {string_idx}: {e}")
                    
                    # Reshape spectrogram to match timesteps
                    spec_frames = []
                    for t in range(timesteps):
                        frame_idx = int(t * S_db.shape[1] / timesteps)
                        if frame_idx < S_db.shape[1]:
                            spec_frames.append(S_db[:, frame_idx])
                        else:
                            # Pad with zeros if needed
                            spec_frames.append(np.zeros(S_db.shape[0]))
                    
                    spec_reshaped = np.array(spec_frames)
                    
                    # Flatten tablature for LSTM input/output
                    tab_flattened = tab_matrix.reshape(timesteps, -1)
                    
                    # Store processed data
                    X_spec.append(spec_reshaped)
                    X_lstm.append(tab_flattened)
                    y_tab.append(tab_flattened)  # Target is the same as input for LSTM refinement
                    
                    file_count += 1
                    processed_count += 1
                    if file_count % 10 == 0:
                        print(f"Processed {file_count} files")
                
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    continue
    
    # Convert lists to numpy arrays if we have data
    if processed_count > 0:
        X_spec = np.array(X_spec)
        X_lstm = np.array(X_lstm)
        y_tab = np.array(y_tab)
        
        # Save processed data
        output_path = os.path.join(output_dir, "guitarset_training_data.npz")
        np.savez_compressed(output_path, X_spec=X_spec, X_lstm=X_lstm, y_tab=y_tab)
        print(f"Saved processed data to {output_dir}")
    else:
        # Return empty arrays if no files were processed
        X_spec = np.array([])
        X_lstm = np.array([])
        y_tab = np.array([])
    
    return X_spec, X_lstm, y_tab

def generate_synthetic_data(num_samples=100, timesteps=50):
    """Generate synthetic data for training when real data is not available."""
    print(f"Generating {num_samples} synthetic samples...")
    
    # Use GPU for data generation if available
    if has_gpu:
        with tf.device('/GPU:0'):
            # Generate random spectrograms (X_spec)
            X_spec = tf.random.normal((num_samples, timesteps, 128), mean=0, stddev=1).numpy()
            
            # Generate random tablature (X_lstm and y_tab)
            # For simplicity, we'll create sparse matrices where each string has at most one active fret
            X_lstm = np.zeros((num_samples, timesteps, 126))  # 6 strings * 21 frets = 126 features
            
            for i in range(num_samples):
                for t in range(timesteps):
                    # Randomly activate 0-3 strings at this timestep
                    active_strings = np.random.choice(6, size=np.random.randint(0, 4), replace=False)
                    
                    for string in active_strings:
                        # Choose a fret (0-20) for this string
                        fret = np.random.randint(0, 21)
                        X_lstm[i, t, string * 21 + fret] = 1.0
            
            # For synthetic data, y_tab is the same as X_lstm
            y_tab = X_lstm.copy()
    else:
        # Fallback to CPU
        X_spec = np.random.normal(0, 1, (num_samples, timesteps, 128)).astype(np.float32)
        X_lstm = np.zeros((num_samples, timesteps, 126), dtype=np.float32)
        
        for i in range(num_samples):
            for t in range(timesteps):
                active_strings = np.random.choice(6, size=np.random.randint(0, 4), replace=False)
                for string in active_strings:
                    fret = np.random.randint(0, 21)
                    X_lstm[i, t, string * 21 + fret] = 1.0
        
        y_tab = X_lstm.copy()
    
    # Save synthetic data
    output_dir = os.path.join(os.path.dirname(__file__), "data", "guitarset_training_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "guitarset_training_data.npz")
    np.savez_compressed(output_path, X_spec=X_spec, X_lstm=X_lstm, y_tab=y_tab)
    print(f"Saved synthetic data to {output_path}")
    
    return X_spec, X_lstm, y_tab

# Example usage
if __name__ == "__main__":
    # Use the correct paths to GuitarSet
    guitarset_dir = r"C:\AIstuffing\Datasets\GuitarSet\audio"
    annotations_dir = r"C:\AIstuffing\Datasets\GuitarSet\annotations"
    output_dir = os.path.join(os.path.dirname(__file__), "data", "guitarset_training_data")
    
    print(f"Looking for GuitarSet data in: {guitarset_dir}")
    print(f"Looking for annotations in: {annotations_dir}")
    print(f"Output will be saved to: {output_dir}")
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if guitarset directory exists and has files
    try:
        if os.path.exists(guitarset_dir) and os.path.exists(annotations_dir):
            files = [f for f in os.listdir(guitarset_dir) if f.endswith('.wav')]
            print(f"Found {len(files)} .wav files in the GuitarSet directory")
            
            if files:
                print("Starting GuitarSet processing...")
                X_spec, X_lstm, y_tab = process_guitarset(guitarset_dir, annotations_dir, output_dir)
                print(f"Processed {X_spec.shape[0]} samples: X_spec={X_spec.shape}, X_lstm={X_lstm.shape}, y_tab={y_tab.shape}")
            else:
                print(f"No .wav files found in {guitarset_dir}. Creating synthetic data instead.")
                X_spec, X_lstm, y_tab = generate_synthetic_data(100)
                
                # Save synthetic data
                np.savez_compressed(os.path.join(output_dir, 'guitarset_training_data.npz'), X_spec=X_spec, X_lstm=X_lstm, y_tab=y_tab)
                
                print(f"Saved synthetic data: X_spec={X_spec.shape}, X_lstm={X_lstm.shape}, y_tab={y_tab.shape}")
        else:
            print(f"GuitarSet directory {guitarset_dir} or annotations directory {annotations_dir} not found. Creating synthetic data instead.")
            X_spec, X_lstm, y_tab = generate_synthetic_data(100)
            
            # Save synthetic data
            np.savez_compressed(os.path.join(output_dir, 'guitarset_training_data.npz'), X_spec=X_spec, X_lstm=X_lstm, y_tab=y_tab)
            
            print(f"Saved synthetic data: X_spec={X_spec.shape}, X_lstm={X_lstm.shape}, y_tab={y_tab.shape}")
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        
        print("Falling back to synthetic data generation...")
        X_spec, X_lstm, y_tab = generate_synthetic_data(100)
        
        # Save synthetic data
        np.savez_compressed(os.path.join(output_dir, 'guitarset_training_data.npz'), X_spec=X_spec, X_lstm=X_lstm, y_tab=y_tab)
        
        print(f"Saved synthetic data: X_spec={X_spec.shape}, X_lstm={X_lstm.shape}, y_tab={y_tab.shape}")
