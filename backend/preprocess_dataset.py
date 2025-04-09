import os
import numpy as np
import librosa
import jams
import tensorflow as tf
from unified_tab_processor2 import UnifiedTabProcessor

# Standard guitar tuning frequencies (E2, A2, D3, G3, B3, E4)
GUITAR_STRINGS = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]
# Allowed frequency ranges for each string (min, max) with much wider tolerance
STRING_FREQ_RANGES = [
    (70.0, 105.0),    # E2 string (low E)
    (95.0, 135.0),    # A2 string
    (130.0, 180.0),   # D3 string
    (175.0, 230.0),   # G3 string
    (210.0, 280.0),   # B3 string
    (290.0, 370.0)    # E4 string (high E)
]

def frequency_to_fret(frequency, string_idx, debug=False):
    """
    Convert a frequency to the closest fret position on a guitar string.
    
    Args:
        frequency: The frequency in Hz
        string_idx: The string index (0-5, where 0 is the lowest E string)
        debug: Whether to print debug information
        
    Returns:
        fret: The fret position (0-24) or None if outside valid range
    """
    if string_idx < 0 or string_idx >= len(GUITAR_STRINGS):
        if debug:
            print(f"Invalid string index: {string_idx}")
        return None
    
    # Get the open string frequency
    open_freq = GUITAR_STRINGS[string_idx]
    
    # Check if frequency is in a reasonable range for this string
    min_freq, max_freq = STRING_FREQ_RANGES[string_idx]
    
    if frequency <= 0:
        if debug:
            print(f"Invalid frequency: {frequency} <= 0")
        return None
    
    # If the frequency is within the allowed range for this string
    if min_freq <= frequency <= max_freq:
        # Calculate closest fret based on equal temperament
        semitones_from_standard = 12 * np.log2(frequency / open_freq)
        fret = round(semitones_from_standard)
        
        # Allow a wider range of frets
        if 0 <= fret <= 24:
            return fret
        else:
            # If outside normal range but frequency is reasonable, 
            # map to closest valid fret
            return max(0, min(24, fret))
    
    # Check if the frequency might belong to a neighboring string
    # This handles cases where the annotation might have the wrong string index
    for alt_string_idx, (alt_min, alt_max) in enumerate(STRING_FREQ_RANGES):
        if alt_string_idx != string_idx and alt_min <= frequency <= alt_max:
            alt_open_freq = GUITAR_STRINGS[alt_string_idx]
            alt_semitones = 12 * np.log2(frequency / alt_open_freq)
            alt_fret = round(alt_semitones)
            
            if 0 <= alt_fret <= 24:
                if debug:
                    print(f"Frequency {frequency} on string {string_idx} might belong to string {alt_string_idx} at fret {alt_fret}")
                return alt_fret
    
    # Last resort: try to find the closest string and fret combination
    # This is useful for frequencies that are just outside our defined ranges
    closest_string = string_idx
    closest_fret = None
    min_distance = float('inf')
    
    for test_string_idx in range(len(GUITAR_STRINGS)):
        test_open_freq = GUITAR_STRINGS[test_string_idx]
        # Calculate how many semitones away this frequency is
        semitones = 12 * np.log2(frequency / test_open_freq)
        test_fret = round(semitones)
        
        # Calculate how far this frequency is from the "ideal" frequency of this fret
        ideal_freq = test_open_freq * (2 ** (test_fret / 12))
        distance = abs(frequency - ideal_freq)
        
        if distance < min_distance and 0 <= test_fret <= 24:
            min_distance = distance
            closest_string = test_string_idx
            closest_fret = test_fret
    
    # If we found a reasonable match and it's not too far off
    if closest_fret is not None and min_distance / frequency < 0.15:  # Within 15% error
        if debug:
            print(f"Frequency {frequency} on string {string_idx} best matches string {closest_string} fret {closest_fret}")
        return closest_fret
    
    # If we get here, the frequency is outside all reasonable ranges
    if debug:
        print(f"Frequency {frequency} is outside reasonable range for string {string_idx}")
    return None

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

def process_guitarset(guitarset_dir: str, annotations_dir: str, output_dir: str, hop_length: int = 512, timesteps: int = 50, debug_mode: bool = False, batch_size: int = 1000):
    """Convert GuitarSet to spectrograms and tablature sequences.
    
    Args:
        guitarset_dir: Directory containing GuitarSet audio files
        annotations_dir: Directory containing JAMS annotation files
        output_dir: Directory to save processed data
        hop_length: Hop length for spectrogram computation
        timesteps: Number of timesteps for each training example
        debug_mode: Whether to print detailed debug information
        batch_size: Number of examples to process before saving to disk
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for batch storage
    batch_dir = os.path.join(output_dir, "batches")
    os.makedirs(batch_dir, exist_ok=True)
    
    # Initialize batch counters
    batch_count = 0
    current_batch_size = 0
    
    # Initialize batch data
    X_spec_batch, X_lstm_batch, y_tab_batch = [], [], []
    
    print(f"Processing GuitarSet data from {guitarset_dir}...")
    print(f"Using annotations from {annotations_dir}...")
    file_count = 0
    processed_count = 0
    total_examples = 0
    
    # Get all available JAMS files
    jams_files = [f for f in os.listdir(annotations_dir) if f.endswith('.jams')]
    print(f"Found {len(jams_files)} JAMS files in annotations directory")
    
    # Create a mapping from audio file pattern to JAMS file
    jams_mapping = {}
    for jams_file in jams_files:
        # Extract the base part without _comp.jams or _solo.jams
        base_part = jams_file.rsplit('_', 1)[0]
        jams_mapping[base_part] = jams_file
    
    # Track statistics for debugging
    total_notes = 0
    successful_notes = 0
    failed_notes_by_string = [0, 0, 0, 0, 0, 0]  # Count of failed notes per string
    
    # Function to save a batch of data
    def save_batch(batch_idx, X_spec, X_lstm, y_tab):
        batch_path = os.path.join(batch_dir, f"batch_{batch_idx}")
        os.makedirs(batch_path, exist_ok=True)
        
        # Convert to numpy arrays
        X_spec_np = np.array(X_spec)
        X_lstm_np = np.array(X_lstm)
        y_tab_np = np.array(y_tab)
        
        # Save batch data
        np.save(os.path.join(batch_path, 'X_spec.npy'), X_spec_np)
        np.save(os.path.join(batch_path, 'X_lstm.npy'), X_lstm_np)
        np.save(os.path.join(batch_path, 'y_tab.npy'), y_tab_np)
        
        print(f"Saved batch {batch_idx} with {len(X_spec)} examples")
        print(f"  X_spec shape: {X_spec_np.shape}")
        print(f"  X_lstm shape: {X_lstm_np.shape}")
        print(f"  y_tab shape: {y_tab_np.shape}")
        
        # Clear memory
        del X_spec_np, X_lstm_np, y_tab_np
        
        return [], [], []  # Return empty lists for next batch
    
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
                    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=80, fmax=4000, hop_length=hop_length)
                    spec = librosa.power_to_db(spec, ref=np.max)  # Convert to dB scale
                    spec = (spec - spec.mean()) / (spec.std() + 1e-8)  # Normalize
                    spec = spec.T  # (time, freq)
                    
                    # Load tablature from JAMS
                    jam = jams.load(jams_path)
                    tab_data = []
                    
                    # Extract note events from JAMS annotations
                    for string_idx in range(6):
                        string_anno = jam.annotations[string_idx]
                        string_note_count = 0
                        string_success_count = 0
                        
                        for note in string_anno:
                            total_notes += 1
                            string_note_count += 1
                            
                            start_time = note.time
                            duration = note.duration
                            
                            # Handle the case where note.value is a dictionary
                            if isinstance(note.value, dict):
                                if 'frequency' in note.value:
                                    frequency = note.value['frequency']
                                else:
                                    failed_notes_by_string[string_idx] += 1
                                    if debug_mode:
                                        print(f"  Warning: Missing frequency in note value: {note.value}")
                                    continue
                            else:
                                try:
                                    frequency = float(note.value)
                                except (ValueError, TypeError):
                                    failed_notes_by_string[string_idx] += 1
                                    if debug_mode:
                                        print(f"  Warning: Could not convert note value to float: {note.value}")
                                    continue
                            
                            # Calculate fret position from frequency with improved function
                            fret = frequency_to_fret(frequency, string_idx, debug=debug_mode)
                            if fret is None:
                                failed_notes_by_string[string_idx] += 1
                                if debug_mode:
                                    print(f"  Warning: Could not calculate fret position for frequency {frequency} on string {string_idx}")
                                continue
                            
                            successful_notes += 1
                            string_success_count += 1
                            tab_data.append((start_time, duration, string_idx, fret))
                        
                        # Report string-specific success rate
                        if string_note_count > 0:
                            success_rate = (string_success_count / string_note_count) * 100
                            print(f"  String {string_idx}: Processed {string_success_count}/{string_note_count} notes ({success_rate:.1f}% success rate)")
                    
                    # Define the maximum fret value to include in the matrix
                    max_fret_in_matrix = 24  # Increased from 20 to handle more frets
                    
                    # Convert to tablature matrix
                    tab_matrix = np.zeros((len(spec), 6, max_fret_in_matrix + 1))  # (time, string, fret)
                    for start_time, duration, string_idx, fret in tab_data:
                        start_frame = int(start_time * sr / hop_length)
                        end_frame = int((start_time + duration) * sr / hop_length)
                        if start_frame < len(spec) and end_frame < len(spec):
                            # Ensure fret is within matrix bounds (0-max_fret_in_matrix)
                            fret_int = int(round(fret))  # Ensure fret is an integer
                            if 0 <= fret_int <= max_fret_in_matrix:
                                tab_matrix[start_frame:end_frame, string_idx, fret_int] = 1
                            else:
                                # For frets > max_fret_in_matrix, map to max_fret_in_matrix
                                tab_matrix[start_frame:end_frame, string_idx, max_fret_in_matrix] = 1
                    
                    # Create context windows for TabCNN input
                    for i in range(len(spec) - timesteps + 1):
                        spec_window = spec[i:i+timesteps]
                        tab_window = tab_matrix[i:i+timesteps]
                        
                        # For TabCNN
                        X_spec_batch.append(spec_window)
                        
                        # For LSTM - reshape to (timesteps, 6*(max_fret_in_matrix+1))
                        lstm_input = tab_window.reshape(timesteps, -1)
                        X_lstm_batch.append(lstm_input)
                        y_tab_batch.append(lstm_input)  # Target is the same as input for LSTM refinement
                        
                        current_batch_size += 1
                        total_examples += 1
                        
                        # If batch is full, save it and start a new one
                        if current_batch_size >= batch_size:
                            X_spec_batch, X_lstm_batch, y_tab_batch = save_batch(
                                batch_count, X_spec_batch, X_lstm_batch, y_tab_batch
                            )
                            batch_count += 1
                            current_batch_size = 0
                    
                    file_count += 1
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # Save any remaining examples in the last batch
    if current_batch_size > 0:
        save_batch(batch_count, X_spec_batch, X_lstm_batch, y_tab_batch)
        batch_count += 1
    
    # Report overall statistics
    if total_notes > 0:
        overall_success_rate = (successful_notes / total_notes) * 100
        print(f"\nOverall note processing: {successful_notes}/{total_notes} notes ({overall_success_rate:.1f}% success rate)")
        for string_idx in range(6):
            failed_count = failed_notes_by_string[string_idx]
            if failed_count > 0:
                print(f"String {string_idx}: {failed_count} failed notes")
    
    print(f"\nProcessed {total_examples} examples in {batch_count} batches")
    print(f"Data saved to {batch_dir}")
    
    # Create a metadata file with batch information
    with open(os.path.join(output_dir, 'metadata.txt'), 'w') as f:
        f.write(f"Total examples: {total_examples}\n")
        f.write(f"Total batches: {batch_count}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Timesteps: {timesteps}\n")
        f.write(f"Hop length: {hop_length}\n")
        f.write(f"Max fret: {max_fret_in_matrix}\n")
        f.write(f"Total notes processed: {total_notes}\n")
        f.write(f"Successful notes: {successful_notes}\n")
        f.write(f"Success rate: {overall_success_rate:.1f}%\n")
    
    return total_examples, batch_count, overall_success_rate

# Example usage
if __name__ == "__main__":
    # Use the correct path to GuitarSet
    guitarset_dir = r"C:\AIstuffing\Datasets\GuitarSet\audio"
    output_dir = os.path.join(os.path.dirname(__file__), "data", "guitarset_training_data")
    
    print(f"Looking for GuitarSet data in: {guitarset_dir}")
    print(f"Output will be saved to: {output_dir}")
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if guitarset directory has files
    try:
        files = [f for f in os.listdir(guitarset_dir) if f.endswith('.wav')]
        print(f"Found {len(files)} .wav files in the GuitarSet directory")
        
        if not files:
            print(f"No .wav files found in {guitarset_dir}. Please add GuitarSet data files.")
            # Create dummy data for testing
            print("Creating dummy data for testing...")
            dummy_size = 10
            X_spec = np.random.random((dummy_size, 50, 128))
            X_lstm = np.random.random((dummy_size, 50, 6*21))
            y_tab = np.random.random((dummy_size, 50, 6*21))
            
            np.save(os.path.join(output_dir, 'X_spec_train.npy'), X_spec)
            np.save(os.path.join(output_dir, 'X_lstm_train.npy'), X_lstm)
            np.save(os.path.join(output_dir, 'y_lstm_train.npy'), y_tab)
            
            print(f"Saved dummy data: X_spec={X_spec.shape}, X_lstm={X_lstm.shape}, y_tab={y_tab.shape}")
        else:
            print("Starting GuitarSet processing...")
            # Define annotations directory
            annotations_dir = os.path.join(guitarset_dir, "..", "annotations")
            print(f"Using annotations from: {annotations_dir}")
            total_examples, batch_count, overall_success_rate = process_guitarset(guitarset_dir, annotations_dir, output_dir, debug_mode=True)
            print(f"Processed {total_examples} examples in {batch_count} batches with {overall_success_rate:.1f}% success rate")
    except Exception as e:
        print(f"Error processing GuitarSet data: {e}")
        import traceback
        traceback.print_exc()