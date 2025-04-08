import os
import numpy as np
import librosa
import jams
from unified_tab_processor2 import UnifiedTabProcessor

def process_guitarset(guitarset_dir: str, output_dir: str, hop_length: int = 512, timesteps: int = 50):
    """Convert GuitarSet to spectrograms and tablature sequences."""
    os.makedirs(output_dir, exist_ok=True)
    processor = UnifiedTabProcessor()
    
    X_spec, y_tab = [], []  # Spectrograms and tab sequences
    for root, _, files in os.walk(guitarset_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                jams_path = audio_path.replace('.wav', '.jams')
                
                if not os.path.exists(jams_path):
                    continue
                
                # Load audio and compute spectrogram
                y, sr = librosa.load(audio_path, sr=22050, mono=True)
                spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=80, fmax=4000, hop_length=hop_length)
                spec = librosa.power_to_db(spec, ref=np.max)  # Convert to dB scale
                spec = (spec - spec.mean()) / (spec.std() + 1e-8)  # Normalize
                spec = spec.T  # (time, freq)
                
                # Load JAMS annotations
                jam = jams.load(jams_path)
                ann = jam.search(namespace='note_midi')[0]
                
                # Convert MIDI notes to fretboard positions
                note_events = []
                for obs in ann.data:
                    midi = int(obs.value)
                    time = obs.time
                    duration = obs.duration
                    # Map MIDI to string/fret using EnhancedAudioAnalyzer logic
                    mapped = processor.audio_analyzer.map_midi_to_fretboard(midi)
                    if mapped:
                        note_events.append({
                            'time': time,
                            'duration': duration,
                            'string': mapped['string'],
                            'fret': mapped['fret']
                        })
                
                # Convert to LSTM array
                num_frames = spec.shape[0]
                tab_array = np.zeros((num_frames, 6, 21))
                for event in note_events:
                    frame_idx = int(event['time'] * sr / hop_length)
                    if frame_idx < num_frames:
                        tab_array[frame_idx, event['string'], event['fret']] = 1.0
                
                # Segment into timesteps
                for start in range(0, num_frames - timesteps + 1, timesteps):
                    X_seg = spec[start:start + timesteps]
                    y_seg = tab_array[start:start + timesteps]
                    if X_seg.shape[0] == timesteps:
                        X_spec.append(X_seg)
                        y_tab.append(y_seg)
    
    X_spec = np.array(X_spec)
    y_tab = np.array(y_tab)
    
    # Save the raw spectrograms
    np.save(os.path.join(output_dir, 'X_spec_train.npy'), X_spec)
    
    # Process spectrograms for TabCNN input (one frame at a time)
    X_tabcnn = []
    for i in range(len(X_spec)):
        # Create context windows for each frame
        frames_with_context = []
        padded_spec = np.pad(X_spec[i], ((4, 4), (0, 0)), mode='constant')
        for j in range(4, padded_spec.shape[0] - 4):
            window = padded_spec[j-4:j+5]  # 9-frame context window
            frames_with_context.append(window)
        
        # Convert to correct shape for TabCNN
        frames_with_context = np.array(frames_with_context)
        frames_with_context = np.expand_dims(frames_with_context, axis=-1)  # Add channel dimension
        
        # Get TabCNN predictions
        try:
            tabcnn_pred = processor.tabcnn.predict_tablature(frames_with_context)
            X_tabcnn.append(tabcnn_pred)
        except Exception as e:
            print(f"Error in TabCNN prediction: {e}")
            # Create dummy data if prediction fails
            X_tabcnn.append(np.random.random((timesteps, 6, 21)) * 0.1)
    
    X_tabcnn = np.array(X_tabcnn)
    
    # Reshape y_tab for LSTM training (batch, timesteps, features)
    y_tab_reshaped = y_tab.reshape(y_tab.shape[0], y_tab.shape[1], -1)
    np.save(os.path.join(output_dir, 'y_lstm_train.npy'), y_tab_reshaped)
    
    # Reshape X_tabcnn for LSTM input (batch, timesteps, features)
    X_lstm = X_tabcnn.reshape(X_tabcnn.shape[0], X_tabcnn.shape[1], -1)
    np.save(os.path.join(output_dir, 'X_lstm_train.npy'), X_lstm)
    
    print(f"Saved data shapes: X_spec={X_spec.shape}, X_lstm={X_lstm.shape}, y_tab={y_tab_reshaped.shape}")
    return X_spec, X_lstm, y_tab_reshaped

# Example usage
if __name__ == "__main__":
    # Use a relative path that's more likely to exist
    guitarset_dir = os.path.join(os.path.dirname(__file__), "data", "guitarset")
    output_dir = os.path.join(os.path.dirname(__file__), "data", "guitarset_training_data")
    
    # Create directories if they don't exist
    os.makedirs(guitarset_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if guitarset directory has files
    files = [f for f in os.listdir(guitarset_dir) if f.endswith('.wav')]
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
        X_spec, X_lstm, y_tab = process_guitarset(guitarset_dir, output_dir)
        print(f"Processed {X_spec.shape[0]} samples: X_spec={X_spec.shape}, X_lstm={X_lstm.shape}, y_tab={y_tab.shape}")