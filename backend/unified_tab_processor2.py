import os
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Any, Tuple
import json
import warnings
from scipy import signal

# Suppress FFmpeg warnings from pydub
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Couldn't find ffmpeg or avconv")

# Assuming these are your existing imports from your project structure
from interpreter import DemucsProcessor, TabCNNProcessor, BasicPitchProcessor, basic_pitch_available
from models.lstm_model.LSTMPredictor import LSTMPredictor
from models.fretboard_position_model.position_predictor import FretboardPositionPredictor

# New imports from your provided modules
from confidence_evaluator import ConfidenceEvaluator
from midi_converter import MidiConverter
from music_theory_analyzer import MusicTheoryAnalyzer
from tab_text_generator import TabTextGenerator
from enhanced_audio_analyzer import EnhancedAudioAnalyzer
from lilypond_tab_generator import LilypondTabGenerator, lilypond_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedTabProcessor:
    """Enhanced processor for generating guitar tablature from audio."""
    
    def __init__(self):
        try:
            import essentia
            self.essentia_available = True
        except ImportError:
            self.essentia_available = False
            logger.warning("Essentia not available - using simplified audio analysis")
        
        self.demucs = DemucsProcessor()
        self.tabcnn = TabCNNProcessor()
        # Try to load weights, but don't fail if there's an issue
        try:
            # Get the model directory
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "tab-cnn", "model", "saved")
            
            if os.path.exists(model_dir):
                # Find the most recent model directory
                model_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
                if model_dirs:
                    latest_model_dir = sorted(model_dirs)[-1]
                    version_dirs = os.path.join(model_dir, latest_model_dir)
                    if os.path.exists(version_dirs):
                        version_subdirs = [d for d in os.listdir(version_dirs) if os.path.isdir(os.path.join(version_dirs, d))]
                        if version_subdirs:
                            latest_version = sorted(version_subdirs)[-1]
                            custom_weights_path = os.path.join(version_dirs, latest_version, "model.weights.h5")
                            
                            if os.path.exists(custom_weights_path):
                                # Rebuild the model to match the saved weights format
                                # Create a simpler model that only has a single output to match saved weights
                                try:
                                    import tensorflow as tf
                                    from tensorflow.keras.models import Model
                                    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Add
                                    
                                    # Build a simpler model with only one output
                                    inputs = Input(shape=(128, 9, 1))
                                    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
                                    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
                                    residual = x
                                    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
                                    x = Add()([x, residual])  # Residual connection
                                    x = MaxPooling2D((2, 2))(x)
                                    x = Dropout(0.25)(x)
                                    x = Flatten()(x)
                                    x = Dense(128, activation='relu')(x)
                                    x = Dropout(0.5)(x)
                                    outputs = Dense(126, activation='sigmoid')(x)  # Single output for all strings and frets
                                    
                                    # Create and compile the model
                                    self.tabcnn.model = Model(inputs, outputs)
                                    self.tabcnn.model.compile(loss='binary_crossentropy',
                                              optimizer='adam', metrics=['accuracy'])
                                    
                                    # Now load the weights
                                    self.tabcnn.model.load_weights(custom_weights_path)
                                    logger.info(f"Loaded TabCNN weights successfully from {custom_weights_path}")
                                    self.tabcnn.initialized = True
                                    
                                except Exception as e:
                                    logger.error(f"Error rebuilding TabCNN model: {e}")
                                    # Continue with default model
                                    logger.warning("Falling back to default model")
                            else:
                                logger.warning(f"Custom weights file not found at {custom_weights_path}")
                        else:
                            logger.warning("No version subdirectories found")
                    else:
                        logger.warning(f"Version directory not found at {version_dirs}")
                else:
                    logger.warning(f"No model directories found in {model_dir}")
            else:
                logger.warning(f"Model directory not found at {model_dir}")
                
        except Exception as e:
            logger.warning(f"Could not load custom TabCNN weights: {e}. Using default model.")
        
        self._load_lstm_model()
        self._load_fretboard_position_model()
        self.text_gen = TabTextGenerator()
        self.confidence_evaluator = ConfidenceEvaluator(threshold=0.7, min_threshold=0.4, fallback_threshold=0.2)
        self.midi_converter = MidiConverter()
        self.theory_analyzer = MusicTheoryAnalyzer()
        self.audio_analyzer = EnhancedAudioAnalyzer()
        
        # Initialize LilyPond generator
        try:
            from .lilypond_tab_generator import LilypondTabGenerator, lilypond_available
            self.lilypond_generator = LilypondTabGenerator()
            self.lilypond_available = lilypond_available
            logger.info(f"LilyPond generator initialized. Available: {lilypond_available}")
        except ImportError as e:
            logger.warning(f"LilyPond generator module not available: {e}")
            self.lilypond_generator = None
            self.lilypond_available = False

        # Initialize Basic Pitch processor if available
        try:
            if basic_pitch_available:
                self.basic_pitch = BasicPitchProcessor()
                logger.info("Basic Pitch processor initialized successfully")
            else:
                self.basic_pitch = None
                logger.warning("Basic Pitch is not available. Will use fallback methods for MIDI conversion.")
        except ImportError:
            self.basic_pitch = None
            logger.warning("Could not import Basic Pitch. Will use fallback methods for MIDI conversion.")
            
        # Load style configurations
        with open('F:/newrepos/olitunes/backend/data/tab_data/style_configs.json') as f:
            self.style_configs = json.load(f)

    def _load_lstm_model(self):
        """Load the LSTM model for tablature refinement."""
        try:
            from models.lstm_model.lstm_predictor import LSTMPredictor
            self.lstm = LSTMPredictor()
            
            saved_models_dir = os.path.join(os.path.dirname(__file__), "models", "lstm_model", "saved")
            
            model_files = []
            if os.path.exists(saved_models_dir):
                for file in os.listdir(saved_models_dir):
                    if file.endswith('.keras') and 'final' in file:
                        model_files.append(os.path.join(saved_models_dir, file))
            
            if model_files:
                model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                newest_model = model_files[0]
                logging.info(f"Loading newest LSTM model: {newest_model}")
                
                try:
                    self.lstm.model.load_weights(newest_model)
                    logging.info("LSTM model loaded successfully")
                except Exception as e:
                    logging.warning(f"Could not load LSTM weights with by_name=True: {e}")
                    try:
                        self.lstm.model.load_weights(newest_model)
                        logging.info("LSTM model loaded successfully without by_name parameter")
                    except Exception as e2:
                        logging.error(f"Failed to load LSTM model: {e2}")
            else:
                default_model_path = os.path.join(os.path.dirname(__file__), "models", "lstm_model", "lstm_best.h5")
                if os.path.exists(default_model_path):
                    try:
                        self.lstm.model.load_weights(default_model_path)
                        logging.info("LSTM model loaded from default path")
                    except Exception as e:
                        logging.warning(f"Could not load LSTM weights: {e}")
                else:
                    logging.warning("No LSTM model found. Using untrained model.")
        except ImportError as e:
            logging.warning(f"Could not import LSTMPredictor: {e}")
            self.lstm = None

    def _load_fretboard_position_model(self):
        """Load the fretboard position model for optimizing tablature positions."""
        try:
            self.fretboard_model = FretboardPositionPredictor()
            logging.info("Fretboard position model loaded successfully")
        except Exception as e:
            logging.warning(f"Could not load FretboardPositionPredictor: {e}")
            self.fretboard_model = None

    def preprocess_audio(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Preprocess audio for TabCNN model
        
        Args:
            y: Audio data
            sr: Sample rate
            
        Returns:
            tuple: (mel-spectrogram, onsets, tempo)
        """
        # Resample to 22050 Hz if necessary
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050

        # Noise reduction
        try:
            import noisereduce as nr
            y = nr.reduce_noise(y=y, sr=sr)
        except ImportError:
            logger.warning("Noisereduce unavailable, skipping noise reduction.")

        # Harmonic separation
        y = librosa.effects.hpss(y)[0]

        # Mel-spectrogram with guitar-tuned bins
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=80, fmax=4000, hop_length=512)
        
        # Convert to dB scale
        spec = librosa.power_to_db(spec, ref=np.max)
        
        # Normalize
        spec = (spec - np.mean(spec)) / (np.std(spec) if np.std(spec) > 0 else 1)
        
        # Reshape to match the expected model input shape (None, 128, 9, 1)
        # First, ensure we have the right width dimension (9)
        target_width = 9  # Model expects 9 time steps
        
        if spec.shape[1] > target_width:
            # Truncate
            spec = spec[:, :target_width]
        elif spec.shape[1] < target_width:
            # Pad with zeros
            padding = np.zeros((spec.shape[0], target_width - spec.shape[1]))
            spec = np.hstack((spec, padding))
        
        # Add batch and channel dimensions to match (batch, height, width, channels)
        # For a single input: (1, 128, 9, 1)
        spec = np.expand_dims(spec, axis=0)  # Add batch dimension (1, 128, 9)
        spec = np.expand_dims(spec, axis=3)  # Add channel dimension (1, 128, 9, 1)

        # Onset detection
        onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)

        # Tempo detection
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]

        return spec, onsets, tempo

    def enhance_guitar_audio(self, audio_path: str, output_path: str) -> str:
        """Enhanced guitar audio processing."""
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        y = self._enhance_guitar_audio(y, sr)
        sf.write(output_path, y, sr)
        return output_path

    def _enhance_guitar_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Enhance guitar audio by:
        - Cleaning non-finite values (NaN, infinity)
        - Normalizing volume
        - Applying EQ to boost guitar frequencies
        - Reducing noise
        
        Args:
            audio (np.ndarray): Input audio data
            sr (int): Sample rate of the audio
        
        Returns:
            np.ndarray: Enhanced audio data
        """
        try:
            # Clean non-finite values
            if not np.all(np.isfinite(audio)):
                logger.warning("Audio contains non-finite values, replacing with zeros")
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize the audio
            audio = librosa.util.normalize(audio)
            
            # Apply bandpass filter to boost guitar frequencies (80 Hz - 1200 Hz)
            b, a = signal.butter(4, [80/(sr/2), 1200/(sr/2)], btype='bandpass')
            audio = signal.filtfilt(b, a, audio)
            
            # Amplify slightly
            audio = audio * 1.2
            
            # Clip to prevent distortion
            return np.clip(audio, -1, 1)
        except Exception as e:
            logger.warning(f"Audio enhancement failed: {str(e)}")
            return audio  # Return original audio if enhancement fails

    def _contains_guitar_frequencies(self, audio: np.ndarray, sr: int) -> bool:
        """Check if audio contains guitar frequency range."""
        try:
            S = np.abs(librosa.stft(audio))
            freqs = librosa.fft_frequencies(sr=sr)
            guitar_band = (freqs >= 80) & (freqs <= 400)
            energy = np.mean(S[guitar_band, :])
            return energy > 0.1
        except:
            return True

    def midi_to_note(self, fret: int, string: int) -> str:
        """Convert fret and string to note name."""
        midi_base = self.audio_analyzer.string_midi_values[string]
        midi_note = midi_base + fret
        return librosa.midi_to_note(midi_note)

    def optimize_fretboard(self, predictions: np.ndarray, key: str = 'C') -> np.ndarray:
        """Optimize fretboard positions using the FretboardPositionPredictor."""
        key_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 
                   'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 
                   'A#': 10, 'Bb': 10, 'B': 11}
        key_context = key_map.get(key.split(' ')[0], 0)
        
        if hasattr(self, 'fretboard_model') and self.fretboard_model is not None:
            midi_notes = []
            for t, pred in enumerate(predictions):
                for string in range(6):
                    if np.max(pred[string]) > 0:
                        fret = np.argmax(pred[string])
                        midi_base = self.audio_analyzer.string_midi_values[string]
                        midi_note = midi_base + fret
                        midi_notes.append((t, midi_note))
            
            optimized = np.zeros_like(predictions)
            prev_position = 0
            prev_string = 0
            
            for t, midi_note in midi_notes:
                string, fret = self.fretboard_model.predict_position(
                    midi_note, 
                    previous_position=prev_position,
                    string=prev_string,
                    key_context=key_context,
                    style=0
                )
                string = max(0, min(5, string))
                fret = max(0, min(24, fret))
                if t < optimized.shape[0]:
                    optimized[t, string, fret] = 1.0
                    prev_position = fret
                    prev_string = string
            
            return optimized
        else:
            notes = [
                {
                    'note': self.midi_to_note(np.argmax(pred[string]), string),
                    'time': t * 0.0232,
                    'duration': 0.1
                }
                for t, pred in enumerate(predictions) for string in range(6) if np.max(pred[string]) > 0
            ]
            mapped = self.audio_analyzer.map_notes_to_fretboard(notes)
            optimized = np.zeros_like(predictions)
            for note in mapped:
                t = int(note['time'] / 0.0232)
                if t < optimized.shape[0]:
                    optimized[t, note['string'], note['fret']] = 1.0
            return optimized

    def process_audio(self, audio_path: str, style: str = "metalcore", reference_tabs: list = None) -> dict:
        """
        Full processing pipeline:
        1. Demucs6 - Source separation
        2. Basic Pitch - Note detection
        3. Music theory analysis
        4. MIDI conversion
        5. TabCNN prediction
        6. Fretboard position modeling
        7. LilyPond generation
        """
        try:
            # 1. Source separation with Demucs6
            logger.info("Separating audio tracks with Demucs6...")
            separated = self.demucs.separate(audio_path)
            guitar_track = separated["guitar"]
            
            # 2. Note detection with Basic Pitch
            logger.info("Detecting notes and frequencies...")
            midi_data = self.basic_pitch.predict(guitar_track)
            
            # 3. Music theory refinement
            logger.info("Applying music theory analysis...")
            analyzed = self.music_theory_analyzer.analyze(midi_data)
            
            # 4. Convert to MIDI
            logger.info("Creating MIDI representation...")
            midi_path = os.path.join(self.output_dir, "guitar_track.mid")
            midi_data.write(midi_path)
            
            # 5. TabCNN prediction
            logger.info("Predicting tablature with TabCNN...")
            tab_prediction = self.tab_cnn.predict(midi_path)
            
            # 6. Fretboard position modeling
            logger.info("Optimizing fretboard positions...")
            optimized_tab = self.fretboard_model.optimize_positions(
                tab_prediction, 
                style=style,
                reference_tabs=reference_tabs
            )
            
            # 7. Generate LilyPond output
            logger.info("Generating LilyPond tablature...")
            lilypond_output = self.lilypond_generator.generate_tab(
                optimized_tab,
                title=os.path.basename(audio_path),
                tuning="Drop D" if style == "metalcore" else "Standard",
                show_techniques=True
            )
            
            return {
                "status": "success",
                "midi_path": midi_path,
                "tab_data": optimized_tab,
                "lilypond_path": lilypond_output,
                "audio_path": audio_path
            }
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return {"status": "error", "message": str(e)}

    def _enhance_predictions_with_midi(self, predictions: np.ndarray, midi_notes: List[Dict]) -> np.ndarray:
        """Enhance TabCNN predictions with MIDI note data
        
        Args:
            predictions: TabCNN predictions array
            midi_notes: List of MIDI note events from Basic Pitch
            
        Returns:
            Enhanced predictions array
        """
        if len(midi_notes) == 0:
            return predictions
            
        # Create a copy of the predictions to modify
        enhanced = np.copy(predictions)
        
        # Map MIDI notes to fretboard positions
        for note in midi_notes:
            # Get pitch and time information
            pitch = note['pitch']
            start_time = note['start_time']
            
            # Map pitch to string and fret
            string, fret = self._map_midi_to_guitar(pitch)
            
            if string is not None and fret is not None:
                # Convert time to frame index (assuming 0.0232s per frame)
                frame_idx = int(start_time / 0.0232)
                
                if 0 <= frame_idx < len(enhanced):
                    # Set the predicted fret for this string
                    # Only override if the confidence is high enough
                    confidence = note.get('confidence', 0.7)
                    if confidence > 0.6:
                        # Clear existing predictions for this string
                        enhanced[frame_idx, string, :] = 0
                        # Set the new prediction
                        enhanced[frame_idx, string, fret] = 1.0
        
        return enhanced
        
    def _map_midi_to_guitar(self, midi_pitch: int) -> Tuple[Optional[int], Optional[int]]:
        """Map MIDI pitch to guitar string and fret
        
        Args:
            midi_pitch: MIDI pitch value
            
        Returns:
            Tuple of (string, fret) or (None, None) if mapping not possible
        """
        # Standard guitar tuning (from low to high): E2(40), A2(45), D3(50), G3(55), B3(59), E4(64)
        string_tunings = [40, 45, 50, 55, 59, 64]
        
        # Try to find the best string/fret combination
        best_string = None
        best_fret = None
        best_distance = float('inf')
        
        for string_idx, base_pitch in enumerate(string_tunings):
            # Calculate fret number
            fret = midi_pitch - base_pitch
            
            # Check if note is playable on this string
            if 0 <= fret <= 24:  # Assuming 24 frets max
                # Prefer lower frets and lower strings
                distance = fret + (string_idx * 0.5)  # Slight preference for lower strings
                if distance < best_distance:
                    best_string = string_idx
                    best_fret = fret
                    best_distance = distance
        
        return best_string, best_fret

    def _chords_to_notes(self, chords: List[Dict], tempo: float) -> List[Dict]:
        """Convert chord progression to note data."""
        notes = []
        for chord in chords:
            start_time = chord['time']
            duration = chord['duration']
            for note_name in chord['notes']:
                notes.append({
                    'time': start_time,
                    'duration': duration,
                    'note': note_name,
                    'velocity': 100
                })
        return self.audio_analyzer.map_notes_to_fretboard(notes)

    def _notes_to_predictions(self, notes: List[Dict], shape: tuple) -> np.ndarray:
        """Convert notes to prediction-like array."""
        preds = np.zeros(shape)
        for note in notes:
            t = int(note['time'] / 0.0232)
            if t < shape[0]:
                preds[t, note['string'], note['fret']] = 1.0
        return preds

    def _predictions_to_notes(self, predictions: np.ndarray) -> List[Dict]:
        """Convert predictions to note data."""
        notes = []
        
        # Handle different prediction shapes
        if len(predictions.shape) == 3:
            # Original 3D format (time, string, fret)
            for t in range(predictions.shape[0]):
                for s in range(6):
                    fret = np.argmax(predictions[t, s])
                    if fret > 0:
                        notes.append({
                            'time': t * 0.0232,
                            'duration': 0.1,
                            'note': self.midi_to_note(fret, s),
                            'string': s,
                            'fret': fret
                        })
        elif len(predictions.shape) == 2:
            # Handle 2D predictions - common output format from CNN models
            if predictions.shape[0] == 1:
                # Batch size of 1 with features
                features = predictions[0]
                
                # For TabCNN, the output is often a flattened representation
                # Each group of consecutive values represents frets for a string
                # Assuming 21 frets per string (0-20) for 6 strings = 126 values
                num_frets = 21  # Including open string (0)
                num_strings = 6
                
                if len(features) == num_strings * num_frets:
                    # Format is likely [string1_frets, string2_frets, ...]
                    for s in range(num_strings):
                        string_start = s * num_frets
                        string_end = (s + 1) * num_frets
                        string_preds = features[string_start:string_end]
                        
                        # Find the highest probability fret
                        if np.max(string_preds) > 0.1:  # Confidence threshold
                            fret = np.argmax(string_preds)
                            if fret > 0:  # Skip if it's a 0 (open string)
                                notes.append({
                                    'time': 0,  # Single time step
                                    'duration': 0.1,
                                    'note': self.midi_to_note(fret, s),
                                    'string': s,
                                    'fret': fret
                                })
            else:
                # Multiple time steps
                for t in range(predictions.shape[0]):
                    row_data = predictions[t]
                    # Handle row data as needed
                    # This would need specific knowledge of the format
                    logger.warning(f"Unsupported prediction shape: {predictions.shape}")
        
        return notes

    def _create_fretboard_data(self, predictions: np.ndarray) -> List[Dict]:
        """Generate fretboard visualization data."""
        frames = []
        
        # Handle different prediction shapes
        if len(predictions.shape) == 3:
            # Original 3D format (time, string, fret)
            for frame_idx in range(predictions.shape[0]):
                frame_data = {'time': frame_idx * 0.0232, 'notes': []}
                for string in range(6):
                    fret = np.argmax(predictions[frame_idx, string])
                    if fret > 0:
                        frame_data['notes'].append({
                            'string': string,
                            'fret': int(fret),
                            'note': self.midi_to_note(fret, string)
                        })
                frames.append(frame_data)
        elif len(predictions.shape) == 2:
            # Handle 2D predictions
            if predictions.shape[0] == 1:
                # Single frame with features
                frame_data = {'time': 0, 'notes': []}
                features = predictions[0]
                
                # For TabCNN, assuming 21 frets per string (0-20) for 6 strings = 126 values
                num_frets = 21
                num_strings = 6
                
                if len(features) == num_strings * num_frets:
                    for s in range(num_strings):
                        string_start = s * num_frets
                        string_end = (s + 1) * num_frets
                        string_preds = features[string_start:string_end]
                        
                        if np.max(string_preds) > 0.1:  # Confidence threshold
                            fret = np.argmax(string_preds)
                            if fret > 0:  # Skip if it's a 0 (open string)
                                frame_data['notes'].append({
                                    'string': s,
                                    'fret': int(fret),
                                    'note': self.midi_to_note(fret, s)
                                })
                    
                    # Add the frame if it has any notes
                    if frame_data['notes']:
                        frames.append(frame_data)
            else:
                # Multiple time steps
                logger.warning(f"Unsupported prediction shape for fretboard visualization: {predictions.shape}")
        
        return frames

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python unified_tab_processor2.py <audio_file_path> [style]")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    style = sys.argv[2] if len(sys.argv) > 2 else "metalcore"
    
    processor = UnifiedTabProcessor()
    results = processor.process_audio(audio_path, style)
    
    if "error" in results:
        print(f"Processing failed: {results['error']}")
        sys.exit(1)
    else:
        print(f"Successfully processed {audio_path}")
        print(f"Results saved to {os.path.join(os.path.dirname(audio_path), 'tab_results.json')}")