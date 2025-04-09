import os
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Any
import json
import warnings
from scipy import signal

# Suppress FFmpeg warnings from pydub
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Couldn't find ffmpeg or avconv")

# Assuming these are your existing imports from your project structure
from interpreter import DemucsProcessor, TabCNNProcessor
from models.lstm_model.LSTMPredictor import LSTMPredictor
from models.fretboard_position_model.position_predictor import FretboardPositionPredictor

# New imports from your provided modules
from confidence_evaluator import ConfidenceEvaluator
from midi_converter import MidiConverter
from music_theory_analyzer import MusicTheoryAnalyzer
from tab_text_generator import TabTextGenerator
from enhanced_audio_analyzer import EnhancedAudioAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedTabProcessor:
    """Enhanced processor for generating guitar tablature from audio."""
    
    def __init__(self):
        self.demucs = DemucsProcessor()
        self.tabcnn = TabCNNProcessor()
        # Try to load weights, but don't fail if there's an issue
        try:
            custom_weights_path = 'F:/newrepos/olitunes/backend/models/tab-cnn/model/saved/c 2025-04-06 082819/4/model.weights.h5'
            if os.path.exists(custom_weights_path):
                try:
                    self.tabcnn.model.load_weights(custom_weights_path)
                    logger.info(f"Loaded TabCNN weights successfully")
                except TypeError as type_error:
                    if "by_name" in str(type_error):
                        logger.info(f"TabCNN model doesn't support by_name parameter, trying without it")
                        self.tabcnn.model.load_weights(custom_weights_path)
                        logger.info(f"Loaded TabCNN weights successfully")
                    else:
                        raise
        except Exception as e:
            logger.warning(f"Could not load custom TabCNN weights: {e}. Using default model.")
        
        self._load_lstm_model()
        self._load_fretboard_position_model()
        self.text_gen = TabTextGenerator()
        self.confidence_evaluator = ConfidenceEvaluator(threshold=0.7, min_threshold=0.4, fallback_threshold=0.2)
        self.midi_converter = MidiConverter()
        self.theory_analyzer = MusicTheoryAnalyzer()
        self.audio_analyzer = EnhancedAudioAnalyzer()
        
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

    def preprocess_audio(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Enhanced audio preprocessing with onset detection and tempo.

        Args:
            y (np.ndarray): Audio time series.
            sr (int): Sample rate of the audio.

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

    def process_audio(self, audio_path: str, style: str = "rock") -> Dict[str, Any]:
        """Process audio file to generate guitar tablature."""
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
            logger.info(f"Processing audio file: {audio_path}")
            
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            logger.info(f"Loaded audio with sample rate: {sr}Hz")
            
            style_config = self.style_configs.get(style, self.style_configs["rock"])
            
            stems = self.demucs.separate_audio(audio_path)
            for stem_name, stem_data in stems.items():
                if not np.all(np.isfinite(stem_data)):
                    logger.warning(f"Stem '{stem_name}' contains non-finite values, cleaning")
                    stems[stem_name] = np.nan_to_num(stem_data, nan=0.0, posinf=0.0, neginf=0.0)
            logger.info(f"Raw stems: {list(stems.keys())}")
            
            guitar_audio = None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            if stems.get('other'):
                logger.info("Checking 'other' stem for guitar content")
                guitar_audio = self._enhance_guitar_audio(stems['other'], sr)
                if not self._contains_guitar_frequencies(guitar_audio, sr):
                    logger.warning("'other' stem doesn't contain guitar frequencies")
                    guitar_audio = None
            
            if guitar_audio is None:
                logger.warning("Using original audio - no guitar stem found")
                guitar_audio = self._enhance_guitar_audio(audio, sr)
            
            # Preprocess guitar audio
            spec, onsets, tempo = self.preprocess_audio(guitar_audio, sr)
            
            # TabCNN processing
            tabcnn_results = self.tabcnn.predict_tablature(spec)
            if not tabcnn_results:
                raise RuntimeError("TabCNN processing failed")
            
            # LSTM refinement
            if hasattr(self, 'lstm') and self.lstm.model:
                lstm_results = self.lstm.predict(tabcnn_results['predictions'])
            
            # Music theory analysis
            key_info = self.theory_analyzer.detect_key(audio, sr)
            key = key_info['key']
            chords = self.theory_analyzer.analyze_chord_progression(audio, sr)

            raw_tab = tabcnn_results['predictions']
            use_pred, conf, rec = self.confidence_evaluator.should_use_prediction(raw_tab, 'tablature')
            
            if not use_pred and rec == 'use_rules':
                text_tab = self.text_gen.generate_text_tab(
                    self._chords_to_notes(chords, tempo), 
                    {'tempo': tempo, 'key': key, 'chords': chords}
                )
                refined_tab = raw_tab
            else:
                refined_tab = lstm_results['refined_tab']
                if rec == 'hybrid':
                    rule_notes = self._chords_to_notes(chords, tempo)
                    rule_tab = self._notes_to_predictions(rule_notes, refined_tab.shape)
                    refined_tab = self.confidence_evaluator.blend_predictions(refined_tab, rule_tab, conf)

            optimized_tab = self.optimize_fretboard(refined_tab, key)
            note_data = self._predictions_to_notes(optimized_tab)
            text_tab = self.text_gen.generate_text_tab(
                note_data,
                {'tempo': tempo, 'key': key, 'chords': chords}
            )

            midi_notes = [
                {'time': note['time'], 'note': note['note'], 'duration': note['duration'], 'velocity': 100}
                for note in note_data
            ]
            midi_path = os.path.join(os.path.dirname(audio_path), "guitar_tab.mid")
            midi_result = self.midi_converter.notes_to_midi(midi_notes, midi_path, bpm=tempo)

            results = {
                'guitar': {
                    'text_tab': text_tab,
                    'fretboard_data': self._create_fretboard_data(optimized_tab),
                    'stem_path': stems.get('guitar', audio_path),
                    'onsets': onsets.tolist(),
                    'midi_path': midi_path if midi_result == midi_path else None,
                    'confidence': conf,
                    'key': key,
                    'chords': chords,
                    'tempo': float(tempo)
                }
            }
            
            with open(os.path.join(os.path.dirname(audio_path), "tab_results.json"), 'w') as f:
                json.dump({
                    k: {sk: sv if not isinstance(sv, np.ndarray) else sv.tolist() 
                        for sk, sv in v.items()} 
                    for k, v in results.items()
                }, f, indent=2)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process audio: {str(e)}")
            return {"error": str(e)}

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
        return notes

    def _create_fretboard_data(self, predictions: np.ndarray) -> List[Dict]:
        """Generate fretboard visualization data."""
        frames = []
        for frame_idx in range(predictions.shape[0]):
            frame_data = {'time': frame_idx * 0.0232, 'notes': []}
            for string in range(6):
                fret = np.argmax(predictions[frame_idx, string])
                if fret > 0:
                    frame_data['notes'].append({'string': string, 'fret': fret})
            if frame_data['notes']:
                frames.append(frame_data)
        return frames

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python unified_tab_processor2.py <audio_file_path> [style]")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    style = sys.argv[2] if len(sys.argv) > 2 else "rock"
    
    processor = UnifiedTabProcessor()
    results = processor.process_audio(audio_path, style)
    
    if "error" in results:
        print(f"Processing failed: {results['error']}")
        sys.exit(1)
    else:
        print(f"Successfully processed {audio_path}")
        print(f"Results saved to {os.path.join(os.path.dirname(audio_path), 'tab_results.json')}")