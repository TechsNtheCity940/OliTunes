"""
Unified Tablature Processing Pipeline

1. Audio Separation (Demucs)
2. Tab Prediction (TabCNN)
3. Tab Refinement (LSTM)
4. Visualization
"""
import os
import logging
from typing import Dict, Optional, List
import numpy as np
import json
import soundfile as sf
import librosa

# Import existing modules
from interpreter import DemucsProcessor, TabCNNProcessor
from models.lstm_model.predictor import LSTMPredictor
from tab_text_generator import TabTextGenerator

# New imports for enhanced functionality
from music_theory import MusicTheoryAnalyzer
from midi_conversion import MidiConverter
from audio_analyzer_connector import EnhancedAudioAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_audio(audio_path: str) -> np.ndarray:
    """Basic audio preprocessing"""
    y, sr = librosa.load(audio_path)
    
    # Apply basic preprocessing
    y = librosa.effects.preemphasis(y)
    y = librosa.util.normalize(y)
    
    return y

def enhance_guitar_audio(audio_path: str, output_path: str = None) -> str:
    """
    Apply advanced audio processing to enhance the guitar stem for better tablature prediction
    
    Args:
        audio_path: Path to the guitar audio file
        output_path: Path to save the enhanced audio (if None, will modify the original path)
        
    Returns:
        Path to the enhanced audio file
    """
    import librosa
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Create output path if not provided
    if output_path is None:
        dirname = os.path.dirname(audio_path)
        basename = os.path.basename(audio_path)
        output_path = os.path.join(dirname, f"enhanced_{basename}")
    
    logger.info(f"Enhancing guitar audio from {audio_path}")
    logger.info(f"Output will be saved to {output_path}")
    
    try:
        # Load the audio file
        logger.info("Loading audio file...")
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        logger.info(f"Loaded audio with shape {y.shape} and sample rate {sr}")
        
        # Apply a series of audio enhancements
        logger.info("Starting audio enhancement process...")
        
        # 1. Noise reduction using spectral gating
        try:
            import noisereduce as nr
            logger.info("Applying noise reduction...")
            y = nr.reduce_noise(y=y, sr=sr)
            logger.info("Noise reduction completed")
        except ImportError:
            logger.warning("Noisereduce library not available, skipping noise reduction")
        except Exception as e:
            logger.warning(f"Error during noise reduction: {str(e)}")
        
        # 2. Apply high-pass filter to remove low frequency rumble
        try:
            from scipy import signal
            logger.info("Applying high-pass filter...")
            # Create high-pass filter
            b, a = signal.butter(4, 80/(sr/2), btype='highpass')
            y = signal.filtfilt(b, a, y)
            logger.info("High-pass filtering completed")
        except ImportError:
            logger.warning("SciPy not available, skipping high-pass filtering")
        except Exception as e:
            logger.warning(f"Error during high-pass filtering: {str(e)}")
        
        # 3. Apply EQ to enhance mid frequencies where guitar is prominent
        try:
            from scipy import signal
            logger.info("Applying mid-frequency boost...")
            # Boost frequencies between 500Hz and 5000Hz
            b, a = signal.butter(4, [500/(sr/2), 5000/(sr/2)], btype='bandpass')
            y_filtered = signal.filtfilt(b, a, y)
            # Mix with original
            y = y * 0.5 + y_filtered * 0.8
            logger.info("Mid-frequency boost completed")
        except ImportError:
            logger.warning("SciPy not available, skipping EQ enhancement")
        except Exception as e:
            logger.warning(f"Error during EQ enhancement: {str(e)}")
        
        # 4. Apply compression to even out dynamics
        try:
            logger.info("Applying dynamic compression...")
            # Simple compression algorithm
            threshold = 0.5
            ratio = 4.0
            makeup_gain = 1.5
            
            # Apply compression
            mask = np.abs(y) > threshold
            y[mask] = threshold + (np.abs(y[mask]) - threshold) / ratio * np.sign(y[mask])
            
            # Apply makeup gain
            y = y * makeup_gain
            logger.info("Dynamic compression completed")
        except Exception as e:
            logger.warning(f"Error applying compression: {str(e)}")
        
        # 5. Normalize the audio
        try:
            logger.info("Normalizing audio...")
            y = librosa.util.normalize(y)
            logger.info("Normalization completed")
        except Exception as e:
            logger.warning(f"Error during normalization: {str(e)}")
        
        # Save the processed audio
        logger.info(f"Saving enhanced audio to {output_path}...")
        sf.write(output_path, y, sr)
        
        logger.info(f"Enhanced guitar audio successfully saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error enhancing guitar audio: {str(e)}")
        # Return the original file as fallback
        logger.warning(f"Using original audio file as fallback: {audio_path}")
        return audio_path

class ConfidenceEvaluator:
    """Evaluates prediction confidence and determines when to use rule-based approaches."""
    
    def __init__(self, threshold=0.7, min_threshold=0.4, fallback_threshold=0.2):
        self.threshold = threshold
        self.min_threshold = min_threshold
        self.fallback_threshold = fallback_threshold
    
    def should_use_prediction(self, predictions, prediction_type='tablature'):
        """Determine if ML prediction should be used or fallback to rules."""
        if predictions is None:
            return False, 0.0, 'use_rules'
        
        # Calculate confidence based on prediction type
        if prediction_type == 'tablature':
            # For tablature, confidence is based on the max value in each frame
            confidence = np.mean([np.max(frame) for frame in predictions if np.any(frame)])
        else:
            # Generic confidence calculation
            confidence = np.mean(predictions)
        
        # Decision logic
        if confidence >= self.threshold:
            return True, confidence, 'use_prediction'
        elif confidence >= self.min_threshold:
            return True, confidence, 'hybrid'
        elif confidence >= self.fallback_threshold:
            return False, confidence, 'hybrid'
        else:
            return False, confidence, 'use_rules'
    
    def blend_predictions(self, ml_pred, rule_pred, confidence):
        """Blend ML and rule-based predictions based on confidence."""
        # Linear interpolation based on confidence
        weight = (confidence - self.fallback_threshold) / (self.threshold - self.fallback_threshold)
        weight = max(0.0, min(1.0, weight))  # Clamp to [0, 1]
        
        return ml_pred * weight + rule_pred * (1 - weight)
    
    def calibrate_with_dataset(self, model, X_val, y_val):
        """Calibrate thresholds using validation data."""
        # Placeholder for actual calibration logic
        results = {'thresholds': {}, 'results': {}}
        
        # Try different thresholds
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            self.threshold = threshold
            self.min_threshold = threshold * 0.6
            self.fallback_threshold = threshold * 0.3
            
            # Evaluate with current thresholds
            correct = 0
            total = 0
            used = 0
            
            for x, y in zip(X_val, y_val):
                pred = model.predict(np.expand_dims(x, axis=0))[0]
                use_pred, conf, _ = self.should_use_prediction(pred)
                
                if use_pred:
                    used += 1
                    # Check if prediction is correct (simplified)
                    pred_classes = np.argmax(pred, axis=-1)
                    true_classes = np.argmax(y, axis=-1)
                    if np.array_equal(pred_classes, true_classes):
                        correct += 1
                
                total += 1
            
            # Calculate metrics
            accuracy = correct / used if used > 0 else 0
            coverage = used / total if total > 0 else 0
            
            results['results'][threshold] = {
                'accuracy': accuracy,
                'coverage': coverage,
                'count': used
            }
        
        # Find optimal threshold (80% accuracy target)
        best_threshold = 0.7  # Default
        for threshold, metrics in results['results'].items():
            if metrics['accuracy'] >= 0.8 and metrics['coverage'] > results['results'].get(best_threshold, {}).get('coverage', 0):
                best_threshold = threshold
        
        # Update thresholds
        self.threshold = best_threshold
        self.min_threshold = best_threshold * 0.6
        self.fallback_threshold = best_threshold * 0.3
        
        results['thresholds'] = {
            'main': self.threshold,
            'min': self.min_threshold,
            'fallback': self.fallback_threshold
        }
        
        return results

class UnifiedTabProcessor:
    """Enhanced processor for generating guitar tablature from audio."""
    
    def __init__(self):
        self.demucs = DemucsProcessor()
        self.tabcnn = TabCNNProcessor()
        self.lstm = LSTMPredictor()
        self.text_gen = TabTextGenerator()
        self.confidence_evaluator = ConfidenceEvaluator(threshold=0.7, min_threshold=0.4, fallback_threshold=0.2)
        self.midi_converter = MidiConverter()
        self.theory_analyzer = MusicTheoryAnalyzer()
        self.audio_analyzer = EnhancedAudioAnalyzer()
        
        # Load fretboard position model
        try:
            from models.fretboard_position_model.position_predictor import FretboardPositionPredictor
            self.fretboard_model = FretboardPositionPredictor()
            self.use_fretboard_model = self.fretboard_model.model is not None
            logger.info("Fretboard position model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load fretboard position model: {str(e)}")
            self.use_fretboard_model = False
            self.fretboard_model = None
        
        # Load style configurations
        with open('backend/data/tab_data/style_configs.json') as f:
            self.style_configs = json.load(f)

    def preprocess_audio(self, audio_path: str) -> tuple:
        """Enhanced audio preprocessing with onset detection and tempo."""
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
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
        tempo = librosa.beat.tempo(y, sr=sr)[0]
        
        return spec, onsets, tempo

    def enhance_guitar_audio(self, audio_path: str, output_path: str) -> str:
        """Enhanced guitar audio processing."""
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        y = librosa.effects.preemphasis(y)
        y = librosa.util.normalize(y)
        sf.write(output_path, y, sr)
        return output_path

    def midi_to_note(self, fret: int, string: int) -> str:
        """Convert fret and string to note name."""
        midi_base = self.audio_analyzer.string_midi_values[string]
        midi_note = midi_base + fret
        return librosa.midi_to_note(midi_note)

    def optimize_fretboard(self, predictions: np.ndarray, key: str = 'C') -> np.ndarray:
        """
        Optimize fretboard positions using the trained fretboard position model when available,
        falling back to EnhancedAudioAnalyzer when the model is not available.
        
        Args:
            predictions: Raw tablature predictions
            key: Musical key of the piece
            
        Returns:
            Optimized tablature with improved fretboard positions
        """
        # Extract notes from predictions
        notes = [
            {
                'note': self.midi_to_note(np.argmax(pred[string]), string),
                'midi_note': self.audio_analyzer.string_midi_values[string] + np.argmax(pred[string]),
                'time': t * 0.0232,  # 512/22050 ≈ 0.0232s
                'duration': 0.1,  # Default duration
                'string': string,
                'fret': np.argmax(pred[string])
            }
            for t, pred in enumerate(predictions) for string in range(6) if np.max(pred[string]) > 0
        ]
        
        # Use fretboard position model if available
        if self.use_fretboard_model:
            logger.info("Using fretboard position model for position optimization")
            optimized = np.zeros_like(predictions)
            
            # Process notes sequentially to maintain context
            prev_string, prev_fret = 0, 0
            key_context = self.theory_analyzer.key_to_index(key) if key else 0
            style_context = 0  # Default style
            
            for note in notes:
                # Get optimized position from the model
                midi_note = note['midi_note']
                t = int(note['time'] / 0.0232)
                
                if t < optimized.shape[0]:
                    # Use model to predict optimal string/fret
                    string, fret = self.fretboard_model.predict_position(
                        midi_note, 
                        previous_position=prev_fret,
                        string=prev_string,
                        key_context=key_context,
                        style=style_context
                    )
                    
                    # Update the optimized tab
                    optimized[t, string, fret] = 1.0  # One-hot encoding
                    
                    # Update context for next note
                    prev_string, prev_fret = string, fret
            
            return optimized
        else:
            # Fallback to original method using EnhancedAudioAnalyzer
            logger.info("Fretboard position model not available, using fallback method")
            mapped = self.audio_analyzer.map_notes_to_fretboard(notes)
            
            optimized = np.zeros_like(predictions)
            for note in mapped:
                t = int(note['time'] / 0.0232)
                if t < optimized.shape[0]:
                    string = note['string']
                    fret = note['fret']
                    optimized[t, string, fret] = 1.0  # One-hot encoding
            
            return optimized

    def process_audio(self, audio_path: str, style: str = 'rock', key: Optional[str] = None) -> Dict[str, Dict]:
        """Full audio-to-tab pipeline with enhancements."""
        try:
            base_dir = os.path.dirname(audio_path)
            song_name = os.path.splitext(os.path.basename(audio_path))[0]
            stems_dir = os.path.join(base_dir, "olitunes_stems", song_name)
            os.makedirs(stems_dir, exist_ok=True)

            # 1. Separate audio
            stems = self.demucs.separate_audio(audio_path, output_dir=stems_dir)
            guitar_track = stems.get('guitar', stems.get('other', audio_path))
            enhanced_path = os.path.join(stems_dir, "enhanced_guitar.wav")
            guitar_track = self.enhance_guitar_audio(guitar_track, enhanced_path)

            # 2. Preprocess audio
            y, sr = librosa.load(guitar_track, sr=22050, mono=True)
            spec, onsets, tempo = self.preprocess_audio(guitar_track)
            
            # 3. Music theory analysis
            if not key:
                key_info = self.theory_analyzer.detect_key(y, sr)
                key = key_info['key']
                chords = self.theory_analyzer.analyze_chord_progression(y, sr)
            else:
                chords = self.theory_analyzer.analyze_chord_progression(y, sr)

            # 4. Predict raw tablature
            raw_tab = self.tabcnn.predict_tablature(spec)
            
            # 5. Evaluate confidence and refine
            use_pred, conf, rec = self.confidence_evaluator.should_use_prediction(raw_tab, 'tablature')
            if not use_pred and rec == 'use_rules':
                # Fallback to chord-based tab
                text_tab = self.text_gen.generate_text_tab(
                    self._chords_to_notes(chords, tempo), 
                    {'tempo': tempo, 'key': key, 'chords': chords}
                )
                refined_tab = raw_tab  # Keep raw for MIDI/fretboard data
            else:
                # Refine with LSTM
                refined_tab = self.lstm.predict(raw_tab)
                if rec == 'hybrid':
                    rule_notes = self._chords_to_notes(chords, tempo)
                    rule_tab = self._notes_to_predictions(rule_notes, refined_tab.shape)
                    refined_tab = self.confidence_evaluator.blend_predictions(refined_tab, rule_tab, conf)

            # 6. Optimize fretboard positions
            optimized_tab = self.optimize_fretboard(refined_tab, key)

            # 7. Generate text tab
            config = self.style_configs.get(style, {})
            note_data = self._predictions_to_notes(optimized_tab)
            text_tab = self.text_gen.generate_text_tab(
                note_data,
                {'tempo': tempo, 'key': key, 'chords': chords}
            )

            # 8. Generate MIDI
            midi_notes = [
                {'time': note['time'], 'note': note['note'], 'duration': note['duration'], 'velocity': 100}
                for note in note_data
            ]
            midi_path = os.path.join(stems_dir, "guitar_tab.mid")
            midi_result = self.midi_converter.notes_to_midi(midi_notes, midi_path, bpm=tempo)

            # 9. Prepare results
            results = {
                'guitar': {
                    'text_tab': text_tab,
                    'fretboard_data': self._create_fretboard_data(optimized_tab),
                    'stem_path': guitar_track,
                    'onsets': onsets.tolist(),
                    'midi_path': midi_path if midi_result == midi_path else None,
                    'confidence': conf,
                    'key': key,
                    'chords': chords,
                    'tempo': float(tempo)
                }
            }
            
            # Save results
            with open(os.path.join(stems_dir, "tab_results.json"), 'w') as f:
                json.dump({
                    k: {sk: sv if not isinstance(sv, np.ndarray) else sv.tolist() 
                        for sk, sv in v.items()} 
                    for k, v in results.items()
                }, f, indent=2)
            
            return results

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return {}

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
                    'note': note_name
                })
        return notes

    def _notes_to_predictions(self, notes: List[Dict], shape: tuple) -> np.ndarray:
        """Convert note data to prediction array format."""
        predictions = np.zeros(shape)
        for note in notes:
            t = int(note['time'] / 0.0232)
            if t < shape[0]:
                # Find best string/fret for this note
                mapped = self.audio_analyzer.map_note_to_fretboard(note['note'])
                if mapped:
                    string = mapped['string']
                    fret = mapped['fret']
                    predictions[t, string, fret] = 1.0
        return predictions

    def _predictions_to_notes(self, predictions: np.ndarray) -> List[Dict]:
        """Convert prediction array to note data."""
        notes = []
        for t, frame in enumerate(predictions):
            for string in range(6):
                fret_idx = np.argmax(frame[string])
                if frame[string, fret_idx] > 0:
                    note_name = self.midi_to_note(fret_idx, string)
                    notes.append({
                        'time': t * 0.0232,
                        'duration': 0.1,  # Default duration
                        'note': note_name,
                        'string': string,
                        'fret': int(fret_idx)
                    })
        return notes

    def _create_fretboard_data(self, optimized_tab: np.ndarray) -> List[Dict]:
        """Convert optimized tab to fretboard visualization data."""
        fretboard_data = []
        for t, frame in enumerate(optimized_tab):
            notes = []
            for string in range(6):
                fret_idx = np.argmax(frame[string])
                if frame[string, fret_idx] > 0:
                    notes.append({
                        'string': int(string),
                        'fret': int(fret_idx),
                        'note': self.midi_to_note(fret_idx, string)
                    })
            if notes:
                fretboard_data.append({
                    'time': t * 0.0232,
                    'notes': notes
                })
        return fretboard_data

# Helper function for app.py integration
def create_tablature(audio_file: str, style: str = 'rock'):
    processor = UnifiedTabProcessor()
    return processor.process_audio(audio_file, style)

# Test function
def test_with_audio_file(audio_path: str):
    """
    Test the unified tab processor with a specific audio file
    
    Args:
        audio_path: Path to the audio file to process
    """
    import matplotlib.pyplot as plt
    import time
    
    start_time = time.time()
    
    # Initialize processor
    processor = UnifiedTabProcessor()
    
    # Process audio
    print(f"Processing audio file: {audio_path}")
    results = processor.process_audio(audio_path, style='rock')
    
    # Print processing time
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    # Print results summary
    if results and 'guitar' in results:
        guitar_data = results['guitar']
        print("\n=== Guitar Tablature Results ===")
        print(f"Key: {guitar_data.get('key', 'Unknown')}")
        print(f"Tempo: {guitar_data.get('tempo', 0):.1f} BPM")
        print(f"Confidence: {guitar_data.get('confidence', 0):.2f}")
        
        # Print chord progression
        chords = guitar_data.get('chords', [])
        if chords:
            print("\nChord Progression:")
            for i, chord in enumerate(chords[:10]):  # Show first 10 chords
                print(f"{i+1}. {chord['chord']} ({chord['time']:.2f}s - {chord['time'] + chord['duration']:.2f}s)")
            if len(chords) > 10:
                print(f"... and {len(chords) - 10} more chords")
        
        # Print tablature excerpt
        tab_text = guitar_data.get('text_tab', '')
        if tab_text:
            print("\nTablature Excerpt (first 10 lines):")
            tab_lines = tab_text.split('\n')
            for line in tab_lines[:10]:
                print(line)
            if len(tab_lines) > 10:
                print(f"... and {len(tab_lines) - 10} more lines")
        
        # Print MIDI path
        midi_path = guitar_data.get('midi_path')
        if midi_path:
            print(f"\nMIDI file saved to: {midi_path}")
        
        # Print stem path
        stem_path = guitar_data.get('stem_path')
        if stem_path:
            print(f"Guitar stem saved to: {stem_path}")
        
        # Visualize fretboard data if available
        fretboard_data = guitar_data.get('fretboard_data', [])
        if fretboard_data:
            print(f"\nGenerated {len(fretboard_data)} fretboard positions")
            
            # Plot first few positions
            try:
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                axes = axes.flatten()
                
                for i, ax in enumerate(axes):
                    if i < min(6, len(fretboard_data)):
                        frame = fretboard_data[i]
                        time = frame['time']
                        notes = frame['notes']
                        
                        # Create fretboard grid
                        fretboard = np.zeros((6, 21))
                        for note in notes:
                            fretboard[note['string'], note['fret']] = 1
                        
                        ax.imshow(fretboard, aspect='auto', cmap='Blues')
                        ax.set_title(f"Time: {time:.2f}s")
                        ax.set_xlabel("Fret")
                        ax.set_ylabel("String")
                        ax.set_yticks(range(6))
                        ax.set_yticklabels(['E', 'A', 'D', 'G', 'B', 'e'])
                        
                        # Annotate notes
                        for note in notes:
                            ax.text(note['fret'], note['string'], note['note'], 
                                   ha='center', va='center', color='red')
                    else:
                        ax.axis('off')
                
                plt.tight_layout()
                plt.savefig("fretboard_visualization.png")
                print("Fretboard visualization saved to: fretboard_visualization.png")
            except Exception as e:
                print(f"Could not create visualization: {str(e)}")
    else:
        print("No results or guitar data found")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python unified_tab_processor.py <path_to_audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        sys.exit(1)
    
    test_with_audio_file(audio_path)
