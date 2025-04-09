import os
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional
import json

# Assuming these are your existing imports from your project structure
from interpreter import DemucsProcessor, TabCNNProcessor
from models.lstm_model.predictor import LSTMPredictor

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
        self.lstm = LSTMPredictor()
        self.text_gen = TabTextGenerator()
        self.confidence_evaluator = ConfidenceEvaluator(threshold=0.7, min_threshold=0.4, fallback_threshold=0.2)
        self.midi_converter = MidiConverter()
        self.theory_analyzer = MusicTheoryAnalyzer()
        self.audio_analyzer = EnhancedAudioAnalyzer()
        
        # Load style configurations
        with open('backend/data/tab_data/style_configs.json') as f:
            self.style_configs = json.load(f)

    def preprocess_audio(self, audio_path: str) -> tuple[np.ndarray, np.ndarray, float]:
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
        """Optimize fretboard positions using EnhancedAudioAnalyzer."""
        notes = [
            {
                'note': self.midi_to_note(np.argmax(pred[string]), string),
                'time': t * 0.0232,  # 512/22050 ≈ 0.0232s
                'duration': 0.1  # Default duration
            }
            for t, pred in enumerate(predictions) for string in range(6) if np.max(pred[string]) > 0
        ]
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
    processor = UnifiedTabProcessor()
    results = processor.process_audio("sample_audio.wav", style="rock")
    print(json.dumps(results, indent=2))