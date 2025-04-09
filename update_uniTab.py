import os
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List
from interpreter import DemucsProcessor, TabCNNProcessor
from tab_text_generator import TabTextGenerator
from models.lstm_model.predictor import LSTMPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedTabProcessor:
    def __init__(self):
        self.demucs = DemucsProcessor()
        self.tabcnn = TabCNNProcessor()
        self.lstm = LSTMPredictor()
        self.text_gen = TabTextGenerator()
        with open('backend/data/tab_data/style_configs.json') as f:
            self.style_configs = json.load(f)

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Enhanced audio preprocessing for guitar."""
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        # Noise reduction
        try:
            import noisereduce as nr
            y = nr.reduce_noise(y=y, sr=sr)
        except ImportError:
            logger.warning("Noisereduce unavailable, skipping noise reduction.")
        # High-pass filter (80 Hz cutoff)
        y = librosa.effects.hpss(y)[0]  # Harmonic component only
        # Onset detection for note segmentation
        onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
        # Mel-spectrogram with guitar-tuned bins
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=80, fmax=4000, hop_length=512)
        return spec, onsets

    def enhance_guitar_audio(self, audio_path: str, output_path: str) -> str:
        """Enhanced guitar audio processing."""
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        # Compression and EQ
        y = librosa.effects.preemphasis(y)
        y = librosa.util.normalize(y)
        sf.write(output_path, y, sr)
        return output_path

    def optimize_fretboard(self, predictions: np.ndarray, key: str = 'C') -> np.ndarray:
        """Optimize fretboard positions for playability."""
        from music_theory import fretboard_optimizer
        optimized = np.zeros_like(predictions)
        for frame_idx in range(predictions.shape[0]):
            frame = predictions[frame_idx]
            prev_frame = predictions[frame_idx - 1] if frame_idx > 0 else None
            optimized[frame_idx] = fretboard_optimizer.optimize_frame(frame, prev_frame, key)
        return optimized

    def process_audio(self, audio_path: str, style: str = 'rock') -> Dict[str, Dict]:
        """Full audio-to-tab pipeline with improvements."""
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

            # 2. Preprocess and predict
            spec, onsets = self.preprocess_audio(guitar_track)
            raw_tab = self.tabcnn.predict_tablature(spec)

            # 3. Refine with LSTM
            refined_tab = self.lstm.predict(raw_tab)

            # 4. Optimize fretboard positions
            key = 'C'  # Placeholder; get from app.py analysis
            optimized_tab = self.optimize_fretboard(refined_tab, key)

            # 5. Generate text tab
            config = self.style_configs.get(style, {})
            text_tab = self.text_gen.generate(optimized_tab, config.get('bpm', 120))

            results = {
                'guitar': {
                    'text_tab': text_tab,
                    'fretboard_data': self._create_fretboard_data(optimized_tab),
                    'stem_path': guitar_track,
                    'onsets': onsets.tolist()
                }
            }
            with open(os.path.join(stems_dir, "tab_results.json"), 'w') as f:
                json.dump({k: {sk: sv if not isinstance(sv, np.ndarray) else sv.tolist() 
                              for sk, sv in v.items()} for k, v in results.items()}, f, indent=2)
            return results

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return {}

    def _create_fretboard_data(self, predictions: np.ndarray) -> List[Dict]:
        """Generate fretboard visualization data."""
        frames = []
        for frame_idx in range(predictions.shape[0]):
            frame_data = {'time': frame_idx * 0.0232, 'notes': []}  # 512/22050 ≈ 0.0232s
            for string in range(6):
                fret = np.argmax(predictions[frame_idx, string])
                if fret > 0:  # Ignore open strings or no note
                    frame_data['notes'].append({'string': string, 'fret': fret})
            if frame_data['notes']:
                frames.append(frame_data)
        return frames