import librosa
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union
import os
import logging
from scipy import signal
import json
import time
import wave
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load ffmpeg configuration before importing audio libraries
try:
    # Import our custom ffmpeg configuration
    from pydub_config import setup_ffmpeg, force_ffmpeg_path
    # First try the normal setup
    if not setup_ffmpeg():
        # If that fails, force the path
        force_ffmpeg_path()
    logger.info("FFmpeg configuration loaded successfully")
except Exception as e:
    logger.warning(f"Failed to load FFmpeg configuration: {e}")

# Core audio processing libraries
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logging.warning("PyAudio not available. Real-time audio processing will be limited.")

try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    logging.warning("Essentia not available. Advanced audio analysis features will be limited.")
    
    # Create fallback implementations for essential Essentia features
    class EssentiaFallback:
        """Fallback implementation of essential Essentia functionality"""
        
        class MonoLoader:
            def __init__(self, filename=None):
                self.filename = filename
            
            def __call__(self):
                """Load audio file using librosa instead"""
                logging.info(f"Using librosa to load audio file: {self.filename}")
                import librosa
                y, sr = librosa.load(self.filename, sr=44100, mono=True)
                return y
        
        class Windowing:
            def __init__(self, type='hann'):
                self.type = type
            
            def __call__(self, frame):
                """Apply window function to a frame"""
                import numpy as np
                import scipy.signal as signal
                
                if self.type == 'hann':
                    return frame * signal.windows.hann(len(frame))
                elif self.type == 'hamming':
                    return frame * signal.windows.hamming(len(frame))
                else:
                    return frame
        
        class Spectrum:
            def __call__(self, frame):
                """Compute spectrum from a frame"""
                import numpy as np
                return np.abs(np.fft.rfft(frame))
        
        class SpectralPeaks:
            def __call__(self, spec):
                """Find spectral peaks in the spectrum"""
                import numpy as np
                from scipy.signal import find_peaks
                
                # Find peaks with simple threshold
                peaks, _ = find_peaks(spec, height=np.mean(spec)*1.5)
                freqs = peaks
                mags = spec[peaks]
                
                return freqs, mags
        
        class HPCP:
            def __call__(self, freqs, mags):
                """Generate a Harmonic Pitch Class Profile (simplified)"""
                import numpy as np
                
                # Simple fallback implementation - create 12 bins for pitch classes
                hpcp = np.zeros(12)
                
                if len(freqs) == 0:
                    return hpcp
                
                # Map frequencies to pitch classes (very simplified)
                for i, freq in enumerate(freqs):
                    if freq > 0:
                        # Convert frequency bin to approximate note
                        note_bin = int(round(12 * np.log2(freq / 440.0) + 69) % 12)
                        hpcp[note_bin] += mags[i]
                
                # Normalize
                if np.sum(hpcp) > 0:
                    hpcp = hpcp / np.max(hpcp)
                
                return hpcp
        
        class Key:
            def __call__(self, hpcp):
                """Estimate key from HPCP (simplified)"""
                import numpy as np
                
                # Major and minor key profiles (simplified Krumhansl-Kessler profiles)
                major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
                minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
                
                major_corrs = []
                minor_corrs = []
                
                # Calculate correlation with each possible key
                for i in range(12):
                    # Rotate profiles to different keys
                    shifted_major = np.roll(major_profile, i)
                    shifted_minor = np.roll(minor_profile, i)
                    
                    # Calculate correlation
                    major_corr = np.corrcoef(hpcp, shifted_major)[0, 1]
                    minor_corr = np.corrcoef(hpcp, shifted_minor)[0, 1]
                    
                    major_corrs.append(major_corr)
                    minor_corrs.append(minor_corr)
                
                # Find best correlation
                best_major_idx = np.argmax(major_corrs)
                best_minor_idx = np.argmax(minor_corrs)
                
                if major_corrs[best_major_idx] > minor_corrs[best_minor_idx]:
                    key_idx = best_major_idx
                    scale = "major"
                else:
                    key_idx = best_minor_idx
                    scale = "minor"
                
                # Convert index to key name
                key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                key = key_names[key_idx]
                
                return key, scale
        
        class ChordsDetection:
            def __call__(self, hpcp):
                """Detect chord from HPCP (simplified)"""
                import numpy as np
                
                # Basic chord templates
                major_template = np.zeros(12)
                major_template[[0, 4, 7]] = 1  # Root, major third, perfect fifth
                
                minor_template = np.zeros(12)
                minor_template[[0, 3, 7]] = 1  # Root, minor third, perfect fifth
                
                major7_template = np.zeros(12)
                major7_template[[0, 4, 7, 11]] = 1  # Root, major third, perfect fifth, major seventh
                
                minor7_template = np.zeros(12)
                minor7_template[[0, 3, 7, 10]] = 1  # Root, minor third, perfect fifth, minor seventh
                
                dom7_template = np.zeros(12)
                dom7_template[[0, 4, 7, 10]] = 1  # Root, major third, perfect fifth, minor seventh
                
                # Calculate correlation for each chord type and each root note
                best_corr = -1
                best_chord = "N"  # No chord
                best_scale = "none"
                
                # Check each root note
                key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                
                for root in range(12):
                    # Shift templates to current root
                    shifted_major = np.roll(major_template, root)
                    shifted_minor = np.roll(minor_template, root)
                    shifted_major7 = np.roll(major7_template, root)
                    shifted_minor7 = np.roll(minor7_template, root)
                    shifted_dom7 = np.roll(dom7_template, root)
                    
                    # Calculate correlations
                    major_corr = np.corrcoef(hpcp, shifted_major)[0, 1]
                    minor_corr = np.corrcoef(hpcp, shifted_minor)[0, 1]
                    major7_corr = np.corrcoef(hpcp, shifted_major7)[0, 1]
                    minor7_corr = np.corrcoef(hpcp, shifted_minor7)[0, 1]
                    dom7_corr = np.corrcoef(hpcp, shifted_dom7)[0, 1]
                    
                    # Handle NaN correlations (happens with all-zero hpcp)
                    if np.isnan(major_corr): major_corr = -1
                    if np.isnan(minor_corr): minor_corr = -1
                    if np.isnan(major7_corr): major7_corr = -1
                    if np.isnan(minor7_corr): minor7_corr = -1
                    if np.isnan(dom7_corr): dom7_corr = -1
                    
                    # Find best match
                    corrs = [major_corr, minor_corr, major7_corr, minor7_corr, dom7_corr]
                    chord_types = ["major", "minor", "maj7", "min7", "7"]
                    
                    best_type_idx = np.argmax(corrs)
                    corr = corrs[best_type_idx]
                    
                    if corr > best_corr:
                        best_corr = corr
                        chord_type = chord_types[best_type_idx]
                        
                        # Format chord name
                        if chord_type == "major":
                            best_chord = key_names[root]
                        elif chord_type == "minor":
                            best_chord = key_names[root] + "m"
                        else:
                            best_chord = key_names[root] + chord_type
                        
                        best_scale = "major" if chord_type in ["major", "maj7", "7"] else "minor"
                
                return best_chord, best_scale
        
        class FrameGenerator:
            def __init__(self, audio, frameSize=2048, hopSize=1024):
                self.audio = audio
                self.frame_size = frameSize
                self.hop_size = hopSize
                self.current = 0
                
            def __iter__(self):
                self.current = 0
                return self
                
            def __next__(self):
                if self.current + self.frame_size > len(self.audio):
                    raise StopIteration
                
                frame = self.audio[self.current:self.current + self.frame_size]
                self.current += self.hop_size
                
                # Zero-pad if necessary
                if len(frame) < self.frame_size:
                    import numpy as np
                    frame = np.pad(frame, (0, self.frame_size - len(frame)))
                    
                return frame
    
    # Create the fallback namespace
    es = EssentiaFallback()

try:
    import madmom
    MADMOM_AVAILABLE = True
except ImportError:
    MADMOM_AVAILABLE = False
    logging.warning("Madmom not available. Advanced rhythm analysis will be limited.")

try:
    import aubio
    AUBIO_AVAILABLE = True
except ImportError:
    AUBIO_AVAILABLE = False
    logging.warning("Aubio not available. Some pitch detection features will be limited.")

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    logging.warning("Mido not available. MIDI export/import will be limited.")

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False
    logging.warning("Pretty_midi not available. Advanced MIDI processing will be limited.")

try:
    import pedalboard
    from pedalboard import Pedalboard, Chorus, Reverb, Delay
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    logging.warning("Pedalboard not available. Audio effects processing will be limited.")

try:
    import music21
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    logging.warning("Music21 not available. Music theory analysis will be limited.")

# Already in requirements
from pydub import AudioSegment

# Helper function for time formatting
def format_time(seconds):
    """Format seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"
