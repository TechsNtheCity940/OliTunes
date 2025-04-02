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
