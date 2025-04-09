"""
Simple test script to verify Essentia fallback functionality
"""
import os
import sys
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# First load the ffmpeg configuration
import backend.pydub_config

# Then import from audio_imports
from backend.audio_imports import ESSENTIA_AVAILABLE, es
print(f'Essentia available: {ESSENTIA_AVAILABLE}')

if not ESSENTIA_AVAILABLE:
    print('Testing Essentia fallback with simple signal...')
    
    # Create a simple test signal (sine wave)
    sample_rate = 44100
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    freq = 440.0  # A4 note
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    
    # Test the frame generator
    frame_gen = es.FrameGenerator(signal, frameSize=2048, hopSize=1024)
    frames = list(frame_gen)
    print(f"Generated {len(frames)} frames from audio signal")
    
    # Test windowing
    windowing = es.Windowing(type='hann')
    windowed_frame = windowing(frames[0])
    print(f"Applied windowing to frame, shape: {windowed_frame.shape}")
    
    # Test spectrum
    spectrum = es.Spectrum()
    spec = spectrum(windowed_frame)
    print(f"Computed spectrum, shape: {spec.shape}")
    
    # Test spectral peaks
    spectral_peaks = es.SpectralPeaks()
    freqs, mags = spectral_peaks(spec)
    print(f"Found {len(freqs)} spectral peaks")
    
    # Test HPCP
    hpcp = es.HPCP()
    hpcp_values = hpcp(freqs, mags)
    print(f"Generated HPCP profile with shape: {hpcp_values.shape}")
    
    # Test key detection
    key = es.Key()
    detected_key, scale = key(hpcp_values)
    print(f"Detected key: {detected_key} {scale}")
    
    # Test chord detection
    chord_detector = es.ChordsDetection()
    chord, scale = chord_detector(hpcp_values)
    print(f"Detected chord: {chord} {scale}")
    
    print('Essentia fallback implementation working!')
else:
    print('Essentia is available, no need to test fallback implementation.')
