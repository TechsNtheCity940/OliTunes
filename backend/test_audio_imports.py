"""
Test script to verify that our audio analysis system works with the fallback implementation
"""
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from backend.audio_imports import ESSENTIA_AVAILABLE, es

def test_essentia_fallback():
    """Test whether our Essentia fallback implementation works"""
    logger.info(f"Essentia available: {ESSENTIA_AVAILABLE}")
    
    if not ESSENTIA_AVAILABLE:
        logger.info("Testing Essentia fallback implementation...")
        
        # Test basic functionality
        import numpy as np
        
        # Create a simple audio signal (a sine wave)
        sr = 44100
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(sr * duration))
        freq = 440.0  # A4 note
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Test the frame generator
        frame_gen = es.FrameGenerator(audio, frameSize=2048, hopSize=1024)
        frames = list(frame_gen)
        logger.info(f"Generated {len(frames)} frames from audio signal")
        
        # Test windowing
        windowing = es.Windowing(type='hann')
        windowed_frame = windowing(frames[0])
        logger.info(f"Applied windowing to frame, shape: {windowed_frame.shape}")
        
        # Test spectrum
        spectrum = es.Spectrum()
        spec = spectrum(windowed_frame)
        logger.info(f"Computed spectrum, shape: {spec.shape}")
        
        # Test spectral peaks
        spectral_peaks = es.SpectralPeaks()
        freqs, mags = spectral_peaks(spec)
        logger.info(f"Found {len(freqs)} spectral peaks")
        
        # Test HPCP
        hpcp = es.HPCP()
        hpcp_values = hpcp(freqs, mags)
        logger.info(f"Generated HPCP profile with shape: {hpcp_values.shape}")
        
        # Test key detection
        key = es.Key()
        detected_key, scale = key(hpcp_values)
        logger.info(f"Detected key: {detected_key} {scale}")
        
        # Test chord detection
        chord_detector = es.ChordsDetection()
        chord, scale = chord_detector(hpcp_values)
        logger.info(f"Detected chord: {chord} {scale}")
        
        logger.info("Essentia fallback implementation test completed successfully!")
        return True
    else:
        logger.info("Essentia is available, no need to test fallback implementation")
        return True

if __name__ == "__main__":
    test_essentia_fallback()
