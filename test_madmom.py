"""
Test script to verify madmom functionality in OliTunes
"""
import logging
import os
import sys
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import madmom
from backend.audio_imports import MADMOM_AVAILABLE

def test_madmom():
    """Test whether madmom is correctly installed and functioning"""
    logger.info(f"Madmom available: {MADMOM_AVAILABLE}")
    
    if MADMOM_AVAILABLE:
        try:
            # Import key madmom modules
            import madmom
            from madmom.audio.signal import Signal
            from madmom.features.beats import RNNBeatProcessor
            from madmom.features.downbeats import RNNDownBeatProcessor
            
            logger.info(f"Madmom version: {madmom.__version__}")
            
            # Create a simple test signal (1 second of silence)
            sr = 44100
            signal = np.zeros(sr, dtype=np.float32)
            
            # Create madmom Signal object
            madmom_signal = Signal(signal, sample_rate=sr)
            logger.info(f"Created madmom Signal with sample rate: {madmom_signal.sample_rate}")
            
            # Initialize beat tracking processor (this will fail if madmom isn't properly installed)
            rnn_processor = RNNBeatProcessor()
            logger.info("Successfully initialized RNNBeatProcessor")
            
            # Initialize downbeat tracking processor
            downbeat_processor = RNNDownBeatProcessor()
            logger.info("Successfully initialized RNNDownBeatProcessor")
            
            logger.info("Madmom test completed successfully!")
            return True
        except Exception as e:
            logger.error(f"Error testing madmom: {e}")
            return False
    else:
        logger.warning("Madmom is not available")
        return False

if __name__ == "__main__":
    test_madmom()
