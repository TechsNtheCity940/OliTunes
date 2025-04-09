"""
Direct test of madmom functionality without going through audio_imports.py
"""
import sys
import os
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import madmom
    logger.info(f"Successfully imported madmom version: {madmom.__version__}")
    
    # Try importing some key madmom components
    from madmom.audio.signal import Signal
    logger.info("Successfully imported madmom.audio.signal.Signal")
    
    from madmom.features.beats import RNNBeatProcessor
    logger.info("Successfully imported madmom.features.beats.RNNBeatProcessor")
    
    from madmom.features.downbeats import RNNDownBeatProcessor
    logger.info("Successfully imported madmom.features.downbeats.RNNDownBeatProcessor")
    
    from madmom.audio.filters import LogarithmicFilterbank
    logger.info("Successfully imported madmom.audio.filters.LogarithmicFilterbank")
    
    # Print Python path to see where it's looking for packages
    logger.info(f"Python path: {sys.path}")
    
    # Try to initialize a basic madmom processor
    processor = RNNBeatProcessor()
    logger.info("Successfully initialized RNNBeatProcessor")
    
    logger.info("All madmom imports and initializations successful!")
except Exception as e:
    logger.error(f"Error with madmom: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    logger.error(f"Python path: {sys.path}")
