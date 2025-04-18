"""
Test script for the fretboard position model.
"""
import os
import sys
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the FretboardPositionPredictor directly
from backend.models.fretboard_position_model.position_predictor import FretboardPositionPredictor

def test_fretboard_model():
    """Test the fretboard position model directly."""
    logger.info("Testing fretboard position model")
    
    # Create FretboardPositionPredictor instance
    predictor = FretboardPositionPredictor()
    
    # Check if model was loaded
    if predictor.model is not None:
        logger.info("Fretboard position model loaded successfully")
    else:
        logger.warning("Fretboard position model not available")
        return False
    
    # Test with some MIDI notes
    test_notes = [
        # E string (40) + fret 5 = 45 (A note)
        45,
        # A string (45) + fret 2 = 47 (B note)
        47,
        # D string (50) + fret 0 = 50 (D note)
        50,
        # G string (55) + fret 2 = 57 (A note)
        57,
        # B string (59) + fret 3 = 62 (D note)
        62,
        # E string (64) + fret 0 = 64 (E note)
        64
    ]
    
    # Define context
    prev_fret = 0
    prev_string = 0
    key = 0  # C key
    style = 0  # Default style
    
    # Run predictions
    logger.info("Running position predictions")
    results = []
    
    for midi_note in test_notes:
        string, fret = predictor.predict_position(
            midi_note, 
            previous_position=prev_fret,
            string=prev_string,
            key_context=key,
            style=style
        )
        
        results.append((midi_note, string, fret))
        logger.info(f"MIDI Note {midi_note}: Predicted String {string}, Fret {fret}")
        
        # Update context for next note
        prev_string, prev_fret = string, fret
    
    return True

if __name__ == "__main__":
    success = test_fretboard_model()
    if success:
        logger.info("Fretboard position model test completed successfully")
    else:
        logger.warning("Fretboard position model test failed")
