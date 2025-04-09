"""
Test script for the integrated fretboard position model with the UnifiedTabProcessor.
This script demonstrates how the fretboard position model is used in the full tablature generation pipeline.
"""
import os
import sys
import logging
import numpy as np
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add necessary paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Simplified MusicTheoryAnalyzer class for testing
class SimpleMusicTheoryAnalyzer:
    """
    Simplified version of MusicTheoryAnalyzer for testing without audio_imports.
    """
    def __init__(self):
        self.has_music21 = False
        
    def key_to_index(self, key: str) -> int:
        """
        Simplified key to index conversion for testing.
        """
        if not key:
            return 0
            
        # Extract tonic from key string
        tonic = key.split()[0] if ' ' in key else key
        
        # Map tonic to index
        tonic_map = {
            'C': 0, 'C#': 1, 'Db': 1, 
            'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 
            'F': 5, 'F#': 6, 'Gb': 6,
            'G': 7, 'G#': 8, 'Ab': 8,
            'A': 9, 'A#': 10, 'Bb': 10,
            'B': 11
        }
        
        return tonic_map.get(tonic, 0)

def test_fretboard_model_with_mock_data():
    """
    Test the fretboard position model integration with mock tablature data.
    This simulates the tablature generation pipeline without requiring the full audio processing stack.
    """
    try:
        # Import only what we need for testing
        from backend.models.fretboard_position_model.position_predictor import FretboardPositionPredictor
        
        logger.info("Testing fretboard position model integration with mock data")
        
        # Create instances
        fretboard_model = FretboardPositionPredictor()
        theory_analyzer = SimpleMusicTheoryAnalyzer()
        
        # Check if fretboard model was loaded
        if fretboard_model.model is None:
            logger.warning("Fretboard position model not available")
            return False
        
        logger.info("Fretboard position model loaded successfully")
        
        # Create mock tablature data (time, string, fret)
        # Shape: (time_steps, 6 strings, 25 frets)
        mock_tab = np.zeros((10, 6, 25))
        
        # Add some test notes (one per time step)
        # Time 0: E string (0), fret 5 (A note)
        mock_tab[0, 0, 5] = 1.0
        
        # Time 1: A string (1), fret 2 (B note)
        mock_tab[1, 1, 2] = 1.0
        
        # Time 2: D string (2), fret 0 (D note)
        mock_tab[2, 2, 0] = 1.0
        
        # Time 3: G string (3), fret 2 (A note)
        mock_tab[3, 3, 2] = 1.0
        
        # Time 4: B string (4), fret 3 (D note)
        mock_tab[4, 4, 3] = 1.0
        
        # Time 5: E string (5), fret 0 (E note)
        mock_tab[5, 5, 0] = 1.0
        
        # Define string tunings (E2, A2, D3, G3, B3, E4)
        string_midi_values = [40, 45, 50, 55, 59, 64]
        
        # Extract notes from mock tablature
        notes = []
        for t in range(mock_tab.shape[0]):
            for s in range(6):
                for f in range(25):
                    if mock_tab[t, s, f] > 0:
                        midi_note = string_midi_values[s] + f
                        notes.append({
                            'time': t * 0.0232,  # 512/22050 ≈ 0.0232s
                            'duration': 0.1,
                            'string': s,
                            'fret': f,
                            'midi_note': midi_note
                        })
                        logger.info(f"Original: Time {t}: String {s}, Fret {f}, MIDI {midi_note}")
        
        # Optimize fretboard positions using the model
        optimized = np.zeros_like(mock_tab)
        prev_string, prev_fret = 0, 0
        key_context = theory_analyzer.key_to_index('C major')
        style_context = 0  # Default style
        
        for note in notes:
            midi_note = note['midi_note']
            t = int(note['time'] / 0.0232)
            
            # Use model to predict optimal string/fret
            string, fret = fretboard_model.predict_position(
                midi_note, 
                previous_position=prev_fret,
                string=prev_string,
                key_context=key_context,
                style=style_context
            )
            
            # Update the optimized tab
            optimized[t, string, fret] = 1.0
            
            logger.info(f"Optimized: Time {t}: String {string}, Fret {fret}, MIDI {midi_note}")
            
            # Update context for next note
            prev_string, prev_fret = string, fret
        
        # Compare original and optimized positions
        changes = 0
        for t in range(mock_tab.shape[0]):
            for s in range(6):
                for f in range(25):
                    if mock_tab[t, s, f] > 0 and optimized[t, s, f] == 0:
                        changes += 1
        
        logger.info(f"Position changes: {changes} out of {len(notes)} notes")
        
        # Save results to a JSON file for visualization
        results = {
            'original': [],
            'optimized': []
        }
        
        for t in range(mock_tab.shape[0]):
            for s in range(6):
                for f in range(25):
                    if mock_tab[t, s, f] > 0:
                        results['original'].append({
                            'time': t,
                            'string': s,
                            'fret': f,
                            'midi': string_midi_values[s] + f
                        })
                    if optimized[t, s, f] > 0:
                        results['optimized'].append({
                            'time': t,
                            'string': s,
                            'fret': f,
                            'midi': string_midi_values[s] + f
                        })
        
        # Save results
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_results')
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'fretboard_optimization_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {os.path.join(output_dir, 'fretboard_optimization_results.json')}")
        
        return True
    
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fretboard_model_with_mock_data()
    if success:
        logger.info("Fretboard position model integration test completed successfully")
    else:
        logger.warning("Fretboard position model integration test failed")
