"""
Simplified test script to verify music21 functionality.
"""
import os
from music21 import chord, key, stream, note, scale

def test_chord_identification():
    """Test music21's chord identification capabilities."""
    print("Testing chord identification...")
    
    # Test common chords
    chord_tests = [
        ([60, 64, 67], "major triad"),
        ([60, 63, 67], "minor triad"),
        ([60, 64, 67, 71], "major seventh"),
        ([60, 63, 67, 70], "minor seventh"),
        ([60, 64, 68], "augmented triad"),
        ([60, 63, 66], "diminished triad"),
        ([60, 64, 67, 69], "major sixth")
    ]
    
    success = True
    for pitches, expected_name in chord_tests:
        # Create chord from pitches
        c = chord.Chord(pitches)
        
        # Get chord name
        chord_name = c.commonName
        
        # Print result
        print(f"Pitches {pitches}: Identified as '{chord_name}'")
        print(f"Expected: '{expected_name}'")
        
        # Basic check (might need adjustment based on exact string formatting)
        if expected_name.lower() in chord_name.lower():
            print("✅ Correct")
        else:
            print("❌ Incorrect")
            success = False
        print("-" * 30)
    
    if success:
        print("✅ Chord identification test PASSED!")
    else:
        print("❌ Chord identification test FAILED!")
    
    return success

if __name__ == "__main__":
    print("=== MUSIC21 SIMPLIFIED TEST ===")
    chord_test = test_chord_identification()
    print("\nTest complete.")
