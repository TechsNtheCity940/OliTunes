"""
Test script to verify music21 scale generation functionality.
"""
from music21 import scale

def test_scale_generation():
    """Test music21's scale generation capabilities."""
    print("Testing scale generation...")
    
    # Test various scale types
    scale_tests = [
        ("C", scale.MajorScale, ["C", "D", "E", "F", "G", "A", "B"]),
        ("A", scale.MinorScale, ["A", "B", "C", "D", "E", "F", "G"]),
        ("G", scale.MixolydianScale, ["G", "A", "B", "C", "D", "E", "F"]),
        ("D", scale.DorianScale, ["D", "E", "F", "G", "A", "B", "C"]),
        ("F", scale.LydianScale, ["F", "G", "A", "B", "C", "D", "E"])
    ]
    
    success = True
    for tonic, scale_type, expected_pitches in scale_tests:
        # Create scale
        test_scale = scale_type(tonic)
        
        # Get pitches - exclude the last one (octave)
        pitches = [p.name for p in test_scale.getPitches()[:-1]]
        
        # Print result
        print(f"{tonic} {scale_type.__name__}: {pitches}")
        print(f"Expected: {expected_pitches}")
        
        # Check if all expected notes are in the scale (regardless of order)
        all_notes_present = all(note in pitches for note in expected_pitches)
        if all_notes_present:
            print("✅ All expected notes present")
        else:
            print("❌ Some expected notes missing")
            success = False
        
        # Check first pitch
        if pitches[0] == expected_pitches[0]:
            print("✅ Correct starting pitch")
        else:
            print(f"❌ Incorrect starting pitch: {pitches[0]}, expected {expected_pitches[0]}")
            success = False
        
        print("-" * 30)
    
    if success:
        print("✅ Scale generation test PASSED!")
    else:
        print("❌ Scale generation test FAILED!")
    
    return success

if __name__ == "__main__":
    print("=== MUSIC21 SCALE GENERATION TEST ===")
    scale_test = test_scale_generation()
    print("\nTest complete.")
