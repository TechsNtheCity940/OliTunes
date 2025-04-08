"""
Test script to verify pretty_midi functionality.
"""
import os
import numpy as np
import pretty_midi

def test_pretty_midi():
    """Test pretty_midi's basic functionality by creating and reading a MIDI file."""
    print("Testing pretty_midi...")
    
    # Create a new PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    
    # Create a new piano instrument
    piano_program = 0  # Piano
    piano = pretty_midi.Instrument(program=piano_program)
    
    # Create a C major scale
    start_time = 0.0
    duration = 0.5  # Half second notes
    for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):  # C4 to C5
        # Create a Note instance
        note = pretty_midi.Note(
            velocity=100,  # Velocity (loudness)
            pitch=pitch,   # MIDI note number
            start=start_time + i * duration,  # Start time
            end=start_time + (i + 1) * duration  # End time
        )
        # Add note to the instrument
        piano.notes.append(note)
    
    # Add instrument to the PrettyMIDI object
    midi.instruments.append(piano)
    
    # Create test directory if it doesn't exist
    test_dir = os.path.join(os.path.dirname(__file__), 'test_output')
    os.makedirs(test_dir, exist_ok=True)
    
    # Write the MIDI file to disk
    test_midi_path = os.path.join(test_dir, 'test_scale.mid')
    midi.write(test_midi_path)
    
    # Reading the MIDI file back
    try:
        loaded_midi = pretty_midi.PrettyMIDI(test_midi_path)
        
        # Verify the tempo
        print(f"Tempo: {loaded_midi.estimate_tempo()} BPM")
        
        # Verify the instrument
        if loaded_midi.instruments:
            instr = loaded_midi.instruments[0]
            print(f"Instrument: {pretty_midi.program_to_instrument_name(instr.program)}")
            print(f"Number of notes: {len(instr.notes)}")
            
            # Check that all expected notes are present
            if len(instr.notes) == 8:
                print("✅ All notes present")
                
                # Check note timing
                start_times = [note.start for note in instr.notes]
                if all(abs(start_times[i+1] - start_times[i] - duration) < 0.001 for i in range(len(start_times)-1)):
                    print("✅ Note timing correct")
                else:
                    print("❌ Note timing incorrect")
                    
                # Check pitches
                pitches = [note.pitch for note in instr.notes]
                expected_pitches = [60, 62, 64, 65, 67, 69, 71, 72]
                if pitches == expected_pitches:
                    print("✅ Note pitches correct")
                else:
                    print("❌ Note pitches incorrect")
                    print(f"Expected: {expected_pitches}")
                    print(f"Actual: {pitches}")
                
                # Overall test passed if all sub-tests passed
                if (len(instr.notes) == 8 and 
                    all(abs(start_times[i+1] - start_times[i] - duration) < 0.001 for i in range(len(start_times)-1)) and
                    pitches == expected_pitches):
                    print("\n✅ pretty_midi test PASSED!")
                    return True
                else:
                    print("\n❌ pretty_midi test FAILED!")
                    return False
            else:
                print(f"❌ Expected 8 notes, found {len(instr.notes)}")
                print("\n❌ pretty_midi test FAILED!")
                return False
        else:
            print("❌ No instruments found in loaded MIDI")
            print("\n❌ pretty_midi test FAILED!")
            return False
    except Exception as e:
        print(f"❌ Error reading MIDI file: {e}")
        print("\n❌ pretty_midi test FAILED!")
        return False

def test_pretty_midi_analysis():
    """Test pretty_midi's analysis capabilities using proper API methods."""
    print("\nTesting pretty_midi analysis capabilities...")
    
    # Create two separate MIDI objects for each chord
    midi_c = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano_c = pretty_midi.Instrument(program=0)
    
    # C major chord (C, E, G)
    for pitch in [60, 64, 67]:  # C4, E4, G4
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=0.0,
            end=1.0
        )
        piano_c.notes.append(note)
    
    midi_c.instruments.append(piano_c)
    
    # Create a G7 chord in separate MIDI object
    midi_g7 = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano_g7 = pretty_midi.Instrument(program=0)
    
    # G7 chord (G, B, D, F)
    for pitch in [67, 71, 74, 77]:  # G4, B4, D5, F5
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=0.0,
            end=1.0
        )
        piano_g7.notes.append(note)
    
    midi_g7.instruments.append(piano_g7)
    
    # Test chord recognition using get_chroma (entire MIDI file)
    chroma_c_major = midi_c.get_chroma()
    chroma_g7 = midi_g7.get_chroma()
    
    print("C major chord chroma:")
    print(chroma_c_major)
    
    print("G7 chord chroma:")
    print(chroma_g7)
    
    # Verify C major chord (C, E, G) has high values at indices 0, 4, 7
    c_major_indices = [0, 4, 7]  # C, E, G in chroma
    other_indices = [i for i in range(12) if i not in c_major_indices]
    
    c_major_avg = np.mean(chroma_c_major[c_major_indices])
    other_avg = np.mean(chroma_c_major[other_indices])
    
    print(f"C major chord notes average: {c_major_avg:.4f}")
    print(f"Other notes average: {other_avg:.4f}")
    
    # Verify G7 chord (G, B, D, F) has high values at indices 7, 11, 2, 5
    g7_indices = [7, 11, 2, 5]  # G, B, D, F in chroma
    other_indices = [i for i in range(12) if i not in g7_indices]
    
    g7_avg = np.mean(chroma_g7[g7_indices])
    other_avg = np.mean(chroma_g7[other_indices])
    
    print(f"G7 chord notes average: {g7_avg:.4f}")
    print(f"Other notes average: {other_avg:.4f}")
    
    # Test passes if chord notes have significantly higher average than other notes
    if c_major_avg > 2 * other_avg and g7_avg > 2 * other_avg:
        print("✅ Chord analysis test PASSED!")
        return True
    else:
        print("❌ Chord analysis test FAILED!")
        return False

if __name__ == "__main__":
    print("=== PRETTY_MIDI FUNCTIONALITY TEST ===")
    basic_test = test_pretty_midi()
    analysis_test = test_pretty_midi_analysis()
    
    if basic_test and analysis_test:
        print("\n✅ All pretty_midi tests PASSED!")
    else:
        print("\n❌ Some pretty_midi tests FAILED!")
