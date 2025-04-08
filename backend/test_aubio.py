"""
Test script to verify aubio functionality.
"""
import os
import numpy as np
import aubio

def test_aubio_pitch_detection():
    """Test aubio's pitch detection capabilities."""
    print("Testing aubio pitch detection...")
    
    # Create a simple test signal (440Hz sine wave - A4 note)
    samplerate = 44100
    seconds = 1
    t = np.linspace(0, seconds, samplerate * seconds)
    test_signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # Explicitly cast to float32
    
    # Initialize pitch detection algorithm
    win_s = 2048
    hop_s = 512
    
    pitch_detector = aubio.pitch("yin", win_s, hop_s, samplerate)
    pitch_detector.set_unit("Hz")
    pitch_detector.set_silence(-40)
    
    # Process the signal in chunks to detect pitch
    detected_pitches = []
    
    for i in range(0, len(test_signal) - win_s, hop_s):
        chunk = test_signal[i:i+win_s].astype(np.float32)
        
        # Aubio expects hop_s size input, so we need to process win_s sized windows
        # and feed hop_s sized chunks to the pitch detector
        pitch = pitch_detector(chunk[:hop_s])[0]
        confidence = pitch_detector.get_confidence()
        
        if confidence > 0.8:  # Only include high-confidence readings
            detected_pitches.append(pitch)
    
    # Calculate average detected pitch
    if detected_pitches:
        avg_pitch = sum(detected_pitches) / len(detected_pitches)
        print(f"Average detected pitch: {avg_pitch:.2f} Hz")
        print(f"Expected pitch: 440.00 Hz")
        print(f"Difference: {abs(avg_pitch - 440.0):.2f} Hz")
        
        # Test passes if within 5Hz of expected value
        if abs(avg_pitch - 440.0) < 5:
            print("✅ Aubio pitch detection test PASSED")
            return True
        else:
            print("❌ Aubio pitch detection test FAILED")
            return False
    else:
        print("❌ No pitches detected with high confidence")
        return False

def test_aubio_onset_detection():
    """Test aubio's onset detection capabilities."""
    print("\nTesting aubio onset detection...")
    
    # Create a test signal with clear onsets - use a more realistic approach
    samplerate = 44100
    seconds = 2
    t = np.linspace(0, seconds, samplerate * seconds)
    
    # Create a signal with 4 clear drum-like onsets
    test_signal = np.zeros(int(seconds * samplerate), dtype=np.float32)
    
    # Function to create percussive sound
    def create_percussive_hit(start_sample, duration_samples, amplitude=0.9):
        t_hit = np.arange(duration_samples)
        # Exponential decay to simulate drum hit
        envelope = amplitude * np.exp(-5.0 * t_hit / duration_samples)
        # Add some noise for more percussive character
        hit = envelope * (np.random.random(duration_samples) * 2 - 1)
        test_signal[start_sample:start_sample + duration_samples] = hit
    
    # Create 4 clear onsets at 0.0s, 0.5s, 1.0s, and 1.5s
    for i, time in enumerate([0.0, 0.5, 1.0, 1.5]):
        sample_index = int(time * samplerate)
        create_percussive_hit(sample_index, 4000, 0.9)
    
    # Initialize onset detection with parameters better suited for percussive onsets
    win_s = 1024
    hop_s = 512
    
    onset_detector = aubio.onset("hfc", win_s, hop_s, samplerate)
    onset_detector.set_threshold(0.3)  # Adjust threshold for better detection
    
    # Process the signal to detect onsets
    detected_onsets = []
    total_frames = 0
    
    for i in range(0, len(test_signal) - win_s, hop_s):
        chunk = test_signal[i:i+win_s].astype(np.float32)
        
        # Aubio expects hop_s size input
        if onset_detector(chunk[:hop_s]):
            onset_time = total_frames / float(samplerate)
            detected_onsets.append(onset_time)
        
        total_frames += hop_s
    
    print(f"Expected number of onsets: 4")
    print(f"Detected onsets: {len(detected_onsets)}")
    print(f"Onset times (seconds): {[f'{t:.2f}' for t in detected_onsets]}")
    
    # Test passes if it detects 3-5 onsets (allowing for some margin of error)
    if 3 <= len(detected_onsets) <= 5:
        print("✅ Aubio onset detection test PASSED")
        return True
    else:
        print("❌ Aubio onset detection test FAILED - Incorrect number of onsets detected")
        return False

if __name__ == "__main__":
    print("=== AUBIO FUNCTIONALITY TEST ===")
    pitch_result = test_aubio_pitch_detection()
    onset_result = test_aubio_onset_detection()
    
    if pitch_result and onset_result:
        print("\n✅ All aubio tests PASSED!")
    else:
        print("\n❌ Some aubio tests FAILED!")
