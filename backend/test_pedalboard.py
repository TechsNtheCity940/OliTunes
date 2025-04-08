"""
Test script to verify Pedalboard functionality.
"""
import os
import numpy as np
try:
    from pedalboard import Pedalboard, Chorus, Reverb, Delay, Gain
    pedalboard_available = True
except ImportError:
    pedalboard_available = False
    print("❌ Pedalboard not available")

def test_pedalboard():
    """Test basic Pedalboard effects."""
    if not pedalboard_available:
        print("Skipping Pedalboard tests as the library is not available")
        return False
    
    print("Testing Pedalboard effects chain...")
    
    # Create a simple sine wave as audio input
    sample_rate = 44100
    duration = 2.0  # seconds
    frequency = 440.0  # Hz (A4 note)
    
    # Generate a sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Create a pedalboard with various effects
    board = Pedalboard([
        Chorus(rate_hz=0.5, depth=0.7, mix=0.5),
        Reverb(room_size=0.8, damping=0.5, wet_level=0.3, dry_level=0.8),
        Delay(delay_seconds=0.25, feedback=0.4, mix=0.3),
        Gain(gain_db=3.0)
    ])
    
    # Process the audio through the pedalboard
    try:
        effected_audio = board(audio, sample_rate)
        
        # Check if the output is different from the input (effects were applied)
        if not np.array_equal(audio, effected_audio):
            print("✅ Effects successfully applied to audio")
            
            # Check if the output has the expected length
            if len(effected_audio) >= len(audio):
                print("✅ Output length is as expected")
            else:
                print(f"❌ Output length ({len(effected_audio)}) is shorter than input ({len(audio)})")
                
            # Check if gain was applied (output should have higher amplitude)
            if np.max(np.abs(effected_audio)) > np.max(np.abs(audio)):
                print("✅ Gain effect was successfully applied")
            else:
                print("❌ Gain effect was not applied correctly")
            
            # Calculate and display some basic stats
            original_rms = np.sqrt(np.mean(audio**2))
            effected_rms = np.sqrt(np.mean(effected_audio**2))
            
            print(f"Original RMS amplitude: {original_rms:.4f}")
            print(f"Effected RMS amplitude: {effected_rms:.4f}")
            print(f"RMS ratio: {effected_rms/original_rms:.4f}")
            
            # Create test directory if it doesn't exist
            test_dir = os.path.join(os.path.dirname(__file__), 'test_output')
            os.makedirs(test_dir, exist_ok=True)
            
            # Test individual effects
            print("\nTesting individual effects:")
            
            # Test reverb
            reverb = Pedalboard([Reverb(room_size=0.9, wet_level=0.8, dry_level=0.2)])
            reverb_audio = reverb(audio, sample_rate)
            if not np.array_equal(audio, reverb_audio):
                print("✅ Reverb effect working")
            else:
                print("❌ Reverb effect not working")
                
            # Test chorus
            chorus = Pedalboard([Chorus(rate_hz=0.9, depth=0.9, mix=0.8)])
            chorus_audio = chorus(audio, sample_rate)
            if not np.array_equal(audio, chorus_audio):
                print("✅ Chorus effect working")
            else:
                print("❌ Chorus effect not working")
                
            # Test delay
            delay = Pedalboard([Delay(delay_seconds=0.5, feedback=0.7, mix=0.6)])
            delay_audio = delay(audio, sample_rate)
            if not np.array_equal(audio, delay_audio):
                print("✅ Delay effect working")
            else:
                print("❌ Delay effect not working")
            
            # Overall success
            print("\n✅ Pedalboard test PASSED!")
            return True
            
        else:
            print("❌ Effects were not applied to audio")
            print("\n❌ Pedalboard test FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Error when processing audio: {e}")
        print("\n❌ Pedalboard test FAILED!")
        return False

if __name__ == "__main__":
    print("=== PEDALBOARD FUNCTIONALITY TEST ===")
    
    if pedalboard_available:
        test_result = test_pedalboard()
    else:
        print("Skipping tests as Pedalboard is not available.")
        print("Try reinstalling with: pip install pedalboard")
        test_result = False
    
    print("\nTest complete.")
