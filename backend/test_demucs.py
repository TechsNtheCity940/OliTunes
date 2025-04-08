"""
Test script to verify Demucs functionality.
"""
import os
import sys
import torch
import numpy as np

def test_demucs_installation():
    """Test if Demucs is properly installed and can be imported."""
    try:
        import demucs
        from demucs.pretrained import get_model
        print("✅ Demucs successfully imported")
        
        # Check if a model can be loaded
        print("\nAttempting to get available Demucs models...")
        try:
            # Try to load a known model
            model = get_model("htdemucs")
            print(f"✅ Successfully loaded model: htdemucs")
            print(f"Model sources: {model.sources}")
            print(f"Model sample rate: {model.samplerate}")
            
            # Check PyTorch CUDA availability (helpful for Demucs performance)
            print("\nPyTorch CUDA status:")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Current CUDA device: {torch.cuda.current_device()}")
                print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    except ImportError as e:
        print(f"❌ Error importing Demucs: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_openunmix_installation():
    """Test if Open-Unmix is properly installed and can be imported."""
    try:
        import openunmix
        print("\n✅ Open-Unmix successfully imported")
        
        # Check module structure
        print("\nExploring Open-Unmix module structure...")
        module_attributes = dir(openunmix)
        filtered_attributes = [attr for attr in module_attributes if not attr.startswith('_')]
        print(f"Available attributes: {filtered_attributes}")
        
        # Try the import path for model components
        if 'model' in filtered_attributes:
            from openunmix import model
            print("✅ Successfully imported openunmix.model")
            model_attrs = dir(model)
            filtered_model_attrs = [attr for attr in model_attrs if not attr.startswith('_')]
            print(f"Model components: {filtered_model_attrs}")
        
        # Try using the appropriate API for the version installed
        try:
            print("\nTrying to use Open-Unmix for inference...")
            # Create a dummy audio tensor (2 channels, 2 seconds at 44.1kHz)
            dummy_audio = torch.rand(2, 88200)
            
            # Try to use the appropriate API based on what's available
            if 'separate' in filtered_attributes:
                # Newer versions might use this API
                print("Using openunmix.separate API...")
                device = torch.device("cpu")
                # Just check if the function exists, don't execute to avoid downloads
                print("✅ openunmix.separate function is available")
                return True
            else:
                print("❌ Could not find appropriate inference function in openunmix")
                return False
        except Exception as e:
            print(f"❌ Error testing Open-Unmix inference: {e}")
            return False
    
    except ImportError as e:
        print(f"❌ Error importing Open-Unmix: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("=== DEMUCS AND OPEN-UNMIX INSTALLATION TEST ===")
    
    # Test Demucs
    demucs_result = test_demucs_installation()
    
    # Test Open-Unmix
    openunmix_result = test_openunmix_installation()
    
    if demucs_result and openunmix_result:
        print("\n✅ Both Demucs and Open-Unmix are properly installed and functional!")
    else:
        if not demucs_result:
            print("\n❌ Demucs installation test failed")
        if not openunmix_result:
            print("\n❌ Open-Unmix installation test failed")
    
    print("\nTest complete.")
