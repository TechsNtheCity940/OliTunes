"""
Script to test if all necessary music libraries are installed correctly.
"""
import sys
import importlib

def check_library(library_name, version_attr="_version_"):
    """Check if a library is installed and optionally report its version."""
    try:
        lib = importlib.import_module(library_name)
        if hasattr(lib, version_attr):
            version = getattr(lib, version_attr)
            return True, version
        else:
            return True, "Unknown version"
    except ImportError:
        return False, None

def main():
    """Main function to test all libraries."""
    print("=== MUSIC LIBRARY AVAILABILITY TEST ===\n")
    
    # List of libraries to check
    libraries = [
        ("aubio", "__version__"),
        ("pretty_midi", "__version__"),
        ("music21", "__version__"),
        ("pedalboard", ""),  # No version attribute
        ("demucs", ""),  # No standard version attribute
        ("openunmix", ""),  # No standard version attribute
        ("pyaudio", "__version__"),  # For future testing
    ]
    
    all_installed = True
    
    # Check each library
    for library_name, version_attr in libraries:
        installed, version = check_library(library_name, version_attr)
        if installed:
            if version:
                print(f" {library_name} (version {version}) is installed")
            else:
                print(f" {library_name} is installed")
        else:
            print(f" {library_name} is NOT installed")
            all_installed = False
    
    # Overall result
    print("\n=== TEST RESULTS ===")
    if all_installed:
        print(" All libraries are successfully installed!")
    else:
        print(" Some libraries are missing. Please install them to ensure full functionality.")
    
    # Recommended test scripts
    print("\n=== DETAILED TESTING ===")
    print("For detailed functionality testing, run these scripts:")
    print("1. python backend/test_aubio.py - Test pitch and onset detection")
    print("2. python backend/test_pretty_midi.py - Test MIDI file handling")
    print("3. python backend/test_music21.py - Test music theory analysis")
    print("4. python backend/test_music21_scales.py - Test scale generation")
    print("5. python backend/test_pedalboard.py - Test audio effects")
    print("6. python backend/test_demucs.py - Test source separation")
    
if __name__ == "__main__":
    main()
