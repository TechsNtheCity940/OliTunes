# Auto-generated configuration for pydub to find ffmpeg
import os
import sys
from pathlib import Path

# Function to setup ffmpeg
def setup_ffmpeg():
    # Get the full path to ffmpeg executables
    tools_dir = Path(__file__).parent.parent / "tools"
    ffmpeg_dirs = [
        # Look in specific directories based on different extraction methods
        tools_dir / "ffmpeg" / "bin",  # Prioritize this one based on our directory structure
        tools_dir / "ffmpeg-essentials" / "bin",
        tools_dir / "ffmpeg-master-latest-win64-gpl" / "bin",
        tools_dir / "ffmpeg-7.1.1-essentials_build" / "bin",
        # Direct extraction might put executables directly in these folders
        tools_dir / "bin",
        # Try the extracted zip's top-level directory
        tools_dir
    ]

    # Find the directory that contains ffmpeg.exe
    ffmpeg_path = None
    for directory in ffmpeg_dirs:
        potential_path = directory / "ffmpeg.exe" if isinstance(directory, Path) else Path(directory) / "ffmpeg.exe"
        if potential_path.exists():
            ffmpeg_path = directory
            break

    # If still not found, search recursively through the tools directory
    if not ffmpeg_path:
        print("Searching for ffmpeg.exe recursively...")
        for root, dirs, files in os.walk(str(tools_dir)):
            if "ffmpeg.exe" in files:
                ffmpeg_path = Path(root)
                break

    if ffmpeg_path:
        print(f"Found ffmpeg at: {ffmpeg_path}")
        # Add to PATH environment variable
        os.environ['PATH'] = str(ffmpeg_path) + os.pathsep + os.environ.get('PATH', '')
        
        # Also set ffmpeg path directly for pydub
        os.environ['FFMPEG_BINARY'] = str(ffmpeg_path / "ffmpeg.exe")
        os.environ['FFPROBE_BINARY'] = str(ffmpeg_path / "ffprobe.exe")
        
        # Direct modification of pydub internals
        try:
            import pydub
            # The simplest and most direct approach - override pydub's _cmd_runner list
            # with our known correct paths
            ffmpeg_exe = str(ffmpeg_path / "ffmpeg.exe")
            ffprobe_exe = str(ffmpeg_path / "ffprobe.exe")
            
            # Directly modify pydub's internal command paths
            pydub.utils.FFMPEG_PATH = ffmpeg_exe
            pydub.utils.FFPROBE_PATH = ffprobe_exe
            
            # This will ensure pydub uses our specified paths
            print("Successfully set pydub's FFmpeg paths directly")
        except Exception as e:
            print(f"Note: Direct pydub modification failed: {e} - but PATH is still set correctly")
        
        return True
    else:
        print("Warning: ffmpeg not found. Audio processing with pydub may not work correctly.")
        return False

# Direct known path setup for when detection fails
def force_ffmpeg_path():
    """Force the ffmpeg path to the known location"""
    tools_dir = Path(__file__).parent.parent / "tools"
    ffmpeg_bin = tools_dir / "ffmpeg" / "bin"
    
    if ffmpeg_bin.exists() and (ffmpeg_bin / "ffmpeg.exe").exists():
        print(f"Forcing ffmpeg path to known location: {ffmpeg_bin}")
        os.environ['PATH'] = str(ffmpeg_bin) + os.pathsep + os.environ.get('PATH', '')
        os.environ['FFMPEG_BINARY'] = str(ffmpeg_bin / "ffmpeg.exe")
        os.environ['FFPROBE_BINARY'] = str(ffmpeg_bin / "ffprobe.exe")
        
        # Direct modification of pydub internals
        try:
            import pydub
            # Set the paths directly
            ffmpeg_exe = str(ffmpeg_bin / "ffmpeg.exe")
            ffprobe_exe = str(ffmpeg_bin / "ffprobe.exe")
            
            pydub.utils.FFMPEG_PATH = ffmpeg_exe
            pydub.utils.FFPROBE_PATH = ffprobe_exe
            
            print("Successfully set pydub's FFmpeg paths directly")
        except Exception as e:
            print(f"Note: Direct pydub modification failed: {e} - but PATH is still set correctly")
        
        return True
    return False

# Try the automatic setup first, then fall back to forced path if needed
if not setup_ffmpeg():
    force_ffmpeg_path()