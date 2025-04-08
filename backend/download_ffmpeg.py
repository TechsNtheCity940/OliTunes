"""
Download and extract ffmpeg for Windows to make it available for pydub.
"""
import os
import requests
import zipfile
import shutil
import sys
from pathlib import Path

def download_file(url, target_file):
    """Download a file from a URL to a target path."""
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(target_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded to {target_file}")
    return target_file

def extract_zip(zip_path, extract_to):
    """Extract a zip file to a target directory."""
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def setup_ffmpeg():
    """Download and set up ffmpeg for Windows."""
    # Define paths
    tools_dir = Path("../tools")
    ffmpeg_dir = tools_dir / "ffmpeg"
    ffmpeg_zip = tools_dir / "ffmpeg.zip"
    
    # Create directories if they don't exist
    os.makedirs(tools_dir, exist_ok=True)
    
    # Download ffmpeg (latest build for Windows)
    ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    
    try:
        # Download if the file doesn't exist
        if not ffmpeg_zip.exists():
            download_file(ffmpeg_url, ffmpeg_zip)
        
        # Clear existing ffmpeg directory if it exists
        if ffmpeg_dir.exists():
            shutil.rmtree(ffmpeg_dir)
        
        # Extract the zip file
        extract_zip(ffmpeg_zip, tools_dir)
        
        # Rename the extracted directory to just 'ffmpeg'
        extracted_dirs = [d for d in tools_dir.iterdir() if d.is_dir() and d.name.startswith("ffmpeg-")]
        if extracted_dirs:
            shutil.move(str(extracted_dirs[0]), str(ffmpeg_dir))
            
        # Create a .pydub_ffmpeg file in the user's home directory
        ffmpeg_bin_dir = ffmpeg_dir / "bin"
        ffmpeg_exe = ffmpeg_bin_dir / "ffmpeg.exe"
        ffprobe_exe = ffmpeg_bin_dir / "ffprobe.exe"
        
        if ffmpeg_exe.exists() and ffprobe_exe.exists():
            # Configuration for current project
            pydub_config = Path("pydub_config.py")
            with open(pydub_config, 'w') as f:
                f.write(f"""
# Auto-generated configuration for pydub to find ffmpeg
import os

# Set environment variables for ffmpeg executables
os.environ['PATH'] += os.pathsep + r'{ffmpeg_bin_dir}'
                """)
                
            print(f"\nSuccess! ffmpeg has been set up at {ffmpeg_dir}")
            print("Add this to your app.py or any module that uses pydub:")
            print("import pydub_config  # This will set up ffmpeg paths\n")
            return True
        else:
            print(f"Could not find ffmpeg.exe and ffprobe.exe in {ffmpeg_bin_dir}")
            return False
    
    except Exception as e:
        print(f"Error setting up ffmpeg: {e}")
        return False

if __name__ == "__main__":
    if setup_ffmpeg():
        print("ffmpeg setup complete!")
    else:
        print("ffmpeg setup failed.")
        sys.exit(1)
