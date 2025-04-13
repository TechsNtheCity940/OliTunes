"""
Parse Ultimate Guitar tabs and convert them to training data format for TabCNN model.
"""

import os
import re
import numpy as np
import glob
from pathlib import Path
import random

# Get the backend directory path
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(script_dir))

def extract_tab_sections(tab_content):
    """
    Extract tab sections from the tab content.
    
    Args:
        tab_content: String containing the tab content
        
    Returns:
        List of tab sections
    """
    # Find tab sections using regex - look for [tab] tags or similar
    tab_patterns = [
        r'\[tab\](.*?)\[/tab\]',
        r'\[/tab\](.*?)\[/tab\]',
        r'\[Tab\](.*?)\[/Tab\]',
        r'\[TAB\](.*?)\[/TAB\]'
    ]
    
    tab_sections = []
    for pattern in tab_patterns:
        sections = re.findall(pattern, tab_content, re.DOTALL)
        if sections:
            tab_sections.extend(sections)
    
    if not tab_sections:
        # If no [tab] tags, try to find sections with standard tab notation
        lines = tab_content.split('\n')
        current_section = []
        in_tab_section = False
        
        for line in lines:
            # Check if line looks like a tab line (contains |----|)
            if '|' in line and '-' in line and any(s in line.strip()[:2] for s in ['E|', 'B|', 'G|', 'D|', 'A|', 'e|', 'b|', 'g|', 'd|', 'a|']):
                current_section.append(line)
                in_tab_section = True
            elif in_tab_section and not line.strip():
                # Empty line within a tab section - keep it
                current_section.append(line)
            elif in_tab_section:
                # If we have a current section and encounter a non-tab line, save the section
                if len(current_section) >= 4:  # Ensure we have at least 4 strings
                    tab_sections.append('\n'.join(current_section))
                current_section = []
                in_tab_section = False
        
        # Add the last section if it exists
        if current_section and len(current_section) >= 4:
            tab_sections.append('\n'.join(current_section))
    
    return tab_sections

def parse_tab_line(line):
    """
    Parse a single line of tab notation.
    
    Args:
        line: String containing a line of tab notation
        
    Returns:
        Tuple of (string_number, fret_positions)
    """
    # Extract the string name and fret positions
    parts = line.split('|')
    if len(parts) < 2:
        return None, []
    
    # Get the string name (first character before the first |)
    string_name = parts[0].strip()
    
    # Handle common variations in string naming
    if string_name.startswith('e') or string_name.startswith('E'):
        string_name = 'E'
    elif string_name.startswith('B') or string_name.startswith('b'):
        string_name = 'B'
    elif string_name.startswith('G') or string_name.startswith('g'):
        string_name = 'G'
    elif string_name.startswith('D') or string_name.startswith('d'):
        string_name = 'D'
    elif string_name.startswith('A') or string_name.startswith('a'):
        string_name = 'A'
    else:
        # If not a valid string name, return None
        return None, []
    
    # Map string names to string numbers (0-5, where 0 is the high E string)
    string_map = {'E': 0, 'B': 1, 'G': 2, 'D': 3, 'A': 4}
    string_number = string_map.get(string_name, None)
    
    if string_number is None:
        return None, []
    
    # Extract fret positions
    fret_positions = []
    content = '|'.join(parts[1:])
    
    # Find all fret numbers in the line
    # Support for various tab notations: numbers, h (hammer-on), p (pull-off), b (bend), etc.
    fret_pattern = r'(\d+)(?:[hpbr/\\](?:\d+))?'
    position = 0
    
    for match in re.finditer(fret_pattern, content):
        fret_number = int(match.group(1))
        if 0 <= fret_number <= 24:  # Ensure fret is within range
            fret_positions.append((position, fret_number))
            position += 1
    
    return string_number, fret_positions

def convert_tab_to_training_data(tab_section, time_steps=50, mel_bins=128):
    """
    Convert a tab section to training data format.
    
    Args:
        tab_section: String containing a tab section
        time_steps: Number of time steps in the input data
        mel_bins: Number of mel bins in the input data
        
    Returns:
        Tuple of (X, y) where X is the input features and y is the target positions
    """
    # Split the tab section into lines
    lines = tab_section.strip().split('\n')
    
    # Initialize the output array (6 strings x 25 frets = 150 positions)
    y = np.zeros((time_steps, 150))
    
    # Parse each line
    string_data = {}
    for line in lines:
        string_number, fret_positions = parse_tab_line(line)
        if string_number is not None and fret_positions:
            string_data[string_number] = fret_positions
    
    # Skip if no valid tab data was found
    if not string_data:
        return None, None
    
    # Determine the maximum number of positions
    max_positions = max([len(pos) for pos in string_data.values()] + [0])
    if max_positions == 0:
        return None, None
    
    # Scale the positions to fit into time_steps
    for string_number, positions in string_data.items():
        for i, (_, fret) in enumerate(positions):
            if i < max_positions:
                # Calculate the time step for this position
                t = min(int(i * time_steps / max_positions), time_steps - 1)
                if 0 <= fret <= 24:  # Ensure fret is within range
                    y[t, string_number * 25 + fret] = 1
    
    # Generate synthetic input features (mel spectrogram) that correlate with the tab data
    X = np.zeros((time_steps, mel_bins))
    
    # Base frequency for each string (in Hz)
    string_base_freq = {
        0: 329.63,  # E4 (high E)
        1: 246.94,  # B3
        2: 196.00,  # G3
        3: 146.83,  # D3
        4: 110.00,  # A2
        5: 82.41    # E2 (low E)
    }
    
    # For each time step, add energy in frequency bands corresponding to active notes
    for t in range(time_steps):
        # Add some background noise
        X[t] = np.random.rand(mel_bins) * 0.05
        
        # Find active positions at this time step
        active_positions = np.where(y[t] > 0)[0]
        
        for pos in active_positions:
            # Determine which string and fret
            string_number = pos // 25
            fret = pos % 25
            
            if string_number in string_base_freq:
                # Calculate the frequency of the note (base frequency * 2^(fret/12))
                freq = string_base_freq[string_number] * (2 ** (fret / 12))
                
                # Convert frequency to mel bin (approximate)
                # Mel scale formula: m = 2595 * log10(1 + f/700)
                mel = 2595 * np.log10(1 + freq / 700)
                
                # Scale to our mel_bins range (0 to mel_bins-1)
                max_mel = 2595 * np.log10(1 + 20000 / 700)  # Max frequency ~20kHz
                bin_index = int((mel / max_mel) * mel_bins)
                
                # Ensure bin_index is within range
                bin_index = max(0, min(bin_index, mel_bins - 1))
                
                # Add energy in a band around this frequency (with harmonics)
                # Fundamental
                width = max(3, int(mel_bins / 40))  # Width of the peak
                for i in range(max(0, bin_index - width), min(mel_bins, bin_index + width + 1)):
                    X[t, i] += 0.8 * np.exp(-0.5 * ((i - bin_index) / (width / 2)) ** 2)
                
                # Add harmonics (2x, 3x, 4x the fundamental frequency)
                for harmonic in [2, 3, 4]:
                    harmonic_bin = min(int(bin_index * harmonic), mel_bins - 1)
                    harmonic_width = max(2, int(width / harmonic))
                    for i in range(max(0, harmonic_bin - harmonic_width), min(mel_bins, harmonic_bin + harmonic_width + 1)):
                        X[t, i] += (0.8 / harmonic) * np.exp(-0.5 * ((i - harmonic_bin) / (harmonic_width / 2)) ** 2)
    
    # Apply some temporal smoothing to simulate real audio
    for t in range(1, time_steps):
        X[t] = 0.7 * X[t] + 0.3 * X[t-1]
    
    # Normalize the input features
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    
    return X, y

def process_tab_file(tab_file_path, output_dir=None, max_sections=10):
    """
    Process a tab file and convert it to training data.
    
    Args:
        tab_file_path: Path to the tab file
        output_dir: Directory to save the training data
        max_sections: Maximum number of sections to process
        
    Returns:
        List of (X, y) tuples
    """
    if output_dir is None:
        output_dir = os.path.join(backend_dir, "data", "guitarset_training_data", "ultimate_guitar")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the file name for logging
    file_name = os.path.basename(tab_file_path)
    print(f"Processing {file_name}...", flush=True)
    
    # Read the tab file
    try:
        with open(tab_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            tab_content = f.read()
    except Exception as e:
        print(f"  Error reading file {file_name}: {str(e)}", flush=True)
        return []
    
    # Extract tab sections
    tab_sections = extract_tab_sections(tab_content)
    print(f"  Found {len(tab_sections)} tab sections", flush=True)
    
    if not tab_sections:
        print(f"  No tab sections found in {file_name}", flush=True)
        return []
    
    # Process each tab section
    training_data = []
    valid_sections = 0
    
    for i, section in enumerate(tab_sections[:max_sections]):
        try:
            # Skip very short sections
            if len(section.strip().split('\n')) < 4:
                continue
                
            X, y = convert_tab_to_training_data(section)
            
            # Skip sections with no valid data
            if X is None or y is None:
                continue
                
            # Skip sections with no active positions
            if np.count_nonzero(y) == 0:
                continue
            
            # Save the training data
            base_name = file_name.replace('.txt', '').replace(' ', '_')
            x_path = os.path.join(output_dir, f"{base_name}_section_{i}_x.npy")
            y_path = os.path.join(output_dir, f"{base_name}_section_{i}_y.npy")
            
            np.save(x_path, X)
            np.save(y_path, y)
            
            training_data.append((X, y))
            valid_sections += 1
            
        except Exception as e:
            print(f"  Error processing section {i} of {file_name}: {str(e)}", flush=True)
    
    print(f"  Successfully processed {valid_sections} sections from {file_name}", flush=True)
    return training_data

def process_all_tab_files(tab_dir=None, output_dir=None, max_files=None):
    """
    Process all tab files in a directory.
    
    Args:
        tab_dir: Directory containing tab files
        output_dir: Directory to save the training data
        max_files: Maximum number of files to process
        
    Returns:
        Number of processed files
    """
    if tab_dir is None:
        tab_dir = os.path.join(backend_dir, "data", "tab_data", "tab_files")
    
    if output_dir is None:
        output_dir = os.path.join(backend_dir, "data", "guitarset_training_data", "ultimate_guitar")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Looking for tab files in {tab_dir}...", flush=True)
    
    # Find all tab files
    tab_files = []
    
    # First, check for files directly in the tab_dir
    direct_files = glob.glob(os.path.join(tab_dir, "*.txt"))
    tab_files.extend(direct_files)
    
    # Then check for files in genre subdirectories
    for genre in ['rock', 'metal', 'blues', 'jazz', 'folk']:
        genre_dir = os.path.join(tab_dir, genre)
        if os.path.exists(genre_dir) and os.path.isdir(genre_dir):
            genre_files = glob.glob(os.path.join(genre_dir, "*.txt"))
            tab_files.extend(genre_files)
    
    # Also check for any other txt files in subdirectories
    other_files = glob.glob(os.path.join(tab_dir, "**", "*.txt"), recursive=True)
    for file in other_files:
        if file not in tab_files:
            tab_files.append(file)
    
    print(f"Found {len(tab_files)} tab files", flush=True)
    
    if max_files is not None:
        tab_files = tab_files[:max_files]
        print(f"Processing up to {max_files} files", flush=True)
    
    # Process each tab file
    processed_files = 0
    total_sections = 0
    
    for i, file_path in enumerate(tab_files):
        try:
            print(f"[{i+1}/{len(tab_files)}] Processing {os.path.basename(file_path)}...", flush=True)
            training_data = process_tab_file(file_path, output_dir)
            
            if training_data:
                processed_files += 1
                total_sections += len(training_data)
                print(f"  Generated {len(training_data)} training samples", flush=True)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
    
    print(f"Processed {processed_files} files with a total of {total_sections} valid tab sections", flush=True)
    print(f"Training data saved to {output_dir}", flush=True)
    
    return processed_files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse Ultimate Guitar tabs and convert them to training data")
    parser.add_argument("--tab_dir", type=str, help="Directory containing tab files")
    parser.add_argument("--output_dir", type=str, help="Directory to save the training data")
    parser.add_argument("--max_files", type=int, help="Maximum number of files to process")
    
    args = parser.parse_args()
    
    process_all_tab_files(args.tab_dir, args.output_dir, args.max_files)
