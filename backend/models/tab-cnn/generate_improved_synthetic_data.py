import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import json
import random
from datetime import datetime

# Get the backend directory path
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(script_dir, "../.."))

# Guitar constants
NUM_STRINGS = 6
MAX_FRET = 24  # Maximum fret number
POSITIONS_PER_STRING = 25  # 0-24 frets
TOTAL_POSITIONS = NUM_STRINGS * POSITIONS_PER_STRING  # 6 strings x 25 positions

# Music theory constants
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4 in MIDI notes
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]  # Whole, whole, half, whole, whole, whole, half steps
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]  # Whole, half, whole, whole, half, whole, whole steps

# Common chord shapes (fret positions relative to root)
CHORD_SHAPES = {
    'major': [(0, 0), (1, 2), (2, 2), (2, 1), (3, 0), (3, 0)],  # E shape
    'minor': [(0, 0), (1, 2), (2, 2), (2, 0), (3, 0), (3, 0)],  # Em shape
    'major7': [(0, 0), (1, 2), (2, 1), (2, 1), (3, 0), (3, 0)],  # Emaj7 shape
    'dom7': [(0, 0), (1, 2), (2, 0), (2, 1), (3, 0), (3, 0)],    # E7 shape
    'minor7': [(0, 0), (1, 2), (2, 0), (2, 0), (3, 0), (3, 0)],  # Em7 shape
}

# Common scale patterns (fret positions for each string)
SCALE_PATTERNS = {
    'major_pattern1': [
        [0, 2, 3, 5],       # Low E string
        [0, 2, 3, 5],       # A string
        [0, 2, 4, 5],       # D string
        [0, 2, 4, 5, 7],    # G string
        [0, 3, 5, 7],       # B string
        [0, 2, 3, 5]        # High E string
    ],
    'pentatonic_minor': [
        [0, 3],             # Low E string
        [0, 3],             # A string
        [0, 2],             # D string
        [0, 2],             # G string
        [0, 3],             # B string
        [0, 3]              # High E string
    ]
}

def generate_chord_progression(key, progression_type='basic', num_chords=4):
    """Generate a chord progression in a given key"""
    # Define common chord progressions (using Roman numerals)
    progressions = {
        'basic': ['I', 'IV', 'V', 'I'],
        'pop': ['I', 'V', 'vi', 'IV'],
        'blues': ['I', 'I', 'I', 'I', 'IV', 'IV', 'I', 'I', 'V', 'IV', 'I', 'V'],
        'jazz': ['ii', 'V', 'I', 'vi'],
    }
    
    # Get the requested progression or use a random one
    if progression_type in progressions:
        progression = progressions[progression_type]
    else:
        progression = random.choice(list(progressions.values()))
    
    # Limit to requested number of chords
    progression = progression[:num_chords]
    
    # Map Roman numerals to chord types and scale degrees
    chord_map = {
        'I': ('major', 0),
        'ii': ('minor', 1),
        'iii': ('minor', 2),
        'IV': ('major', 3),
        'V': ('major', 4),
        'vi': ('minor', 5),
        'vii': ('minor', 6),
    }
    
    # Convert progression to actual chords
    chords = []
    for numeral in progression:
        chord_type, scale_degree = chord_map[numeral]
        chords.append((chord_type, scale_degree))
    
    return chords

def generate_scale_run(key, scale_type='major', pattern='major_pattern1', position=0):
    """Generate a scale run in a given key"""
    # Choose the scale pattern
    if pattern in SCALE_PATTERNS:
        pattern = SCALE_PATTERNS[pattern]
    else:
        pattern = random.choice(list(SCALE_PATTERNS.values()))
    
    # Choose the scale type
    if scale_type == 'major':
        scale = MAJOR_SCALE
    else:
        scale = MINOR_SCALE
    
    # Generate the scale run
    scale_positions = []
    
    # For each string
    for string_idx in range(NUM_STRINGS):
        # Get the pattern for this string
        string_pattern = pattern[string_idx]
        
        # Add the positions for this string
        for fret in string_pattern:
            # Adjust for position on the neck
            adjusted_fret = fret + position
            if adjusted_fret <= MAX_FRET:
                scale_positions.append((string_idx, adjusted_fret))
    
    return scale_positions

def generate_chord_shape(chord_type, position):
    """Generate a chord shape at a given position"""
    if chord_type not in CHORD_SHAPES:
        chord_type = random.choice(list(CHORD_SHAPES.keys()))
    
    shape = CHORD_SHAPES[chord_type]
    chord_positions = []
    
    for string_idx, (string, fret_offset) in enumerate(shape):
        fret = position + fret_offset
        if fret <= MAX_FRET:
            chord_positions.append((string_idx, fret))
    
    return chord_positions

def generate_melody(key, scale_type='major', length=8, position=0):
    """Generate a simple melody in a given key"""
    # Choose the scale
    if scale_type == 'major':
        scale = MAJOR_SCALE
    else:
        scale = MINOR_SCALE
    
    # Generate a random melody
    melody_positions = []
    
    # Choose a string for the melody (usually higher strings for melodies)
    melody_string = random.choice([2, 3, 4, 5])  # D, G, B, or high E
    
    # Generate notes
    prev_fret = None
    for _ in range(length):
        # Prefer smaller intervals for more musical melodies
        if prev_fret is not None:
            # Choose a fret near the previous one
            fret_options = [f for f in range(max(0, prev_fret - 3), min(MAX_FRET, prev_fret + 4))]
            fret = random.choice(fret_options)
        else:
            # First note - choose something in the middle of the neck
            fret = random.randint(position, position + 7)
        
        melody_positions.append((melody_string, fret))
        prev_fret = fret
    
    return melody_positions

def generate_tablature_sequence(length=50, style='mixed'):
    """Generate a sequence of tablature positions"""
    # Initialize an empty tablature grid
    tab_grid = np.zeros((length, NUM_STRINGS, POSITIONS_PER_STRING), dtype=np.float32)
    
    # Choose a random key
    key = random.randint(0, 11)  # 0 = C, 1 = C#, etc.
    
    # Generate different types of musical content based on style
    if style == 'chords':
        # Generate a chord progression
        progression = generate_chord_progression(key, 'basic', num_chords=4)
        
        # Place chords in the sequence
        for i, (chord_type, scale_degree) in enumerate(progression):
            # Calculate position based on key and scale degree
            position = key + MAJOR_SCALE[scale_degree]
            position = position % 12  # Keep within first octave
            
            # Generate the chord shape
            chord_positions = generate_chord_shape(chord_type, position)
            
            # Place the chord in the tab grid
            start_idx = i * (length // len(progression))
            end_idx = min(start_idx + (length // len(progression)), length)
            
            # Strum the chord
            for t in range(start_idx, end_idx):
                # Randomly choose whether to include each note in the chord
                for string, fret in chord_positions:
                    if random.random() < 0.7:  # 70% chance to include each note
                        tab_grid[t, string, fret] = 1.0
    
    elif style == 'scales':
        # Generate a scale run
        scale_positions = generate_scale_run(key, 'major', 'major_pattern1', position=random.randint(0, 5))
        
        # Place the scale in the sequence
        for i, (string, fret) in enumerate(scale_positions):
            if i < length:
                tab_grid[i, string, fret] = 1.0
    
    elif style == 'melody':
        # Generate a melody
        melody_positions = generate_melody(key, 'major', length=length, position=random.randint(0, 7))
        
        # Place the melody in the sequence
        for i, (string, fret) in enumerate(melody_positions):
            if i < length:
                tab_grid[i, string, fret] = 1.0
    
    else:  # mixed - combine different styles
        # Divide the sequence into sections
        section_length = length // 3
        
        # First section: chords
        progression = generate_chord_progression(key, 'basic', num_chords=2)
        for i, (chord_type, scale_degree) in enumerate(progression):
            position = key + MAJOR_SCALE[scale_degree]
            position = position % 12
            chord_positions = generate_chord_shape(chord_type, position)
            
            start_idx = i * (section_length // len(progression))
            end_idx = min(start_idx + (section_length // len(progression)), section_length)
            
            for t in range(start_idx, end_idx):
                for string, fret in chord_positions:
                    if random.random() < 0.7:
                        tab_grid[t, string, fret] = 1.0
        
        # Second section: scale run
        scale_positions = generate_scale_run(key, 'major', 'major_pattern1', position=random.randint(0, 5))
        for i, (string, fret) in enumerate(scale_positions):
            if i < section_length:
                tab_grid[section_length + i, string, fret] = 1.0
        
        # Third section: melody
        melody_positions = generate_melody(key, 'major', length=section_length, position=random.randint(0, 7))
        for i, (string, fret) in enumerate(melody_positions):
            if i < section_length:
                tab_grid[2 * section_length + i, string, fret] = 1.0
    
    # Flatten the grid to match the expected output format (time_steps, 150)
    flattened_tab = np.zeros((length, TOTAL_POSITIONS), dtype=np.float32)
    
    for t in range(length):
        for string in range(NUM_STRINGS):
            for fret in range(POSITIONS_PER_STRING):
                idx = string * POSITIONS_PER_STRING + fret
                flattened_tab[t, idx] = tab_grid[t, string, fret]
    
    return flattened_tab

def generate_spectrogram_features(tab_data, time_steps=50, mel_bins=128):
    """Generate synthetic spectrogram features based on tablature data"""
    # Create a random spectrogram-like feature matrix
    X = np.random.randn(time_steps, mel_bins).astype(np.float32) * 0.1
    
    # Enhance the features based on the tablature data
    for t in range(time_steps):
        if t < tab_data.shape[0]:
            # Find active positions at this time step
            active_positions = np.where(tab_data[t] > 0.5)[0]
            
            # For each active position, add energy to the spectrogram
            for pos in active_positions:
                string = pos // POSITIONS_PER_STRING
                fret = pos % POSITIONS_PER_STRING
                
                # Calculate approximate frequency based on string and fret
                base_note = STANDARD_TUNING[string]
                note = base_note + fret
                
                # Convert MIDI note to frequency bin
                freq_bin = int((note - 40) * 2) + 20  # Rough mapping to mel bins
                
                # Ensure the bin is within range
                if 0 <= freq_bin < mel_bins:
                    # Add energy at the fundamental frequency
                    X[t, freq_bin] += 1.0
                    
                    # Add some energy at harmonics
                    for harmonic in [2, 3, 4]:
                        harmonic_bin = freq_bin + int(12 * np.log2(harmonic))
                        if harmonic_bin < mel_bins:
                            X[t, harmonic_bin] += 0.5 / harmonic
    
    # Normalize
    X = (X - X.mean()) / (X.std() + 1e-8)
    
    return X

def generate_batch(batch_id, num_samples=100, time_steps=50, mel_bins=128, output_dir=None):
    """Generate a batch of synthetic training data"""
    if output_dir is None:
        output_dir = os.path.join(backend_dir, "data", "guitarset_training_data", f"batch_{batch_id}")
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples
    styles = ['chords', 'scales', 'melody', 'mixed']
    
    for i in range(num_samples):
        # Choose a random style
        style = random.choice(styles)
        
        # Generate tablature data
        y_tab = generate_tablature_sequence(length=time_steps, style=style)
        
        # Generate corresponding spectrogram features
        x_spec = generate_spectrogram_features(y_tab, time_steps, mel_bins)
        
        # Save the data
        x_path = os.path.join(output_dir, f"x_{i}.npy")
        y_path = os.path.join(output_dir, f"y_{i}.npy")
        
        np.save(x_path, x_spec)
        np.save(y_path, y_tab)
    
    print(f"Generated {num_samples} samples in {output_dir}")
    
    # Create a metadata file
    metadata = {
        "batch_id": batch_id,
        "num_samples": num_samples,
        "time_steps": time_steps,
        "mel_bins": mel_bins,
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "styles_used": styles
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return output_dir

def visualize_sample(batch_dir, sample_idx=0):
    """Visualize a sample from a batch"""
    x_path = os.path.join(batch_dir, f"x_{sample_idx}.npy")
    y_path = os.path.join(batch_dir, f"y_{sample_idx}.npy")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"Sample {sample_idx} not found in {batch_dir}")
        return
    
    # Load the data
    x = np.load(x_path)
    y = np.load(y_path)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot the spectrogram
    ax1.imshow(x.T, aspect='auto', origin='lower')
    ax1.set_title(f"Synthetic Spectrogram (Sample {sample_idx})")
    ax1.set_ylabel("Mel Bins")
    ax1.set_xlabel("Time Steps")
    
    # Plot the tablature
    ax2.imshow(y, aspect='auto', origin='lower')
    ax2.set_title(f"Synthetic Tablature (Sample {sample_idx})")
    ax2.set_ylabel("Position Index")
    ax2.set_xlabel("Time Steps")
    
    plt.tight_layout()
    
    # Save the visualization
    vis_dir = os.path.join(batch_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, f"sample_{sample_idx}.png"))
    
    plt.close()
    
    # Also visualize as ASCII tab
    visualize_as_ascii_tab(y)

def visualize_as_ascii_tab(tab_data, sample_idx=0, time_start=0, time_end=None):
    """
    Visualize tablature data as ASCII guitar tab notation
    
    Args:
        tab_data: Tablature data with shape (time_steps, 150)
        sample_idx: Which sample to visualize (if tab_data has multiple samples)
        time_start: Starting time step
        time_end: Ending time step (if None, use time_start + 16)
    """
    # Handle 3D data (multiple samples)
    if len(tab_data.shape) == 3:
        tab_data = tab_data[sample_idx]
    
    if time_end is None:
        time_end = min(time_start + 16, tab_data.shape[0])
    
    # Extract the data for the specified time range
    tab = tab_data[time_start:time_end]
    
    # Convert from flattened 150-dim to 6x25 grid (6 strings, 25 frets)
    tab_grid = np.zeros((6, tab.shape[0], 25))
    
    for t in range(tab.shape[0]):
        for string in range(6):
            for fret in range(25):
                idx = string * 25 + fret
                if idx < tab.shape[1]:
                    tab_grid[string, t, fret] = tab[t, idx]
    
    # Create ASCII tab notation
    string_names = ["e", "B", "G", "D", "A", "E"]  # Standard tuning (high to low)
    
    print("\nGuitar Tablature (ASCII notation):")
    print("=================================")
    
    for string in range(6):
        # Print string name
        line = f"{string_names[string]}|"
        
        # For each time step, find the active fret position (if any)
        for t in range(tab_grid.shape[1]):
            active_frets = np.where(tab_grid[string, t] > 0.5)[0]
            
            if len(active_frets) > 0:
                # If multiple frets are active, choose the one with highest probability
                if len(active_frets) > 1:
                    probs = [tab_grid[string, t, f] for f in active_frets]
                    active_fret = active_frets[np.argmax(probs)]
                else:
                    active_fret = active_frets[0]
                
                # Add the fret number to the line
                fret_str = str(active_fret)
                if len(fret_str) == 1:
                    line += f"{fret_str}--"
                else:
                    line += f"{fret_str}-"
            else:
                # No active fret, add dashes
                line += "---"
        
        # Print the completed line
        print(line)
    
    print("=================================")

def main():
    parser = argparse.ArgumentParser(description="Generate improved synthetic tablature data")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to generate")
    parser.add_argument("--samples_per_batch", type=int, default=100, help="Number of samples per batch")
    parser.add_argument("--time_steps", type=int, default=50, help="Number of time steps per sample")
    parser.add_argument("--mel_bins", type=int, default=128, help="Number of mel bins per sample")
    parser.add_argument("--output_dir", type=str, help="Base output directory")
    parser.add_argument("--visualize", action="store_true", help="Visualize samples after generation")
    
    args = parser.parse_args()
    
    # Set up the output directory
    if args.output_dir:
        output_base_dir = args.output_dir
    else:
        output_base_dir = os.path.join(backend_dir, "data", "guitarset_training_data")
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Generate batches
    for batch_id in range(args.num_batches):
        print(f"Generating batch {batch_id}...")
        batch_dir = generate_batch(
            batch_id=batch_id,
            num_samples=args.samples_per_batch,
            time_steps=args.time_steps,
            mel_bins=args.mel_bins,
            output_dir=os.path.join(output_base_dir, f"batch_{batch_id}")
        )
        
        # Visualize a few samples
        if args.visualize:
            for i in range(min(3, args.samples_per_batch)):
                visualize_sample(batch_dir, sample_idx=i)
    
    print(f"Done! Generated {args.num_batches} batches with {args.samples_per_batch} samples each.")

if __name__ == "__main__":
    main()
