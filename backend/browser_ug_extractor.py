import re
import numpy as np
import os
from typing import List, Dict
from unified_tab_processor2 import UnifiedTabProcessor

def parse_ug_tab(tab_content: str) -> List[Dict]:
    """Parse UG ASCII tab into a list of note events with timing."""
    note_events = []
    current_time = 0.0
    time_step = 0.1  # Assume 100ms per tab "frame" (adjustable)
    
    # Split into lines and filter tab sections
    lines = tab_content.split('\n')
    tab_blocks = []
    current_block = []
    
    for line in lines:
        if any(s in line for s in ['D|', 'A|', 'F|', 'C|', 'G|', 'C|']):  # Guitar strings
            current_block.append(line)
        elif current_block:  # End of a block
            tab_blocks.append(current_block)
            current_block = []
    
    if current_block:
        tab_blocks.append(current_block)
    
    # Parse each block
    for block in tab_blocks:
        block_length = max(len(line.split('|')[1]) for line in block if '|' in line)
        for pos in range(block_length):
            frame = {'time': current_time, 'notes': []}
            for line in block:
                if '|' not in line:
                    continue
                string = line.split('|')[0].strip()
                content = line.split('|')[1]
                if pos < len(content):
                    char = content[pos]
                    if char.isdigit() or char in ['h', 'p', '/', '\\', '~', '*', 'v']:  # Note or articulation
                        fret = int(char) if char.isdigit() else 0  # Articulations assigned fret 0 for simplicity
                        string_idx = {'D': 0, 'A': 1, 'F': 2, 'C': 3, 'G': 4, 'C': 5}.get(string)
                        if string_idx is not None and fret <= 20:  # Max fret 20
                            frame['notes'].append({'string': string_idx, 'fret': fret})
            if frame['notes']:
                note_events.append(frame)
            current_time += time_step
    
    return note_events

def tab_to_lstm_array(note_events: List[Dict], timesteps: int = 50) -> np.ndarray:
    """Convert note events to LSTM input array."""
    num_frames = len(note_events)
    time_step = 0.1  # Default time step between frames
    
    if num_frames < timesteps:
        # Calculate the last time in the sequence
        last_time = note_events[-1]['time'] if note_events else 0
        
        # Create padding events with incrementing times
        padding = []
        for i in range(timesteps - num_frames):
            padding.append({'time': last_time + (i + 1) * time_step, 'notes': []})
        
        padded_events = note_events + padding
    else:
        padded_events = note_events[:timesteps]
    
    array = np.zeros((timesteps, 6, 21))  # 6 strings, 21 frets
    for t, event in enumerate(padded_events):
        for note in event['notes']:
            array[t, note['string'], note['fret']] = 1.0
    
    return array.reshape(timesteps, 6 * 21)  # Flatten to (timesteps, 126)

def process_ug_tabs(tab_dir: str, output_dir: str, audio_dir: str = None):
    """Process UG tabs and pair with TabCNN predictions if audio available."""
    os.makedirs(output_dir, exist_ok=True)
    processor = UnifiedTabProcessor()  # From your consolidated code
    
    X_train, y_train = [], []
    for tab_file in os.listdir(tab_dir):
        if not tab_file.endswith('.txt'):
            continue
        
        tab_path = os.path.join(tab_dir, tab_file)
        with open(tab_path, 'r', encoding='utf-8') as f:
            tab_content = f.read()
        
        # Parse tab
        note_events = parse_ug_tab(tab_content)
        y_array = tab_to_lstm_array(note_events)
        y_train.append(y_array)
        
        # If audio available, generate TabCNN predictions
        audio_file = os.path.join(audio_dir, tab_file.replace('.txt', '.wav')) if audio_dir else None
        if audio_file and os.path.exists(audio_file):
            results = processor.process_audio(audio_file, style='rock')
            raw_tab = results['guitar']['fretboard_data']  # Assuming this is (timesteps, strings, frets)
            if raw_tab:
                X_array = np.array([frame['notes'] for frame in raw_tab]).reshape(-1, 6 * 21)
                if X_array.shape[0] >= 50:
                    X_array = X_array[:50]
                else:
                    X_array = np.pad(X_array, ((0, 50 - X_array.shape[0]), (0, 0)), 'constant')
                X_train.append(X_array)
            else:
                X_train.append(np.random.random((50, 6 * 21)) * 0.5)  # Fallback noisy data
        else:
            # Simulate noisy TabCNN output
            noisy_array = y_array.copy()
            noisy_array += np.random.random(noisy_array.shape) * 0.3  # Add noise
            noisy_array = np.clip(noisy_array, 0, 1)
            X_train.append(noisy_array)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    np.save(os.path.join(output_dir, 'X_lstm_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_lstm_train.npy'), y_train)
    return X_train, y_train

# Example usage
if __name__ == "__main__":
    tab_dir = "F:/newrepos/olitunes/backend/data/tab_data/tab_files"  # Where UG tabs are saved
    output_dir = "F:/newrepos/olitunes/backend/data/lstm_training_data"
    audio_dir = None  # Optional: directory with corresponding audio files
    X_train, y_train = process_ug_tabs(tab_dir, output_dir, audio_dir)
    print(f"Processed {X_train.shape[0]} samples for LSTM training.")