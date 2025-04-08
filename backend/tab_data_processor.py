#!/usr/bin/env python
"""
Tab Data Processor - Processes guitar tab files and generates training data
"""

import os
import re
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tab_processor")

# Constants
TAB_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "tab_data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATASET_PATH = os.path.join(TAB_DATA_DIR, "fretboard_positions.csv")
TAB_FILES_DIR = os.path.join(TAB_DATA_DIR, "tab_files")

# Guitar string standard tuning (E2, A2, D3, G3, B3, E4)
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]

# Ensure directories exist
os.makedirs(TAB_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TAB_FILES_DIR, exist_ok=True)

class TabDataProcessor:
    """Processes guitar tablature data from files and generates training datasets"""
    
    def __init__(self):
        """Initialize the tab data processor"""
        self.position_data = []
        
        # Load existing data if available
        if os.path.exists(DATASET_PATH):
            try:
                self.dataset = pd.read_csv(DATASET_PATH, low_memory=False)
                logger.info(f"Loaded existing dataset with {len(self.dataset)} records")
            except Exception as e:
                logger.error(f"Error loading existing dataset: {e}")
                self.dataset = pd.DataFrame()
        else:
            self.dataset = pd.DataFrame()
    
    def parse_tab_file(self, filepath, metadata=None):
        """
        Parse guitar positions from a tab file
        
        Args:
            filepath: Path to the tab file
            metadata: Optional dictionary with artist, song, genre, etc.
            
        Returns:
            Number of positions extracted
        """
        if metadata is None:
            metadata = {
                "artist": "Unknown",
                "song": Path(filepath).stem,
                "genre": "Unknown",
                "rating": 5.0
            }
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract positions
            positions = self._extract_positions_from_tab(content, metadata)
            self.position_data.extend(positions)
            
            logger.info(f"Extracted {len(positions)} positions from {filepath}")
            return len(positions)
            
        except Exception as e:
            logger.error(f"Error parsing tab file {filepath}: {e}")
            return 0
    
    def _extract_positions_from_tab(self, content, metadata):
        """Extract fretboard positions from tab content"""
        positions = []
        
        # First, look for tab sections enclosed in [tab] tags
        tag_pattern = re.compile(r'\[tab\](.*?)\[/tab\]', re.DOTALL)
        tag_matches = tag_pattern.findall(content)
        
        if tag_matches:
            # Process each [tab] section
            for tab_section in tag_matches:
                # Split into lines
                tab_lines = tab_section.strip().split('\n')
                processed_positions = self._process_tab_section(tab_lines, metadata)
                positions.extend(processed_positions)
        else:
            # No [tab] tags found, try the older regex approach
            # Regex pattern for standard tab format with 6 strings
            tab_line_pattern = re.compile(r'([eEbBgGdDaA])[|-](.+?)[\r\n]')
            
            # Look for tab sections - groups of 6 strings
            tab_sections = []
            current_section = []
            
            for line in content.split('\n'):
                match = tab_line_pattern.match(line.strip())
                if match:
                    current_section.append((match.group(1).lower(), match.group(2)))
                    # If we've found a complete 6-string section, add it
                    if len(current_section) == 6:
                        tab_sections.append(current_section)
                        current_section = []
            
            # Process each tab section with the old method
            for section in tab_sections:
                # Map string names to tuning and indices
                string_map = {}
                for i, (string_name, content) in enumerate(section):
                    # Handle different notations (e, E, high e, etc.)
                    if string_name == 'e' and i == 0:  # High E
                        string_map[string_name] = 0
                    elif string_name == 'b':
                        string_map[string_name] = 1
                    elif string_name == 'g':
                        string_map[string_name] = 2
                    elif string_name == 'd':
                        string_map[string_name] = 3
                    elif string_name == 'a':
                        string_map[string_name] = 4
                    elif string_name == 'e' and i > 0:  # Low E
                        string_map[string_name] = 5
                
                # Extract fret positions
                for string_name, string_content in section:
                    if string_name not in string_map:
                        continue
                    
                    string_idx = string_map[string_name]
                    
                    # Find all fret numbers
                    fret_matches = re.finditer(r'(\d+)', string_content)
                    for match in fret_matches:
                        fret = int(match.group(1))
                        position = match.start()
                        
                        # Skip unrealistic frets
                        if fret > 24:
                            continue
                        
                        # Calculate MIDI note
                        midi_note = STANDARD_TUNING[string_idx] + fret
                        
                        positions.append({
                            "artist": metadata["artist"],
                            "song": metadata["song"],
                            "genre": metadata.get("genre", "Unknown"),
                            "rating": metadata.get("rating", 5.0),
                            "string": string_idx,
                            "fret": fret,
                            "midi_note": midi_note,
                            "position_in_tab": position
                        })
        
        return positions
    
    def _process_tab_section(self, tab_lines, metadata):
        """Process a tab section in standard ASCII format"""
        positions = []
        
        # Filter out empty lines and collect only tab lines that look like guitar tablature
        valid_tab_lines = []
        for line in tab_lines:
            line = line.strip()
            if line and '|' in line and '-' in line:
                valid_tab_lines.append(line)
        
        # We need at least 6 lines for a complete guitar tab (6 strings)
        if len(valid_tab_lines) < 6:
            return positions
        
        # Standard guitar has 6 strings, so find continuous sections of 6 lines
        for i in range(0, len(valid_tab_lines) - 5, 6):
            section = valid_tab_lines[i:i+6]
            
            # Skip if we don't have exactly 6 strings
            if len(section) != 6:
                continue
                
            # Map the strings in order (high E to low E)
            string_indices = [0, 1, 2, 3, 4, 5]  # E, B, G, D, A, E (high to low)
            
            # Process each string
            for string_idx, tab_line in zip(string_indices, section):
                # Clean the line to the part that contains the actual tab data
                # Typically between the first | and last |
                parts = tab_line.split('|')
                if len(parts) < 3:
                    continue
                
                # The tab content is after the first | and before the last |
                tab_content = '|'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
                
                # Find all fret numbers and technical notations
                # This regex handles:
                # - Regular fret numbers
                # - Hammer-ons (h), pull-offs (p), slides (/), bends (b)
                # - Simple formatting like fret+string combinations (2p0, 0h2, etc.)
                chord_pattern = re.compile(r'(\d+)(?:[hpbs/\\](\d+))?')
                
                for match in chord_pattern.finditer(tab_content):
                    fret = int(match.group(1))
                    position = match.start()
                    
                    # Skip unrealistic frets
                    if fret > 24:
                        continue
                    
                    # Calculate MIDI note
                    midi_note = STANDARD_TUNING[string_idx] + fret
                    
                    # Add the base position
                    positions.append({
                        "artist": metadata["artist"],
                        "song": metadata["song"],
                        "genre": metadata.get("genre", "Unknown"),
                        "rating": metadata.get("rating", 5.0),
                        "string": string_idx,
                        "fret": fret,
                        "midi_note": midi_note,
                        "position_in_tab": position,
                        "technique": "normal"
                    })
                    
                    # Check if there's a secondary note (hammer-on, pull-off, etc.)
                    if match.group(2):
                        secondary_fret = int(match.group(2))
                        
                        # Skip unrealistic frets
                        if secondary_fret > 24:
                            continue
                        
                        # Determine technique based on notation
                        technique = "normal"
                        full_match = match.group(0)
                        if 'h' in full_match:
                            technique = "hammer-on"
                        elif 'p' in full_match:
                            technique = "pull-off"
                        elif '/' in full_match or '\\' in full_match:
                            technique = "slide"
                        elif 'b' in full_match:
                            technique = "bend"
                        
                        # Calculate MIDI note for secondary fret
                        secondary_midi_note = STANDARD_TUNING[string_idx] + secondary_fret
                        
                        # Add the secondary position
                        positions.append({
                            "artist": metadata["artist"],
                            "song": metadata["song"],
                            "genre": metadata.get("genre", "Unknown"),
                            "rating": metadata.get("rating", 5.0),
                            "string": string_idx,
                            "fret": secondary_fret,
                            "midi_note": secondary_midi_note,
                            "position_in_tab": position,
                            "technique": technique,
                            "related_fret": fret
                        })
                
        return positions
    
    def generate_synthetic_data(self, num_samples=1000, styles=None):
        """
        Generate synthetic training data based on music theory principles
        
        Args:
            num_samples: Number of samples to generate
            styles: List of styles to generate data for
            
        Returns:
            Number of samples generated
        """
        if styles is None:
            styles = ["blues", "rock", "jazz", "folk", "metal"]
        
        logger.info(f"Generating {num_samples} synthetic samples across {len(styles)} styles")
        
        # Common scale patterns (represented as half-step intervals from root)
        scales = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "pentatonic_major": [0, 2, 4, 7, 9],
            "pentatonic_minor": [0, 3, 5, 7, 10],
            "blues": [0, 3, 5, 6, 7, 10]
        }
        
        # Style-specific preferences
        style_prefs = {
            "blues": {
                "scales": ["blues", "pentatonic_minor"],
                "position_bias": [0.5, 0.3, 0.2, 0.0, 0.0],  # Low, mid-low, mid, mid-high, high
                "common_keys": ["A", "E", "G", "D", "C"]
            },
            "rock": {
                "scales": ["pentatonic_minor", "pentatonic_major", "major"],
                "position_bias": [0.3, 0.4, 0.2, 0.1, 0.0],
                "common_keys": ["E", "A", "D", "G", "C"]
            },
            "jazz": {
                "scales": ["major", "minor"],
                "position_bias": [0.1, 0.2, 0.3, 0.3, 0.1],
                "common_keys": ["C", "F", "Bb", "Eb", "G"]
            },
            "folk": {
                "scales": ["major", "pentatonic_major"],
                "position_bias": [0.6, 0.3, 0.1, 0.0, 0.0],  # Strong preference for open positions
                "common_keys": ["G", "C", "D", "E", "A"]
            },
            "metal": {
                "scales": ["pentatonic_minor", "minor"],
                "position_bias": [0.2, 0.3, 0.3, 0.2, 0.0],
                "common_keys": ["E", "A", "D", "C", "G"]
            }
        }
        
        # Note to MIDI mapping (C4 = 60)
        note_to_midi = {
            "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
            "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
            "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
        }
        
        # Generate samples
        samples_per_style = num_samples // len(styles)
        generated = 0
        
        for style in styles:
            style_config = style_prefs.get(style, style_prefs["rock"])
            
            for _ in range(samples_per_style):
                # Select a key
                key = np.random.choice(style_config["common_keys"])
                key_root = note_to_midi[key]
                
                # Select a scale
                scale_type = np.random.choice(style_config["scales"])
                scale_intervals = scales[scale_type]
                
                # Generate a sequence of notes in this key/scale
                sequence_length = np.random.randint(4, 12)
                midi_notes = []
                
                for _ in range(sequence_length):
                    # Choose a scale degree
                    scale_degree = np.random.randint(0, len(scale_intervals))
                    octave = np.random.randint(4, 6)  # Between C4 and B5
                    
                    # Calculate MIDI note
                    midi_note = 60 + (octave - 4) * 12 + key_root + scale_intervals[scale_degree]
                    midi_notes.append(midi_note)
                
                # For each note, find valid positions on the fretboard
                prev_string = None
                prev_fret = None
                
                for i, midi_note in enumerate(midi_notes):
                    # Find all valid positions for this note
                    valid_positions = []
                    
                    for string_idx, open_string_midi in enumerate(STANDARD_TUNING):
                        fret = midi_note - open_string_midi
                        if 0 <= fret <= 24:
                            # Weight by style position bias - divide fretboard into 5 regions
                            region = min(4, fret // 5)  # 0-4, 5-9, 10-14, 15-19, 20-24
                            bias = style_config["position_bias"][region]
                            
                            # Adjust for previous position if available
                            position_score = bias
                            if prev_string is not None and prev_fret is not None:
                                # Prefer minimal movement
                                string_distance = abs(string_idx - prev_string)
                                fret_distance = abs(fret - prev_fret)
                                total_distance = string_distance + fret_distance * 0.5
                                
                                # Reduce score based on distance (closer is better)
                                position_score *= max(0.1, 1.0 - (total_distance / 10.0))
                            
                            valid_positions.append((string_idx, fret, position_score))
                    
                    if valid_positions:
                        # Select position based on weighted probabilities
                        weights = [p[2] for p in valid_positions]
                        total_weight = sum(weights)
                        if total_weight > 0:
                            probabilities = [w / total_weight for w in weights]
                            chosen_idx = np.random.choice(len(valid_positions), p=probabilities)
                            chosen = valid_positions[chosen_idx]
                            
                            string_idx, fret = chosen[0], chosen[1]
                            
                            # Add to dataset
                            self.position_data.append({
                                "artist": f"Synthetic_{style}",
                                "song": f"Generated_{style}_{generated}",
                                "genre": style,
                                "rating": 5.0,  # All synthetic data gets top rating
                                "string": string_idx,
                                "fret": fret,
                                "midi_note": midi_note,
                                "position_in_tab": i,
                                "synthetic": True
                            })
                            
                            # Update previous position
                            prev_string, prev_fret = string_idx, fret
                
                generated += 1
        
        logger.info(f"Generated {len(self.position_data)} synthetic positions")
        return len(self.position_data)
    
    def save_positions(self, output_path):
        """
        Save the extracted positions to a CSV file
        
        Args:
            output_path: Path to save the CSV file
            
        Returns:
            Number of positions saved
        """
        if not self.position_data:
            logger.warning("No position data to save")
            return 0
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.position_data)
            
            # Save to CSV
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Saved {len(df)} positions to {output_path}")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error saving positions data: {e}")
            return 0
    
    def _save_dataset(self):
        """Save collected position data to CSV"""
        if not self.position_data:
            logger.warning("No position data to save")
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.position_data)
            
            # Combine with existing data if any
            if not self.dataset.empty:
                df = pd.concat([self.dataset, df], ignore_index=True)
                
                # Remove duplicates
                df = df.drop_duplicates()
            
            # Save to CSV
            df.to_csv(DATASET_PATH, index=False)
            logger.info(f"Saved {len(df)} positions to {DATASET_PATH}")
            
            # Update instance reference
            self.dataset = df
            
        except Exception as e:
            logger.error(f"Error saving position data: {e}")
    
    def prepare_training_data(self):
        """
        Prepare the collected data for training the neural network
        
        Returns:
            Dictionary with training features and labels
        """
        if not os.path.exists(DATASET_PATH):
            logger.error(f"Dataset file {DATASET_PATH} not found")
            return None
        
        try:
            # Load the dataset
            df = pd.read_csv(DATASET_PATH, low_memory=False)
            logger.info(f"Preparing training data from {len(df)} positions")
            
            # Create input features
            X = []
            y = []
            
            # Group by song to maintain context
            for _, group in df.groupby(['artist', 'song']):
                # Sort by position in tab to maintain sequence
                group = group.sort_values('position_in_tab')
                
                # Process each position with better error handling
                for i in range(len(group)):
                    try:
                        row = group.iloc[i]
                        
                        # Create input vector (one-hot for MIDI note, etc.)
                        input_vector = np.zeros(176)
                        
                        # Set MIDI note (first 128 elements)
                        midi_note = int(row['midi_note'])
                        if 0 <= midi_note < 128:
                            input_vector[midi_note] = 1.0
                        
                        # Set previous position if available (next 25 elements)
                        if i > 0:
                            prev_row = group.iloc[i-1]
                            prev_fret = int(prev_row['fret'])
                            if 0 <= prev_fret <= 24:
                                input_vector[128 + prev_fret] = 1.0
                        
                        # Set string (next 6 elements)
                        string_idx = int(row['string'])
                        if 0 <= string_idx < 6:
                            input_vector[153 + string_idx] = 1.0
                        
                        # Set style based on genre (last 5 elements)
                        genre = str(row.get('genre', '')).lower()
                        style_idx = 0  # Default style
                        if 'blues' in genre:
                            style_idx = 0
                        elif 'rock' in genre:
                            style_idx = 1
                        elif 'jazz' in genre:
                            style_idx = 2
                        elif 'folk' in genre or 'country' in genre:
                            style_idx = 3
                        elif 'metal' in genre:
                            style_idx = 4
                        input_vector[171 + style_idx] = 1.0
                        
                        # Create output vector (one-hot for string/fret combination)
                        output_vector = np.zeros(150)  # 6 strings * 25 frets
                        output_idx = string_idx * 25 + int(row['fret'])
                        if 0 <= output_idx < 150:
                            output_vector[output_idx] = 1.0
                        
                        X.append(input_vector)
                        y.append(output_vector)
                    except Exception as e:
                        # Log the error and continue with next row
                        logger.warning(f"Error processing row {i}: {e}")
                        continue
            
            if len(X) == 0:
                logger.error("No valid training examples were generated")
                return None
                
            return {
                'X': np.array(X),
                'y': np.array(y),
            }
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None

    def process_downloaded_tabs(self, directory=None):
        """
        Process downloaded tab files from Ultimate Guitar or similar sources
        
        Args:
            directory: Directory containing downloaded tab files
            
        Returns:
            Number of positions extracted
        """
        if directory is None:
            directory = TAB_FILES_DIR
        
        if not os.path.exists(directory):
            logger.error(f"Directory {directory} does not exist")
            return 0
        
        tab_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                    if f.endswith('.txt') or f.endswith('.tab')]
        
        total_positions = 0
        for filepath in tab_files:
            # Extract metadata from filename
            filename = os.path.basename(filepath)
            parts = filename.split('_', 1)
            
            if len(parts) >= 2:
                artist = parts[0]
                song = parts[1].split('.')[0]
                # Remove any text in parentheses
                song = re.sub(r'\([^)]*\)', '', song).strip()
            else:
                artist = "Unknown"
                song = os.path.splitext(filename)[0]
            
            metadata = {
                "artist": artist,
                "song": song,
                "source": "downloaded",
                "format": "ascii"
            }
            
            positions = self.parse_tab_file(filepath, metadata)
            total_positions += positions
        
        if total_positions > 0:
            # Save the new dataset
            self._save_dataset()
            logger.info(f"Processed {len(tab_files)} downloaded tab files, extracted {total_positions} positions")
        else:
            logger.warning("No positions extracted from downloaded tab files")
        
        return total_positions

def process_tab_dir(directory=None):
    """Process all tab files in a directory"""
    if directory is None:
        directory = TAB_FILES_DIR
        
    processor = TabDataProcessor()
    
    # Process tab files
    total = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(('.txt', '.tab')):
                filepath = os.path.join(root, filename)
                
                # Try to extract genre from directory name
                dirname = os.path.basename(root).lower()
                genre = "Unknown"
                for g in ["blues", "rock", "jazz", "folk", "metal", "country", "pop"]:
                    if g in dirname:
                        genre = g
                        break
                
                # Extract file metadata
                artist = "Unknown"
                song = os.path.splitext(filename)[0]
                
                # Look for artist-song pattern in filename
                name_parts = song.split('-', 1)
                if len(name_parts) == 2:
                    artist, song = name_parts[0].strip(), name_parts[1].strip()
                
                metadata = {
                    "artist": artist,
                    "song": song,
                    "genre": genre,
                    "rating": 5.0
                }
                
                count = processor.parse_tab_file(filepath, metadata)
                total += count
    
    # Generate synthetic data if we don't have enough
    if total < 1000:
        required = max(1000, total * 2)
        processor.generate_synthetic_data(num_samples=required)
    
    # Save all the data
    processor._save_dataset()
    
    # Prepare training data
    training_data = processor.prepare_training_data()
    if training_data:
        # Save the prepared data
        np.save(os.path.join(TAB_DATA_DIR, "X_tab_positions.npy"), training_data['X'])
        np.save(os.path.join(TAB_DATA_DIR, "y_tab_positions.npy"), training_data['y'])
        logger.info(f"Saved training data: {len(training_data['X'])} samples")
    
    return total

def main():
    """Main entry point for the script"""
    logger.info("Starting tab data processing")
    
    total = process_tab_dir()
    
    if total > 0:
        logger.info(f"Processed {total} tab positions")
        print(f"\n✓ Successfully processed {total} tab positions!")
        print("  You can now train the model using the collected data.")
    else:
        # Just generate synthetic data
        processor = TabDataProcessor()
        num_generated = processor.generate_synthetic_data(num_samples=5000)
        processor._save_dataset()
        
        # Prepare training data
        training_data = processor.prepare_training_data()
        if training_data:
            np.save(os.path.join(TAB_DATA_DIR, "X_tab_positions.npy"), training_data['X'])
            np.save(os.path.join(TAB_DATA_DIR, "y_tab_positions.npy"), training_data['y'])
            logger.info(f"Saved synthetic training data: {len(training_data['X'])} samples")
            
        print(f"\n✓ Generated {num_generated} synthetic tab positions!")
        print("  No tab files found, but synthetic data is ready for training.")
        print("  You can add .txt or .tab files to the 'data/tab_data/tab_files' directory.")
    
    logger.info("Tab data processing complete")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process guitar tab files for ML model training")
    parser.add_argument('--synthetic', type=int, default=0, help='Number of synthetic samples to generate')
    parser.add_argument('--process-tabs', action='store_true', help='Process all tab files in the tab directory')
    parser.add_argument('--process-downloads', action='store_true', help='Process downloaded tab files')
    parser.add_argument('--convert-dataset', action='store_true', help='Convert dataset to numpy format for training')
    parser.add_argument('--visualize', action='store_true', help='Visualize the dataset')
    
    args = parser.parse_args()
    
    processor = TabDataProcessor()
    
    if args.synthetic > 0:
        processor.generate_synthetic_data(args.synthetic)
    
    if args.process_tabs:
        processor.process_tab_dir()
    
    if args.process_downloads:
        processor.process_downloaded_tabs()
    
    if args.convert_dataset:
        processor.convert_to_numpy_dataset()
    
    if args.visualize:
        processor.visualize_dataset()
        
    if not any([args.synthetic, args.process_tabs, args.process_downloads, args.convert_dataset, args.visualize]):
        # Default behavior if no arguments are provided
        print("No arguments provided. Processing downloaded tabs...")
        processor.process_downloaded_tabs()
        processor.convert_to_numpy_dataset()
