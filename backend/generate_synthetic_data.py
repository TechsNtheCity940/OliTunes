#!/usr/bin/env python
"""
Synthetic Guitar Tab Data Generator

This script generates synthetic guitar tablature data for training the 
fretboard position prediction model, based on music theory principles
and style-specific playing patterns.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("synthetic_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("synthetic_data")

# Constants
TAB_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "tab_data")
DATASET_PATH = os.path.join(TAB_DATA_DIR, "fretboard_positions.csv")
STYLES_CONFIG_PATH = os.path.join(TAB_DATA_DIR, "style_configs.json")

# Guitar string standard tuning (E2, A2, D3, G3, B3, E4)
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]

# Ensure directories exist
os.makedirs(TAB_DATA_DIR, exist_ok=True)

# Style-specific configurations for realistic playing patterns
DEFAULT_STYLE_CONFIGS = {
    "blues": {
        "description": "Blues style emphasizes emotional expression with frequent bends and vibrato",
        "scales": ["blues", "pentatonic_minor", "pentatonic_major"],
        "position_bias": [0.5, 0.4, 0.1, 0.0, 0.0],  # Low, mid-low, mid, mid-high, high
        "common_keys": ["A", "E", "G", "D", "C"],
        "patterns": {
            "bends": 0.3,
            "vibrato": 0.4,
            "slides": 0.2,
            "hammer_ons": 0.1,
            "pull_offs": 0.1,
            "open_strings": 0.2
        },
        "chord_bias": {
            "dominant7": 0.5,
            "minor": 0.3,
            "major": 0.2
        },
        "box_patterns": True,  # Blues often uses box patterns/positions
        "hand_stretch": 3  # Typical max fret span for hand position
    },
    "rock": {
        "description": "Rock style balances rhythm and lead playing across the fretboard",
        "scales": ["pentatonic_minor", "pentatonic_major", "major", "natural_minor"],
        "position_bias": [0.3, 0.4, 0.2, 0.1, 0.0],
        "common_keys": ["E", "A", "D", "G", "C"],
        "patterns": {
            "bends": 0.2,
            "vibrato": 0.2,
            "slides": 0.1,
            "hammer_ons": 0.2,
            "pull_offs": 0.2,
            "open_strings": 0.3
        },
        "chord_bias": {
            "power": 0.4,
            "major": 0.3,
            "minor": 0.2,
            "dominant7": 0.1
        },
        "box_patterns": False,
        "hand_stretch": 4
    },
    "jazz": {
        "description": "Jazz utilizes complex chords and scales across the entire fretboard",
        "scales": ["major", "dorian", "mixolydian", "altered", "diminished"],
        "position_bias": [0.1, 0.2, 0.4, 0.2, 0.1],  # Uses more mid-neck positions
        "common_keys": ["C", "F", "Bb", "Eb", "G"],
        "patterns": {
            "bends": 0.05,
            "vibrato": 0.1,
            "slides": 0.3,
            "hammer_ons": 0.2,
            "pull_offs": 0.2,
            "open_strings": 0.05
        },
        "chord_bias": {
            "maj7": 0.3,
            "min7": 0.3,
            "dom7": 0.2,
            "dim7": 0.1,
            "aug": 0.05,
            "altered": 0.05
        },
        "box_patterns": False,
        "hand_stretch": 5  # Jazz often requires larger stretches
    },
    "folk": {
        "description": "Folk emphasizes open chords and fingerpicking patterns",
        "scales": ["major", "pentatonic_major", "mixolydian"],
        "position_bias": [0.7, 0.2, 0.1, 0.0, 0.0],  # Strong preference for open positions
        "common_keys": ["G", "C", "D", "E", "A"],
        "patterns": {
            "bends": 0.05,
            "vibrato": 0.1,
            "slides": 0.1,
            "hammer_ons": 0.2,
            "pull_offs": 0.2,
            "open_strings": 0.7  # Folk uses many open strings
        },
        "chord_bias": {
            "major": 0.5,
            "minor": 0.3,
            "sus": 0.1,
            "7": 0.1
        },
        "box_patterns": False,
        "hand_stretch": 3
    },
    "metal": {
        "description": "Metal features fast alternate picking, palm muting, and power chords",
        "scales": ["pentatonic_minor", "natural_minor", "phrygian", "harmonic_minor"],
        "position_bias": [0.2, 0.3, 0.3, 0.2, 0.0],
        "common_keys": ["E", "A", "D", "C", "G"],
        "patterns": {
            "bends": 0.1,
            "vibrato": 0.2,
            "slides": 0.1,
            "hammer_ons": 0.3,
            "pull_offs": 0.3,
            "open_strings": 0.3
        },
        "chord_bias": {
            "power": 0.7,
            "minor": 0.2,
            "diminished": 0.1
        },
        "box_patterns": True,
        "hand_stretch": 4
    }
}

# Scale definitions (half-step intervals from root)
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "altered": [0, 1, 3, 4, 6, 8, 10],
    "diminished": [0, 2, 3, 5, 6, 8, 9, 11]
}

# Note to MIDI mapping (C4 = 60)
NOTE_TO_MIDI = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
}

class SyntheticTabGenerator:
    """Generates synthetic guitar tablature data for model training"""
    
    def __init__(self, style_configs=None):
        """
        Initialize the generator
        
        Args:
            style_configs: Configuration dictionary for various playing styles
        """
        # Load style configs or use defaults
        if style_configs is None:
            if os.path.exists(STYLES_CONFIG_PATH):
                try:
                    with open(STYLES_CONFIG_PATH, 'r') as f:
                        self.style_configs = json.load(f)
                    logger.info(f"Loaded style configurations from {STYLES_CONFIG_PATH}")
                except Exception as e:
                    logger.warning(f"Error loading style configs: {e}. Using defaults.")
                    self.style_configs = DEFAULT_STYLE_CONFIGS
            else:
                self.style_configs = DEFAULT_STYLE_CONFIGS
                # Save default configs for future reference
                try:
                    with open(STYLES_CONFIG_PATH, 'w') as f:
                        json.dump(DEFAULT_STYLE_CONFIGS, f, indent=2)
                    logger.info(f"Saved default style configurations to {STYLES_CONFIG_PATH}")
                except Exception as e:
                    logger.warning(f"Error saving default configurations: {e}")
        else:
            self.style_configs = style_configs
        
        # Initialize position data
        self.position_data = []
        
        # Load existing dataset if available
        if os.path.exists(DATASET_PATH):
            try:
                self.dataset = pd.read_csv(DATASET_PATH)
                logger.info(f"Loaded existing dataset with {len(self.dataset)} records")
            except Exception as e:
                logger.error(f"Error loading existing dataset: {e}")
                self.dataset = pd.DataFrame()
        else:
            self.dataset = pd.DataFrame()
    
    def generate_riff(self, style, length=8, key=None, scale_type=None):
        """
        Generate a guitar riff in a specific style
        
        Args:
            style: The playing style (blues, rock, jazz, etc.)
            length: Number of notes in the riff
            key: Optional root key, otherwise randomly selected
            scale_type: Optional scale type, otherwise based on style preference
            
        Returns:
            List of dictionaries with note data
        """
        # Get style configuration
        style_config = self.style_configs.get(style, self.style_configs["rock"])
        
        # Select key if not provided
        if key is None:
            key = random.choice(style_config["common_keys"])
        
        # Convert key to MIDI root note
        key_root = NOTE_TO_MIDI[key]
        
        # Select scale if not provided
        if scale_type is None:
            scale_weights = [1.0] * len(style_config["scales"])
            scale_type = random.choices(style_config["scales"], weights=scale_weights, k=1)[0]
        
        # Get scale intervals
        scale_intervals = SCALES.get(scale_type, SCALES["pentatonic_minor"])
        
        # Generate the riff
        riff = []
        prev_string = None
        prev_fret = None
        
        for i in range(length):
            # Choose a scale degree
            scale_degree = random.randint(0, len(scale_intervals) - 1)
            octave = random.randint(4, 6)  # Between C4 and B6
            
            # Calculate MIDI note
            midi_note = 60 + (octave - 4) * 12 + key_root + scale_intervals[scale_degree]
            
            # Find positions for this note
            positions = self._find_positions_for_note(
                midi_note, 
                style_config,
                prev_string=prev_string,
                prev_fret=prev_fret
            )
            
            if positions:
                # Select a position based on probabilities
                string_idx, fret = self._select_position(positions)
                
                # Add to riff
                riff.append({
                    "midi_note": midi_note,
                    "string": string_idx,
                    "fret": fret,
                    "position_in_tab": i
                })
                
                # Update previous position
                prev_string, prev_fret = string_idx, fret
        
        return riff
    
    def _find_positions_for_note(self, midi_note, style_config, prev_string=None, prev_fret=None):
        """Find valid fretboard positions for a MIDI note with style preferences"""
        valid_positions = []
        
        # Position bias from style config
        position_bias = style_config["position_bias"]
        hand_stretch = style_config.get("hand_stretch", 4)
        
        for string_idx, open_string_midi in enumerate(STANDARD_TUNING):
            fret = midi_note - open_string_midi
            if 0 <= fret <= 24:
                # Determine region for this fret
                region = min(4, fret // 5)  # 0-4, 5-9, 10-14, 15-19, 20-24
                
                # Get base probability from style position bias
                bias = position_bias[region]
                
                # Adjust for previous position if available
                position_score = bias
                if prev_string is not None and prev_fret is not None:
                    # Calculate string and fret distance
                    string_distance = abs(string_idx - prev_string)
                    fret_distance = abs(fret - prev_fret)
                    
                    # Check if within reasonable hand stretch
                    within_hand_stretch = fret_distance <= hand_stretch
                    
                    # Decrease score for positions that would require large hand movements
                    if string_distance > 1 and fret_distance > 1:
                        position_score *= 0.5
                    
                    # Heavily penalize positions outside hand stretch
                    if not within_hand_stretch:
                        position_score *= 0.2
                
                # Adjust for open strings if specified in style
                if fret == 0 and style_config["patterns"]["open_strings"] > 0.3:
                    position_score *= 1.5
                
                valid_positions.append((string_idx, fret, position_score))
        
        return valid_positions
    
    def _select_position(self, positions):
        """Select a position based on weighted probabilities"""
        weights = [p[2] for p in positions]
        total_weight = sum(weights)
        
        if total_weight > 0:
            probabilities = [w / total_weight for w in weights]
            chosen_idx = np.random.choice(len(positions), p=probabilities)
            return positions[chosen_idx][0], positions[chosen_idx][1]
        else:
            # If all weights are 0, select randomly
            chosen = random.choice(positions)
            return chosen[0], chosen[1]
    
    def generate_style_data(self, style, num_riffs=50, notes_per_riff=16):
        """
        Generate tablature data for a specific playing style
        
        Args:
            style: The playing style (blues, rock, jazz, etc.)
            num_riffs: Number of riffs to generate
            notes_per_riff: Notes per riff
            
        Returns:
            Number of positions generated
        """
        logger.info(f"Generating {num_riffs} riffs for {style} style")
        count = 0
        
        for i in range(num_riffs):
            # Vary riff length
            length = random.randint(max(4, notes_per_riff - 4), notes_per_riff + 4)
            
            # Generate a riff
            riff = self.generate_riff(style, length=length)
            
            # Add style metadata to each note
            for note in riff:
                note["genre"] = style
                note["artist"] = f"Synthetic_{style}"
                note["song"] = f"Generated_Riff_{i}"
                note["rating"] = 5.0
                note["synthetic"] = True
                
                self.position_data.append(note)
                count += 1
        
        logger.info(f"Generated {count} positions for {style}")
        return count
    
    def generate_multi_style_data(self, styles=None, positions_per_style=800):
        """
        Generate data across multiple styles
        
        Args:
            styles: List of styles to generate data for
            positions_per_style: Target number of positions per style
            
        Returns:
            Total number of positions generated
        """
        if styles is None:
            styles = list(self.style_configs.keys())
        
        total = 0
        for style in styles:
            # Calculate how many riffs needed (assuming average of 16 notes per riff)
            num_riffs = max(1, positions_per_style // 16)
            count = self.generate_style_data(style, num_riffs=num_riffs)
            total += count
        
        logger.info(f"Total synthetic positions generated: {total}")
        return total
    
    def prepare_training_data(self):
        """
        Prepare the collected data for training the neural network
        
        Returns:
            True if successful, False otherwise
        """
        if not self.position_data and self.dataset.empty:
            logger.error("No data available to prepare for training")
            return False
        
        try:
            # Convert to dataframe if needed
            if not self.position_data and not self.dataset.empty:
                df = self.dataset
            elif self.position_data and self.dataset.empty:
                df = pd.DataFrame(self.position_data)
            else:
                # Combine new data with existing dataset
                new_df = pd.DataFrame(self.position_data)
                df = pd.concat([self.dataset, new_df], ignore_index=True)
            
            # Save as CSV
            df.to_csv(DATASET_PATH, index=False)
            logger.info(f"Saved {len(df)} records to {DATASET_PATH}")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return False
    
    def generate_dataset(self, output_path, num_samples=5000, styles=None):
        """
        Generate a complete synthetic dataset and save it to the specified path.
        
        Args:
            output_path: Path to save the generated dataset
            num_samples: Total number of samples to generate
            styles: List of styles to include
            
        Returns:
            Number of samples generated
        """
        if styles is None:
            styles = list(self.style_configs.keys())
        
        logger.info(f"Generating dataset with {num_samples} samples across {len(styles)} styles")
        
        # Calculate samples per style
        samples_per_style = max(100, num_samples // len(styles))
        
        # Generate data for each style
        self.position_data = []  # Reset existing data
        generated_samples = self.generate_multi_style_data(
            styles=styles, 
            positions_per_style=samples_per_style
        )
        
        # Save to the specified output path
        try:
            if self.position_data:
                # Create output directory if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Convert to DataFrame and save
                df = pd.DataFrame(self.position_data)
                df.to_csv(output_path, index=False)
                
                logger.info(f"Successfully saved {len(df)} samples to {output_path}")
                return len(df)
            else:
                logger.error("No data generated to save")
                return 0
                
        except Exception as e:
            logger.error(f"Error saving generated dataset: {e}")
            return 0

    def save_data(self):
        """Save generated position data to CSV"""
        if not self.position_data and self.dataset.empty:
            logger.warning("No position data to save")
            return False
        
        try:
            # Convert to dataframe if needed
            if not self.position_data and not self.dataset.empty:
                df = self.dataset
            elif self.position_data and self.dataset.empty:
                df = pd.DataFrame(self.position_data)
            else:
                # Combine new data with existing dataset
                new_df = pd.DataFrame(self.position_data)
                df = pd.concat([self.dataset, new_df], ignore_index=True)
            
            # Save to CSV
            df.to_csv(DATASET_PATH, index=False)
            logger.info(f"Saved {len(df)} records to {DATASET_PATH}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving position data: {e}")
            return False
    
def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Generate synthetic guitar tab data")
    parser.add_argument('--positions', type=int, default=1000, help='Positions per style')
    parser.add_argument('--styles', nargs='+', default=None, help='Styles to generate')
    args = parser.parse_args()
    
    logger.info("Starting synthetic data generation")
    
    generator = SyntheticTabGenerator()
    
    # Generate data
    styles = args.styles if args.styles else ["blues", "rock", "jazz", "folk", "metal"]
    positions = args.positions
    
    print(f"Generating approximately {positions} positions for each of these styles: {', '.join(styles)}")
    
    # Generate synthetic data based on music theory and style-specific patterns
    total = generator.generate_multi_style_data(styles=styles, positions_per_style=positions)
    
    # Save the generated data
    success = generator.save_data()
    
    if success and total > 0:
        # Prepare data for training
        training_success = generator.prepare_training_data()
        
        if training_success:
            print(f"\n✅ Successfully generated and prepared {total} synthetic positions!")
            print("  The synthetic data respects music theory principles and style-specific patterns.")
            print("  Each style has unique characteristics to help the model learn style distinctions:")
            for style in styles:
                print(f"  • {style}: {generator.style_configs[style]['description']}")
            
            print("\nNext steps:")
            print("  1. Train the model with: python train_fretboard_model.py")
            print("  2. For an LSTM model (recommended): python train_fretboard_model.py --lstm")
        else:
            print("\n⚠️ Data was generated but there was an error preparing it for training")
    else:
        print("\n❌ Error generating synthetic data")
    
    logger.info("Synthetic data generation complete")

if __name__ == "__main__":
    main()
