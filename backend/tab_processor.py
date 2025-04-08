#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tab Processor for OliTunes ML
=============================

This module processes downloaded guitar tabs to create ML-ready training data.
It handles cleaning, normalization, feature extraction, and preparation of tabs
for use in training the position prediction models.

The processing pipeline:
1. Load raw tabs from downloaded files
2. Clean and normalize tab notation
3. Extract features (notes, positions, techniques)
4. Create training examples with input/output pairs
5. Save processed data in ML-ready format
"""

import os
import re
import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Union, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TAB_FILES_DIR = os.path.join(DATA_DIR, "tab_data", "tab_files")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_tabs")
TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training_data")

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

# Guitar constants
STRINGS = 6
FRETS = 24
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4 in MIDI
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Tab notation patterns
NOTE_PATTERN = re.compile(r'(\d{1,2})')
TECHNIQUE_PATTERNS = {
    'hammer_on': re.compile(r'(\d+)h(\d+)'),
    'pull_off': re.compile(r'(\d+)p(\d+)'),
    'bend': re.compile(r'(\d+)b'),
    'slide': re.compile(r'(\d+)/(\d+)'),
    'vibrato': re.compile(r'(\d+)~'),
}

class TabProcessor:
    """
    Processes guitar tabs from raw text into ML-ready features.
    """
    def __init__(self, input_dir: str = TAB_FILES_DIR, output_dir: str = PROCESSED_DATA_DIR):
        """
        Initialize the tab processor.
        
        Args:
            input_dir: Directory containing raw tab files
            output_dir: Directory to save processed tab data
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Ensure directories exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Tab processor initialized. Input: {input_dir}, Output: {output_dir}")

    def get_available_tabs(self) -> List[str]:
        """
        Get list of all available tab files.
        
        Returns:
            List of tab file paths
        """
        if not os.path.exists(self.input_dir):
            logger.warning(f"Input directory {self.input_dir} does not exist")
            return []
            
        files = []
        for file in os.listdir(self.input_dir):
            if file.endswith('.txt'):
                files.append(os.path.join(self.input_dir, file))
        
        logger.info(f"Found {len(files)} tab files")
        return files

    def load_tab_file(self, file_path: str) -> str:
        """
        Load a tab file and return its contents.
        
        Args:
            file_path: Path to the tab file
            
        Returns:
            Contents of the tab file as a string
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Error loading tab file {file_path}: {str(e)}")
            return ""

    def clean_tab(self, content: str) -> str:
        """
        Clean and normalize tab content.
        
        Args:
            content: Raw tab content
            
        Returns:
            Cleaned tab content
        """
        # Remove unnecessary blank lines
        lines = content.split('\n')
        lines = [line for line in lines if line.strip()]
        
        # Filter out non-tab content
        tab_lines = []
        in_tab_section = False
        
        for line in lines:
            if '|' in line or '-' in line:
                # Likely a tab line
                if any(c.isdigit() for c in line):
                    in_tab_section = True
                    tab_lines.append(line)
            elif in_tab_section and len(line.strip()) < 3:
                # Empty line in tab section
                in_tab_section = False
                tab_lines.append('')  # Add separator
            elif in_tab_section and any(c.isdigit() for c in line):
                # Continuation of tab
                tab_lines.append(line)
        
        # Remove unnecessary content
        cleaned_content = '\n'.join(tab_lines)
        return cleaned_content

    def extract_tab_sections(self, content: str) -> List[List[str]]:
        """
        Extract individual tab sections from the content.
        
        Args:
            content: Cleaned tab content
            
        Returns:
            List of tab sections, each as a list of strings
        """
        sections = []
        current_section = []
        
        lines = content.split('\n')
        for line in lines:
            if line.strip():
                current_section.append(line)
            elif current_section:
                sections.append(current_section)
                current_section = []
        
        if current_section:
            sections.append(current_section)
        
        # Filter out sections that don't look like valid tab sections
        valid_sections = []
        for section in sections:
            if len(section) >= 6:  # At least 6 strings
                valid_sections.append(section)
        
        return valid_sections

    def extract_positions_from_tab_section(self, section: List[str]) -> List[List[int]]:
        """
        Extract note positions from a tab section.
        
        Args:
            section: A tab section as a list of strings
            
        Returns:
            List of positions for each string, where each position is a fret number
        """
        positions = [[] for _ in range(STRINGS)]
        
        # Normalize string order (high E to low E)
        # Most tabs have high E at the top, but some might be reversed
        section = section[:STRINGS]
        
        # Extract positions for each string
        for i, line in enumerate(section):
            if i >= STRINGS:
                break
                
            # Find all numbers in the line
            matches = re.finditer(r'(\d+)', line)
            for match in matches:
                try:
                    positions[i].append(int(match.group(1)))
                except (ValueError, IndexError):
                    continue
        
        return positions

    def extract_techniques_from_tab_section(self, section: List[str]) -> Dict:
        """
        Extract playing techniques from a tab section.
        
        Args:
            section: A tab section as a list of strings
            
        Returns:
            Dictionary of techniques with positions
        """
        techniques = {
            'hammer_on': [],
            'pull_off': [],
            'bend': [],
            'slide': [],
            'vibrato': [],
        }
        
        # Process each string
        for i, line in enumerate(section):
            if i >= STRINGS:
                break
                
            # Check for each technique
            for technique, pattern in TECHNIQUE_PATTERNS.items():
                matches = pattern.finditer(line)
                for match in matches:
                    if technique in ['hammer_on', 'pull_off', 'slide']:
                        # Techniques with start and end positions
                        try:
                            start_pos = int(match.group(1))
                            end_pos = int(match.group(2))
                            techniques[technique].append({
                                'string': i,
                                'start_fret': start_pos,
                                'end_fret': end_pos
                            })
                        except (ValueError, IndexError):
                            continue
                    else:
                        # Techniques with single position
                        try:
                            pos = int(match.group(1))
                            techniques[technique].append({
                                'string': i,
                                'fret': pos
                            })
                        except (ValueError, IndexError):
                            continue
        
        return techniques

    def fret_to_midi_note(self, string_idx: int, fret: int, tuning: List[int] = STANDARD_TUNING) -> int:
        """
        Convert string and fret to MIDI note.
        
        Args:
            string_idx: String index (0-5, where 0 is high E)
            fret: Fret number
            tuning: Tuning as list of MIDI notes
            
        Returns:
            MIDI note number
        """
        return tuning[string_idx] + fret

    def create_position_features(self, positions: List[List[int]]) -> List[Dict]:
        """
        Create feature vectors for ML from extracted positions.
        
        Args:
            positions: List of positions for each string
            
        Returns:
            List of position features
        """
        features = []
        
        # Create features for each position
        for string_idx, string_positions in enumerate(positions):
            for fret in string_positions:
                # Create a feature dictionary for this position
                midi_note = self.fret_to_midi_note(string_idx, fret)
                note_name = NOTE_NAMES[midi_note % 12]
                octave = midi_note // 12 - 1  # MIDI octave formula
                
                feature = {
                    'string': string_idx,
                    'fret': fret,
                    'midi_note': midi_note,
                    'note_name': note_name,
                    'octave': octave,
                    'note_full': f"{note_name}{octave}"
                }
                
                features.append(feature)
        
        return features

    def process_tab_file(self, file_path: str) -> Dict:
        """
        Process a single tab file and extract features.
        
        Args:
            file_path: Path to the tab file
            
        Returns:
            Dictionary with extracted tab data
        """
        logger.info(f"Processing tab file: {file_path}")
        
        # Extract filename for metadata
        filename = os.path.basename(file_path)
        artist_song = os.path.splitext(filename)[0]
        
        try:
            # Split artist and song based on common separator patterns
            if ' - ' in artist_song:
                artist, song = artist_song.split(' - ', 1)
            elif '_' in artist_song:
                artist, song = artist_song.split('_', 1)
            else:
                artist = "Unknown"
                song = artist_song
                
            # Load and clean tab content
            content = self.load_tab_file(file_path)
            if not content:
                logger.warning(f"Empty or invalid tab file: {file_path}")
                return {}
                
            cleaned_content = self.clean_tab(content)
            tab_sections = self.extract_tab_sections(cleaned_content)
            
            if not tab_sections:
                logger.warning(f"No valid tab sections found in {file_path}")
                return {}
            
            # Process all tab sections and combine data
            all_positions = []
            all_techniques = {
                'hammer_on': [],
                'pull_off': [],
                'bend': [],
                'slide': [],
                'vibrato': [],
            }
            
            for section in tab_sections:
                positions = self.extract_positions_from_tab_section(section)
                techniques = self.extract_techniques_from_tab_section(section)
                
                all_positions.extend(self.create_position_features(positions))
                
                for technique, values in techniques.items():
                    all_techniques[technique].extend(values)
            
            # Create a structured data object
            tab_data = {
                'metadata': {
                    'artist': artist,
                    'song': song,
                    'filename': filename
                },
                'positions': all_positions,
                'techniques': all_techniques,
                'stats': {
                    'total_notes': len(all_positions),
                    'section_count': len(tab_sections),
                    'techniques_count': sum(len(t) for t in all_techniques.values())
                }
            }
            
            return tab_data
            
        except Exception as e:
            logger.error(f"Error processing tab file {file_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def save_processed_data(self, tab_data: Dict, output_file: str = None) -> str:
        """
        Save processed tab data to JSON.
        
        Args:
            tab_data: The processed tab data dictionary
            output_file: Optional output filename, otherwise auto-generated
            
        Returns:
            Path to the saved file
        """
        if not tab_data or not tab_data.get('metadata'):
            logger.warning("Cannot save empty or invalid tab data")
            return ""
        
        # Create filename from metadata if not provided
        if not output_file:
            artist = tab_data['metadata']['artist']
            song = tab_data['metadata']['song']
            output_file = f"{artist}_{song}_processed.json"
            output_file = output_file.replace(" ", "_").replace("/", "_")
        
        output_path = os.path.join(self.output_dir, output_file)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(tab_data, f, indent=2)
            
            logger.info(f"Processed tab data saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            return ""

    def process_all_tabs(self) -> List[str]:
        """
        Process all available tab files.
        
        Returns:
            List of paths to processed data files
        """
        tab_files = self.get_available_tabs()
        processed_files = []
        
        logger.info(f"Processing {len(tab_files)} tab files")
        
        for file_path in tab_files:
            tab_data = self.process_tab_file(file_path)
            if tab_data:
                output_path = self.save_processed_data(tab_data)
                if output_path:
                    processed_files.append(output_path)
        
        logger.info(f"Processed {len(processed_files)} tab files successfully")
        return processed_files


class TrainingDataGenerator:
    """
    Generates ML training data from processed tab files.
    """
    def __init__(self, processed_dir: str = PROCESSED_DATA_DIR, output_dir: str = TRAINING_DATA_DIR):
        """
        Initialize the training data generator.
        
        Args:
            processed_dir: Directory containing processed tab data
            output_dir: Directory to save training data
        """
        self.processed_dir = processed_dir
        self.output_dir = output_dir
        
        # Ensure directories exist
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Training data generator initialized. Input: {processed_dir}, Output: {output_dir}")

    def get_processed_files(self) -> List[str]:
        """
        Get list of all processed tab data files.
        
        Returns:
            List of processed data file paths
        """
        if not os.path.exists(self.processed_dir):
            logger.warning(f"Processed directory {self.processed_dir} does not exist")
            return []
            
        files = []
        for file in os.listdir(self.processed_dir):
            if file.endswith('.json'):
                files.append(os.path.join(self.processed_dir, file))
        
        logger.info(f"Found {len(files)} processed tab files")
        return files

    def load_processed_data(self, file_path: str) -> Dict:
        """
        Load a processed data file.
        
        Args:
            file_path: Path to the processed data file
            
        Returns:
            Processed tab data dictionary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading processed data file {file_path}: {str(e)}")
            return {}

    def create_training_examples(self, processed_data: Dict, context_window: int = 3) -> List[Dict]:
        """
        Create training examples with context window for sequence prediction.
        
        Args:
            processed_data: Processed tab data dictionary
            context_window: Number of previous notes to use as context
            
        Returns:
            List of training examples
        """
        if not processed_data or not processed_data.get('positions'):
            return []
        
        positions = processed_data['positions']
        examples = []
        
        # Sort positions by occurrence (assuming they're roughly in order)
        # In a real implementation, we'd need to properly sequence the notes
        for i in range(context_window, len(positions)):
            # Previous notes as context
            context = positions[i-context_window:i]
            
            # Current note as target
            target = positions[i]
            
            # Create a training example
            example = {
                'metadata': processed_data['metadata'],
                'context': context,
                'target': target
            }
            
            examples.append(example)
        
        return examples

    def create_feature_vectors(self, examples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert training examples to numeric feature vectors and targets.
        
        Args:
            examples: List of training examples
            
        Returns:
            Tuple of (X, y) where X is the input features and y is the target
        """
        if not examples:
            return np.array([]), np.array([])
        
        # Initialize empty arrays
        X = []  # Features
        y = []  # Targets
        
        for example in examples:
            # Create input feature vector
            # For each note in context, encode:
            # - String (one-hot)
            # - Fret number (normalized)
            # - MIDI note (normalized)
            
            features = []
            
            for note in example['context']:
                # One-hot encode string
                string_onehot = [0] * STRINGS
                string_onehot[note['string']] = 1
                
                # Normalize fret
                fret_norm = note['fret'] / FRETS
                
                # Normalize MIDI note
                midi_norm = (note['midi_note'] - 40) / (88 - 40)  # Range 40-88
                
                # Combine features
                note_features = string_onehot + [fret_norm, midi_norm]
                features.extend(note_features)
            
            X.append(features)
            
            # Target is string and fret
            target_string = example['target']['string']
            target_fret = example['target']['fret']
            
            # One-hot encode target string
            target_string_onehot = [0] * STRINGS
            target_string_onehot[target_string] = 1
            
            # Target can be either one-hot encoding of string+fret combinations
            # or separate predictions for string and fret
            # Here we'll use the latter approach
            y.append([target_string, target_fret])
        
        return np.array(X), np.array(y)

    def combine_with_synthetic_data(self, X: np.ndarray, y: np.ndarray, 
                                   synthetic_data_path: str = None, 
                                   ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine real tab data with synthetic data.
        
        Args:
            X: Real data features
            y: Real data targets
            synthetic_data_path: Path to synthetic data file
            ratio: Ratio of real to synthetic data (0.5 means 50% real, 50% synthetic)
            
        Returns:
            Tuple of combined (X, y)
        """
        if synthetic_data_path is None or not os.path.exists(synthetic_data_path):
            logger.warning("No synthetic data provided or file not found")
            return X, y
        
        try:
            # Load synthetic data
            with open(synthetic_data_path, 'r') as f:
                synthetic_data = json.load(f)
            
            # Extract synthetic features and targets
            X_synthetic = np.array(synthetic_data['X'])
            y_synthetic = np.array(synthetic_data['y'])
            
            # Calculate sizes
            real_size = len(X)
            synthetic_size = len(X_synthetic)
            
            # Calculate how many synthetic examples to use
            synthetic_to_use = int(real_size * (1 - ratio) / ratio)
            synthetic_to_use = min(synthetic_to_use, synthetic_size)
            
            # Randomly sample synthetic data
            indices = np.random.choice(synthetic_size, synthetic_to_use, replace=False)
            X_synthetic_sample = X_synthetic[indices]
            y_synthetic_sample = y_synthetic[indices]
            
            # Combine data
            X_combined = np.vstack([X, X_synthetic_sample])
            y_combined = np.vstack([y, y_synthetic_sample])
            
            logger.info(f"Combined data: {len(X_combined)} examples " +
                       f"({real_size} real, {synthetic_to_use} synthetic)")
            
            return X_combined, y_combined
            
        except Exception as e:
            logger.error(f"Error combining with synthetic data: {str(e)}")
            return X, y

    def save_training_data(self, X: np.ndarray, y: np.ndarray, output_file: str = "training_data.npz") -> str:
        """
        Save training data to file.
        
        Args:
            X: Input features
            y: Target values
            output_file: Output filename
            
        Returns:
            Path to the saved file
        """
        if X.size == 0 or y.size == 0:
            logger.warning("Cannot save empty training data")
            return ""
        
        output_path = os.path.join(self.output_dir, output_file)
        
        try:
            # Save as numpy compressed file
            np.savez_compressed(output_path, X=X, y=y)
            
            logger.info(f"Training data saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving training data: {str(e)}")
            return ""

    def generate_training_data(self, context_window: int = 3, 
                              synthetic_data_path: str = None,
                              synthetic_ratio: float = 0.5) -> str:
        """
        Generate training data from all processed tab files.
        
        Args:
            context_window: Number of previous notes to use as context
            synthetic_data_path: Path to synthetic data file
            synthetic_ratio: Ratio of real to synthetic data
            
        Returns:
            Path to the saved training data file
        """
        processed_files = self.get_processed_files()
        all_examples = []
        
        logger.info(f"Generating training data from {len(processed_files)} processed files")
        
        for file_path in processed_files:
            processed_data = self.load_processed_data(file_path)
            if processed_data:
                examples = self.create_training_examples(processed_data, context_window)
                all_examples.extend(examples)
        
        logger.info(f"Created {len(all_examples)} training examples")
        
        if not all_examples:
            logger.warning("No training examples generated")
            return ""
        
        # Convert to feature vectors
        X, y = self.create_feature_vectors(all_examples)
        
        # Combine with synthetic data if provided
        if synthetic_data_path:
            X, y = self.combine_with_synthetic_data(X, y, synthetic_data_path, synthetic_ratio)
        
        # Save training data
        output_path = self.save_training_data(X, y)
        return output_path


def main():
    """Main entry point for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tab Processor and Training Data Generator")
    parser.add_argument('--process', action='store_true', help='Process raw tab files')
    parser.add_argument('--generate', action='store_true', help='Generate training data')
    parser.add_argument('--synthetic', help='Path to synthetic data file')
    parser.add_argument('--ratio', type=float, default=0.5, help='Ratio of real to synthetic data')
    parser.add_argument('--context', type=int, default=3, help='Context window size')
    args = parser.parse_args()
    
    try:
        # Process tab files
        if args.process:
            print("\nProcessing tab files...")
            processor = TabProcessor()
            processed_files = processor.process_all_tabs()
            print(f"Processed {len(processed_files)} tab files")
        
        # Generate training data
        if args.generate:
            print("\nGenerating training data...")
            generator = TrainingDataGenerator()
            output_path = generator.generate_training_data(
                context_window=args.context,
                synthetic_data_path=args.synthetic,
                synthetic_ratio=args.ratio
            )
            
            if output_path:
                print(f"Training data saved to {output_path}")
            else:
                print("Failed to generate training data")
                
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
