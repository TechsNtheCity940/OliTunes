"""
Tab Text Generator
Generates human-readable, text-based guitar tablature from note data
"""

import os
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TabTextGenerator:
    """
    Generates ASCII guitar tablature similar to standard tab websites
    """
    
    # Standard guitar tuning (lowest to highest)
    STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'e']
    
    def __init__(self):
        """Initialize the tab generator"""
        self.tab_width = 80  # Characters per line
        self.measure_separators = True  # Whether to add | for measures
    
    def generate_text_tab(self, 
                          notes: List[Dict[str, Any]], 
                          song_info: Dict[str, Any] = None,
                          measures_per_line: int = 4) -> str:
        """
        Generate a text-based guitar tablature from the detected notes.
        
        Args:
            notes: List of note dictionaries with time, string, fret info
            song_info: Dictionary with song metadata (title, artist, etc.)
            measures_per_line: Number of measures to display per line
        
        Returns:
            String containing the formatted tablature
        """
        if not notes:
            return "No notes detected to generate tablature."
        
        # Sort notes by time
        sorted_notes = sorted(notes, key=lambda x: x.get('time', 0))
        
        # Extract or set defaults for song info
        if not song_info:
            song_info = {}
        
        title = song_info.get('title', 'Untitled')
        artist = song_info.get('artist', 'Unknown Artist')
        time_sig = song_info.get('timeSignature', {'numerator': 4, 'denominator': 4})
        tempo = song_info.get('tempo', 120)
        capo = song_info.get('capo', 0)
        key = song_info.get('key', 'C')
        
        # Create header
        tab_text = []
        tab_text.append(f"{title} by {artist}\n\n")
        
        if capo > 0:
            tab_text.append(f"Capo {capo}\n\n")
        
        tab_text.append(f"Time Signature: {time_sig['numerator']}/{time_sig['denominator']}\n")
        tab_text.append(f"Key: {key}\n")
        tab_text.append(f"Tempo: {int(tempo)} BPM\n\n")
        
        # Generate a chord section if chord data is available
        if 'chords' in song_info and song_info['chords']:
            tab_text.append("Chords used:\n")
            for chord in song_info['chords']:
                if isinstance(chord, dict) and 'chord' in chord:
                    tab_text.append(f"{chord['chord']} ")
                elif isinstance(chord, str):
                    tab_text.append(f"{chord} ")
            tab_text.append("\n\n")
        
        # Calculate the time grid (snap notes to a grid based on time signature)
        beats_per_measure = time_sig['numerator']
        beat_value = time_sig['denominator']
        
        # Determine the duration of a measure in seconds
        beat_duration = 60.0 / tempo  # Duration of one beat in seconds
        measure_duration = beat_duration * beats_per_measure  # Duration of a measure
        
        # Initialize the tablature lines
        string_lines = {i: [] for i in range(6)}  # 0=highest string (e), 5=lowest string (E)
        
        # Group notes by measure
        measures = {}
        total_measures = 0
        
        for note in sorted_notes:
            # Get note time, string, and fret
            time = note.get('time', 0)
            string = note.get('string', 0)  # 0 is high E, 5 is low E in the tablature display
            fret = note.get('fret', 0)
            
            # Calculate which measure this note belongs to
            measure_idx = int(time / measure_duration)
            total_measures = max(total_measures, measure_idx + 1)
            
            # Initialize this measure if it doesn't exist
            if measure_idx not in measures:
                measures[measure_idx] = {i: [] for i in range(6)}
            
            # Add note to the correct string in this measure
            position_in_measure = (time % measure_duration) / measure_duration
            # Convert to character position (we'll use 12 chars per measure)
            char_position = int(position_in_measure * 12)
            
            # Add this note to the measure
            measures[measure_idx][string].append((char_position, fret))
        
        # Now build the actual tab
        tab_text.append("\n[tab]\n")
        
        # Generate the tab, line by line
        for start_measure in range(0, total_measures, measures_per_line):
            end_measure = min(start_measure + measures_per_line, total_measures)
            
            # Initialize line for each string
            current_line = {i: "" for i in range(6)}
            
            # Process each measure in this line
            for m in range(start_measure, end_measure):
                if m in measures:
                    # For each string, fill the measure with appropriate fret numbers
                    for string in range(6):
                        # Start with a measure of dashes
                        measure_str = ["-"] * 12
                        
                        # Place each note in its position
                        for pos, fret in measures[m][string]:
                            # Make sure we don't go out of bounds
                            if 0 <= pos < 12:
                                # Format numbers > 9 differently
                                if fret >= 10:
                                    if pos + 1 < 12:  # Make sure we have space
                                        measure_str[pos] = str(fret)[0]
                                        measure_str[pos + 1] = str(fret)[1]
                                else:
                                    measure_str[pos] = str(fret)
                        
                        # Add measure line and bar separator
                        current_line[string] += "".join(measure_str)
                        if self.measure_separators:
                            current_line[string] += "|"
                
                else:
                    # Empty measure
                    for string in range(6):
                        current_line[string] += "-" * 12
                        if self.measure_separators:
                            current_line[string] += "|"
            
            # Add tuning letter at the start of each string
            for string in range(6):
                current_line[string] = f"{self.STRING_NAMES[5-string]}|{current_line[string]}"
            
            # Append to tab
            for string in range(6):
                tab_text.append(current_line[5-string] + "\n")  # Reverse to have high E on top
            
            tab_text.append("\n")  # Space between sections
        
        tab_text.append("[/tab]\n")
        
        return "".join(tab_text)

    def generate_tab(self, 
                     predictions, 
                     key='C', 
                     tempo=120, 
                     style='rock'):
        """
        Generate guitar tablature from model predictions
        
        Args:
            predictions: Model predictions array or list
            key: Detected music key
            tempo: Detected tempo in BPM
            style: Style for tablature generation ('rock', 'blues', etc.)
            
        Returns:
            Formatted tablature as string
        """
        # Convert predictions to notes
        notes = self._predictions_to_notes(predictions)
        
        # Add song info
        song_info = {
            'title': 'Generated Tab',
            'artist': 'OliTunes',
            'timeSignature': {'numerator': 4, 'denominator': 4},
            'tempo': tempo,
            'key': key,
            'style': style
        }
        
        # Generate tab using existing generate_text_tab method
        return self.generate_text_tab(notes, song_info=song_info)
        
    def _predictions_to_notes(self, predictions):
        """
        Convert model predictions to note dictionaries
        
        Args:
            predictions: Model prediction array/matrix
            
        Returns:
            List of note dictionaries
        """
        notes = []
        
        # Handle different prediction formats
        if isinstance(predictions, np.ndarray):
            # Check the shape of predictions
            if len(predictions.shape) == 3:  # [time, string, fret]
                # Handle 3D predictions (time, string, fret)
                for t in range(predictions.shape[0]):
                    for s in range(predictions.shape[1]):
                        # Get the highest probability fret for this string at this time
                        if np.max(predictions[t, s]) > 0.1:  # Confidence threshold
                            fret = np.argmax(predictions[t, s])
                            if fret > 0:  # Skip if it's a 0 (open string or no note)
                                notes.append({
                                    'time': t * 0.25,  # Assuming 16th note grid (0.25 beats)
                                    'string': s,
                                    'fret': int(fret)
                                })
            elif len(predictions.shape) == 2:
                # Handle 2D predictions
                # Check if it's in the format [batch, features]
                if predictions.shape[0] == 1:
                    # Handle the case where first dimension is batch size
                    # Common format for CNN models like TabCNN
                    features = predictions[0]  # Get the features for the single batch item
                    
                    # For TabCNN, the output is often a flattened representation
                    # Each group of consecutive values represents frets for a string
                    # Assuming 21 frets per string (0-20) for 6 strings = 126 values
                    num_frets = 21  # Including open string (0)
                    num_strings = 6
                    
                    if len(features) == num_strings * num_frets:
                        # Format is likely [string1_frets, string2_frets, ...]
                        for s in range(num_strings):
                            string_start = s * num_frets
                            string_end = (s + 1) * num_frets
                            string_preds = features[string_start:string_end]
                            
                            # Find the highest probability fret
                            if np.max(string_preds) > 0.1:  # Confidence threshold
                                fret = np.argmax(string_preds)
                                if fret > 0:  # Skip if it's a 0 (open string)
                                    notes.append({
                                        'time': 0,  # Single time step
                                        'string': s,
                                        'fret': int(fret)
                                    })
                    else:
                        # If we don't know the exact format, make a best guess
                        # Split the features evenly among 6 strings
                        frets_per_string = len(features) // 6
                        for s in range(6):
                            start_idx = s * frets_per_string
                            end_idx = (s + 1) * frets_per_string
                            string_preds = features[start_idx:end_idx]
                            
                            if np.max(string_preds) > 0.1:
                                fret = np.argmax(string_preds) + 1  # +1 because fret 0 is open string
                                notes.append({
                                    'time': 0,
                                    'string': s,
                                    'fret': int(fret)
                                })
                else:
                    # Handle the case where rows might represent time steps
                    for t in range(predictions.shape[0]):
                        row_preds = predictions[t]
                        
                        # If we have 6 strings with 21 frets each (0-20)
                        # and flattened as [string1_frets, string2_frets, ...]
                        if len(row_preds) == 126:  # 6 strings * 21 frets
                            for s in range(6):
                                start_idx = s * 21
                                end_idx = (s + 1) * 21
                                string_preds = row_preds[start_idx:end_idx]
                                
                                if np.max(string_preds) > 0.1:
                                    fret = np.argmax(string_preds)
                                    if fret > 0:
                                        notes.append({
                                            'time': t * 0.25,
                                            'string': s,
                                            'fret': int(fret)
                                        })
                        else:
                            # If we don't know the exact format, make a best guess
                            frets_per_string = len(row_preds) // 6
                            for s in range(6):
                                start_idx = s * frets_per_string
                                end_idx = (s + 1) * frets_per_string
                                string_preds = row_preds[start_idx:end_idx]
                                
                                if np.max(string_preds) > 0.1:
                                    fret = np.argmax(string_preds) + 1
                                    notes.append({
                                        'time': t * 0.25,
                                        'string': s,
                                        'fret': int(fret)
                                    })
        elif isinstance(predictions, list):
            # If it's already a list of note dictionaries
            for note in predictions:
                if isinstance(note, dict) and 'string' in note and 'fret' in note:
                    # Already in the right format
                    notes.append(note)
                    
        # Sort by time
        notes.sort(key=lambda x: x.get('time', 0))
        return notes

    def format_note_as_tab(self, note_data: List[Dict[str, Any]], song_duration: float) -> str:
        """
        Simplified tablature generation for testing or when detailed analysis isn't available.
        
        Args:
            note_data: Note events with time, string, fret information
            song_duration: Total duration of the song in seconds
            
        Returns:
            ASCII tab notation as a string
        """
        # Default empty strings for 6 guitar strings
        lines = ["e|", "B|", "G|", "D|", "A|", "E|"]
        
        # Calculate roughly how many dashes we need
        total_chars = min(int(song_duration * 8), 1000)  # Limit to reasonable size
        
        # Fill with dashes initially
        for i in range(6):
            lines[i] += "-" * total_chars
        
        # Convert dictionary to tab: go through each note and place it
        for note in note_data:
            string_idx = note.get('string', 0)  # String number (0-5)
            fret_num = note.get('fret', 0)      # Fret number
            time = note.get('time', 0)          # Time in seconds
            
            # Calculate position in tab
            position = min(int(time * 8), total_chars - 1)
            
            # Place the fret number at the appropriate position and string
            if 0 <= string_idx <= 5 and 0 <= position < total_chars:
                # Convert the lines to lists to modify them
                line_list = list(lines[string_idx])
                fret_str = str(fret_num)
                
                # Place each digit of the fret number
                for i, digit in enumerate(fret_str):
                    if position + i + 1 < len(line_list):
                        line_list[position + i + 1] = digit
                
                # Convert back to string
                lines[string_idx] = ''.join(line_list)
        
        # Finalize the tab
        for i in range(6):
            lines[i] += "|"
        
        # Join all lines and return
        return "\n".join(lines)

    def generate_professional_tab(self, predictions: np.ndarray, bpm: int = 120) -> str:
        """
        Generate professional tab from model predictions
        Format matches reference metal tabs with:
        - Proper spacing
        - Measure separators
        - Timing indicators
        """
        # Convert predictions to fret positions
        frets = np.argmax(predictions, axis=1)
        
        # Create tab lines
        tab_lines = []
        for string_idx in range(6):
            line = self.STRING_NAMES[5-string_idx] + "|-"
            
            # Add frets with proper spacing
            for frame_idx in range(predictions.shape[0]):
                fret = frets[frame_idx, string_idx]
                if fret > 0:
                    line += f"-{fret}-"
                else:
                    line += "---"
                
                # Add measure separator every 4 beats
                if (frame_idx + 1) % (4 * bpm) == 0:
                    line += "|"
            
            tab_lines.append(line + "|")
        
        # Add header/footer like reference tab
        header = "TAB GENERATED BY OLITUNES\n"
        header += f"Tempo: {bpm} BPM\n"
        header += "="*80 + "\n"
        
        # Reverse order (high to low strings)
        return header + "\n".join(reversed(tab_lines)) + "\n" + "="*80

    def _create_fretboard_data(self, predictions: np.ndarray) -> List[Dict]:
        """Generate standard 6-string tablature visualization"""
        frames = []
        string_names = ['e', 'B', 'G', 'D', 'A', 'E']  # Standard string order
        
        # Create tab lines for each string
        for string_idx in range(6):
            frames.append({
                'time': 0,
                'type': 'string_header',
                'string': 5 - string_idx,
                'name': string_names[string_idx]
            })
        
        # Add notes with measure markers
        for frame_idx in range(predictions.shape[0]):
            for string_idx in range(6):
                fret = np.argmax(predictions[frame_idx, string_idx])
                if fret > 0:
                    frames.append({
                        'time': frame_idx * 0.1,
                        'string': 5 - string_idx,
                        'fret': fret,
                        'style': 'plain'
                    })
            
            # Add measure marker every 4 beats
            if frame_idx % 16 == 0:
                frames.append({
                    'time': frame_idx * 0.1,
                    'type': 'measure',
                    'value': '|'
                })
        
        return frames
