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
