import librosa
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from scipy import signal
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAudioAnalyzer:
    """Enhanced audio analysis for guitar notes and fretboard mapping"""
    
    # Standard guitar tuning
    STANDARD_TUNING = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']  # Low to high
    
    # Pitch classes for easy conversion
    PITCH_CLASSES = {
        'C': 0, 'C#': 1, 'Db': 1,
        'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4,
        'F': 5, 'F#': 6, 'Gb': 6,
        'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10,
        'B': 11
    }
    
    # Common chord positions and finger patterns
    COMMON_POSITIONS = {
        # Format: note: [(string, fret), (string, fret), ...]
        'E': [(0, 0), (1, 5), (2, 9), (3, 14), (4, 19), (5, 24)],
        'A': [(0, 5), (1, 0), (2, 2), (3, 7), (4, 12), (5, 17)],
        'D': [(0, 10), (1, 5), (2, 0), (3, 2), (4, 7), (5, 12)],
        'G': [(0, 3), (1, 10), (2, 5), (3, 0), (4, 2), (5, 7)],
        'B': [(0, 7), (1, 2), (2, 8), (3, 4), (4, 0), (5, 2)],
        'C': [(0, 8), (1, 3), (2, 10), (3, 5), (4, 1), (5, 3)],
        'F': [(0, 1), (1, 6), (2, 10), (3, 3), (4, 8), (5, 13)]
    }
    
    def __init__(self):
        """Initialize the enhanced audio analyzer"""
        self.string_midi_values = [librosa.note_to_midi(note) for note in self.STANDARD_TUNING]
    
    def enhance_note_detection(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance note detection by filling gaps and refining timing"""
        if not notes:
            return []
            
        # Sort by time
        notes.sort(key=lambda x: x['time'])
        
        # Remove notes with too low confidence
        notes = [note for note in notes if note.get('confidence', 0) > 0.25]
        
        # Ensure note durations don't overlap next note
        for i in range(len(notes) - 1):
            next_start = notes[i+1]['time']
            if notes[i]['time'] + notes[i]['duration'] > next_start:
                # Trim current note to end slightly before next starts
                new_duration = next_start - notes[i]['time'] - 0.01
                if new_duration > 0.05:  # Keep reasonable minimum duration
                    notes[i]['duration'] = new_duration
        
        return notes
    
    def map_notes_to_fretboard(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map notes to optimal positions on the guitar fretboard with smart fingering
        
        This function uses sophisticated algorithms to determine the most playable
        tablature fingerings based on guitar playing techniques.
        
        Args:
            notes: List of note dictionaries with 'note', 'time', 'duration', etc.
            
        Returns:
            List of note dictionaries with added 'string' and 'fret' attributes
        """
        if not notes:
            return []
        
        # Create copy of notes and enhance them
        notes = self.enhance_note_detection(notes.copy())
        mapped_notes = []
        
        # Current hand position tracking
        current_position = None  # (mean_fret, mean_string)
        position_memory = []  # Track last few positions for consistency
        
        # Process each note
        for i, note in enumerate(notes):
            # Skip notes without a proper name
            if 'note' not in note or not note['note']:
                continue
            
            # Create a copy of the note to modify
            note_copy = note.copy()
            note_name = note['note']
            
            # Get all possible positions for this note
            positions = self._get_all_fret_positions(note_name)
            if not positions:
                logger.warning(f"Could not find fretboard position for note: {note_name}")
                continue
            
            # Get previous note's position if available
            prev_position = None
            if i > 0 and 'string' in mapped_notes[-1] and 'fret' in mapped_notes[-1]:
                prev_position = (mapped_notes[-1]['string'], mapped_notes[-1]['fret'])
            
            # Score positions based on playability
            scored_positions = self._score_positions(positions, prev_position, current_position, position_memory)
            
            # Choose the best position
            best_position = scored_positions[0]
            best_string, best_fret = best_position['string'], best_position['fret']
            
            # Update note with string and fret info
            note_copy['string'] = best_string
            note_copy['fret'] = best_fret
            mapped_notes.append(note_copy)
            
            # Update current hand position (weighted average of last few positions)
            if best_fret > 0:  # Only update position for non-open strings
                # Add to position memory
                position_memory.append((best_string, best_fret))
                # Keep only last 4 positions
                position_memory = position_memory[-4:]
                
                # Calculate current hand position as weighted average
                if position_memory:
                    frets = [pos[1] for pos in position_memory if pos[1] > 0]
                    strings = [pos[0] for pos in position_memory]
                    if frets:
                        # Weight recent positions more highly
                        weights = [0.4, 0.3, 0.2, 0.1][-len(frets):]
                        mean_fret = sum(f * w for f, w in zip(frets, weights)) / sum(weights)
                        mean_string = sum(s * w for s, w in zip(strings, weights)) / sum(weights)
                        current_position = (mean_string, mean_fret)
        
        return mapped_notes
    
    def _get_all_fret_positions(self, note_name: str) -> List[Dict[str, Any]]:
        """Get all possible fretboard positions for a given note
        
        Args:
            note_name: The note name (e.g., "C#4")
            
        Returns:
            List of dictionaries containing 'string' and 'fret' attributes
        """
        positions = []
        
        # Handle octave and note parsing
        try:
            # Try standard format like "C#4"
            pitch_class = note_name[:-1]
            octave = int(note_name[-1])
            
            # Handle accidentals standardization (flatten sharps/flats representation)
            if pitch_class in ['Db', 'D♭']:
                pitch_class = 'C#'
            elif pitch_class in ['Eb', 'E♭']:
                pitch_class = 'D#'
            elif pitch_class in ['Gb', 'G♭']:
                pitch_class = 'F#'
            elif pitch_class in ['Ab', 'A♭']:
                pitch_class = 'G#'
            elif pitch_class in ['Bb', 'B♭']:
                pitch_class = 'A#'
            
            full_note = f"{pitch_class}{octave}"
        except (IndexError, ValueError):
            # If parsing fails, make a best guess
            if len(note_name) <= 2:  # Just a pitch class like "C" or "F#"
                full_note = f"{note_name}4"  # Assume octave 4
            else:
                # Try to extract octave if there's a digit somewhere
                for i, c in enumerate(note_name):
                    if c.isdigit():
                        try:
                            pitch_class = note_name[:i]
                            octave = int(note_name[i:])
                            full_note = f"{pitch_class}{octave}"
                            break
                        except:
                            full_note = f"{note_name}4"  # Default fallback
                else:
                    full_note = f"{note_name}4"  # Default fallback
        
        # Calculate MIDI note number
        try:
            midi_note = librosa.note_to_midi(full_note)
        except:
            # If conversion fails, try common note name format
            try:
                # Strip any non-standard characters and try again
                clean_name = ''.join(c for c in note_name if c.isalnum() or c in '#b')
                if len(clean_name) <= 2:  # Just a pitch class
                    midi_note = librosa.note_to_midi(f"{clean_name}4")
                else:
                    midi_note = librosa.note_to_midi(clean_name)
            except:
                # If all else fails, return empty list
                return positions
        
        # Find all possible positions on each string
        for string_idx, string_midi in enumerate(self.string_midi_values):
            # Calculate fret number (how many semitones up from open string)
            fret = midi_note - string_midi
            
            # Check if it's in a playable range (0-24)
            if 0 <= fret <= 24:
                # Calculate distance from nut (higher means harder to play)
                # Open strings (fret 0) are easiest
                if fret == 0:
                    difficulty = 0
                else:
                    # Higher frets and extreme strings are harder
                    string_difficulty = abs(string_idx - 2.5) * 0.5  # Middle strings are easier
                    fret_difficulty = fret * 0.8  # Higher frets are harder
                    difficulty = string_difficulty + fret_difficulty
                
                positions.append({
                    'string': string_idx,
                    'fret': fret,
                    'difficulty': difficulty
                })
        
        # Sort by difficulty (easier positions first)
        positions.sort(key=lambda x: x['difficulty'])
        
        # If we have no positions but we can parse the pitch class, check common positions
        if not positions and pitch_class in self.COMMON_POSITIONS:
            # Convert from common positions to our string numbering
            for string, fret in self.COMMON_POSITIONS[pitch_class]:
                # Our strings are indexed from low E to high E (0-5)
                positions.append({
                    'string': string,
                    'fret': fret,
                    'difficulty': fret * 0.8
                })
            
            # Sort by difficulty
            positions.sort(key=lambda x: x['difficulty'])
        
        # If still nothing, return empty list
        if not positions:
            logger.warning(f"No fretboard positions found for note: {note_name}")
        
        return positions
    
    def _score_positions(self, positions: List[Dict[str, Any]], prev_position: Optional[Tuple[int, int]], 
                         current_position: Optional[Tuple[float, float]], position_memory: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Score fretboard positions based on playability
        
        Args:
            positions: List of possible positions, each with 'string' and 'fret'
            prev_position: Previous note position as (string, fret) or None
            current_position: Current hand position as (mean_string, mean_fret) or None
            position_memory: List of recent (string, fret) positions
            
        Returns:
            List of positions sorted by score (best first)
        """
        # Deep copy positions to avoid modifying originals
        scored_positions = [p.copy() for p in positions]
        
        # Default hand span (how many frets a hand can comfortably cover)
        hand_span = 4
        
        # If we have position memory, calculate actual hand span
        if len(position_memory) >= 2:
            active_frets = [p[1] for p in position_memory if p[1] > 0]
            if active_frets:
                hand_span = max(4, max(active_frets) - min(active_frets) + 2)
        
        # Set default scores based on difficulty
        for pos in scored_positions:
            # Start with the base difficulty
            pos['score'] = pos['difficulty']
            
            # Strong preference for open strings
            if pos['fret'] == 0:
                pos['score'] -= 3.0
        
        # If we have a previous position, adjust scores for continuity
        if prev_position:
            prev_string, prev_fret = prev_position
            
            for pos in scored_positions:
                # Penalize string changes
                string_change = abs(pos['string'] - prev_string)
                pos['score'] += string_change * 1.0
                
                # Penalize large fret jumps (but not for open strings)
                if pos['fret'] > 0 and prev_fret > 0:
                    fret_jump = abs(pos['fret'] - prev_fret)
                    # Small jumps are fine, large jumps are harder
                    if fret_jump <= 2:
                        pos['score'] += fret_jump * 0.2
                    else:
                        pos['score'] += fret_jump * 1.0
        
        # If we have a current hand position, favor notes within that position
        if current_position:
            mean_string, mean_fret = current_position
            
            for pos in scored_positions:
                # Skip open strings
                if pos['fret'] == 0:
                    continue
                
                # Distance from current hand position
                string_dist = abs(pos['string'] - mean_string)
                fret_dist = abs(pos['fret'] - mean_fret)
                
                # Heavy penalty for positions outside current hand span
                if fret_dist > hand_span:
                    pos['score'] += (fret_dist - hand_span) * 2.0
                
                # Moderate penalty for string changes within position
                pos['score'] += string_dist * 0.5
        
        # Sort by score (lower is better)
        scored_positions.sort(key=lambda x: x['score'])
        
        return scored_positions
