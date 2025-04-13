"""
LilyPond Tab Generator for OliTunes

This module provides functionality to generate guitar tablature using LilyPond
from MIDI data or note events. It integrates with the existing OliTunes pipeline
to create high-quality, printable tablature.
"""

import os
import json
import tempfile
import subprocess
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

# Check if LilyPond is available
LILYPOND_PATH = "C:\\ProgramData\\lilypond-2.24.4\\bin\\lilypond.exe"

try:
    # First try with the specific path
    result = subprocess.run([LILYPOND_PATH, '--version'], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE,
                           text=True,
                           check=False)
    lilypond_available = result.returncode == 0
    
    # If that fails, try with the general command (for other systems)
    if not lilypond_available:
        result = subprocess.run(['lilypond', '--version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True,
                              check=False)
        lilypond_available = result.returncode == 0
    
    if lilypond_available:
        lilypond_version = result.stdout.split('\n')[0]
        logger.info(f"LilyPond available: {lilypond_version}")
    else:
        logger.warning("LilyPond not found. Tab generation will be limited.")
except Exception as e:
    lilypond_available = False
    logger.warning(f"Error checking for LilyPond: {e}")

class LilypondTabGenerator:
    """
    Generates guitar tablature using LilyPond from MIDI or note data.
    """
    
    def __init__(self):
        """Initialize the LilyPond tab generator."""
        self.initialized = lilypond_available
        
    def generate_tab_from_midi(self, midi_file: str, output_dir: Optional[str] = None,
                              output_format: str = 'pdf', 
                              include_standard_notation: bool = True) -> Dict[str, Any]:
        """
        Generate tablature from a MIDI file using LilyPond.
        
        Args:
            midi_file: Path to the MIDI file
            output_dir: Directory to save the output files (default: same as MIDI file)
            output_format: Output format ('pdf', 'png', or 'both')
            include_standard_notation: Whether to include standard notation above tablature
            
        Returns:
            Dictionary with paths to generated files and status information
        """
        if not self.initialized:
            return {
                'success': False,
                'error': "LilyPond is not available. Please install LilyPond."
            }
            
        if not os.path.exists(midi_file):
            return {
                'success': False,
                'error': f"MIDI file not found: {midi_file}"
            }
            
        if output_dir is None:
            output_dir = os.path.dirname(midi_file)
            
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Import MIDI converter to extract notes
            from midi_converter import MidiConverter
            converter = MidiConverter()
            notes = converter.midi_to_notes(midi_file)
            
            if not notes:
                return {
                    'success': False,
                    'error': "No notes found in MIDI file"
                }
                
            # Generate tablature from notes
            return self.generate_tab_from_notes(
                notes=notes,
                output_dir=output_dir,
                output_format=output_format,
                include_standard_notation=include_standard_notation,
                title=os.path.splitext(os.path.basename(midi_file))[0]
            )
            
        except Exception as e:
            error_msg = f"Error generating tablature from MIDI: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def generate_tab_from_notes(self, notes: List[Dict[str, Any]], 
                               output_dir: str,
                               output_format: str = 'pdf',
                               include_standard_notation: bool = True,
                               title: str = "Generated Tab") -> Dict[str, Any]:
        """
        Generate tablature from note data using LilyPond.
        
        Args:
            notes: List of note dictionaries with pitch, start_time, duration
            output_dir: Directory to save the output files
            output_format: Output format ('pdf', 'png', or 'both')
            include_standard_notation: Whether to include standard notation above tablature
            title: Title for the tablature
            
        Returns:
            Dictionary with paths to generated files and status information
        """
        if not self.initialized:
            return {
                'success': False,
                'error': "LilyPond is not available. Please install LilyPond."
            }
            
        if not notes:
            return {
                'success': False,
                'error': "No notes provided for tablature generation"
            }
            
        try:
            # Create a safe filename from the title
            safe_title = "".join([c if c.isalnum() else "_" for c in title])
            
            # Create LilyPond file path
            ly_file = os.path.join(output_dir, f"{safe_title}.ly")
            
            # Generate LilyPond code
            ly_code = self._generate_lilypond_code(
                notes=notes,
                title=title,
                include_standard_notation=include_standard_notation
            )
            
            # Write LilyPond file
            with open(ly_file, 'w') as f:
                f.write(ly_code)
                
            logger.info(f"LilyPond file created: {ly_file}")
            
            # Run LilyPond to generate output
            output_files = self._run_lilypond(
                ly_file=ly_file,
                output_format=output_format
            )
            
            if not output_files:
                return {
                    'success': False,
                    'error': "LilyPond failed to generate output files"
                }
                
            return {
                'success': True,
                'ly_file': ly_file,
                'output_files': output_files,
                'format': output_format
            }
            
        except Exception as e:
            error_msg = f"Error generating tablature: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def _generate_lilypond_code(self, notes: List[Dict[str, Any]], 
                               title: str,
                               include_standard_notation: bool = True) -> str:
        """
        Generate LilyPond code for guitar tablature.
        
        Args:
            notes: List of note dictionaries
            title: Title for the tablature
            include_standard_notation: Whether to include standard notation
            
        Returns:
            LilyPond code as a string
        """
        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda x: x['start_time'])
        
        # Map MIDI pitches to string/fret combinations
        string_tunings = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4
        
        # Create a list of note events with string/fret information
        tab_notes = []
        
        for note in sorted_notes:
            # Get pitch (either directly or from note name)
            if 'pitch' in note:
                pitch = note['pitch']
            elif 'note' in note:
                # Try to convert note name to MIDI pitch
                try:
                    from pretty_midi import note_name_to_number
                    pitch = note_name_to_number(note['note'])
                except:
                    try:
                        import librosa
                        pitch = librosa.note_to_midi(note['note'])
                    except:
                        logger.warning(f"Could not convert note name: {note['note']}")
                        continue
            else:
                logger.warning("Note missing pitch information")
                continue
                
            # Find best string/fret combination
            string, fret = self._map_pitch_to_guitar(pitch)
            
            if string is not None and fret is not None:
                tab_notes.append({
                    'string': string,
                    'fret': fret,
                    'pitch': pitch,
                    'start_time': note['start_time'],
                    'duration': note.get('duration', 0.25)
                })
        
        # Group notes by start time to handle chords
        time_groups = {}
        for note in tab_notes:
            # Round to nearest 16th note for quantization
            time = round(note['start_time'] * 16) / 16
            if time not in time_groups:
                time_groups[time] = []
            time_groups[time].append(note)
        
        # Generate LilyPond code
        ly_code = f"""\\version "2.22.0"
\\header {{
  title = "{title}"
  tagline = "Generated by OliTunes"
}}

"""
        
        # Add tablature music
        ly_code += "music = {\n"
        
        # Sort times and process each time group
        for time in sorted(time_groups.keys()):
            notes_at_time = time_groups[time]
            
            # If multiple notes at same time, create a chord
            if len(notes_at_time) > 1:
                # Start chord
                ly_code += "  <"
                
                # Add each note in the chord
                for note in sorted(notes_at_time, key=lambda x: x['string']):
                    string_num = 6 - note['string']  # LilyPond counts strings from 1 (high E) to 6 (low E)
                    ly_code += f" {note['fret']}-{string_num}"
                
                # End chord and add duration
                duration = self._get_lilypond_duration(notes_at_time[0]['duration'])
                ly_code += f">{duration} "
            else:
                # Single note
                note = notes_at_time[0]
                string_num = 6 - note['string']  # LilyPond counts strings from 1 (high E) to 6 (low E)
                duration = self._get_lilypond_duration(note['duration'])
                ly_code += f"  {note['fret']}-{string_num}{duration} "
            
            # Add bar lines every 4 beats
            if int(time) % 4 == 0 and time > 0:
                ly_code += "|\n  "
        
        # Close music block
        ly_code += "\n}\n\n"
        
        # Add tablature and score settings
        if include_standard_notation:
            ly_code += """\\score {
  <<
    \\new Staff {
      \\clef "treble_8"
      \\music
    }
    \\new TabStaff {
      \\tabFullNotation
      \\music
    }
  >>
  \\layout { }
  \\midi { }
}
"""
        else:
            ly_code += """\\score {
  \\new TabStaff {
    \\tabFullNotation
    \\music
  }
  \\layout { }
  \\midi { }
}
"""
        
        return ly_code
    
    def _map_pitch_to_guitar(self, midi_pitch: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Map MIDI pitch to guitar string and fret.
        
        Args:
            midi_pitch: MIDI pitch value
            
        Returns:
            Tuple of (string, fret) or (None, None) if mapping not possible
        """
        # Standard guitar tuning (from low to high): E2(40), A2(45), D3(50), G3(55), B3(59), E4(64)
        string_tunings = [40, 45, 50, 55, 59, 64]
        
        # Try to find the best string/fret combination
        best_string = None
        best_fret = None
        best_distance = float('inf')
        
        for string_idx, base_pitch in enumerate(string_tunings):
            # Calculate fret number
            fret = midi_pitch - base_pitch
            
            # Check if note is playable on this string
            if 0 <= fret <= 24:  # Assuming 24 frets max
                # Prefer lower frets and lower strings
                distance = fret + (string_idx * 0.5)  # Slight preference for lower strings
                if distance < best_distance:
                    best_string = string_idx
                    best_fret = fret
                    best_distance = distance
        
        return best_string, best_fret
    
    def _get_lilypond_duration(self, duration: float) -> str:
        """
        Convert duration in seconds to LilyPond duration notation.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            LilyPond duration string
        """
        # Approximate duration to common note values
        # Assuming quarter note = 1 beat at 120 BPM (0.5 seconds)
        duration_beats = duration * 2  # Convert seconds to beats
        
        if duration_beats >= 4:
            return "1"  # Whole note
        elif duration_beats >= 2:
            return "2"  # Half note
        elif duration_beats >= 1:
            return "4"  # Quarter note
        elif duration_beats >= 0.5:
            return "8"  # Eighth note
        elif duration_beats >= 0.25:
            return "16"  # Sixteenth note
        else:
            return "32"  # Thirty-second note
    
    def _run_lilypond(self, ly_file: str, output_format: str = 'pdf') -> Dict[str, Any]:
        """
        Run LilyPond on the generated .ly file to create tablature
        
        Args:
            ly_file: Path to the .ly file
            output_format: Output format ('pdf', 'png', or 'both')
            
        Returns:
            Dictionary with result information
        """
        if not self.initialized:
            return {'success': False, 'error': 'LilyPond not available'}
        
        # Get output directory and base filename
        output_dir = os.path.dirname(ly_file)
        base_filename = os.path.splitext(os.path.basename(ly_file))[0]
        
        try:
            # Build command based on output format
            if output_format == 'png':
                cmd = [LILYPOND_PATH, '-fpng', '--png', '-dresolution=300', '-o', os.path.join(output_dir, base_filename), ly_file]
            else:  # Default to PDF
                cmd = [LILYPOND_PATH, '-fpdf', '-o', os.path.join(output_dir, base_filename), ly_file]
            
            # Run LilyPond
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                output_files = []
                
                # Check for generated files
                if output_format == 'png' or output_format == 'both':
                    png_file = os.path.join(output_dir, f"{base_filename}.png")
                    if os.path.exists(png_file):
                        output_files.append(png_file)
                
                if output_format == 'pdf' or output_format == 'both':
                    pdf_file = os.path.join(output_dir, f"{base_filename}.pdf")
                    if os.path.exists(pdf_file):
                        output_files.append(pdf_file)
                
                return {
                    'success': True,
                    'output_files': output_files,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                return {
                    'success': False,
                    'error': f"LilyPond failed with code {result.returncode}",
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
        except Exception as e:
            logger.error(f"Error running LilyPond: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Standalone function for direct import
def generate_tab_from_midi(midi_file: str, output_dir: Optional[str] = None,
                         output_format: str = 'pdf',
                         include_standard_notation: bool = True) -> Dict[str, Any]:
    """
    Generate tablature from a MIDI file using LilyPond.
    
    Args:
        midi_file: Path to the MIDI file
        output_dir: Directory to save the output files
        output_format: Output format ('pdf', 'png', or 'both')
        include_standard_notation: Whether to include standard notation
        
    Returns:
        Dictionary with paths to generated files and status information
    """
    generator = LilypondTabGenerator()
    return generator.generate_tab_from_midi(
        midi_file=midi_file,
        output_dir=output_dir,
        output_format=output_format,
        include_standard_notation=include_standard_notation
    )


def generate_tab_from_notes(notes: List[Dict[str, Any]], output_dir: str,
                          output_format: str = 'pdf',
                          include_standard_notation: bool = True,
                          title: str = "Generated Tab") -> Dict[str, Any]:
    """
    Generate tablature from note data using LilyPond.
    
    Args:
        notes: List of note dictionaries with pitch, start_time, duration
        output_dir: Directory to save the output files
        output_format: Output format ('pdf', 'png', or 'both')
        include_standard_notation: Whether to include standard notation
        title: Title for the tablature
        
    Returns:
        Dictionary with paths to generated files and status information
    """
    generator = LilypondTabGenerator()
    return generator.generate_tab_from_notes(
        notes=notes,
        output_dir=output_dir,
        output_format=output_format,
        include_standard_notation=include_standard_notation,
        title=title
    )
