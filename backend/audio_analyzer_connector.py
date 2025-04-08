import os
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from audio_analysis import AudioAnalyzer
from audio_analysis_advanced import AdvancedAudioAnalyzer
from note_visualization import NoteVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAudioAnalyzer:
    """
    An enhanced audio analyzer that integrates the original AudioAnalyzer
    with the new AdvancedAudioAnalyzer for improved pitch detection.
    """
    
    def __init__(self, audio_path: str):
        """Initialize with both analyzers"""
        self.audio_path = audio_path
        
        # Initialize both analyzer classes
        self.standard_analyzer = AudioAnalyzer(audio_path)
        self.advanced_analyzer = AdvancedAudioAnalyzer(audio_path)
        
        # Copy useful properties from standard analyzer
        self.y = self.standard_analyzer.y
        self.sr = self.standard_analyzer.sr
        self.cache_dir = self.standard_analyzer.cache_dir
    
    def detect_guitar_notes(self) -> List[Dict[str, Any]]:
        """
        Enhanced guitar note detection using the advanced algorithm.
        Falls back to standard detection if advanced detection fails.
        """
        logger.info("Using advanced note detection algorithm...")
        
        try:
            # Try using the advanced algorithm first
            notes = self.advanced_analyzer.detect_notes()
            logger.info(f"Advanced detection found {len(notes)} notes")
            
            if notes and len(notes) > 0:
                return notes
            
            # If no notes found, fall back to standard method
            logger.warning("Advanced detection returned no notes, falling back to standard algorithm")
        except Exception as e:
            logger.warning(f"Advanced detection failed: {str(e)}, falling back to standard algorithm")
        
        # Fallback to standard analyzer
        return self.standard_analyzer.detect_guitar_notes()
    
    def generate_tablature(self) -> Dict[str, Any]:
        """
        Generate enhanced guitar tablature from detected notes
        """
        logger.info("Generating enhanced tablature...")
        
        try:
            # Get notes with advanced detection
            notes = self.detect_guitar_notes()
            
            # Check if the advanced analyzer has mapped notes
            if all('string' in note and 'fret' in note for note in notes):
                # Notes already have string/fret info
                mapped_notes = notes
            else:
                # Map notes to fretboard if needed
                logger.info("Mapping notes to optimal fretboard positions...")
                mapped_notes = self.advanced_analyzer._map_notes_to_fretboard(notes)
            
            # Generate ASCII tablature
            logger.info("Generating tablature display...")
            tab_text = self.advanced_analyzer.generate_tablature(mapped_notes)
            
            # Create visualization-friendly structure
            tab_data = self._format_tablature_for_display(mapped_notes, tab_text)
            
            return tab_data
        
        except Exception as e:
            logger.warning(f"Enhanced tablature generation failed: {str(e)}, falling back to standard method")
            
            # Fallback to standard analyzer
            return self.standard_analyzer.generate_tablature()
    
    def _format_tablature_for_display(self, notes: List[Dict[str, Any]], tab_ascii: str) -> Dict[str, Any]:
        """
        Format the tablature data for the frontend to display
        """
        # Get audio duration
        duration = len(self.y) / self.sr
        
        # Create structured tablature data
        tab_data = {
            'tablature': tab_ascii.split('\n') if isinstance(tab_ascii, str) else [tab_ascii],
            'bars': self._create_bars_from_notes(notes),
            'totalDuration': float(duration),
            'gridResolution': 0.125,  # 16th note resolution
            'notePositions': []
        }
        
        # Add note positions for UI display
        for note in notes:
            if 'string' in note and 'fret' in note:
                tab_data['notePositions'].append({
                    'note': note['note'],
                    'time': float(note['time']),
                    'duration': float(note['duration']),
                    'string': note['string'],
                    'fret': note['fret'],
                    'confidence': float(note.get('confidence', 0.8))
                })
        
        return tab_data
    
    def _create_bars_from_notes(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create measure and bar structures from notes for tablature display
        """
        if not notes:
            return []
            
        # Sort notes by time
        notes = sorted(notes, key=lambda x: x['time'])
        
        # Get audio duration
        duration = len(self.y) / self.sr
        
        # Analyze tempo (using standard analyzer)
        try:
            tempo_info = self.standard_analyzer.analyze_tempo()
            tempo = tempo_info['tempo']
        except:
            tempo = 120  # Default tempo if analysis fails
        
        # Try to get time signature
        try:
            time_sig = self.standard_analyzer.detect_time_signature()
        except:
            time_sig = (4, 4)  # Default to 4/4
        
        # Setup measure parameters
        beats_per_measure = time_sig[0]
        beat_duration = 60.0 / tempo
        measure_duration = beats_per_measure * beat_duration
        
        # Calculate total number of measures
        num_measures = max(1, int(duration / measure_duration))
        
        # Group measures into bars (4 measures per bar by default)
        measures_per_bar = 4
        num_bars = (num_measures + measures_per_bar - 1) // measures_per_bar
        
        bars = []
        
        # Generate bars and measures
        for bar_idx in range(num_bars):
            bar_start = bar_idx * measures_per_bar * measure_duration
            bar_end = min(duration, (bar_idx + 1) * measures_per_bar * measure_duration)
            
            bar = {
                'barNumber': bar_idx + 1,
                'startTime': float(bar_start),
                'endTime': float(bar_end),
                'timeSignature': f"{time_sig[0]}/{time_sig[1]}",
                'measures': []
            }
            
            # Generate measures in this bar
            for m in range(measures_per_bar):
                measure_idx = bar_idx * measures_per_bar + m
                if measure_idx >= num_measures:
                    break
                    
                m_start = measure_idx * measure_duration
                m_end = min(duration, (measure_idx + 1) * measure_duration)
                
                # Find notes in this measure
                measure_notes = [n for n in notes if m_start <= n['time'] < m_end]
                
                # Create empty tab lines for each string with consistent width
                tab_width = int(beats_per_measure * 4)  # 4 chars per beat for consistent spacing
                tab_lines = [
                    f"e|{'-' * tab_width}|",
                    f"B|{'-' * tab_width}|",
                    f"G|{'-' * tab_width}|", 
                    f"D|{'-' * tab_width}|",
                    f"A|{'-' * tab_width}|",
                    f"E|{'-' * tab_width}|"
                ]
                
                # Add notes to tab lines
                for note in measure_notes:
                    if 'string' not in note or 'fret' not in note:
                        continue
                        
                    string_idx = note['string']
                    fret_str = str(note['fret'])
                    
                    # Calculate position in the tab
                    relative_time = note['time'] - m_start
                    relative_pos = int(relative_time / (m_end - m_start) * tab_width)
                    
                    # Make sure position is within tab width
                    if 0 <= relative_pos < tab_width:
                        # Replace hyphen with fret number, adjusting for multi-digit frets
                        line_chars = list(tab_lines[string_idx])
                        
                        # Position after the | character
                        pos = relative_pos + 2
                        
                        # Handle fret number placement, ensuring it doesn't overflow
                        if len(fret_str) == 1:
                            if pos < len(line_chars):
                                line_chars[pos] = fret_str
                        else:
                            # For multi-digit frets, need multiple positions
                            for i, digit in enumerate(fret_str):
                                if pos + i < len(line_chars):
                                    line_chars[pos + i] = digit
                        
                        tab_lines[string_idx] = ''.join(line_chars)
                
                # Create measure data structure
                beat_positions = [float(m_start + i * beat_duration) for i in range(beats_per_measure + 1)]
                
                measure = {
                    'measureNumber': m + 1,
                    'startTime': float(m_start),
                    'endTime': float(m_end),
                    'tabLines': tab_lines,
                    'beatPositions': beat_positions
                }
                
                bar['measures'].append(measure)
            
            bars.append(bar)
        
        return bars
    
    # Delegate methods to standard analyzer for compatibility
    def detect_chords(self) -> List[Dict[str, Any]]:
        return self.standard_analyzer.detect_chords()
    
    def detect_key(self) -> Dict[str, Any]:
        return self.standard_analyzer.detect_key()
    
    def detect_time_signature(self) -> Tuple[int, int]:
        return self.standard_analyzer.detect_time_signature()
    
    def extract_lyrics(self) -> Dict[str, Any]:
        return self.standard_analyzer.extract_lyrics()
    
    def map_notes_to_fretboard(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map notes to optimal positions on guitar fretboard"""
        return self.advanced_analyzer._map_notes_to_fretboard(notes)
        
    def generate_visualizations(self, output_name: str = None) -> Dict[str, str]:
        """
        Generate RipX-style piano roll and spectral visualizations
        
        Args:
            output_name: Base name for output files (defaults to audio filename)
            
        Returns:
            Dictionary with paths to visualization images
        """
        # Default to audio filename if none provided
        if output_name is None:
            output_name = os.path.splitext(os.path.basename(self.audio_path))[0]
            
        # Detect notes if needed
        notes = self.detect_guitar_notes()
        
        # Create visualizer
        visualizer = NoteVisualizer(self.y, self.sr, notes)
        
        # Generate visualizations
        vis_paths = visualizer.generate_visualizations(output_name)
        
        return vis_paths
