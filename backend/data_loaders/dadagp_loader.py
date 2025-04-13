"""
DadaGP Guitar Pro File Loader

Loads and processes Guitar Pro (.gp3, .gp4, .gp5) files from the DadaGP dataset.
"""
import os
import logging
import guitarpro
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DadaGPLoader:
    """Loader for DadaGP Guitar Pro files"""
    
    def __init__(self, gp_path: str):
        """
        Initialize the loader with path to Guitar Pro files
        
        Args:
            gp_path: Path to directory containing .gp files
        """
        self.gp_path = gp_path
        
    def load_file(self, filename: str) -> Optional[Dict]:
        """
        Load a single Guitar Pro file
        
        Args:
            filename: Name of the .gp file to load
            
        Returns:
            Dictionary containing extracted tablature data or None if failed
        """
        try:
            filepath = os.path.join(self.gp_path, filename)
            gp = guitarpro.parse(filepath)
            return self._extract_tab_data(gp)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None
    
    def load_all(self) -> List[Dict]:
        """
        Load all Guitar Pro files in the directory
        
        Returns:
            List of tablature data dictionaries
        """
        tabs = []
        for filename in os.listdir(self.gp_path):
            if filename.lower().endswith(('.gp3', '.gp4', '.gp5', '.gp')):
                tab_data = self.load_file(filename)
                if tab_data:
                    tabs.append(tab_data)
        return tabs
    
    def _extract_tab_data(self, gp) -> Dict:
        """
        Extract tablature data from a parsed Guitar Pro file
        
        Args:
            gp: Parsed GuitarPro file
            
        Returns:
            Dictionary containing:
            - metadata (title, artist, etc)
            - notes (list of notes with positions)
            - beats (rhythmic information)
            - measures (structural information)
        """
        metadata = {
            'title': gp.title,
            'artist': gp.artist,
            'album': gp.album,
            'copyright': gp.copyright,
            'tempo': gp.tempo,
            'key': gp.key.name if gp.key else None
        }
        
        notes = []
        beats = []
        measures = []
        
        for track in gp.tracks:
            for measure in track.measures:
                measures.append({
                    'number': measure.number,
                    'time_signature': f"{measure.timeSignature.numerator}/{measure.timeSignature.denominator.value}",
                    'marker': measure.marker.text if measure.marker else None
                })
                
                for voice in measure.voices:
                    for beat in voice.beats:
                        beats.append({
                            'duration': beat.duration.value,
                            'tuplet': beat.tuplet.enters if beat.tuplet else None,
                            'notes': len(beat.notes)
                        })
                        
                        for note in beat.notes:
                            notes.append({
                                'string': note.string,
                                'fret': note.value,
                                'duration': beat.duration.value,
                                'effects': {
                                    'hammer': note.effect.hammer,
                                    'pulloff': note.effect.pullOff,
                                    'slide': note.effect.slideType.name if note.effect.slideType else None,
                                    'bend': note.effect.bend is not None
                                }
                            })
        
        return {
            'metadata': metadata,
            'notes': notes,
            'beats': beats,
            'measures': measures,
            'tracks': len(gp.tracks)
        }
