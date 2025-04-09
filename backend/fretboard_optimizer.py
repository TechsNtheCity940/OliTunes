import numpy as np
from typing import List, Dict, Tuple, Optional

class FretboardOptimizer:
    """Optimizes guitar fretboard positions for ergonomic playability."""
    
    STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # MIDI values for E2, A2, D3, G3, B3, E4
    
    def __init__(self, tuning: List[int] = None):
        self.tuning = tuning if tuning else self.STANDARD_TUNING
        self.max_fret = 24  # Typical guitar fretboard limit
        self.hand_span = 4  # Typical comfortable fret span
        
    def optimize_frame(self, frame: np.ndarray, prev_frame: Optional[np.ndarray], key: str) -> np.ndarray:
        """
        Optimize a single frame of fret predictions.
        
        Args:
            frame: Array of shape (6,) with fret values for each string
            prev_frame: Previous frame for temporal consistency, or None
            key: Musical key for context (e.g., 'C', 'G')
            
        Returns:
            Optimized frame as a numpy array
        """
        # Convert frame to MIDI notes
        midi_notes = self._frets_to_midi(frame)
        possible_positions = self._get_all_positions(midi_notes)
        
        if not possible_positions:
            return frame  # Return original if no optimization possible
            
        # Score positions based on playability and context
        scored_positions = self._score_positions(possible_positions, prev_frame, key)
        best_position = scored_positions[0]['frets']
        
        return np.array(best_position, dtype=np.int32)
    
    def _frets_to_midi(self, frets: np.ndarray) -> List[int]:
        """Convert fret values to MIDI notes based on tuning."""
        midi_notes = []
        for string_idx, fret in enumerate(frets):
            if fret >= 0:  # Ignore invalid frets
                midi_notes.append(self.tuning[string_idx] + fret)
        return midi_notes
    
    def _get_all_positions(self, midi_notes: List[int]) -> List[Dict[str, Any]]:
        """Get all possible fretboard positions for a set of MIDI notes."""
        positions = []
        
        # For each note, find all possible (string, fret) combinations
        note_positions = []
        for note in midi_notes:
            note_pos = []
            for string_idx, open_string in enumerate(self.tuning):
                fret = note - open_string
                if 0 <= fret <= self.max_fret:
                    note_pos.append((string_idx, fret))
            note_positions.append(note_pos)
        
        # Generate all combinations (simple for now; could use dynamic programming for chords)
        from itertools import product
        for combo in product(*note_positions):
            # Check if each string is used at most once
            strings_used = [pos[0] for pos in combo]
            if len(set(strings_used)) == len(strings_used):  # No string overlap
                frets = [-1] * 6  # -1 means no note
                for (string, fret) in combo:
                    frets[string] = fret
                positions.append({'frets': frets})
        
        return positions
    
    def _score_positions(self, positions: List[Dict[str, Any]], prev_frame: Optional[np.ndarray], key: str) -> List[Dict[str, Any]]:
        """Score positions based on ergonomic and musical factors."""
        for pos in positions:
            frets = pos['frets']
            score = 0.0
            
            # Penalize high frets
            active_frets = [f for f in frets if f >= 0]
            if active_frets:
                score += sum(f for f in active_frets if f > 0) * 0.5  # Favor open strings
            
            # Penalize large fret spans
            if len(active_frets) > 1:
                fret_span = max(active_frets) - min(active_frets)
                if fret_span > self.hand_span:
                    score += (fret_span - self.hand_span) * 2.0
            
            # Favor positions near previous frame
            if prev_frame is not None:
                prev_frets = [f for f in prev_frame if f >= 0]
                if prev_frets and active_frets:
                    prev_mean = sum(prev_frets) / len(prev_frets)
                    curr_mean = sum(active_frets) / len(active_frets)
                    score += abs(curr_mean - prev_mean) * 1.0
            
            # Bonus for key-relevant positions (simplified)
            if key in ['C', 'G', 'D', 'A', 'E']:  # Open chord-friendly keys
                if any(f in [0, 2, 3, 5] for f in active_frets):  # Common open positions
                    score -= 2.0
            
            pos['score'] = score
        
        # Sort by score (lower is better)
        return sorted(positions, key=lambda x: x['score'])

# Usage in UnifiedTabProcessor
def optimize_fretboard(self, predictions: np.ndarray, key: str = 'C') -> np.ndarray:
    optimizer = FretboardOptimizer()
    optimized = np.zeros_like(predictions)
    for frame_idx in range(predictions.shape[0]):
        frame = predictions[frame_idx]
        prev_frame = predictions[frame_idx - 1] if frame_idx > 0 else None
        optimized[frame_idx] = optimizer.optimize_frame(frame, prev_frame, key)
    return optimized