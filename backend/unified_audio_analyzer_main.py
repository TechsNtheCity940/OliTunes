"""
Unified Audio Analyzer - Main Integration Module
This module integrates all parts of the UnifiedAudioAnalyzer class and provides
a complete API for audio analysis in OliTunes.
"""

# Import basic utilities
import librosa
import numpy as np
import logging
import os
import json
import soundfile as sf

# Set up logging
logger = logging.getLogger(__name__)

# Import main analyzer and guitar methods
from audio_imports import *
from unified_audio_analyzer import UnifiedAudioAnalyzer
from unified_audio_analyzer_part2 import add_note_detection_methods

# Try to import optional dependencies
try:
    import pedalboard
except ImportError:
    logger.warning("Pedalboard not available. Some effects processing features will be limited.")

try:
    import pretty_midi
except ImportError:
    logger.warning("PrettyMIDI not available. MIDI export features will be limited.")

# Check for essentia
ESSENTIA_AVAILABLE = False
try:
    import essentia.standard
    ESSENTIA_AVAILABLE = True
except ImportError:
    logger.warning("Essentia not available in unified_audio_analyzer_main. Some advanced analysis features will be limited.")

# Add guitar note detection methods
UnifiedAudioAnalyzer = add_note_detection_methods(UnifiedAudioAnalyzer)

# Add chord detection methods
def add_chord_detection_methods(UnifiedAudioAnalyzer):
    """Add chord detection methods to the UnifiedAudioAnalyzer class."""
    
    def detect_chords(self):
        """
        Detect chords in audio using chromagram analysis.
        
        Returns:
            List of chord objects with timing and label information
        """
        # Ensure audio is loaded
        self._ensure_audio_loaded()
        
        # Try advanced chord detection first
        try:
            chords = self.detect_chords_advanced()
            if chords and len(chords) > 0:
                return chords
        except Exception as e:
            logger.warning(f"Advanced chord detection failed: {e}")
        
        # Fallback to basic chord detection
        try:
            # Compute the chromagram
            chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
            
            # Enhance harmonic content
            harmonic, _ = librosa.effects.hpss(self.y)
            chroma_harmonic = librosa.feature.chroma_cqt(y=harmonic, sr=self.sr)
            
            # Beat synchronous chroma
            tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
            beat_chroma = librosa.util.sync(chroma_harmonic, beats, aggregate=np.median)
            
            # Convert to chord labels
            chord_labels = []
            
            # Use simple chord templates
            maj_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])  # C major: C, E, G
            min_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])  # C minor: C, Eb, G
            
            # Chord types and their templates
            chord_types = {
                "": maj_template,      # Major
                "m": min_template,     # Minor
            }
            
            # Note names
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            # Detect chords for each beat segment
            chord_results = []
            prev_chord = None
            start_time = 0
            
            for i, beat in enumerate(beats):
                beat_time = librosa.frames_to_time(beat, sr=self.sr)
                
                if i < len(beat_chroma[0]):
                    # Get the chroma for this beat
                    chroma_vector = beat_chroma[:, i]
                    
                    # Normalize
                    if np.sum(chroma_vector) > 0:
                        chroma_vector = chroma_vector / np.sum(chroma_vector)
                    
                    # Find the best matching chord
                    best_score = -1
                    best_chord = "N"  # No chord
                    
                    for root in range(12):
                        for chord_type, template in chord_types.items():
                            # Rotate template to current root note
                            rotated_template = np.roll(template, root)
                            
                            # Calculate correlation
                            score = np.correlate(chroma_vector, rotated_template)[0]
                            
                            if score > best_score:
                                best_score = score
                                chord_name = note_names[root] + chord_type
                                best_chord = chord_name
                    
                    # Only create a new chord event if the chord changes
                    if best_chord != prev_chord:
                        if prev_chord:
                            # End the previous chord
                            chord_results.append({
                                "start": start_time,
                                "end": beat_time,
                                "label": prev_chord,
                                "confidence": 0.7
                            })
                        
                        # Start new chord
                        prev_chord = best_chord
                        start_time = beat_time
            
            # Add the final chord
            if prev_chord and start_time < librosa.get_duration(y=self.y, sr=self.sr):
                chord_results.append({
                    "start": start_time,
                    "end": librosa.get_duration(y=self.y, sr=self.sr),
                    "label": prev_chord,
                    "confidence": 0.7
                })
            
            return chord_results
            
        except Exception as e:
            logger.error(f"Basic chord detection failed: {e}")
            return []
    
    def detect_chords_advanced(self):
        """
        Advanced chord detection using multiple libraries.
        
        Returns:
            List of chord objects with timing and label information
        """
        chord_results = []
        
        # Check if we have essentia available
        if ESSENTIA_AVAILABLE:
            try:
                logger.info("Detecting chords with Essentia...")
                
                # Ensure audio is loaded
                self._ensure_audio_loaded()
                
                # Ensure libraries are initialized
                self._ensure_libraries_initialized()
                
                # Get frame size and hop size
                frame_size = 2048
                hop_size = 1024
                
                # Process audio in frames
                for frame_start in range(0, len(self.y) - frame_size, hop_size):
                    frame = self.y[frame_start:frame_start+frame_size]
                    
                    # Skip silent frames
                    if np.max(np.abs(frame)) < 0.01:
                        continue
                    
                    try:
                        # Use Essentia's chord detection
                        w = essentia.standard.Windowing(type='hann')
                        spectrum = essentia.standard.Spectrum()
                        spectral_peaks = essentia.standard.SpectralPeaks()
                        hpcp = essentia.standard.HPCP()
                        chord_detector = essentia.standard.ChordsDetection()
                        
                        # Process frames
                        frame_windowed = w(frame)
                        frame_spectrum = spectrum(frame_windowed)
                        peaks_freqs, peaks_mags = spectral_peaks(frame_spectrum)
                        hpcp_vals = hpcp(peaks_freqs, peaks_mags)
                        chords_detection_result = chord_detector(hpcp_vals)
                        
                        # Get the chord
                        chord = chords_detection_result[0]  # First element is the chord
                        confidence = chords_detection_result[1]  # Second element is confidence
                        
                        # Calculate time
                        start_time = frame_start / self.sr
                        end_time = (frame_start + frame_size) / self.sr
                        
                        # Store the detected chord
                        chord_results.append({
                            'chord': chord,
                            'start_time': start_time,
                            'end_time': end_time,
                            'confidence': float(confidence)
                        })
                    except Exception as e:
                        logger.warning(f"Error detecting chord in frame: {e}")
                
                # Merge consecutive identical chords
                if chord_results:
                    merged_results = [chord_results[0]]
                    for i in range(1, len(chord_results)):
                        if chord_results[i]['chord'] == merged_results[-1]['chord']:
                            merged_results[-1]['end_time'] = chord_results[i]['end_time']
                            merged_results[-1]['confidence'] = max(merged_results[-1]['confidence'], chord_results[i]['confidence'])
                        else:
                            merged_results.append(chord_results[i])
                    
                    logger.info(f"Detected {len(merged_results)} chords with Essentia")
                    return merged_results
            except Exception as e:
                logger.warning(f"Essentia chord detection failed: {e}")
        
        # Fallback to librosa-based chord detection
        try:
            logger.info("Falling back to librosa for chord detection...")
            
            # Ensure audio is loaded
            self._ensure_audio_loaded()
            
            # Compute the chromagram
            chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
            
            # Get beats
            _, beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
            beat_times = librosa.frames_to_time(beats, sr=self.sr)
            
            # Get beat-synced chroma
            beat_chroma = librosa.util.sync(chroma, beats, aggregate=np.median)
            
            # Define simple chord templates
            templates = {
                'C:maj': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                'C:min': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                'C#:maj': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                'C#:min': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                # Add more templates for other chords...
            }
            
            # Detect chords at each beat
            for i in range(len(beat_times) - 1):
                start_time = beat_times[i]
                end_time = beat_times[i+1]
                
                # Get chroma for this beat
                beat_chroma_vec = beat_chroma[:, i]
                
                # Find best matching chord
                best_chord = 'N'  # No chord
                best_score = -1
                
                for chord_name, template in templates.items():
                    # For a real implementation, you would need to handle all chord types and keys
                    score = np.dot(beat_chroma_vec, template)
                    if score > best_score:
                        best_score = score
                        best_chord = chord_name
                
                chord_results.append({
                    'chord': best_chord,
                    'start_time': start_time,
                    'end_time': end_time,
                    'confidence': float(best_score)
                })
            
            # Merge consecutive identical chords
            if chord_results:
                merged_results = [chord_results[0]]
                for i in range(1, len(chord_results)):
                    if chord_results[i]['chord'] == merged_results[-1]['chord']:
                        merged_results[-1]['end_time'] = chord_results[i]['end_time']
                    else:
                        merged_results.append(chord_results[i])
                
                logger.info(f"Detected {len(merged_results)} chords with librosa")
                return merged_results
        except Exception as e:
            logger.warning(f"Librosa chord detection failed: {e}")
        
        return []
    
    # Add the methods to the class
    UnifiedAudioAnalyzer.detect_chords = detect_chords
    UnifiedAudioAnalyzer.detect_chords_advanced = detect_chords_advanced
    
    return UnifiedAudioAnalyzer

# Add tablature generation methods
def add_tablature_methods(UnifiedAudioAnalyzer):
    """Add tablature generation methods to the UnifiedAudioAnalyzer class."""
    
    def generate_tablature(self):
        """
        Generate guitar tablature from detected notes.
        
        Returns:
            Dictionary with tablature data and rendering information
        """
        # Detect notes
        notes = self.detect_guitar_notes()
        
        # Map notes to fretboard
        tablature_notes = self.map_notes_to_fretboard(notes)
        
        # Organize into measures based on tempo
        tempo_info = self.detect_tempo()
        
        # Default to 120 BPM if detection fails
        bpm = tempo_info.get('tempo', 120.0)
        if bpm <= 0:
            bpm = 120.0
        
        # Get time signature
        time_sig = self.detect_time_signature()
        beats_per_measure = time_sig.get('numerator', 4)
        beat_value = time_sig.get('denominator', 4)
        
        # Calculate seconds per measure
        seconds_per_beat = 60.0 / bpm
        seconds_per_measure = seconds_per_beat * beats_per_measure
        
        # Group notes into measures
        measures = []
        current_measure = []
        measure_start_time = 0
        
        for note in tablature_notes:
            # Skip notes without position data
            if note.get('string') is None or note.get('fret') is None:
                continue
                
            # Check if note belongs in current measure
            if note['start'] >= measure_start_time and note['start'] < measure_start_time + seconds_per_measure:
                # Add to current measure
                current_measure.append(note)
            else:
                # Find appropriate measure for this note
                while note['start'] >= measure_start_time + seconds_per_measure:
                    # Finish current measure
                    if current_measure:
                        measures.append(current_measure)
                    
                    # Start new measure
                    current_measure = []
                    measure_start_time += seconds_per_measure
                
                # Add to new current measure
                current_measure.append(note)
        
        # Add the last measure if it has notes
        if current_measure:
            measures.append(current_measure)
        
        # Generate ASCII tablature
        ascii_tab = self._generate_ascii_tablature(measures, beats_per_measure)
        
        # Get detected chords
        chords = self.detect_chords()
        
        # Associate chords with measures
        measure_chords = []
        for i, measure in enumerate(measures):
            if not measure:
                measure_chords.append(None)
                continue
                
            measure_start = measure_start_time + i * seconds_per_measure
            measure_end = measure_start + seconds_per_measure
            
            # Find chords in this measure
            measure_chord = None
            max_duration = 0
            
            for chord in chords:
                # Check overlap with measure
                overlap_start = max(chord['start'], measure_start)
                overlap_end = min(chord['end'], measure_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > max_duration:
                    max_duration = overlap_duration
                    measure_chord = chord['label']
            
            measure_chords.append(measure_chord)
        
        # Generate data for visualization
        visualization_data = {
            'measures': [],
            'chords': measure_chords,
            'tempo': bpm,
            'timeSignature': f"{beats_per_measure}/{beat_value}"
        }
        
        for measure in measures:
            measure_data = []
            for note in measure:
                measure_data.append({
                    'string': note['string'],
                    'fret': note['fret'],
                    'start': note['start'],
                    'end': note['end'],
                    'name': note['name']
                })
            visualization_data['measures'].append(measure_data)
        
        return {
            'ascii': ascii_tab,
            'data': visualization_data
        }
    
    def _generate_ascii_tablature(self, measures, beats_per_measure):
        """
        Generate ASCII tablature representation from note measures.
        
        Args:
            measures: List of measures, each containing note data
            beats_per_measure: Number of beats per measure
            
        Returns:
            String containing ASCII tablature
        """
        if not measures:
            return "No notes detected for tablature generation."
            
        # Create ASCII tablature
        lines = []
        
        # Add tuning information
        lines.append("Guitar Tablature - Standard Tuning (E A D G B E)")
        lines.append("")
        
        # Generate 4 measures per line
        for i in range(0, len(measures), 4):
            measure_group = measures[i:i+4]
            
            # Create string lines for each guitar string
            string_lines = [[] for _ in range(6)]
            
            # Add measure markers
            for j, measure in enumerate(measure_group):
                # Create empty measure template (5 chars per beat + 1 for measure line)
                measure_width = beats_per_measure * 5 + 1
                
                for string_idx in range(6):
                    if j == 0:
                        # First measure in line needs string name
                        string_name = self.STANDARD_TUNING[5 - string_idx]
                        string_lines[string_idx].append(f"{string_name[0]}|")
                    else:
                        string_lines[string_idx].append("|")
                    
                    # Add empty frets for the measure
                    string_lines[string_idx].append("-" * (measure_width - 1))
            
            # Add notes to the tablature
            for j, measure in enumerate(measure_group):
                measure_width = beats_per_measure * 5 + 1
                
                for note in measure:
                    if note.get('string') is None or note.get('fret') is None:
                        continue
                        
                    string_idx = note['string']
                    fret = note['fret']
                    
                    # Calculate position within the measure
                    start_time = note['start']
                    measure_start = j * beats_per_measure * (60.0 / self.detect_tempo().get('tempo', 120.0))
                    pos_in_measure = (start_time - measure_start) / (beats_per_measure * (60.0 / self.detect_tempo().get('tempo', 120.0)))
                    
                    # Convert to character position
                    char_pos = int(pos_in_measure * (measure_width - 1))
                    
                    # Position from start of line
                    full_pos = j * measure_width + char_pos + 1
                    
                    # Make sure we're not out of bounds
                    if full_pos < len(string_lines[string_idx][0]) and string_idx < len(string_lines):
                        # Replace the dash with fret number
                        line = string_lines[string_idx][0]
                        fret_str = str(fret)
                        
                        # Check if there's room for fret number
                        if full_pos + len(fret_str) <= len(line):
                            string_lines[string_idx][0] = line[:full_pos] + fret_str + line[full_pos + len(fret_str):]
            
            # Join the lines for all strings
            for string_idx in range(6):
                lines.append("".join(string_lines[string_idx]))
            
            # Add chord line if available
            chord_line = ""
            for j, measure in enumerate(measure_group):
                measure_width = beats_per_measure * 5 + 1
                
                # Get chord for this measure
                if j < len(measure_group) and j + i < len(measures):
                    chord = self.detect_chords()[j + i] if j + i < len(self.detect_chords()) else None
                    chord_label = chord['label'] if chord else ""
                else:
                    chord_label = ""
                
                # Create chord marker
                if j == 0:
                    chord_line += f" {chord_label}"
                    chord_line += " " * (measure_width - len(chord_label) - 1)
                else:
                    # Center chord in measure
                    padding = measure_width - len(chord_label)
                    left_padding = padding // 2
                    right_padding = padding - left_padding
                    chord_line += " " * left_padding + chord_label + " " * right_padding
            
            lines.append(chord_line)
            lines.append("")
            
        return "\n".join(lines)
    
    # Add the methods to the class
    UnifiedAudioAnalyzer.generate_tablature = generate_tablature
    UnifiedAudioAnalyzer._generate_ascii_tablature = _generate_ascii_tablature
    
    return UnifiedAudioAnalyzer

# Add MIDI and other export methods
def add_export_methods(UnifiedAudioAnalyzer):
    """Add export methods to the UnifiedAudioAnalyzer class."""
    
    def export_to_midi(self, output_path=None):
        """
        Export audio analysis to MIDI file using pretty_midi.
        
        Args:
            output_path: Optional path to save the MIDI file.
                If not provided, saves to the same location as the audio file with .mid extension.
                
        Returns:
            Path to the saved MIDI file
        """
        if not PRETTY_MIDI_AVAILABLE:
            logger.error("pretty_midi library not available for MIDI export")
            return None
            
        try:
            # Create a pretty_midi object
            midi = pretty_midi.PrettyMIDI()
            
            # Create an instrument
            guitar = pretty_midi.Instrument(program=24)  # Acoustic Guitar (nylon)
            
            # Get detected notes
            notes = self.detect_guitar_notes()
            
            # Add each note to the instrument
            for note in notes:
                # Create a Note object
                midi_note = pretty_midi.Note(
                    velocity=max(64, min(127, int(note['confidence'] * 127))),  # Map confidence to velocity
                    pitch=note['pitch'],
                    start=note['start'],
                    end=note['end']
                )
                
                # Add note to instrument
                guitar.notes.append(midi_note)
            
            # Add the instrument to the PrettyMIDI object
            midi.instruments.append(guitar)
            
            # Set default output path if not provided
            if output_path is None:
                output_path = os.path.splitext(self.audio_path)[0] + '.mid'
                
            # Write out the MIDI data
            midi.write(output_path)
            logger.info(f"MIDI file exported to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting to MIDI: {e}")
            return None
    
    def apply_audio_effects(self, output_path=None, effects=None):
        """
        Apply audio effects using Pedalboard library and export to file.
        
        Args:
            output_path: Optional path to save the processed audio file.
            effects: Dictionary of effects to apply with their parameters.
                
        Returns:
            Path to the processed audio file
        """
        if not PEDALBOARD_AVAILABLE:
            logger.error("Pedalboard library not available for audio effects")
            return None
            
        try:
            # Set default output path if not provided
            if output_path is None:
                base, ext = os.path.splitext(self.audio_path)
                output_path = f"{base}_processed{ext}"
                
            # Set default effects if not provided
            if effects is None:
                effects = {
                    'reverb': {'room_size': 0.5, 'damping': 0.5, 'wet_level': 0.33, 'dry_level': 0.8},
                    'delay': {'delay_seconds': 0.5, 'feedback': 0.3, 'mix': 0.25}
                }
                
            # Create a Pedalboard with the specified effects
            board = pedalboard.Pedalboard([])
            
            # Add reverb if specified
            if 'reverb' in effects:
                params = effects['reverb']
                board.append(pedalboard.Reverb(
                    room_size=params.get('room_size', 0.5),
                    damping=params.get('damping', 0.5),
                    wet_level=params.get('wet_level', 0.33),
                    dry_level=params.get('dry_level', 0.8)
                ))
                
            # Add delay if specified
            if 'delay' in effects:
                params = effects['delay']
                board.append(pedalboard.Delay(
                    delay_seconds=params.get('delay_seconds', 0.5),
                    feedback=params.get('feedback', 0.3),
                    mix=params.get('mix', 0.25)
                ))
                
            # Add chorus if specified
            if 'chorus' in effects:
                params = effects['chorus']
                board.append(pedalboard.Chorus(
                    rate_hz=params.get('rate_hz', 1.0),
                    depth=params.get('depth', 0.25),
                    centre_delay_ms=params.get('centre_delay_ms', 7.0),
                    feedback=params.get('feedback', 0.0),
                    mix=params.get('mix', 0.5)
                ))
                
            # Add distortion if specified
            if 'distortion' in effects:
                params = effects['distortion']
                board.append(pedalboard.Distortion(
                    drive_db=params.get('drive_db', 25.0)
                ))
                
            # Process the audio
            processed_audio = board(self.y, self.sr)
            
            # Save the processed audio
            sf.write(output_path, processed_audio, self.sr)
            
            logger.info(f"Processed audio saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error applying audio effects: {e}")
            return None
    
    # Add the methods to the class
    UnifiedAudioAnalyzer.export_to_midi = export_to_midi
    UnifiedAudioAnalyzer.apply_audio_effects = apply_audio_effects
    
    return UnifiedAudioAnalyzer

# Apply all method additions
UnifiedAudioAnalyzer = add_chord_detection_methods(UnifiedAudioAnalyzer)
UnifiedAudioAnalyzer = add_tablature_methods(UnifiedAudioAnalyzer)
UnifiedAudioAnalyzer = add_export_methods(UnifiedAudioAnalyzer)

# Now the UnifiedAudioAnalyzer class has all methods from the original scattered implementations
