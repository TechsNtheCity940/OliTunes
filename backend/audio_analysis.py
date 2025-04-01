import librosa
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import os
import logging
from scipy import signal
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function for time formatting
def format_time(seconds):
    """Format seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

class AudioAnalyzer:
    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        self.y, self.sr = librosa.load(audio_path)
        
        # Cache directory for analysis results
        self.cache_dir = os.path.join(os.path.dirname(self.audio_path), "analysis_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def detect_guitar_notes(self) -> List[Dict[str, Any]]:
        """Detect guitar notes using pitch detection algorithms
        
        Uses a combination of techniques:
        1. Adaptive CQT analysis with optimized frequency bins
        2. Source separation to isolate melodic content when possible
        3. Advanced post-processing for note segmentation
        """
        logger.info("Starting guitar note detection...")
        
        # Check if results are cached
        cache_file = os.path.join(self.cache_dir, f"{os.path.basename(self.audio_path)}_notes.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_notes = json.load(f)
                    logger.info(f"Using cached note detection results ({len(cached_notes)} notes)")
                    return cached_notes
            except Exception as e:
                logger.warning(f"Failed to load cached notes: {e}")
        
        # Apply preprocessing to enhance guitar signal
        logger.info("Applying preprocessing to isolate guitar frequencies...")
        y_processed = self._preprocess_audio_for_guitar()
        
        # Initialize lists to store results
        all_notes = []
        
        # First try neural network-based detection if available
        # This is wrapped in a try-except to handle cases when CREPE isn't installed
        try:
            # Only attempt to import if we're going to use it
            import crepe
            logger.info("Using CREPE neural network for pitch detection...")
            crepe_notes = self._detect_notes_with_crepe(y_processed)
            if crepe_notes and len(crepe_notes) > 0:
                logger.info(f"CREPE detected {len(crepe_notes)} notes")
                all_notes = crepe_notes
        except (ImportError, ModuleNotFoundError):
            logger.info("CREPE not available, using standard detection methods")
        except Exception as e:
            logger.warning(f"Error in CREPE detection: {str(e)}")
        
        # If no notes detected yet, use CQT-based detection
        if not all_notes:
            try:
                logger.info("Performing CQT-based detection...")
                cqt_notes = self._detect_notes_with_cqt(y_processed)
                if cqt_notes and len(cqt_notes) > 0:
                    logger.info(f"CQT detection found {len(cqt_notes)} notes")
                    all_notes = cqt_notes
            except Exception as e:
                logger.warning(f"Error in CQT detection: {str(e)}")
        
        # If still no notes, fall back to basic detection
        if not all_notes:
            logger.warning("Standard methods failed, falling back to basic detection")
            try:
                fallback_notes = self._detect_notes_basic(y_processed)
                all_notes = fallback_notes
                logger.info(f"Basic detection found {len(all_notes)} notes")
            except Exception as e:
                logger.error(f"Basic detection also failed: {str(e)}")
                return []
        
        # Post-processing for better results
        filtered_notes = self._post_process_notes(all_notes)
        
        # Cache results for future use
        self._cache_notes(filtered_notes, cache_file)
        
        logger.info(f"Final detection returned {len(filtered_notes)} notes")
        return filtered_notes
    
    def _preprocess_audio_for_guitar(self) -> np.ndarray:
        """Preprocess audio to enhance guitar signal components"""
        # Validate input audio
        if len(self.y) == 0:
            logger.warning("Empty audio data, returning original")
            return self.y
            
        # Make a copy of the original audio
        y_processed = self.y.copy()
        
        # Validate sample rate
        if self.sr < 8000:
            logger.warning(f"Low sample rate {self.sr}Hz may affect quality")
        
        # Apply a bandpass filter focusing on guitar frequency range (80Hz-1.2kHz)
        sos = signal.butter(10, [80, 1200], 'bandpass', fs=self.sr, output='sos')
        y_filtered = signal.sosfilt(sos, y_processed)
        
        try:
            # Ensure audio is valid for harmonic separation
            if len(y_filtered) < 2048:
                raise ValueError("Audio too short for harmonic separation")
                
            # Add additional validation to catch NaN values
            if not np.all(np.isfinite(y_filtered)):
                y_filtered = np.nan_to_num(y_filtered)
            
            # Split long audio into chunks for processing
            chunk_size = min(len(y_filtered), 343980)  # Demucs training length
            y_harmonic_chunks = []
            
            # Process each chunk with robust error handling
            for i in range(0, len(y_filtered), chunk_size):
                chunk = y_filtered[i:i+chunk_size]
                if len(chunk) < 2048:
                    # Skip chunks that are too short but keep them in the output
                    y_harmonic_chunks.append(chunk)
                    continue
                
                try:
                    # Process with lower margin for better stability
                    y_harmonic = librosa.effects.harmonic(chunk, margin=3.0)
                    
                    # Validate output
                    if not np.all(np.isfinite(y_harmonic)) or len(y_harmonic) != len(chunk):
                        logger.warning(f"Harmonic separation produced invalid output for chunk {i}")
                        y_harmonic_chunks.append(chunk)  # Use original chunk as fallback
                    else:
                        y_harmonic_chunks.append(y_harmonic)
                except Exception as e:
                    logger.warning(f"Error processing chunk {i}: {e}")
                    y_harmonic_chunks.append(chunk)  # Use original chunk as fallback
            
            # Validate before concatenating
            if not y_harmonic_chunks:
                raise ValueError("No valid chunks to process")
            
            # Recombine processed chunks
            y_harmonic = np.concatenate(y_harmonic_chunks)
            
            # Mix original and processed signals for better quality
            y_processed = 0.7 * y_harmonic + 0.3 * y_filtered
        except Exception as e:
            logger.warning(f"Error in harmonic separation: {e}. Using filtered audio.")
            logger.debug(f"Audio shape: {y_filtered.shape}, Sample rate: {self.sr}")
            y_processed = y_filtered
        
        # Normalize
        if np.max(np.abs(y_processed)) > 0:
            y_processed = y_processed / np.max(np.abs(y_processed))
            
        return y_processed
    
    def _cache_notes(self, notes: List[Dict], cache_file: str) -> None:
        """Cache detected notes to file"""
        try:
            with open(cache_file, 'w') as f:
                json.dump(notes, f)
            logger.info(f"Cached {len(notes)} notes to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache notes: {e}")
    
    def _detect_notes_with_crepe(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """Use CREPE neural network for high-accuracy pitch detection if available"""
        import crepe
        
        # Run CREPE with good accuracy settings
        logger.info("Running CREPE neural network pitch detection...")
        time_array, frequency, confidence, _ = crepe.predict(
            audio, 
            self.sr, 
            model_capacity='full',
            viterbi=True,
            step_size=10
        )
        
        # Filter by confidence
        threshold = 0.5
        valid_indices = confidence > threshold
        
        # Process pitch values
        notes = []
        current_note = None
        current_start = None
        min_note_duration = 0.1  # Minimum duration in seconds
        
        # Process pitch values
        for i, (t, f, conf) in enumerate(zip(time_array[valid_indices], 
                                       frequency[valid_indices], 
                                       confidence[valid_indices])):
            if f > 0:
                # Convert frequency to musical note
                midi_note = librosa.hz_to_midi(f)
                note_name = librosa.midi_to_note(int(round(midi_note)), unicode=False)
                
                # Check if this is a new note or continuation
                if current_note is None:
                    # Start new note
                    current_note = note_name
                    current_start = t
                elif current_note != note_name:
                    # Note changed, add previous note if long enough
                    duration = t - current_start
                    if duration >= min_note_duration:
                        notes.append({
                            'note': current_note,
                            'time': float(current_start),
                            'duration': float(duration),
                            'frequency': float(librosa.note_to_hz(current_note)),
                            'confidence': float(conf),
                            'velocity': float(min(127, conf * 127))
                        })
                    
                    # Start tracking new note
                    current_note = note_name
                    current_start = t
        
        # Add the last note if there is one
        if current_note and current_start is not None:
            duration = time_array[-1] - current_start
            if duration >= min_note_duration:
                notes.append({
                    'note': current_note,
                    'time': float(current_start),
                    'duration': float(duration),
                    'frequency': float(librosa.note_to_hz(current_note)),
                    'confidence': float(confidence[-1]),
                    'velocity': float(min(127, confidence[-1] * 127))
                })
        
        return notes
    
    def _detect_notes_with_cqt(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """Detect notes using enhanced Constant-Q Transform analysis"""
        # Configuration for guitar pitch detection
        hop_length = 512
        bins_per_octave = 36  # Higher resolution for more accurate pitch
        fmin = librosa.note_to_hz('E2')  # Lowest standard guitar note
        n_bins = 6 * bins_per_octave  # 6 octaves
            
        # Compute Constant-Q transform
        C = np.abs(librosa.cqt(
            y=audio,
            sr=self.sr,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=fmin
        ))
        
        # Get times for each frame
        times = librosa.times_like(C, sr=self.sr, hop_length=hop_length)
        
        # Enhanced onset detection for note segmentation
        onset_env = librosa.onset.onset_strength(
            y=audio, 
            sr=self.sr,
            hop_length=hop_length
        )
        
        # Detect onsets
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=hop_length,
            backtrack=True
        )
        
        onset_times = librosa.frames_to_time(onsets, sr=self.sr, hop_length=hop_length)
        
        # Add final frame as an "onset" to catch the last note
        if len(onset_times) > 0:
            onset_times = np.append(onset_times, times[-1])
        else:
            # If no onsets detected, create artificial segments
            onset_times = np.linspace(0, times[-1], 10)
        
        # Find peaks in each CQT frame
        notes = []
        
        # Process each segment between onsets
        for i in range(len(onset_times) - 1):
            onset_time = onset_times[i]
            end_time = onset_times[i+1]
            
            # Skip very short segments
            if end_time - onset_time < 0.1:  # Minimum 100ms
                continue
                
            # Find frames within this segment
            segment_frames = np.where((times >= onset_time) & (times < end_time))[0]
            
            if len(segment_frames) == 0:
                continue
                
            # Calculate average spectrum for this segment
            segment = C[:, segment_frames]
            avg_spectrum = np.mean(segment, axis=1)
            
            # Find peaks in the spectrum
            peak_threshold = np.max(avg_spectrum) * 0.3  # 30% of maximum
            peaks, _ = signal.find_peaks(avg_spectrum, height=peak_threshold, distance=2)
            
            if len(peaks) == 0:
                continue
                
            # Get the strongest peak
            strongest_peak = peaks[np.argmax(avg_spectrum[peaks])]
            
            # Convert peak index to MIDI note
            midi_note = librosa.hz_to_midi(librosa.midi_to_hz('E2') * (2 ** (strongest_peak / bins_per_octave)))
            note_name = librosa.midi_to_note(int(round(midi_note)), unicode=False)
            
            # Calculate confidence based on peak height
            confidence = min(1.0, avg_spectrum[strongest_peak] / np.max(avg_spectrum))
            
            # Calculate estimated frequency
            frequency = librosa.midi_to_hz(midi_note)
            
            # Add to detected notes
            notes.append({
                'note': note_name,
                'time': float(onset_time),
                'duration': float(end_time - onset_time),
                'frequency': float(frequency),
                'confidence': float(confidence),
                'velocity': float(min(127, confidence * 127))
            })
        
        return notes
    
    def _detect_notes_basic(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """Basic note detection as a fallback"""
        # Simple pitch detection using auto-correlation
        hop_length = 512
        
        # Detect pitches
        pitches, magnitudes = librosa.piptrack(
            y=audio, 
            sr=self.sr,
            hop_length=hop_length,
            fmin=librosa.note_to_hz('E2'),
            fmax=librosa.note_to_hz('E6')
        )
        
        # Get times for each frame
        times = librosa.times_like(pitches[0], sr=self.sr, hop_length=hop_length)
        
        # Enhanced onset detection
        onset_env = librosa.onset.onset_strength(
            y=audio, 
            sr=self.sr,
            hop_length=hop_length
        )
        
        # Detect onsets with adaptive thresholding
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=hop_length,
            backtrack=True
        )
        
        onset_times = librosa.frames_to_time(onsets, sr=self.sr, hop_length=hop_length)
        
        # Add final frame as an "onset" to catch the last note
        if len(onset_times) > 0:
            onset_times = np.append(onset_times, times[-1])
        else:
            # If no onsets detected, create artificial segments
            onset_times = np.linspace(0, times[-1], 10)
        
        notes = []
        
        # Process each segment between onsets
        for i in range(len(onset_times) - 1):
            onset_time = onset_times[i]
            end_time = onset_times[i+1]
            
            # Skip very short segments
            if end_time - onset_time < 0.1:
                continue
            
            # Find frames within this segment
            segment_frames = np.where((times >= onset_time) & (times < end_time))[0]
            
            if len(segment_frames) == 0:
                continue
            
            # For each frame, find the frequency with highest magnitude
            frame_pitches = []
            frame_mags = []
            
            for frame in segment_frames:
                idx = np.argmax(magnitudes[:, frame])
                if magnitudes[idx, frame] > 0:  # Only consider if magnitude is positive
                    frame_pitches.append(pitches[idx, frame])
                    frame_mags.append(magnitudes[idx, frame])
            
            if not frame_pitches:
                continue
            
            # Use the median pitch for stability
            median_pitch = np.median(frame_pitches)
            
            # Convert to MIDI note
            midi_note = librosa.hz_to_midi(median_pitch)
            note_name = librosa.midi_to_note(int(round(midi_note)), unicode=False)
            
            # Simple confidence calculation
            confidence = 0.7  # Default confidence level for basic method
            
            notes.append({
                'note': note_name,
                'time': float(onset_time),
                'duration': float(end_time - onset_time),
                'frequency': float(median_pitch),
                'confidence': float(confidence),
                'velocity': float(min(127, confidence * 127))
            })
        
        return notes
    
    def _post_process_notes(self, notes: List[Dict]) -> List[Dict]:
        """Apply post-processing to refine detected notes"""
        if not notes:
            return []
            
        # Sort by time
        notes.sort(key=lambda x: x['time'])
        
        # Filter out very short notes (likely noise)
        min_duration = 0.08  # 80ms minimum note duration
        filtered_notes = [n for n in notes if n['duration'] >= min_duration]
        
        # Remove overlapping notes, keeping the one with higher confidence
        non_overlapping = []
        i = 0
        while i < len(filtered_notes):
            current = filtered_notes[i]
            j = i + 1
            overlap_found = False
            
            # Look for overlapping notes
            while j < len(filtered_notes) and filtered_notes[j]['time'] < current['time'] + current['duration']:
                # If significant overlap (>50%), keep the one with higher confidence
                overlap_start = filtered_notes[j]['time']
                overlap_end = min(current['time'] + current['duration'], 
                                 filtered_notes[j]['time'] + filtered_notes[j]['duration'])
                overlap_duration = overlap_end - overlap_start
                
                min_duration = min(current['duration'], filtered_notes[j]['duration'])
                if overlap_duration > 0.5 * min_duration:
                    overlap_found = True
                    if filtered_notes[j]['confidence'] > current['confidence']:
                        # Keep the next note instead
                        current = filtered_notes[j]
                j += 1
            
            non_overlapping.append(current)
            i = j if overlap_found else i + 1
        
        # Make sure note durations don't overlap the next note's start time
        for i in range(len(non_overlapping) - 1):
            next_start = non_overlapping[i+1]['time']
            if non_overlapping[i]['time'] + non_overlapping[i]['duration'] > next_start:
                # Adjust duration to end slightly before next note starts
                new_duration = next_start - non_overlapping[i]['time'] - 0.01
                if new_duration > 0.1:  # Keep reasonable minimum duration
                    non_overlapping[i]['duration'] = new_duration
        
        return non_overlapping
    
    def map_notes_to_fretboard(self, notes: List[Dict]) -> List[Dict]:
        """Map notes to optimal positions on guitar fretboard
        
        Uses algorithms to determine the best fingering and 
        string choices for playability.
        """
        logger.info("Mapping notes to fretboard positions...")
        
        if not notes:
            return []
            
        # Create copy of notes
        mapped_notes = []
        
        # Define standard tuning
        strings = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']  # Low to high
        
        # Calculate fret positions for each note
        for note in notes:
            note_copy = note.copy()
            
            # Skip notes without proper note name
            if 'note' not in note or not note['note']:
                continue
                
            note_name = note['note']
            
            # Try to find position on fretboard
            fretboard_position = self._find_best_fret_position(note_name, strings)
            
            if fretboard_position:
                note_copy['string'] = fretboard_position['string']
                note_copy['fret'] = fretboard_position['fret']
                mapped_notes.append(note_copy)
            else:
                logger.warning(f"Could not find fretboard position for note: {note_name}")
                
        logger.info(f"Mapped {len(mapped_notes)} notes to fretboard positions")
        return mapped_notes
    
    def _find_best_fret_position(self, note_name: str, strings: List[str]) -> Optional[Dict[str, int]]:
        """Find the best position on the fretboard for a given note"""
        # Convert to pitch class and octave
        try:
            pitch_class = note_name[:-1]  # e.g. "C#" from "C#4"
            octave = int(note_name[-1])
        except (IndexError, ValueError):
            # If no octave in the name, guess a reasonable default
            if any(c.isdigit() for c in note_name):
                # Try to extract octave number if it's somewhere in the string
                for i, c in enumerate(note_name):
                    if c.isdigit():
                        pitch_class = note_name[:i]
                        try:
                            octave = int(note_name[i:])
                        except ValueError:
                            octave = 4
                        break
            else:
                pitch_class = note_name
                octave = 4
        
        # Full note with octave
        full_note = f"{pitch_class}{octave}"
        
        # Calculate MIDI note number
        try:
            midi_note = librosa.note_to_midi(full_note)
        except Exception:
            # If conversion fails, try a different approach
            # Try to map common notation like "C" to "C4"
            if len(note_name) <= 2:  # Just a note name like "C" or "F#"
                full_note = f"{note_name}4"  # Assume octave 4
                try:
                    midi_note = librosa.note_to_midi(full_note)
                except Exception:
                    return None
            else:
                return None
        
        # Find all possible positions
        positions = []
        
        for string_idx, string_note in enumerate(strings):
            string_midi = librosa.note_to_midi(string_note)
            
            # How many frets up from the open string
            fret = midi_note - string_midi
            
            # If fret is in a playable range (0-24)
            if 0 <= fret <= 24:
                positions.append({
                    'string': string_idx,
                    'fret': fret,
                    'distance_from_nut': fret,  # Higher is harder
                    'string_idx': string_idx,  # 0 is low E
                })
        
        if not positions:
            return None
            
        # Sort positions by fret number (prefer lower positions)
        positions.sort(key=lambda x: x['fret'])
        
        # Return the best position (lowest fret for now)
        return positions[0]
    
    def analyze_tempo(self) -> Dict[str, Any]:
        """Analyze tempo and time signature"""
        # Get tempo
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        
        # Estimate time signature (simplified)
        y_harmonic = librosa.effects.harmonic(self.y)
        y_percussive = librosa.effects.percussive(self.y)
        
        # Extract beats
        _, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=self.sr, trim=False)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        
        # Simple heuristic for time signature
        # Look at beat patterns in groups of 2, 3, 4
        # This is very simplified and will often guess 4/4
        time_sig_num = 4  # Default to 4/4
        
        # Try to check beat strength patterns
        if len(beat_frames) > 8:
            # Get onset strengths at beat frames
            beat_strengths = onset_env[beat_frames]
            
            # Look at patterns of emphasized beats
            strength_4s = beat_strengths[::4]
            strength_3s = beat_strengths[::3]
            
            # If first beat of every 3 is consistently stronger
            if np.mean(strength_3s) > np.mean(beat_strengths) * 1.2:
                time_sig_num = 3  # Might be 3/4 or 6/8
                
        return {
            'tempo': float(tempo),
            'time_signature': f"{time_sig_num}/4",
            'beat_times': beat_times.tolist()
        }
    
    def detect_chords(self) -> List[Dict[str, Any]]:
        """Detect chords in audio using chromagram analysis"""
        logger.info("Detecting chords using chromagram analysis...")
        
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
        
        # Detect beats for segmentation
        tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        beat_times = librosa.frames_to_time(beats, sr=self.sr)
        
        # Group beats into measures (assuming 4/4 time)
        measure_size = 4  # beats per measure
        measure_starts = beat_times[::measure_size]
        if len(measure_starts) > 0 and measure_starts[0] > 0:
            measure_starts = np.insert(measure_starts, 0, 0)
        
        # If no measures detected, create artificial segments
        if len(measure_starts) < 2:
            # Create segments every 2 seconds
            duration = librosa.get_duration(y=self.y, sr=self.sr)
            segment_duration = 2.0
            num_segments = max(1, int(duration / segment_duration))
            measure_starts = np.linspace(0, duration, num_segments + 1)
        
        # Create chord progression
        chords = []
        
        # Chord templates (major, minor)
        chord_templates = {
            'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C-E-G
            'Cm': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # C-Eb-G
            'C#': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            'C#m': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            'D': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            'Dm': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'Eb': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            'Ebm': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            'E': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            'Em': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            'F': [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'Fm': [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            'F#': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            'F#m': [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            'G': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'Gm': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            'Ab': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            'Abm': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
            'A': [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            'Am': [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            'Bb': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            'Bbm': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            'B': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            'Bm': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
        }
        
        # Analyze each measure
        for i in range(len(measure_starts) - 1):
            start_time = measure_starts[i]
            end_time = measure_starts[i + 1]
            
            # Get frames within this measure
            start_frame = librosa.time_to_frames(start_time, sr=self.sr)
            end_frame = librosa.time_to_frames(end_time, sr=self.sr)
            
            # Skip if invalid frames
            if start_frame >= chroma.shape[1] or end_frame >= chroma.shape[1] or start_frame == end_frame:
                continue
                
            # Get average chroma in this segment
            measure_chroma = np.mean(chroma[:, start_frame:end_frame], axis=1)
            
            # Compare with chord templates to find best match
            max_corr = -np.inf
            best_chord = "N/C"  # No chord
            
            for chord_name, template in chord_templates.items():
                # Calculate correlation between template and measure chroma
                corr = np.corrcoef(measure_chroma, template)[0, 1]
                
                if corr > max_corr:
                    max_corr = corr
                    best_chord = chord_name
            
            # Add to chord progression if confidence is high enough
            confidence = float(max(0.2, min(1.0, (max_corr + 1) / 2)))
            
            chords.append({
                'chord': best_chord,
                'start_time': float(start_time),
                'end_time': float(end_time),
                'confidence': confidence
            })
        
        return chords
    
    def detect_key(self) -> Dict[str, Any]:
        """Detect the musical key of the audio"""
        logger.info("Detecting musical key...")
        
        # Calculate chroma
        chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
        
        # Calculate global key profile
        chroma_avg = np.mean(chroma, axis=1)
        
        # Profiles for major and minor keys
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Normalize profiles
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)
        
        # Calculate correlation with profiles for all possible keys
        key_correlations = np.zeros(24)  # 12 major + 12 minor
        
        # Normalize chroma
        if np.sum(chroma_avg) > 0:
            chroma_avg = chroma_avg / np.sum(chroma_avg)
        
        for i in range(12):  # For each possible key
            # Shift the profile to match the key
            maj_shifted = np.roll(major_profile, i)
            min_shifted = np.roll(minor_profile, i)
            
            # Calculate the correlation
            key_correlations[i] = np.corrcoef(chroma_avg, maj_shifted)[0, 1]
            key_correlations[i + 12] = np.corrcoef(chroma_avg, min_shifted)[0, 1]
        
        # Find best key
        best_key_idx = np.argmax(key_correlations)
        
        # Key names (C, C#, D, etc.)
        key_names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        
        # Get key name and mode
        if best_key_idx < 12:
            key_name = key_names[best_key_idx]
            mode = "major"
        else:
            key_name = key_names[best_key_idx - 12]
            mode = "minor"
        
        confidence = float(max(0.2, min(0.9, (key_correlations[best_key_idx] + 1) / 2)))
        
        return {
            'key': key_name,
            'mode': mode,
            'confidence': confidence
        }
    
    def detect_time_signature(self) -> Tuple[int, int]:
        """Detect the time signature of the audio"""
        logger.info("Detecting time signature...")
        
        # Extract onset envelope
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        
        # Estimate tempo
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        
        # Calculate autocorrelation of onset strength
        ac = librosa.autocorrelate(onset_env, max_size=4 * self.sr // 512)
        
        # Find peaks in the autocorrelation
        peaks = signal.find_peaks(ac)[0]
        
        # Default to 4/4 if no peaks found
        if len(peaks) == 0:
            return (4, 4)
        
        # Find peaks that might correspond to measures
        # Looking for 4-beat and 3-beat patterns
        candidates = [2, 3, 4, 6, 8]  # Common time signatures: 2/4, 3/4, 4/4, 6/8, 8/8
        scores = np.zeros(len(candidates))
        
        for i, candidate in enumerate(candidates):
            if candidate < len(ac):
                # Check correlation at this beat pattern
                scores[i] = ac[candidate]
        
        best_idx = np.argmax(scores)
        
        # Map to time signature
        if candidates[best_idx] == 3:
            return (3, 4)  # 3/4
        elif candidates[best_idx] == 6:
            return (6, 8)  # 6/8
        elif candidates[best_idx] == 2:
            return (2, 4)  # 2/4
        elif candidates[best_idx] == 8:
            return (4, 4)  # Using 4/4 instead of 8/8
        else:
            return (4, 4)  # Most common: 4/4
    
    def extract_lyrics(self) -> Dict[str, Any]:
        """Extract lyrics from audio (placeholder)"""
        logger.info("Extracting lyrics (placeholder)...")
        
        # This is a placeholder since we don't have lyrics extraction capability
        # without specialized libraries
        
        return {
            'status': 'success',
            'message': 'Lyrics extraction requires specialized libraries',
            'lyrics': 'Lyrics not available in this build.\n\nTo enable lyrics analysis, please install specialized libraries.',
            'timestamps': []
        }
    
    def analyze_guitar_track(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze guitar track using optimized processing"""
        # This is a wrapper around detect_guitar_notes that accepts an audio array
        analyzer = AudioAnalyzer(self.audio_path)
        analyzer.y = audio
        return analyzer.detect_guitar_notes()
    
    def note_to_guitar_position(self, note_name: str) -> List[Tuple[int, int]]:
        """Convert a note name to all possible positions on guitar fretboard
        
        Returns list of tuples (string_index, fret_number)
        string_index: 0=high E, 5=low E
        fret_number: 0=open string, 1=first fret, etc.
        """
        # Define standard tuning
        strings = ['E4', 'B3', 'G3', 'D3', 'A2', 'E2']  # High to low
        
        positions = []
        
        # Try to parse note name
        try:
            pitch_class = note_name[:-1]  # e.g. "C#" from "C#4"
            octave = int(note_name[-1])
            full_note = f"{pitch_class}{octave}"
        except (IndexError, ValueError):
            # If no octave in the name or other issue, try to guess
            if any(c.isdigit() for c in note_name):
                # Try to extract octave if it exists
                for i, c in enumerate(note_name):
                    if c.isdigit():
                        pitch_class = note_name[:i]
                        try:
                            octave = int(note_name[i:])
                        except ValueError:
                            octave = 4
                        break
            else:
                # Default to octave 4 if no octave information
                pitch_class = note_name
                octave = 4
            
            full_note = f"{pitch_class}{octave}"
        
        # Convert note to MIDI
        try:
            note_midi = librosa.note_to_midi(full_note)
        except ValueError:
            # If conversion fails, try with default octave
            try:
                note_midi = librosa.note_to_midi(f"{pitch_class}4")
            except:
                return positions  # Return empty if conversion fails
        
        # For each string, find possible fret positions
        for string_idx, string_note in enumerate(strings):
            try:
                string_midi = librosa.note_to_midi(string_note)
                fret = note_midi - string_midi
                
                if 0 <= fret <= 24:  # Check if in playable range
                    positions.append((string_idx, fret))
            except:
                continue  # Skip if string note conversion fails
        
        return positions
    
    def find_best_guitar_position(self, note_name: str, prev_position: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Find best playable position for a note on guitar
        
        Args:
            note_name: The name of the note (e.g., "C4")
            prev_position: Previous (string, fret) position for continuity
            
        Returns:
            Tuple of (string_index, fret_number)
        """
        positions = self.note_to_guitar_position(note_name)
        
        if not positions:
            # If no valid position, default to high E string 1st fret
            return (0, 1)
        
        # If we have a previous position, try to find closest one for playability
        if prev_position:
            prev_string, prev_fret = prev_position
            
            # Define a score function to find the most playable transition
            def position_score(pos):
                string, fret = pos
                
                # Calculate distance metrics
                string_distance = abs(string - prev_string)
                fret_distance = abs(fret - prev_fret) if prev_fret > 0 and fret > 0 else 0
                
                # Prefer positions on same string
                if string == prev_string:
                    string_penalty = 0
                else:
                    string_penalty = string_distance * 2
                
                # Prefer positions in same hand position (fret proximity)
                fret_penalty = fret_distance
                
                # Prefer open strings (fret 0)
                open_string_bonus = -3 if fret == 0 else 0
                
                # Prefer lower positions on neck
                position_penalty = 0.2 * fret if fret > 0 else 0
                
                return string_penalty + fret_penalty + open_string_bonus + position_penalty
            
            # Sort by score (lower is better)
            positions.sort(key=position_score)
        else:
            # Without previous position, prefer lower frets
            positions.sort(key=lambda pos: pos[1])
        
        return positions[0]
    
    def generate_tablature(self) -> Dict[str, Any]:
        """Generate guitar tablature from detected notes"""
        logger.info("Generating guitar tablature...")
        
        # Get notes from audio
        notes = self.detect_guitar_notes()
        
        # Map to fretboard positions
        mapped_notes = self.map_notes_to_fretboard(notes)
        
        if not mapped_notes:
            logger.warning("No notes detected for tablature generation")
            return {
                'bars': [],
                'totalDuration': float(librosa.get_duration(y=self.y, sr=self.sr)),
                'gridResolution': 0.125,
                'notePositions': []
            }
        
        # Get audio duration
        duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        # Analyze tempo
        tempo_info = self.analyze_tempo()
        tempo = tempo_info['tempo']
        time_sig = self.detect_time_signature()
        
        # Create measures based on tempo and time signature
        beats_per_measure = time_sig[0]
        beat_duration = 60.0 / tempo
        measure_duration = beats_per_measure * beat_duration
        
        # Create a new tablature data structure
        tab_data = {
            'tablature': ['Guitar Tablature:'],
            'bars': [],
            'totalDuration': float(duration),
            'gridResolution': beat_duration / 4,  # 16th note resolution
            'notePositions': []
        }
        
        # Calculate total number of measures
        num_measures = max(1, int(duration / measure_duration))
        
        # Group measures into bars (4 measures per bar by default)
        measures_per_bar = 4
        num_bars = (num_measures + measures_per_bar - 1) // measures_per_bar
        
        # Generate tablature notation
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
                measure_notes = [n for n in mapped_notes if m_start <= n['time'] < m_end]
                
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
            
            tab_data['bars'].append(bar)
        
        # Add note positions for UI display
        for note in mapped_notes:
            tab_data['notePositions'].append({
                'note': note['note'],
                'time': float(note['time']),
                'duration': float(note['duration']),
                'string': note['string'],
                'fret': note['fret'],
                'confidence': float(note.get('confidence', 0.8))
            })
        
        return tab_data
