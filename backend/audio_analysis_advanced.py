import os
import logging
import json
import numpy as np
import librosa
import scipy
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAudioAnalyzer:
    """
    Advanced audio analysis toolkit with specialized guitar-focused algorithms.
    Implements state-of-the-art pitch detection and tablature generation.
    """
    
    # Standard guitar tuning (E2, A2, D3, G3, B3, E4)
    GUITAR_STRINGS = [40, 45, 50, 55, 59, 64]  # MIDI note numbers
    STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'e']
    
    # Guitar fretboard limits
    MAX_FRET = 24
    MAX_REACH = 5  # Maximum reasonable fret span for human hand
    
    def __init__(self, audio_path: str):
        """Initialize the analyzer with an audio file"""
        self.audio_path = audio_path
        self.cache_dir = os.path.join(os.path.dirname(audio_path), "analysis_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load audio file
        self.y, self.sr = librosa.load(audio_path, sr=None)
        
        # Compute basic features
        self._compute_chromagram()
        
    def _compute_chromagram(self):
        """Compute enhanced chromagram for more accurate pitch detection"""
        # We'll use a multi-resolution CQT for better pitch tracking
        hop_length = 512
        
        # Apply a high-pass filter to focus on guitar frequencies
        y_filt = self._preprocess_audio(self.y)
        
        # Compute multi-resolution CQT
        # First pass with higher frequency resolution
        self.C1 = np.abs(librosa.cqt(
            y_filt, 
            sr=self.sr,
            hop_length=hop_length,
            bins_per_octave=36,  # Higher resolution
            n_bins=8*36,  # 8 octaves
            fmin=librosa.note_to_hz('E2')  # Lowest guitar string
        ))
        
        # Second pass with better time resolution
        self.C2 = np.abs(librosa.cqt(
            y_filt, 
            sr=self.sr,
            hop_length=hop_length//2,  # Faster time resolution
            bins_per_octave=24,
            n_bins=8*24,
            fmin=librosa.note_to_hz('E2')
        ))
        
        # Compute standard chromagram for chord detection
        self.chroma = librosa.feature.chroma_cqt(
            y=y_filt, 
            sr=self.sr,
            hop_length=hop_length
        )
        
        # Time base for the CQT
        self.times = librosa.times_like(self.C1, sr=self.sr, hop_length=hop_length)
        
    def _preprocess_audio(self, y: np.ndarray) -> np.ndarray:
        """
        Enhance audio signal for guitar note detection:
        - High-pass filter to remove low frequency noise
        - Apply bandpass filter to focus on guitar frequency range
        - Non-linear normalization to enhance quieter notes
        """
        try:
            # High-pass filter to remove rumble and focus on guitar frequencies (>70Hz)
            y_filt = librosa.effects.preemphasis(y)
            
            # Apply a bandpass filter focusing on guitar frequency range (80Hz-1200Hz)
            nyquist = self.sr // 2
            cutoff_low = 80 / nyquist
            cutoff_high = 1200 / nyquist
            
            # Create bandpass filter
            b, a = scipy.signal.butter(4, [cutoff_low, cutoff_high], btype='band')
            y_eq = scipy.signal.filtfilt(b, a, y_filt)
            
            # Try to apply harmonic-percussive separation if it works with this version of librosa
            try:
                # Validate audio length for HPSS
                if len(y_eq) < 2048:
                    raise ValueError("Audio segment too short for HPSS")
                
                # Ensure audio is valid for HPSS (non-empty, finite values)
                if not np.all(np.isfinite(y_eq)):
                    y_eq = np.nan_to_num(y_eq)
                    
                # Process in chunks for long audio files
                # This helps prevent array bounds issues with very long files
                max_length = 2000000  # ~45 sec at 44.1kHz
                if len(y_eq) > max_length:
                    # Process in chunks and concatenate
                    chunks = []
                    for i in range(0, len(y_eq), max_length):
                        chunk = y_eq[i:i+max_length]
                        if len(chunk) >= 2048:  # Minimum viable chunk size
                            chunk_harmonic, _ = librosa.decompose.hpss(
                                chunk,
                                kernel_size=31,
                                power=2.0,
                                mask=False
                            )
                            chunks.append(chunk_harmonic)
                        else:
                            chunks.append(chunk)
                    y_harmonic = np.concatenate(chunks)
                else:
                    # Standard HPSS for normal length audio
                    y_harmonic, _ = librosa.decompose.hpss(
                        y_eq,
                        kernel_size=31,  # Smaller kernel size
                        power=2.0,
                        mask=False
                    )
                
                # First check if audio has proper shape and values
                if not np.all(np.isfinite(y_eq)):
                    # Fix NaN values before processing
                    y_eq = np.nan_to_num(y_eq)
                
                # Ensure minimum length for processing
                if len(y_eq) < 2048:
                    raise ValueError("Audio segment too short for harmonic separation")
                
                # Apply HPSS with gentler parameters
                y_harmonic = librosa.effects.harmonic(
                    y_eq,
                    kernel_size=31,  # Smaller kernel size
                    power=2.0,
                    mask=False
                )
                
                # Validate harmonic output before mixing
                if len(y_harmonic) != len(y_eq):
                    logger.warning(f"Harmonic separation produced mismatched array dimensions. Expected {len(y_eq)}, got {len(y_harmonic)}")
                    # Resize to match (duplicate ending or truncate)
                    if len(y_harmonic) < len(y_eq):
                        # Pad the harmonic array to match the original size
                        padding = len(y_eq) - len(y_harmonic)
                        y_harmonic = np.pad(y_harmonic, (0, padding), mode='edge')
                    else:
                        # Truncate the harmonic array to the original size
                        y_harmonic = y_harmonic[:len(y_eq)]
                
                # Check for NaN values in the harmonic output
                if not np.all(np.isfinite(y_harmonic)):
                    logger.warning("Harmonic separation produced NaN values, using filtered signal")
                    y_enhanced = y_eq
                else:
                    # Mix with the filtered content for better results
                    y_enhanced = 0.7 * y_harmonic + 0.3 * y_eq
            except Exception as e:
                # Fall back to just using the equalized signal if HPSS fails
                logger.warning(f"Harmonic separation failed, using filtered signal: {str(e)}")
                y_enhanced = y_eq
            
            # Normalize
            y_enhanced = librosa.util.normalize(y_enhanced)
            
            return y_enhanced
        except Exception as e:
            # If any preprocessing step fails, return the original audio
            logger.warning(f"Audio preprocessing failed: {str(e)}. Using original signal.")
            return librosa.util.normalize(y)
    
    def _yin_pitch_detection(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Implement the YIN algorithm for accurate monophonic pitch detection.
        Returns (frequency, confidence)
        """
        # Implementation of YIN algorithm
        # Reference: "YIN, a fundamental frequency estimator for speech and music"
        window_size = len(frame)
        half_window = window_size // 2
        
        # Step 1: Autocorrelation function
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[window_size-1:window_size+half_window]
        
        # Step 2: Difference function
        d = np.zeros(half_window)
        for tau in range(half_window):
            for j in range(window_size - tau):
                d[tau] += (frame[j] - frame[j + tau])**2
        
        # Step 3: Cumulative mean normalized difference
        d_norm = np.zeros(half_window)
        d_norm[0] = 1.0
        cumsum = np.cumsum(d)
        for tau in range(1, half_window):
            d_norm[tau] = d[tau] / (cumsum[tau] / tau)
        
        # Step 4: Absolute threshold
        tau_min = None
        threshold = 0.1  # YIN threshold parameter
        for tau in range(2, half_window):
            if d_norm[tau] < threshold:
                if tau_min is None or d_norm[tau] < d_norm[tau_min]:
                    tau_min = tau
                    
        if tau_min is None:
            tau_min = np.argmin(d_norm[2:]) + 2
            
        # Step 5: Parabolic interpolation for higher accuracy
        if 1 < tau_min < half_window - 1:
            a = d_norm[tau_min-1]
            b = d_norm[tau_min]
            c = d_norm[tau_min+1]
            adjustment = 0.5 * (a - c) / (a - 2*b + c)
            tau_star = tau_min + adjustment
        else:
            tau_star = tau_min
            
        # Calculate pitch and confidence
        if tau_star > 0:
            frequency = self.sr / tau_star
            confidence = 1 - d_norm[tau_min]  # Higher confidence for lower difference
        else:
            frequency = 0
            confidence = 0
            
        return frequency, confidence
    
    def detect_notes(self) -> List[Dict[str, Any]]:
        """
        Advanced guitar note detection using multiple pitch detection algorithms
        and consensus voting.
        """
        # Check if we have a cached analysis
        cache_path = os.path.join(self.cache_dir, Path(self.audio_path).name + "_notes.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    notes = json.load(f)
                    logger.info(f"Loaded {len(notes)} notes from cache")
                    # Ensure important fields exist
                    if all('note' in note and 'time' in note and 'duration' in note for note in notes):
                        return notes
            except Exception as e:
                logger.warning(f"Error loading from cache: {str(e)}")
        
        logger.info("Performing advanced note detection...")
        
        # Parameters for note detection
        hop_length = 512
        frame_length = 2048
        onset_env = librosa.onset.onset_strength(
            y=self.y, 
            sr=self.sr,
            hop_length=hop_length,
            aggregate=np.median  # More robust onset detection
        )
        
        # Detect note onsets (start of each note)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=hop_length,
            backtrack=True,  # Find true onset boundaries
            units='time'
        )
        
        # Dynamic threshold for onset detection
        peak_thresh = 0.5 * np.mean(onset_env) + 0.5 * np.std(onset_env)
        peaks = scipy.signal.find_peaks(onset_env, height=peak_thresh, distance=2)[0]
        peak_times = librosa.frames_to_time(peaks, sr=self.sr, hop_length=hop_length)
        
        # Merge onset detection methods
        all_onsets = np.unique(np.concatenate([onsets, peak_times]))
        
        # Convert to frames for processing
        onset_frames = librosa.time_to_frames(all_onsets, sr=self.sr, hop_length=hop_length)
        
        # Add an artificial final offset
        duration = librosa.get_duration(y=self.y, sr=self.sr)
        onset_frames = np.append(onset_frames, [len(self.times) - 1])
        
        # Detect notes for each onset-offset segment
        notes = []
        prev_midi = None
        
        for i in range(len(onset_frames) - 1):
            start_frame = onset_frames[i]
            end_frame = onset_frames[i+1]
            
            if end_frame <= start_frame:
                continue
                
            start_time = self.times[start_frame]
            end_time = self.times[end_frame]
            
            # Extract segment from audio
            y_segment = self.y[librosa.frames_to_samples([start_frame], hop_length=hop_length)[0]:
                              librosa.frames_to_samples([end_frame], hop_length=hop_length)[0]]
            
            if len(y_segment) < frame_length:
                # Too short segment, probably noise
                continue
                
            # Compute pitch using multiple methods for robustness
            # 1. YIN algorithm for monophonic pitch
            freq, confidence = self._yin_pitch_detection(y_segment[:frame_length])
            
            # 2. CQT-based pitch tracking (more stable for guitar)
            segment_frames = end_frame - start_frame
            chroma_segment = self.chroma[:, start_frame:end_frame]
            pitch_class = np.argmax(np.mean(chroma_segment, axis=1))
            
            # Determine octave by analyzing energy distribution in CQT
            cqt_segment = self.C1[:, start_frame:end_frame]
            cqt_mean = np.mean(cqt_segment, axis=1)
            
            # Try to match pitch from YIN with chromagram
            # This combines time-domain and frequency-domain analysis
            if 20 < freq < 1000 and confidence > 0.6:
                # Convert frequency to MIDI note
                midi_note = round(librosa.hz_to_midi(freq))
                midi_class = midi_note % 12
                
                # Validate that YIN pitch class matches chroma
                if abs((midi_class - pitch_class) % 12) <= 1:
                    # Good match, use YIN pitch
                    pass
                else:
                    # Calculate best matching octave for the chroma pitch class
                    # Find peaks in CQT that match the pitch class
                    octaves = []
                    for bin_idx in range(len(cqt_mean)):
                        if bin_idx % 36 == pitch_class * 3:  # 36 bins per octave, 3 bins per semitone
                            octaves.append((bin_idx, cqt_mean[bin_idx]))
                    
                    if octaves:
                        # Get highest energy matching bin
                        best_bin = max(octaves, key=lambda x: x[1])[0]
                        # Convert to MIDI note
                        midi_octave = best_bin // 36 + 1  # Adjusted for fmin = E2
                        midi_note = 12 * midi_octave + pitch_class
                    else:
                        midi_note = None
            else:
                # YIN not reliable, use CQT for pitch estimation
                # Find peaks in CQT
                peaks = scipy.signal.find_peaks(cqt_mean)[0]
                if len(peaks) > 0:
                    # Use highest energy peak
                    best_peak = peaks[np.argmax(cqt_mean[peaks])]
                    # Convert to MIDI
                    midi_note = int(librosa.hz_to_midi(librosa.cqt_frequencies(
                        n_bins=best_peak+1, 
                        fmin=librosa.note_to_hz('E2'),
                        bins_per_octave=36
                    )[-1]))
                else:
                    midi_note = None
            
            # Handle case where no pitch detected
            if midi_note is None:
                continue
                
            # For guitar, only consider notes in a reasonable range (E2 to E6)
            if not (40 <= midi_note <= 88):
                continue
                
            # Duration must be reasonable
            note_duration = end_time - start_time
            if note_duration < 0.05:  # Filter out very short detections (glitches)
                continue
                
            # Convert MIDI note to name
            note_name = librosa.midi_to_note(midi_note)
            
            # Add human-friendly features
            octave = midi_note // 12 - 1  # MIDI octave
            note_class = midi_note % 12   # C=0, C#=1, etc.
                
            # Add to notes list
            notes.append({
                'time': float(start_time),
                'duration': float(note_duration),
                'note': note_name,
                'midi': int(midi_note),
                'octave': octave,
                'pitch_class': int(note_class),
                'confidence': float(confidence)
            })
            
            prev_midi = midi_note
            
        # Final post-processing
        if notes:
            # Remove very short notes surrounded by the same pitch (false triggers)
            i = 1
            while i < len(notes) - 1:
                if (notes[i]['duration'] < 0.08 and
                    notes[i-1]['midi'] == notes[i+1]['midi'] and
                    notes[i]['time'] - notes[i-1]['time'] - notes[i-1]['duration'] < 0.1 and
                    notes[i+1]['time'] - notes[i]['time'] - notes[i]['duration'] < 0.1):
                    # This is likely a false trigger, remove
                    notes.pop(i)
                else:
                    i += 1
            
            # Map notes to fretboard positions
            notes = self._map_notes_to_fretboard(notes)
            
            # Save to cache
            try:
                with open(cache_path, 'w') as f:
                    json.dump(notes, f)
                    logger.info(f"Saved {len(notes)} notes to cache")
            except Exception as e:
                logger.warning(f"Error saving to cache: {str(e)}")
                
        return notes
    
    def _map_notes_to_fretboard(self, notes: List[Dict]) -> List[Dict]:
        """Intelligent algorithm to map notes with improved musical positioning"""
        if not notes:
            return []
            
        # Filter out notes without MIDI information
        valid_notes = []
        for note in notes:
            if 'midi' in note:
                # Create a deep copy of the note to prevent modifying the original
                valid_notes.append(note.copy())
        
        if not valid_notes:
            return []
            
        # Sort by time for sequential processing
        valid_notes.sort(key=lambda x: x['time'])
        
        # Track the current hand position and string usage over time
        current_position = 0  # Starting fret position (open strings)
        string_end_times = [-1.0] * 6  # When each string becomes free again
        
        for note in valid_notes:
            midi_note = note['midi']
            note_time = note['time']
            
            # Find all possible positions for this note on the fretboard
            positions = []
            for string, base_note in enumerate(self.GUITAR_STRINGS):
                # Check if note is playable on this string (within fret range)
                if midi_note >= base_note and midi_note - base_note <= self.MAX_FRET:
                    fret = midi_note - base_note
                    
                    # Calculate position quality score based on multiple factors:
                    # 1. Distance from current hand position (lower is better)
                    # 2. String availability (is the string already being played?)
                    # 3. Prefer lower frets when possible
                    # 4. Prefer logical string choice for the note's range
                    
                    # Distance from current position (0-5 points, lower is better)
                    position_score = min(5, abs(fret - current_position))
                    
                    # String availability (0-3 points, lower is better)
                    availability_score = 3 if note_time <= string_end_times[string] else 0
                    
                    # Prefer lower frets (0-2 points, lower is better)
                    fret_score = min(2, fret / 5)
                    
                    # Prefer logical string choice - lower strings for lower notes, higher strings for higher notes
                    # This creates more natural patterns (0-3 points, lower is better)
                    logical_choice_score = abs((midi_note % 12) / 12 - (string / 5)) * 3
                    
                    # Total score (lower is better)
                    total_score = position_score + availability_score + fret_score + logical_choice_score
                    
                    # Store position with its score
                    positions.append({
                        'string': string,
                        'fret': fret,
                        'score': total_score
                    })
            
            # If we found valid positions, choose the best one
            if positions:
                # Sort by score (lower is better)
                positions.sort(key=lambda p: p['score'])
                best_pos = positions[0]
                
                # Assign position to the note
                note['string'] = best_pos['string']
                note['fret'] = best_pos['fret']
                
                # Update current hand position
                current_position = best_pos['fret']
                
                # Update string end time (when it becomes available again)
                # Add a small buffer to note duration to prevent unrealistic fast changes
                buffer_time = min(0.1, note.get('duration', 0) * 0.5)
                string_end_times[best_pos['string']] = note_time + note.get('duration', 0.25) + buffer_time
            else:
                # Fallback for unmappable notes - use highest string with reasonable fret
                best_guess_string = 0  # E string (highest pitch)
                best_guess_fret = min(12, max(0, midi_note - self.GUITAR_STRINGS[0]))
                
                note['string'] = best_guess_string
                note['fret'] = best_guess_fret
        
        return valid_notes
    
    def generate_tablature(self, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate guitar tablature from detected notes with improved accuracy.
        Returns formatted tablature data for display.
        """
        if not notes:
            return {
                'bars': [],
                'totalDuration': float(librosa.get_duration(y=self.y, sr=self.sr)),
                'gridResolution': 0.125,
                'notePositions': []
            }
            
        # Sort notes by time
        notes = sorted(notes, key=lambda x: x['time'])
        
        # Ensure all notes have string and fret information using the improved mapping algorithm
        if not all('string' in note and 'fret' in note for note in notes):
            notes = self._map_notes_to_fretboard(notes)
            
        # Apply musical cleanup - remove overlapping notes on same string that would be unplayable
        cleaned_notes = []
        string_end_times = [-1.0] * 6  # Track when each string is free
        
        for note in notes:
            string_idx = note.get('string')
            if string_idx is None or 'time' not in note:
                continue
                
            note_time = note['time']
            
            # Check if string is available (not being played)
            if note_time < string_end_times[string_idx]:
                # String is already being played, try to reassign to another string
                original_midi = note.get('midi')
                if original_midi:
                    # Find an available string that can play this note
                    for alt_string, base_note in enumerate(self.GUITAR_STRINGS):
                        if (note_time >= string_end_times[alt_string] and 
                            original_midi >= base_note and 
                            original_midi - base_note <= self.MAX_FRET):
                            # Found an available string - reassign the note
                            note = note.copy()  # Create a copy to avoid modifying original
                            note['string'] = alt_string
                            note['fret'] = original_midi - base_note
                            string_idx = alt_string
                            break
                    else:
                        # No available string found, skip this note 
                        # to avoid unplayable overlaps
                        continue
            
            # Add the note and update string end time
            cleaned_notes.append(note)
            string_end_times[string_idx] = note_time + note.get('duration', 0.25)
            
        # Use cleaned notes for tablature generation
        notes = cleaned_notes
        
        # Detect tempo and time signature
        tempo = self._detect_tempo()
        beats_per_minute = tempo['tempo']
        time_sig = self._detect_time_signature()
        beats_per_measure = time_sig[0]
        
        # Calculate beat and measure durations
        beat_duration = 60.0 / beats_per_minute
        measure_duration = beats_per_measure * beat_duration
        
        # Get audio duration
        audio_duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        # Calculate total measures and group into bars
        num_measures = max(1, int(audio_duration / measure_duration))
        measures_per_bar = 4  # Standard for guitar tab display
        num_bars = (num_measures + measures_per_bar - 1) // measures_per_bar
        
        # Generate bars and measures
        bars = []
        for bar_idx in range(num_bars):
            bar_start = bar_idx * measures_per_bar * measure_duration
            bar_end = min(audio_duration, (bar_idx + 1) * measures_per_bar * measure_duration)
            
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
                m_end = min(audio_duration, (measure_idx + 1) * measure_duration)
                
                # Find notes in this measure
                measure_notes = [note for note in notes if m_start <= note['time'] < m_end]
                
                # Create tab lines for this measure
                tab_width = int(beats_per_measure * 4)  # 4 chars per beat
                tab_lines = [
                    f"e|{'-' * tab_width}|",
                    f"B|{'-' * tab_width}|",
                    f"G|{'-' * tab_width}|", 
                    f"D|{'-' * tab_width}|",
                    f"A|{'-' * tab_width}|",
                    f"E|{'-' * tab_width}|"
                ]
                
                # Add notes to tab lines with collision detection
                measure_notes_by_position = {}
                
                # First, group notes by their position in the tab to detect collisions
                for note in measure_notes:
                    string_idx = note.get('string')
                    fret = note.get('fret')
                    
                    if string_idx is None or fret is None:
                        continue
                        
                    # Calculate position in the tab
                    relative_time = note['time'] - m_start
                    position_ratio = relative_time / (m_end - m_start)
                    position = int(position_ratio * tab_width)
                    
                    # Ensure position is within bounds
                    position = max(0, min(position, tab_width - 1))
                    
                    # Group by position
                    if position not in measure_notes_by_position:
                        measure_notes_by_position[position] = []
                    
                    measure_notes_by_position[position].append({
                        'string': string_idx,
                        'fret': fret
                    })
                
                # Now iterate through positions and handle collisions
                for position, notes_at_pos in measure_notes_by_position.items():
                    # Sort by string to ensure consistent rendering
                    notes_at_pos.sort(key=lambda n: n['string'])
                    
                    # Add each note to the tab, adjusting position if needed to avoid collisions
                    for note_data in notes_at_pos:
                        string_idx = note_data['string']
                        fret = note_data['fret']
                        fret_str = str(fret)
                        
                        # Add fret number to tab
                        line = list(tab_lines[string_idx])
                        
                        # Position in the tab line (adding 1 to account for the | character)
                        pos = position + 1
                        
                        # Check for collision - look a few positions ahead for multi-digit frets
                        need_shift = False
                        pos_offset = 0
                        
                        # Try positions to the right first (up to 2 positions)
                        for test_offset in range(3):
                            if pos + test_offset < len(line) - 1 and line[pos + test_offset] != '-':
                                need_shift = True
                                continue
                            
                            # Found a clean spot
                            if need_shift:
                                pos_offset = test_offset
                                break
                        
                        # If we still have a collision, try to the left
                        if need_shift and pos_offset == 0:
                            for test_offset in range(-1, -3, -1):
                                if pos + test_offset > 0 and line[pos + test_offset] != '-':
                                    continue
                                
                                # Found a clean spot
                                pos_offset = test_offset
                                break
                        
                        # Place the fret number with adjusted position
                        adjusted_pos = pos + pos_offset
                        
                        # Place the fret number, handling multi-digit frets
                        if len(fret_str) == 1 and 0 < adjusted_pos < len(line) - 1:
                            line[adjusted_pos] = fret_str
                        else:
                            # For multi-digit frets, ensure we have space
                            for i, digit in enumerate(fret_str):
                                if 0 < adjusted_pos + i < len(line) - 1:  # Ensure we don't overwrite the borders
                                    line[adjusted_pos + i] = digit
                        
                        tab_lines[string_idx] = ''.join(line)
                
                # Create measure object
                beat_positions = [float(m_start + b * beat_duration) for b in range(beats_per_measure + 1)]
                
                measure = {
                    'measureNumber': measure_idx + 1,
                    'startTime': float(m_start),
                    'endTime': float(m_end),
                    'tabLines': tab_lines,
                    'beatPositions': beat_positions
                }
                
                bar['measures'].append(measure)
            
            # Only add bar if it has measures
            if bar['measures']:
                bars.append(bar)
        
        # Create full tablature data structure
        tablature = {
            'bars': bars,
            'totalDuration': float(audio_duration),
            'gridResolution': 0.125,  # 16th note resolution
            'notePositions': []
        }
        
        # Add note positions for visualization
        for note in notes:
            if 'string' in note and 'fret' in note:
                tablature['notePositions'].append({
                    'note': note['note'],
                    'time': float(note['time']),
                    'duration': float(note['duration']),
                    'string': note['string'],
                    'fret': note['fret'],
                    'confidence': note.get('confidence', 0.8)
                })
        
        # Final output includes ASCII tablature
        tab_lines = []
        for bar in bars:
            for measure in bar['measures']:
                tab_lines.extend(measure['tabLines'])
                tab_lines.append('')  # Empty line between measures
        
        # Join tab lines into a single string
        tab_string = '\n'.join(tab_lines)
        tablature['tablature'] = tab_lines
        
        return tablature
    
    def _detect_tempo(self) -> Dict[str, Any]:
        """Detect tempo from audio"""
        # Start with onset strength
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        
        # Detect tempo
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sr)[0]
        
        # Calculate confidence based on clarity of beat
        S = np.abs(librosa.stft(self.y))
        fourier_tempo = librosa.beat.tempo(onset_envelope=S.mean(axis=0), sr=self.sr)[0]
        
        # If both methods agree, higher confidence
        confidence = 1.0 - min(0.5, abs(tempo - fourier_tempo) / tempo)
        
        return {
            'tempo': float(tempo),
            'confidence': float(confidence)
        }
        
    def _detect_time_signature(self) -> Tuple[int, int]:
        """Detect time signature from audio"""
        # Simplified algorithm that favors 4/4 as the default
        # This is reasonable since 4/4 is by far the most common time signature
        
        # Detect beats
        tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        
        # Convert to seconds
        beat_times = librosa.frames_to_time(beats, sr=self.sr)
        
        # Calculate intervals between beats
        if len(beat_times) >= 8:
            # Try to group beats into measures and detect the pattern
            beat_intervals = np.diff(beat_times)
            
            # Check for consistent patterns of emphasis
            # For a more robust implementation, we would analyze the audio's
            # dynamic profile at each beat and detect patterns of emphasis
            
            # For now, default to 4/4 time as it's most common for guitar music
            return (4, 4)
        else:
            return (4, 4)  # Default to 4/4 if not enough beats
