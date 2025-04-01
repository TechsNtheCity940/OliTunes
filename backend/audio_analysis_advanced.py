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
        """Intelligent algorithm to map notes with fallback defaults"""
        mapped = []
        for note in notes:
            note = note.copy()
            if 'midi' not in note:
                continue
            
            # Add default values first
            note.setdefault('string', 0)
            note.setdefault('fret', 0)
            if not notes:
                return []
            
        # Sort notes by time
        notes = sorted(notes, key=lambda x: x['time'])
        
        # Ensure all notes have string and fret information
        if not all('string' in note and 'fret' in note for note in notes):
            notes = self._map_notes_to_fretboard(notes)
        
        # Create a deep copy to avoid modifying original
        result = []
        for note in notes:
            # Create a deep copy of each note
            new_note = dict(note)
            result.append(new_note)
            
        # Current hand position tracker
        current_position = 0  # Initial position is open strings
        
        # For each note, find optimal string/fret position
        for i, note in enumerate(result):
            midi_note = note.get('midi')
            if midi_note is None:
                continue
                
            # Find all possible positions for this note on the fretboard
            positions = []
            for string_idx, open_string in enumerate(self.GUITAR_STRINGS):
                # Calculate which fret on this string produces the target note
                fret = midi_note - open_string
                if 0 <= fret <= self.MAX_FRET:
                    # Score this position based on several factors
                    # 1. Distance from current position
                    position_change = abs(fret - current_position)
                    position_score = max(0, 1 - position_change / 12)
                    
                    # 2. Preference for lower positions (more common)
                    fret_score = max(0, 1 - fret / self.MAX_FRET)
                    
                    # 3. String preference (middle strings often preferable)
                    string_score = 1 - abs(string_idx - 2.5) / 5
                    
                    # 4. Context from adjacent notes
                    context_score = 0
                    if i > 0 and 'string' in result[i-1] and 'fret' in result[i-1]:
                        # Prefer same string for sequential notes (typical for melodies)
                        if string_idx == result[i-1]['string']:
                            context_score += 0.5
                        
                        # Penalty for large jumps
                        fret_jump = abs(fret - result[i-1].get('fret', 0))
                        if fret_jump > self.MAX_REACH and string_idx == result[i-1]['string']:
                            context_score -= 0.5
                            
                    # Calculate total score (weighted sum)
                    total_score = (
                        0.3 * position_score +
                        0.3 * fret_score +
                        0.2 * string_score +
                        0.2 * context_score
                    )
                    
                    positions.append({
                        'string': string_idx,
                        'fret': fret,
                        'score': total_score
                    })
            
            # Choose best position based on score
            if positions:
                best_pos = max(positions, key=lambda p: p['score'])
                note['string'] = best_pos['string']
                note['fret'] = best_pos['fret']
                
                # Update current position
                current_position = best_pos['fret']
        
        return mapped
        
    def generate_tablature(self, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate guitar tablature from detected notes.
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
        
        # Ensure all notes have string and fret information
        if not all('string' in note and 'fret' in note for note in notes):
            notes = self._map_notes_to_fretboard(notes)
        
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
                
                # Add notes to tab lines
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
                    
                    # Add fret number to tab
                    line = list(tab_lines[string_idx])
                    fret_str = str(fret)
                    
                    # Position in the tab line (adding 1 to account for the | character)
                    pos = position + 1
                    
                    # Place the fret number, handling multi-digit frets
                    if len(fret_str) == 1:
                        line[pos] = fret_str
                    else:
                        # For multi-digit frets, ensure we have space
                        for i, digit in enumerate(fret_str):
                            if pos + i < len(line) - 1:  # Ensure we don't overwrite the ending |
                                line[pos + i] = digit
                    
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
