"""
Unified Audio Analyzer Part 2 - Note Detection and Guitar Analysis
This file contains the guitar-specific analysis methods for the UnifiedAudioAnalyzer class.
"""

# This will be imported by unified_audio_analyzer.py

def add_note_detection_methods(UnifiedAudioAnalyzer):
    """Add note detection methods to the UnifiedAudioAnalyzer class."""
    
    def detect_guitar_notes(self):
        """
        Detect guitar notes using the best available pitch detection algorithms.
        
        Uses a combination of techniques:
        1. Adaptive CQT analysis with optimized frequency bins
        2. Source separation to isolate melodic content when possible
        3. Advanced post-processing for note segmentation
        
        Returns:
            List of detected notes with timing and pitch information
        """
        logger.info("Detecting guitar notes...")
        
        # First, try to load from cache
        cache_file = os.path.join(self.cache_dir, f"{os.path.basename(self.audio_path)}_notes.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    notes = json.load(f)
                    logger.info(f"Loaded {len(notes)} notes from cache")
                    return notes
            except Exception as e:
                logger.warning(f"Failed to load notes from cache: {e}")
        
        # Ensure audio is loaded
        self._ensure_audio_loaded()
        
        # Preprocess audio to enhance guitar signal
        preprocessed_audio = self._preprocess_audio_for_guitar()
        
        # Try different note detection methods with fallbacks
        notes = []
        
        # Try CREPE first if available (highest accuracy)
        if CREPE_AVAILABLE:
            try:
                notes = self._detect_notes_with_crepe(preprocessed_audio)
                if notes and len(notes) > 0:
                    logger.info(f"CREPE detected {len(notes)} notes")
                    self._cache_notes(notes, cache_file)
                    return notes
            except Exception as e:
                logger.warning(f"CREPE note detection failed: {e}")
        
        # Try using Basic Pitch if available
        if BASIC_PITCH_AVAILABLE:
            try:
                basic_pitch_model = BasicPitch(self.audio_path)
                midi_data = basic_pitch_model.predict()
                
                if midi_data and hasattr(midi_data, 'instruments') and len(midi_data.instruments) > 0:
                    for note in midi_data.instruments[0].notes:
                        notes.append({
                            'start': note.start,
                            'end': note.end,
                            'pitch': note.pitch,
                            'velocity': note.velocity,
                            'frequency': librosa.midi_to_hz(note.pitch),
                            'confidence': 0.85
                        })
                    
                    if len(notes) > 0:
                        logger.info(f"Basic Pitch detected {len(notes)} notes")
                        self._cache_notes(notes, cache_file)
                        return notes
            except Exception as e:
                logger.warning(f"Basic Pitch note detection failed: {e}")
        
        # Fallback to CQT method
        try:
            notes = self._detect_notes_with_cqt(preprocessed_audio)
            if notes and len(notes) > 0:
                logger.info(f"CQT detected {len(notes)} notes")
                self._cache_notes(notes, cache_file)
                return notes
        except Exception as e:
            logger.warning(f"CQT note detection failed: {e}")
        
        # Last resort: basic note detection
        try:
            notes = self._detect_notes_basic(preprocessed_audio)
            if notes and len(notes) > 0:
                logger.info(f"Basic method detected {len(notes)} notes")
                self._cache_notes(notes, cache_file)
                return notes
        except Exception as e:
            logger.error(f"All note detection methods failed: {e}")
        
        return []
    
    def _preprocess_audio_for_guitar(self):
        """
        Preprocess audio to enhance guitar signal components.
        
        Returns:
            Processed audio signal
        """
        # Ensure audio is loaded
        self._ensure_audio_loaded()
        
        # Apply bandpass filter for guitar frequency range (80Hz to 1200Hz)
        nyquist = self.sr / 2.0
        low = 80 / nyquist
        high = 1200 / nyquist
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, self.y)
            
            # Apply harmonic-percussive separation if librosa version supports it
            try:
                harmonic, _ = librosa.effects.hpss(filtered)
                return harmonic
            except Exception as e:
                logger.warning(f"Harmonic separation failed: {e}")
                return filtered
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}")
            return self.y
    
    def _detect_notes_with_crepe(self, audio):
        """
        Use CREPE neural network for high-accuracy pitch detection if available.
        
        Args:
            audio: Preprocessed audio signal
            
        Returns:
            List of detected notes
        """
        import crepe
        
        # Run CREPE on the audio signal
        time, frequency, confidence, _ = crepe.predict(audio, self.sr, viterbi=True)
        
        # Convert continuous pitch to discrete notes
        notes = []
        current_note = None
        
        for i, (t, f, c) in enumerate(zip(time, frequency, confidence)):
            if c < 0.5:  # Skip low-confidence frames
                if current_note:
                    current_note['end'] = t
                    notes.append(current_note)
                    current_note = None
                continue
            
            # Convert frequency to MIDI pitch
            midi_pitch = librosa.hz_to_midi(f)
            
            # Round to nearest semitone
            midi_pitch_int = int(round(midi_pitch))
            
            # Check if continuing the same note
            if current_note and abs(midi_pitch - current_note['pitch']) < 0.5:
                # Same note continues
                pass
            else:
                # New note starts
                if current_note:
                    current_note['end'] = t
                    notes.append(current_note)
                
                current_note = {
                    'start': t,
                    'end': None,
                    'pitch': midi_pitch_int,
                    'frequency': f,
                    'confidence': c
                }
        
        # Add the last note if it exists
        if current_note:
            current_note['end'] = time[-1]
            notes.append(current_note)
        
        # Post-process notes
        return self._post_process_notes(notes)
    
    def _detect_notes_with_cqt(self, audio):
        """
        Detect notes using enhanced Constant-Q Transform analysis.
        
        Args:
            audio: Preprocessed audio signal
            
        Returns:
            List of detected notes
        """
        # Compute CQT spectrogram
        # Use smaller hop length for better time resolution
        hop_length = 512
        n_bins = 96  # 8 octaves
        bins_per_octave = 12  # semitone resolution
        
        # Calculate CQT
        C = np.abs(librosa.cqt(audio, sr=self.sr, hop_length=hop_length,
                              n_bins=n_bins, bins_per_octave=bins_per_octave))
        
        # Apply threshold to identify strong activations
        threshold = np.mean(C) + 1.5 * np.std(C)
        
        # Create a binary activation matrix
        activation = C > threshold
        
        # Extract notes from the activation matrix
        notes = []
        active_pitches = {}  # Dictionary to track active pitches
        
        for t in range(activation.shape[1]):
            time = librosa.frames_to_time(t, sr=self.sr, hop_length=hop_length)
            
            # Find active pitches at this time frame
            active_at_t = np.where(activation[:, t])[0]
            
            # Close notes that are no longer active
            for pitch in list(active_pitches.keys()):
                if pitch not in active_at_t:
                    note = active_pitches[pitch]
                    note['end'] = time
                    notes.append(note)
                    del active_pitches[pitch]
            
            # Start new notes
            for pitch in active_at_t:
                if pitch not in active_pitches:
                    # Convert CQT bin to MIDI pitch
                    midi_pitch = pitch + 24  # CQT bins typically start at C1 (MIDI 24)
                    
                    # Create new note
                    active_pitches[pitch] = {
                        'start': time,
                        'end': None,
                        'pitch': midi_pitch,
                        'frequency': librosa.midi_to_hz(midi_pitch),
                        'confidence': C[pitch, t] / np.max(C[:, t]) if np.max(C[:, t]) > 0 else 0.5
                    }
        
        # Close any remaining notes
        for pitch, note in active_pitches.items():
            note['end'] = librosa.frames_to_time(activation.shape[1] - 1, sr=self.sr, hop_length=hop_length)
            notes.append(note)
        
        # Post-process notes
        return self._post_process_notes(notes)
    
    def _detect_notes_basic(self, audio):
        """
        Basic note detection as a fallback.
        
        Args:
            audio: Preprocessed audio signal
            
        Returns:
            List of detected notes
        """
        # Use onset detection with pitch estimation
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sr, 
                                                 backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)
        
        # Estimate durations based on onset spacing
        durations = np.diff(onset_times, append=0.5)  # Default 0.5s for last note
        
        # Simple pitch estimation at each onset
        notes = []
        for i, (onset, duration) in enumerate(zip(onset_times, durations)):
            # Extract segment around onset
            start_idx = int(onset * self.sr)
            end_idx = min(len(audio) - 1, start_idx + int(duration * self.sr))
            
            if end_idx <= start_idx or end_idx >= len(audio):
                continue
                
            segment = audio[start_idx:end_idx]
            
            if len(segment) < 1:
                continue
            
            # Estimate pitch using YIN algorithm
            pitches, magnitudes = librosa.piptrack(y=segment, sr=self.sr)
            
            if pitches.size == 0 or magnitudes.size == 0:
                continue
                
            # Find pitch with highest magnitude
            i_pitch, i_mag = np.unravel_index(magnitudes.argmax(), magnitudes.shape)
            pitch = pitches[i_pitch, i_mag]
            
            if pitch > 0:
                midi_pitch = librosa.hz_to_midi(pitch)
                
                notes.append({
                    'start': onset,
                    'end': onset + duration,
                    'pitch': int(round(midi_pitch)),
                    'frequency': pitch,
                    'confidence': 0.6
                })
        
        return self._post_process_notes(notes)
    
    def _post_process_notes(self, notes):
        """
        Apply post-processing to refine detected notes.
        
        Args:
            notes: List of detected note dictionaries
            
        Returns:
            Processed notes with improved timing and pitch
        """
        if not notes:
            return []
            
        # Sort notes by start time
        notes = sorted(notes, key=lambda x: x['start'])
        
        # Merge very short notes of the same pitch
        MIN_DURATION = 0.05  # 50ms
        merged_notes = []
        
        i = 0
        while i < len(notes):
            current = notes[i]
            
            # Skip invalid notes
            if current['end'] is None or current['start'] is None:
                i += 1
                continue
                
            # Calculate duration
            duration = current['end'] - current['start']
            
            # Try to merge with next note if it's the same pitch and close in time
            while (i + 1 < len(notes) and 
                   notes[i + 1]['pitch'] == current['pitch'] and
                   notes[i + 1]['start'] - current['end'] < MIN_DURATION):
                
                # Merge with next note
                current['end'] = notes[i + 1]['end']
                current['confidence'] = max(current['confidence'], notes[i + 1]['confidence'])
                i += 1
            
            # Add to merged notes if duration is sufficient
            if current['end'] - current['start'] >= MIN_DURATION:
                merged_notes.append(current)
            
            i += 1
        
        # Remove overlapping notes, keeping the higher confidence one
        clean_notes = []
        for note in merged_notes:
            # Check if this note overlaps with any in clean_notes
            overlapping = False
            for i, clean_note in enumerate(clean_notes):
                # Check for overlap
                if (note['start'] < clean_note['end'] and
                    note['end'] > clean_note['start']):
                    
                    # If notes overlap significantly and have the same pitch
                    if (note['pitch'] == clean_note['pitch'] and
                        min(note['end'], clean_note['end']) - max(note['start'], clean_note['start']) > 
                        0.5 * (note['end'] - note['start'])):
                        
                        # Keep the one with higher confidence
                        if note['confidence'] > clean_note['confidence']:
                            clean_notes[i] = note
                    
                    overlapping = True
                    break
            
            if not overlapping:
                clean_notes.append(note)
        
        # Add note names and octaves
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for note in clean_notes:
            midi_pitch = note['pitch']
            note_idx = midi_pitch % 12
            octave = midi_pitch // 12 - 1
            
            note['name'] = f"{note_names[note_idx]}{octave}"
        
        return clean_notes
    
    def _cache_notes(self, notes, cache_file):
        """
        Cache detected notes to file.
        
        Args:
            notes: List of note dictionaries
            cache_file: Path to cache file
        """
        try:
            with open(cache_file, 'w') as f:
                json.dump(notes, f)
            logger.info(f"Cached {len(notes)} notes to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache notes: {e}")
    
    def map_notes_to_fretboard(self, notes):
        """
        Map notes to optimal positions on guitar fretboard.
        
        Uses algorithms to determine the best fingering and 
        string choices for playability.
        
        Args:
            notes: List of note dictionaries
            
        Returns:
            Notes with added fretboard position information
        """
        if not notes:
            return []
            
        # Sort notes by start time
        notes = sorted(notes, key=lambda x: x['start'])
        
        # Previous position for continuity
        prev_position = None
        
        # Get key for musical context
        key_info = self.detect_key()
        key = key_info.get('key', 'C')
        
        # Map each note to the best position
        for note in notes:
            position = self.find_best_guitar_position(note['name'], prev_position, key)
            
            if position:
                string_idx, fret = position
                note['string'] = string_idx
                note['fret'] = fret
                prev_position = position
            else:
                # Fallback if no position found
                note['string'] = None
                note['fret'] = None
        
        return notes
    
    def find_best_guitar_position(self, note_name, prev_position=None, key=None):
        """
        Find best playable position for a note on guitar.
        
        Args:
            note_name: The name of the note (e.g., "C4")
            prev_position: Previous (string, fret) position for continuity
            key: Musical key for context
            
        Returns:
            Tuple of (string_index, fret_number)
        """
        # Get all possible positions for this note
        positions = self.note_to_guitar_position(note_name)
        
        if not positions:
            return None
            
        # If only one position exists, return it
        if len(positions) == 1:
            return positions[0]
            
        # Score each position
        position_scores = {}
        
        for pos in positions:
            string_idx, fret = pos
            
            # Base score starts at 10
            score = 10.0
            
            # Prefer lower frets for playability
            if fret <= 5:
                score += 5.0
            elif fret <= 12:
                score += 2.0
                
            # Prefer middle strings for melody
            if string_idx in [1, 2, 3]:
                score += 2.0
                
            # Prefer open strings in certain keys (C, G, D, A, E)
            if fret == 0 and key in ['C', 'G', 'D', 'A', 'E']:
                score += 3.0
                
            # Consider previous position for continuity
            if prev_position:
                prev_string, prev_fret = prev_position
                
                # Distance metric (simplified Manhattan distance)
                distance = abs(string_idx - prev_string) + abs(fret - prev_fret)
                
                # Prefer positions close to previous note
                if distance <= 2:
                    score += 4.0
                elif distance <= 4:
                    score += 2.0
                elif distance <= 7:
                    score += 0.5
                    
                # Penalize string skipping
                if abs(string_idx - prev_string) > 1:
                    score -= 1.0 * abs(string_idx - prev_string)
                    
                # Penalize large fret jumps
                if abs(fret - prev_fret) > 3:
                    score -= 0.5 * abs(fret - prev_fret)
            
            position_scores[pos] = score
        
        # Find position with highest score
        best_position = max(position_scores.items(), key=lambda x: x[1])[0]
        return best_position
    
    def note_to_guitar_position(self, note_name):
        """
        Convert a note name to all possible positions on guitar fretboard.
        
        Args:
            note_name: The name of the note (e.g., "C4")
            
        Returns:
            List of tuples (string_index, fret_number)
            string_index: 0=high E, 5=low E
            fret_number: 0=open string, 1=first fret, etc.
        """
        # Extract pitch and octave
        if len(note_name) < 2:
            return []
            
        try:
            note_pitch = note_name[:-1]  # Everything except the last character
            octave = int(note_name[-1])  # Last character as octave number
            
            # Convert to MIDI pitch
            pitch_class = self.PITCH_CLASSES.get(note_pitch)
            if pitch_class is None:
                return []
                
            midi_pitch = pitch_class + (octave + 1) * 12
            
            # Find all positions on the fretboard
            positions = []
            
            for string_idx, open_string_pitch in enumerate(self.TUNING_MIDI):
                # Calculate fret number
                fret = midi_pitch - open_string_pitch
                
                # Add if within playable range (0-24 typically)
                if 0 <= fret <= 24:
                    positions.append((string_idx, fret))
                    
            return positions
        except Exception as e:
            logger.warning(f"Error converting note {note_name} to guitar position: {e}")
            return []
    
    # Add the methods to the class
    UnifiedAudioAnalyzer.detect_guitar_notes = detect_guitar_notes
    UnifiedAudioAnalyzer._preprocess_audio_for_guitar = _preprocess_audio_for_guitar
    UnifiedAudioAnalyzer._detect_notes_with_crepe = _detect_notes_with_crepe
    UnifiedAudioAnalyzer._detect_notes_with_cqt = _detect_notes_with_cqt
    UnifiedAudioAnalyzer._detect_notes_basic = _detect_notes_basic
    UnifiedAudioAnalyzer._post_process_notes = _post_process_notes
    UnifiedAudioAnalyzer._cache_notes = _cache_notes
    UnifiedAudioAnalyzer.map_notes_to_fretboard = map_notes_to_fretboard
    UnifiedAudioAnalyzer.find_best_guitar_position = find_best_guitar_position
    UnifiedAudioAnalyzer.note_to_guitar_position = note_to_guitar_position
    
    return UnifiedAudioAnalyzer
