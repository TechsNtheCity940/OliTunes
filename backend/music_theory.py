from audio_imports import *

class MusicTheoryAnalyzer:
    """
    Analyzes musical key, chord progressions, scales, and other music theory components.
    Uses Music21 when available with graceful fallback to simpler algorithms.
    """
    
    # Major and minor key profiles (Krumhansl-Schmuckler key-finding algorithm)
    MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    def __init__(self):
        self.has_music21 = MUSIC21_AVAILABLE
    
    def detect_key(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Detect the musical key of the audio
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Dictionary with key information
        """
        # Advanced method with Music21 if available
        if self.has_music21:
            try:
                return self._detect_key_music21(audio, sr)
            except Exception as e:
                logger.warning(f"Music21 key detection failed: {str(e)}")
        
        # Fallback to chromagram-based detection
        return self._detect_key_chromagram(audio, sr)
    
    def _detect_key_music21(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Use Music21 for key detection"""
        import music21
        
        # Get the chromagram
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        
        # Average the chroma over time
        chroma_avg = np.mean(chroma, axis=1)
        
        # Normalize
        if np.sum(chroma_avg) > 0:
            chroma_avg = chroma_avg / np.sum(chroma_avg)
        
        # Convert to Music21 pitch class distribution
        pc_distribution = [float(val) for val in chroma_avg]
        
        # Use Music21's key analyzer
        analyzer = music21.analysis.discrete.KeyWeightKeyAnalysis()
        music21_key = analyzer.solve(pc_distribution)
        
        # Get key details
        key_name = str(music21_key)
        mode = 'minor' if key_name.endswith('minor') else 'major'
        tonic = key_name.split()[0]
        
        # Get scale degrees and chords
        key_obj = music21.key.Key(tonic, mode)
        scale = key_obj.getPitches()
        scale_names = [p.name for p in scale]
        
        # Get common chord progression for this key
        chords = self._get_key_chords(key_obj)
        
        return {
            'key': key_name,
            'tonic': tonic,
            'mode': mode,
            'scale': scale_names,
            'confidence': 0.85,  # Music21 doesn't provide confidence
            'chords': chords
        }
    
    def _detect_key_chromagram(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect key using Krumhansl-Schmuckler algorithm"""
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        
        # Average over all frames
        chroma_avg = np.mean(chroma, axis=1)
        
        # Normalize
        if np.sum(chroma_avg) > 0:
            chroma_avg = chroma_avg / np.sum(chroma_avg)
        
        # Correlate with key profiles
        key_correlations = []
        
        # Check correlation with all possible keys
        for i in range(12):  # 12 possible tonics
            # Major key correlation
            major_corr = np.corrcoef(np.roll(self.MAJOR_PROFILE, i), chroma_avg)[0, 1]
            key_correlations.append((major_corr, i, 'major'))
            
            # Minor key correlation
            minor_corr = np.corrcoef(np.roll(self.MINOR_PROFILE, i), chroma_avg)[0, 1]
            key_correlations.append((minor_corr, i, 'minor'))
        
        # Sort by correlation (highest first)
        key_correlations.sort(reverse=True)
        
        # Get the best match
        best_corr, tonic_idx, mode = key_correlations[0]
        
        # Map index to note name
        tonic_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        tonic = tonic_names[tonic_idx]
        
        # Format key name
        key_name = f"{tonic} {mode}"
        
        # Get scale
        scale = self._get_scale_notes(tonic, mode)
        
        # Generate chords
        chords = self._get_simple_chord_progression(tonic, mode)
        
        return {
            'key': key_name,
            'tonic': tonic,
            'mode': mode,
            'scale': scale,
            'confidence': (best_corr + 1) / 2,  # Convert correlation (-1 to 1) to confidence (0 to 1)
            'chords': chords
        }
    
    def _get_scale_notes(self, tonic: str, mode: str) -> List[str]:
        """Get the notes of a scale given the tonic and mode"""
        # Intervals for major and minor scales (in semitones)
        major_intervals = [0, 2, 4, 5, 7, 9, 11]
        minor_intervals = [0, 2, 3, 5, 7, 8, 10]
        
        # Select intervals based on mode
        intervals = major_intervals if mode == 'major' else minor_intervals
        
        # Map tonic name to MIDI note number for C0
        tonic_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        tonic_idx = tonic_names.index(tonic)
        
        # Generate scale notes
        scale = []
        for interval in intervals:
            note_idx = (tonic_idx + interval) % 12
            scale.append(tonic_names[note_idx])
        
        return scale
    
    def _get_key_chords(self, key_obj) -> List[Dict[str, Any]]:
        """Get common chord progression for a Music21 key"""
        import music21
        
        # Generate diatonic chords
        chords = []
        for scale_degree in range(1, 8):
            chord = key_obj.getDiatonicTriad(scale_degree)
            
            # Get chord symbol
            chord_symbol = self._get_chord_symbol(chord, key_obj, scale_degree)
            
            # Get chord notes
            notes = [p.nameWithOctave for p in chord.pitches]
            
            chords.append({
                'name': chord_symbol,
                'notes': notes,
                'scale_degree': scale_degree,
                'roman': music21.roman.romanNumeralFromChord(chord, key_obj).figure
            })
        
        return chords
    
    def _get_chord_symbol(self, chord, key, scale_degree):
        """Get chord symbol (e.g., Cmaj, Dmin)"""
        root = chord.root().name
        chord_type = chord.commonName
        
        # Simple mapping of chord types
        if 'minor' in chord_type:
            return f"{root}min"
        elif 'major' in chord_type:
            return f"{root}maj"
        elif 'diminished' in chord_type:
            return f"{root}dim"
        elif 'augmented' in chord_type:
            return f"{root}aug"
        else:
            return root
    
    def _get_simple_chord_progression(self, tonic: str, mode: str) -> List[Dict[str, Any]]:
        """Generate a simple chord progression for a key when Music21 is not available"""
        # Note indices for major and minor scales
        major_indices = [0, 2, 4, 5, 7, 9, 11]
        minor_indices = [0, 2, 3, 5, 7, 8, 10]
        
        # Chord quality patterns for major and minor keys
        major_qualities = ['maj', 'min', 'min', 'maj', 'maj', 'min', 'dim']
        minor_qualities = ['min', 'dim', 'maj', 'min', 'min', 'maj', 'maj']
        
        # Roman numeral notation
        major_romans = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°']
        minor_romans = ['i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII']
        
        # Select appropriate arrays based on mode
        indices = major_indices if mode == 'major' else minor_indices
        qualities = major_qualities if mode == 'major' else minor_qualities
        romans = major_romans if mode == 'major' else minor_romans
        
        # Map tonic to index
        tonic_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        tonic_idx = tonic_names.index(tonic)
        
        # Generate chords
        chords = []
        for i, (interval, quality, roman) in enumerate(zip(indices, qualities, romans)):
            # Calculate root note index
            root_idx = (tonic_idx + interval) % 12
            root = tonic_names[root_idx]
            
            # Generate chord notes (simple triads)
            if quality == 'maj':
                # Major triad: root, major 3rd (4 semitones), perfect 5th (7 semitones)
                note_indices = [root_idx, (root_idx + 4) % 12, (root_idx + 7) % 12]
            elif quality == 'min':
                # Minor triad: root, minor 3rd (3 semitones), perfect 5th (7 semitones)
                note_indices = [root_idx, (root_idx + 3) % 12, (root_idx + 7) % 12]
            elif quality == 'dim':
                # Diminished triad: root, minor 3rd (3 semitones), diminished 5th (6 semitones)
                note_indices = [root_idx, (root_idx + 3) % 12, (root_idx + 6) % 12]
            else:  # 'aug'
                # Augmented triad: root, major 3rd (4 semitones), augmented 5th (8 semitones)
                note_indices = [root_idx, (root_idx + 4) % 12, (root_idx + 8) % 12]
            
            # Convert indices to note names
            notes = [tonic_names[idx] for idx in note_indices]
            
            # Add chord to list
            chords.append({
                'name': f"{root}{quality}",
                'notes': notes,
                'scale_degree': i + 1,
                'roman': roman
            })
        
        return chords
    
    def analyze_chord_progression(self, audio: np.ndarray, sr: int, 
                                segment_duration: float = 1.0) -> List[Dict[str, Any]]:
        """
        Analyze the chord progression in the audio
        
        Args:
            audio: Audio data
            sr: Sample rate
            segment_duration: Duration of each segment to analyze (in seconds)
            
        Returns:
            List of detected chords with timing information
        """
        if self.has_music21:
            try:
                return self._analyze_chords_music21(audio, sr, segment_duration)
            except Exception as e:
                logger.warning(f"Music21 chord analysis failed: {str(e)}")
        
        # Fallback to simple chord detection
        return self._analyze_chords_simple(audio, sr, segment_duration)
    
    def _analyze_chords_music21(self, audio: np.ndarray, sr: int, 
                              segment_duration: float) -> List[Dict[str, Any]]:
        """Use Music21 for chord analysis"""
        import music21
        
        # Compute chromagram
        hop_length = 512
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
        
        # Get times for each chroma frame
        chroma_times = librosa.times_like(chroma, sr=sr, hop_length=hop_length)
        
        # First detect the overall key
        key_info = self.detect_key(audio, sr)
        key_obj = music21.key.Key(key_info['tonic'], key_info['mode'])
        
        # Define segment boundaries (in frames)
        segment_frames = int(segment_duration * sr / hop_length)
        num_segments = max(1, chroma.shape[1] // segment_frames)
        
        chords = []
        
        for i in range(num_segments):
            # Get the segment range
            start_frame = i * segment_frames
            end_frame = min(chroma.shape[1], (i + 1) * segment_frames)
            
            if end_frame <= start_frame:
                continue
            
            # Get the segment times
            start_time = chroma_times[start_frame]
            end_time = chroma_times[min(len(chroma_times) - 1, end_frame - 1)]
            
            # Get the average chroma for this segment
            segment_chroma = np.mean(chroma[:, start_frame:end_frame], axis=1)
            
            # Convert to pitch class distribution for Music21
            pc_distribution = [float(val) for val in segment_chroma]
            
            # Use Music21 to identify the chord
            try:
                analyzer = music21.analysis.discrete.ChordWeightAnalysis()
                music21_chord = analyzer.solveProblem(pc_distribution)
                
                # Get chord details
                chord_root = music21_chord.root().name
                chord_type = music21_chord.commonName
                chord_name = f"{chord_root} {chord_type}"
                
                # Get chord notes
                notes = [p.nameWithOctave for p in music21_chord.pitches]
                
                # Get Roman numeral
                roman = music21.roman.romanNumeralFromChord(music21_chord, key_obj).figure
                
                chords.append({
                    'name': chord_name,
                    'time': float(start_time),
                    'duration': float(end_time - start_time),
                    'notes': notes,
                    'roman': roman
                })
            except Exception as e:
                logger.warning(f"Error analyzing chord at {start_time}s: {str(e)}")
        
        return chords
    
    def _analyze_chords_simple(self, audio: np.ndarray, sr: int, 
                             segment_duration: float) -> List[Dict[str, Any]]:
        """Simple chord detection using chromagram templates"""
        # Define chord templates for major and minor triads
        chord_templates = {}
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Create templates for major and minor chords
        for i, root in enumerate(note_names):
            # Major chord: root, major third, perfect fifth
            major_template = np.zeros(12)
            major_template[i] = 1.0  # root
            major_template[(i + 4) % 12] = 0.8  # major third
            major_template[(i + 7) % 12] = 0.9  # perfect fifth
            chord_templates[f"{root}maj"] = major_template
            
            # Minor chord: root, minor third, perfect fifth
            minor_template = np.zeros(12)
            minor_template[i] = 1.0  # root
            minor_template[(i + 3) % 12] = 0.8  # minor third
            minor_template[(i + 7) % 12] = 0.9  # perfect fifth
            chord_templates[f"{root}min"] = minor_template
        
        # Compute chromagram
        hop_length = 512
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
        
        # Get times for each chroma frame
        chroma_times = librosa.times_like(chroma, sr=sr, hop_length=hop_length)
        
        # Define segment boundaries (in frames)
        segment_frames = int(segment_duration * sr / hop_length)
        num_segments = max(1, chroma.shape[1] // segment_frames)
        
        # First detect overall key for context
        key_info = self._detect_key_chromagram(audio, sr)
        
        chords = []
        
        for i in range(num_segments):
            # Get the segment range
            start_frame = i * segment_frames
            end_frame = min(chroma.shape[1], (i + 1) * segment_frames)
            
            if end_frame <= start_frame:
                continue
            
            # Get the segment times
            start_time = chroma_times[start_frame]
            end_time = chroma_times[min(len(chroma_times) - 1, end_frame - 1)]
            
            # Get the average chroma for this segment
            segment_chroma = np.mean(chroma[:, start_frame:end_frame], axis=1)
            
            # Normalize
            if np.sum(segment_chroma) > 0:
                segment_chroma = segment_chroma / np.sum(segment_chroma)
            
            # Compare with chord templates
            chord_matches = []
            for chord_name, template in chord_templates.items():
                # Calculate correlation
                correlation = np.corrcoef(segment_chroma, template)[0, 1]
                chord_matches.append((correlation, chord_name))
            
            # Sort by correlation
            chord_matches.sort(reverse=True)
            
            # Get the best match
            if chord_matches and not np.isnan(chord_matches[0][0]):
                best_corr, chord_name = chord_matches[0]
                
                # Only add if correlation is reasonable
                if best_corr > 0.5:
                    # Parse chord name to get root and quality
                    if chord_name.endswith('maj'):
                        root = chord_name[:-3]
                        quality = 'maj'
                    elif chord_name.endswith('min'):
                        root = chord_name[:-3]
                        quality = 'min'
                    else:
                        root = chord_name
                        quality = ''
                    
                    # Generate notes for this chord
                    root_idx = note_names.index(root)
                    if quality == 'maj':
                        note_indices = [root_idx, (root_idx + 4) % 12, (root_idx + 7) % 12]
                    else:  # 'min'
                        note_indices = [root_idx, (root_idx + 3) % 12, (root_idx + 7) % 12]
                    
                    notes = [note_names[idx] for idx in note_indices]
                    
                    # Determine roman numeral (simplified)
                    roman = self._get_simple_roman_numeral(root, quality, key_info)
                    
                    chords.append({
                        'name': chord_name,
                        'time': float(start_time),
                        'duration': float(end_time - start_time),
                        'notes': notes,
                        'roman': roman,
                        'confidence': float((best_corr + 1) / 2)  # Convert correlation to confidence
                    })
        
        return chords
    
    def _get_simple_roman_numeral(self, chord_root: str, chord_quality: str, key_info: Dict) -> str:
        """Get a simple Roman numeral for a chord relative to the key"""
        tonic = key_info['tonic']
        mode = key_info['mode']
        
        # Get scale degrees
        scale = key_info['scale']
        
        # Find the position of the chord root in the scale
        if chord_root in scale:
            scale_degree = scale.index(chord_root) + 1
        else:
            # Out of key chord
            return chord_root
        
        # Determine case and formatting
        if chord_quality == 'maj':
            # Major chord
            if mode == 'major':
                # In major key, degrees 1, 4, 5 are major
                if scale_degree in [1, 4, 5]:
                    return ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°'][scale_degree - 1]
                else:
                    return ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'][scale_degree - 1]
            else:
                # In minor key, degrees 3, 6, 7 are major
                if scale_degree in [3, 6, 7]:
                    return ['i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII'][scale_degree - 1]
                else:
                    return ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'][scale_degree - 1]
        else:
            # Minor chord
            if mode == 'major':
                # In major key, degrees 2, 3, 6 are minor
                if scale_degree in [2, 3, 6]:
                    return ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°'][scale_degree - 1]
                else:
                    return ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii'][scale_degree - 1]
            else:
                # In minor key, degrees 1, 4, 5 are minor
                if scale_degree in [1, 4, 5]:
                    return ['i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII'][scale_degree - 1]
                else:
                    return ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii'][scale_degree - 1]
    
    def suggest_scales(self, key_info: Dict[str, Any], 
                     purpose: str = 'improvisation') -> List[Dict[str, Any]]:
        """
        Suggest appropriate scales for improvisation or composition
        based on detected key
        
        Args:
            key_info: Key information dict from detect_key
            purpose: 'improvisation', 'composition', 'soloing'
            
        Returns:
            List of scale suggestions with notes
        """
        tonic = key_info['tonic']
        mode = key_info['mode']
        
        # Define scale patterns (intervals from tonic)
        scale_patterns = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'minor_pentatonic': [0, 3, 5, 7, 10],
            'major_pentatonic': [0, 2, 4, 7, 9],
            'blues': [0, 3, 5, 6, 7, 10],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],
            'lydian': [0, 2, 4, 6, 7, 9, 11],
            'phrygian': [0, 1, 3, 5, 7, 8, 10],
            'locrian': [0, 1, 3, 5, 6, 8, 10],
            'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
            'melodic_minor': [0, 2, 3, 5, 7, 9, 11]
        }
        
        # Map tonic to index
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        tonic_idx = note_names.index(tonic)
        
        suggestions = []
        
        # Based on purpose and detected key, select appropriate scales
        if purpose == 'improvisation':
            if mode == 'major':
                scale_types = ['major', 'major_pentatonic', 'mixolydian', 'lydian', 'blues']
            else:  # minor
                scale_types = ['minor', 'minor_pentatonic', 'blues', 'dorian', 'phrygian']
        elif purpose == 'composition':
            if mode == 'major':
                scale_types = ['major', 'lydian', 'mixolydian', 'dorian', 'major_pentatonic']
            else:  # minor
                scale_types = ['minor', 'harmonic_minor', 'melodic_minor', 'dorian', 'phrygian']
        elif purpose == 'soloing':
            if mode == 'major':
                scale_types = ['major_pentatonic', 'blues', 'mixolydian', 'lydian']
            else:  # minor
                scale_types = ['minor_pentatonic', 'blues', 'dorian', 'harmonic_minor']
        else:
            # Default - show all scales
            scale_types = list(scale_patterns.keys())
        
        # Generate scale notes for each suggested scale
        for scale_type in scale_types:
            pattern = scale_patterns[scale_type]
            
            # Generate notes
            notes = []
            for interval in pattern:
                note_idx = (tonic_idx + interval) % 12
                notes.append(note_names[note_idx])
            
            # Format scale name
            if scale_type == 'major':
                scale_name = f"{tonic} Major"
            elif scale_type == 'minor':
                scale_name = f"{tonic} Natural Minor"
            elif scale_type == 'minor_pentatonic':
                scale_name = f"{tonic} Minor Pentatonic"
            elif scale_type == 'major_pentatonic':
                scale_name = f"{tonic} Major Pentatonic"
            elif scale_type == 'blues':
                scale_name = f"{tonic} Blues"
            elif scale_type == 'dorian':
                scale_name = f"{tonic} Dorian"
            elif scale_type == 'mixolydian':
                scale_name = f"{tonic} Mixolydian"
            elif scale_type == 'lydian':
                scale_name = f"{tonic} Lydian"
            elif scale_type == 'phrygian':
                scale_name = f"{tonic} Phrygian"
            elif scale_type == 'locrian':
                scale_name = f"{tonic} Locrian"
            elif scale_type == 'harmonic_minor':
                scale_name = f"{tonic} Harmonic Minor"
            elif scale_type == 'melodic_minor':
                scale_name = f"{tonic} Melodic Minor"
            else:
                scale_name = f"{tonic} {scale_type.replace('_', ' ').title()}"
            
            suggestions.append({
                'name': scale_name,
                'notes': notes,
                'type': scale_type,
                'tonic': tonic,
                'intervals': pattern
            })
        
        return suggestions
    
    def determine_chord_from_notes(self, notes: List[str]) -> Dict[str, Any]:
        """
        Determine the chord type and properties from a list of note names
        
        Args:
            notes: List of note names (e.g., ['C4', 'E4', 'G4'])
            
        Returns:
            Dictionary containing chord information:
            {
                'root': str,    # Root note
                'type': str,    # Chord type (e.g., 'major', 'minor', 'diminished')
                'quality': str, # Chord quality (e.g., 'maj7', 'm7', 'dim7')
                'roman': str,   # Roman numeral notation
                'notes': List[str], # List of notes in the chord
                'intervals': List[str] # List of interval names
            }
        """
        result = {
            'root': None,
            'type': None,
            'quality': None,
            'roman': None,
            'notes': [],
            'intervals': []
        }
        
        try:
            if not self.has_music21:
                raise ImportError("Music21 is not available")
            
            import music21
            
            # Create a chord from the notes
            m21_notes = [music21.note.Note(n) for n in notes]
            chord = music21.chord.Chord(m21_notes)
            
            # Get chord properties
            result['root'] = chord.root().name
            result['type'] = chord.chordKind
            result['quality'] = chord.quality
            result['notes'] = [p.nameWithOctave for p in chord.pitches]
            result['intervals'] = [str(i) for i in chord.intervalVector]
            
            # Try to get Roman numeral if we have a key
            if hasattr(self, '_current_key') and self._current_key:
                key_obj = music21.key.Key(self._current_key)
                roman = music21.roman.romanNumeralFromChord(chord, key_obj)
                result['roman'] = roman.figure
            
        except Exception as e:
            logger.warning(f"Error determining chord from notes: {str(e)}")
            
        return result
