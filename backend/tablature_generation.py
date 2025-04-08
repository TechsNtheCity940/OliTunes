from audio_imports import *

# Try to import optional advanced dependencies
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow available for enhanced tablature generation")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Using fallback algorithms for tablature generation")

try:
    import music21
    MUSIC21_AVAILABLE = True
    logger.info("Music21 available for enhanced music theory integration")
except ImportError:
    MUSIC21_AVAILABLE = False
    logger.warning("Music21 not available. Using basic music theory algorithms")

class TablatureGenerator:
    """
    Generates guitar and bass tablature from detected notes.
    Optimizes fretboard positions for playability using advanced algorithms.
    """
    
    # Define constants for guitar string tunings
    STANDARD_GUITAR_TUNING = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']  # Low to high
    STANDARD_BASS_TUNING = ['E1', 'A1', 'D2', 'G2']  # Low to high
    DROP_D_TUNING = ['D2', 'A2', 'D3', 'G3', 'B3', 'E4']
    OPEN_G_TUNING = ['D2', 'G2', 'D3', 'G3', 'B3', 'D4']
    DADGAD_TUNING = ['D2', 'A2', 'D3', 'G3', 'A3', 'D4']
    
    # Define guitar fretboard range
    MAX_FRET = 24
    
    # Guitar playing parameters for better position inference
    IDEAL_FRET_RANGE = (0, 12)  # Most comfortable fret range
    STRETCH_LIMIT = 4  # Max frets one hand can comfortably stretch
    
    # Scale patterns for position inference
    MAJOR_SCALE_PATTERN = [0, 2, 4, 5, 7, 9, 11]
    MINOR_SCALE_PATTERN = [0, 2, 3, 5, 7, 8, 10]
    PENTATONIC_MAJOR_PATTERN = [0, 2, 4, 7, 9]
    PENTATONIC_MINOR_PATTERN = [0, 3, 5, 7, 10]
    
    def __init__(self, instrument: str = 'guitar', tuning: List[str] = None, 
                 use_ml: bool = True, include_theory: bool = True,
                 position_bias: str = 'optimal'):
        """
        Initialize the tablature generator
        
        Args:
            instrument: 'guitar' or 'bass'
            tuning: List of string tunings from low to high (defaults to standard)
            use_ml: Whether to use machine learning for position inference when available
            include_theory: Whether to use music theory for better mapping
            position_bias: 'optimal', 'low', 'mid', or 'high' position preference on neck
        """
        self.instrument = instrument.lower()
        self.use_ml = use_ml and TENSORFLOW_AVAILABLE
        self.include_theory = include_theory
        self.position_bias = position_bias
        
        # Set default tuning based on instrument
        if tuning is None:
            if self.instrument == 'guitar':
                self.tuning = self.STANDARD_GUITAR_TUNING
            elif self.instrument == 'bass':
                self.tuning = self.STANDARD_BASS_TUNING
            else:
                raise ValueError(f"Unsupported instrument: {instrument}")
        else:
            self.tuning = tuning
        
        # Convert tuning notes to MIDI note numbers for easier calculation
        self.tuning_midi = []
        for note in self.tuning:
            try:
                self.tuning_midi.append(librosa.note_to_midi(note))
            except:
                logger.warning(f"Could not convert note {note} to MIDI number")
                # Default to a reasonable value
                self.tuning_midi.append(40)  # Default to E1
        
        # Create a mapping of MIDI notes to string+fret combinations
        self.note_to_fret_map = self._create_fretboard_map()
        
        # Initialize neural network model for position inference if available
        self.position_model = None
        if self.use_ml:
            self._initialize_position_model()
        
        # Initialize context for tracking playing positions
        self.current_position = None  # Will track last played position
        self.last_chord = None  # Will track last played chord
        self.key_context = None  # Will track detected key
            
    def _initialize_position_model(self):
        """Initialize the neural network model for position inference if available"""
        if not TENSORFLOW_AVAILABLE:
            logger.info("TensorFlow not available, machine learning features will be disabled")
            print("  • TensorFlow not available, using traditional algorithms only")
            return
            
        try:
            # Look for pre-trained model in models directory
            print("  • Initializing machine learning components...")
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'fretboard_position_model')
            
            # Check if model directory exists and has the marker file
            if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, 'model_ready')):
                logger.info(f"Loading pre-trained model from {model_path}")
                
                # Load the model architecture from JSON
                architecture_path = os.path.join(model_path, 'model_architecture.json')
                if os.path.exists(architecture_path):
                    with open(architecture_path, 'r') as f:
                        model_json = f.read()
                    
                    # Create model from JSON
                    self.position_model = tf.keras.models.model_from_json(model_json)
                    
                    # Check if weights file exists
                    weights_path = os.path.join(model_path, 'model_weights.h5')
                    if os.path.exists(weights_path):
                        # Load trained weights if available
                        self.position_model.compile(
                            optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        self.position_model.load_weights(weights_path)
                        logger.info("Successfully loaded pre-trained fretboard position model with weights")
                    else:
                        # If weights don't exist yet, initialize with random weights
                        logger.info("Model architecture loaded, but no trained weights found. Using random initialization.")
                    
                    # Model loaded successfully either way
                    print("  ✓ Machine learning model loaded successfully")
                else:
                    logger.warning("Model architecture file not found. Using fallback algorithms")
                    print("  ⚠ Machine learning model configuration incomplete, using traditional algorithms")
                    self.position_model = None
            else:
                logger.warning("Pre-trained model not found. Using fallback algorithms")
                print("  ⚠ Machine learning model not found, using traditional algorithms")
        except Exception as e:
            logger.error(f"Error loading position model: {str(e)}. Using fallback algorithms")
            print(f"  ✗ Error loading machine learning model: {str(e)}")
            self.position_model = None
            
    def _create_fretboard_map(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        Create a mapping of MIDI note numbers to possible string+fret combinations
        
        Returns:
            Dictionary of MIDI notes to list of (string_index, fret) tuples
        """
        fretboard_map = defaultdict(list)
        
        # For each string
        for string_idx, open_note in enumerate(self.tuning_midi):
            # For each fret position
            for fret in range(self.MAX_FRET + 1):
                note = open_note + fret
                fretboard_map[note].append((string_idx, fret))
        
        return fretboard_map
    
    def generate_tab(self, notes: List[Dict[str, Any]], optimize: bool = True, 
                     key_hint: str = None, style_hint: str = None) -> Dict[str, Any]:
        """
        Generate tablature from a list of notes
        
        Args:
            notes: List of note dictionaries with time, note, and duration
            optimize: Whether to optimize for playability
            key_hint: Optional key hint (e.g., 'C', 'Am') for better position inference
            style_hint: Optional style hint ('blues', 'rock', 'jazz', etc.) for position bias
            
        Returns:
            Dictionary with tablature information
        """
        start_time = time.time()
        
        # Start with logging basic info about the request
        logger.info(f"Generating tablature for {len(notes)} notes, optimize={optimize}, key_hint={key_hint}, style_hint={style_hint}")
        print(f"Generating tablature for {len(notes)} notes...")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Initialize return structure
        result = {
            'success': False,
            'message': '',
            'tab_ascii': '',
            'notes': [],
            'measures': [],
            'chord_symbols': [],
            'tuning': self.tuning,
            'instrument': self.instrument
        }
        
        # Validate input
        if not notes:
            result['message'] = 'No notes provided'
            logger.warning("Tablature generation failed: no notes provided")
            print("✗ Error: No notes provided for tablature generation")
            return result
            
        print("Step 1/5: Preprocessing and validating notes...")
        
        # Clean and standardize note format
        preprocessed_notes = []
        for note in notes:
            # Make sure we have required fields
            if not all(k in note for k in ['time', 'duration']):
                continue
                
            # Create a copy to avoid modifying the original
            new_note = note.copy()
            
            # Convert 'note' field to MIDI if necessary
            if 'midi' not in new_note and 'note' in new_note:
                try:
                    # If it's a string like 'C4', convert to MIDI
                    if isinstance(new_note['note'], str):
                        new_note['midi'] = librosa.note_to_midi(new_note['note'])
                    # If it's already a MIDI number
                    elif isinstance(new_note['note'], int):
                        new_note['midi'] = new_note['note']
                except:
                    logger.warning(f"Could not convert note {new_note['note']} to MIDI")
                    continue
            
            # Skip notes without MIDI values after preprocessing
            if 'midi' not in new_note:
                continue
                
            preprocessed_notes.append(new_note)
        
        # Update notes count after preprocessing
        if len(preprocessed_notes) != len(notes):
            logger.info(f"After preprocessing: {len(preprocessed_notes)}/{len(notes)} notes valid")
            print(f"• Preprocessing complete: {len(preprocessed_notes)}/{len(notes)} valid notes identified")
        
        if not preprocessed_notes:
            result['message'] = 'No valid notes after preprocessing'
            logger.warning("Tablature generation failed: no valid notes after preprocessing")
            print("✗ Error: No valid notes found after preprocessing")
            return result
        
        # Sort notes by time
        preprocessed_notes.sort(key=lambda x: x['time'])
        
        # Detect key if not provided
        print("Step 2/5: Analyzing musical context...")
        if not key_hint and self.include_theory:
            print("• Detecting musical key from notes...")
            key_hint = self._detect_key_from_notes(preprocessed_notes)
            if key_hint:
                logger.info(f"Detected key: {key_hint}")
                print(f"• Detected key: {key_hint}")
                self.key_context = key_hint
        
        # Apply style-based position bias if style is provided
        position_bias = self.position_bias
        if style_hint:
            style_position_bias = self._get_style_position_bias(style_hint)
            if style_position_bias:
                position_bias = style_position_bias
                logger.info(f"Using position bias '{position_bias}' based on style '{style_hint}'")
                print(f"• Using position bias '{position_bias}' based on style '{style_hint}'")
        
        # Optimize positions if requested
        tab_positions = []
        if optimize:
            print("Step 3/5: Optimizing fretboard positions...")
            print(f"• Finding optimal positions using bias: {position_bias}")
            
            # Get position boundaries based on bias
            position_boundaries = self._get_position_boundaries(position_bias)
            
            start_optimize = time.time()
            tab_positions = self._optimize_fret_positions_advanced(
                preprocessed_notes,
                position_bias,
                key_hint
            )
            
            optimize_time = time.time() - start_optimize
            logger.info(f"Position optimization completed in {optimize_time:.2f}s")
            print(f"• Position optimization completed in {optimize_time:.2f}s")
        else:
            # If no optimization, just use the first available position for each note
            print("Step 3/5: Mapping notes to fretboard (no optimization)...")
            for note in preprocessed_notes:
                if 'midi' in note and note['midi'] in self.note_to_fret_map:
                    # Use first available position (usually lowest fret)
                    string_idx, fret = self.note_to_fret_map[note['midi']][0]
                    note_copy = note.copy()
                    note_copy['string'] = string_idx
                    note_copy['fret'] = fret
                    tab_positions.append(note_copy)
            
            print(f"• Simple position mapping complete: {len(tab_positions)}/{len(preprocessed_notes)} notes mapped")
        
        # Check if we have positions
        if not tab_positions:
            result['message'] = 'Could not find valid positions for notes'
            logger.warning("Tablature generation failed: no valid positions found")
            print("✗ Error: Could not find valid positions for notes")
            return result
        
        # Organize into measures
        print("Step 4/5: Organizing tablature into measures...")
        measures = self._organize_into_measures(tab_positions)
        
        if not measures:
            result['message'] = 'Could not organize notes into measures'
            logger.warning("Tablature generation failed: could not organize notes into measures")
            print("✗ Error: Could not organize notes into measures")
            return result
            
        logger.info(f"Organized notes into {len(measures)} measures")
        print(f"• Organized into {len(measures)} measures")
        
        # Add chord symbols
        chord_symbols = []
        if self.include_theory:
            print("• Adding chord symbols using music theory analysis...")
            measures_with_chords = self._add_chord_symbols(measures, key_hint)
            
            # Extract chord symbols for the result
            for measure in measures_with_chords:
                for note in measure:
                    if 'chord_symbol' in note and note['chord_symbol'] and note['chord_symbol'] not in chord_symbols:
                        chord_symbols.append(note['chord_symbol'])
            
            print(f"• Identified {len(chord_symbols)} unique chord symbols")
            
            # Sort and deduplicate chord symbols
            chord_symbols = sorted(list(set(chord_symbols)))
            
            measures = measures_with_chords
        
        # Create ASCII tab
        print("Step 5/5: Generating ASCII tablature representation...")
        tab_ascii = self._create_ascii_tab(measures)
        
        # Update result
        result['success'] = True
        result['message'] = 'Tab generated successfully'
        result['tab_ascii'] = tab_ascii
        result['notes'] = tab_positions
        result['measures'] = measures
        result['chord_symbols'] = chord_symbols
        
        total_time = time.time() - start_time
        logger.info(f"Tab generation completed in {total_time:.2f}s")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"✓ Tablature generation complete in {total_time:.2f}s")
        
        return result
    
    def _get_style_position_bias(self, style: str) -> str:
        """
        Get position bias based on musical style
        
        Args:
            style: Style hint ('blues', 'rock', 'jazz', etc.)
            
        Returns:
            Position bias string
        """
        style_map = {
            'blues': 'low',        # Blues tends to use lower positions
            'rock': 'mid',         # Rock often uses middle positions
            'metal': 'high',       # Metal often uses higher positions
            'jazz': 'extended',    # Jazz uses extended chord positions
            'country': 'open',     # Country often uses open positions
            'folk': 'open',        # Folk often uses open positions
            'classical': 'optimal' # Classical aims for optimal positions
        }
        
        return style_map.get(style.lower(), 'optimal')
    
    def _detect_key_from_notes(self, notes: List[Dict[str, Any]]) -> str:
        """
        Detect musical key from a list of notes
        
        Args:
            notes: List of notes with midi values
            
        Returns:
            Detected key name or None
        """
        if not notes or not self.include_theory:
            return None
            
        try:
            print("  Performing key detection analysis...")
            # Log the start of key detection
            logger.info(f"Attempting to detect key from {len(notes)} notes")
            
            # Extract MIDI pitch values
            pitches = [note['midi'] % 12 for note in notes if 'midi' in note]
            
            if not pitches:
                logger.warning("No valid MIDI pitches found for key detection")
                print("  ✗ No valid pitch data available for key detection")
                return None
                
            # Count pitch class occurrences
            pitch_counts = Counter(pitches)
            logger.info(f"Pitch class distribution: {dict(pitch_counts)}")
            print(f"  • Analyzing pitch distribution across {len(pitch_counts)} unique pitches")
            
            # If music21 is available, use it for more sophisticated key detection
            if MUSIC21_AVAILABLE:
                # Create a stream of notes
                stream = music21.stream.Stream()
                
                # Add notes to the stream in their original order and duration
                for note in notes:
                    if 'midi' not in note:
                        continue
                        
                    # Convert MIDI to music21 note with proper duration
                    midi_num = note['midi']
                    duration = note.get('duration', 1.0)  # Default to quarter note if not specified
                    
                    m21_note = music21.note.Note(midi_num)
                    m21_note.quarterLength = duration * 4  # Convert to quarter note units
                    stream.append(m21_note)
                
                print("  • Using music theory analysis to determine key...")
                # Use music21's key analysis algorithm
                key_analysis = music21.analysis.discrete.analyzeStream(stream, 'key')
                
                if key_analysis:
                    key_name = key_analysis.name
                    key_tonic = key_analysis.tonic.name
                    key_mode = key_analysis.mode
                    
                    logger.info(f"Detected key using music21: {key_name} ({key_tonic} {key_mode})")
                    print(f"  ✓ Detected key: {key_name} ({key_tonic} {key_mode})")
                    return key_name
            
            # Fallback: Use simple algorithm based on pitch frequency
            # The Krumhansl-Schmuckler key-finding algorithm (simplified)
            pitch_weights = {
                # Major key profiles (C, C#, D, etc.)
                'major': [
                    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],  # C
                    [2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29],  # C#
                    [2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66],  # D
                    [3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39],  # D#
                    [2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19],  # E
                    [5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52],  # F
                    [2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09],  # F#
                    [4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38],  # G
                    [4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33],  # G#
                    [2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48],  # A
                    [3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23],  # A#
                    [2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35],  # B
                ],
                # Minor key profiles
                'minor': [
                    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],  # C
                    [3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34],  # C#
                    [3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69],  # D
                    [2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98],  # D#
                    [3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75],  # E
                    [4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54],  # F
                    [2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53],  # F#
                    [3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60],  # G
                    [2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38],  # G#
                    [5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52],  # A
                    [3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68],  # A#
                    [2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33],  # B
                ]
            }
            
            # Convert pitch counts to a 12-element vector
            pitch_vector = [0] * 12
            for pitch, count in pitch_counts.items():
                pitch_vector[pitch] = count
                
            print("  • Using Krumhansl-Schmuckler algorithm for key detection...")
            
            # Calculate correlation with each key profile
            max_corr = -1
            best_key = None
            best_mode = None
            
            for mode in ['major', 'minor']:
                for i, key_profile in enumerate(pitch_weights[mode]):
                    # Calculate correlation
                    correlation = self._correlation(pitch_vector, key_profile)
                    
                    if correlation > max_corr:
                        max_corr = correlation
                        root_note = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][i]
                        best_key = f"{root_note} {mode}"
                        best_mode = mode
            
            if best_key:
                logger.info(f"Detected key using Krumhansl-Schmuckler algorithm: {best_key} (correlation: {max_corr:.3f})")
                print(f"  ✓ Detected key: {best_key} (confidence: {max_corr:.3f})")
                return best_key
                
            logger.warning("Could not detect key from notes")
            print("  ✗ Could not determine key with confidence")
            return None
            
        except Exception as e:
            logger.error(f"Error in key detection: {str(e)}")
            print(f"  ✗ Error in key detection: {str(e)}")
            return None
            
    def _correlation(self, vec1, vec2):
        """Calculate correlation coefficient between two vectors"""
        mean1 = sum(vec1) / len(vec1) if sum(vec1) > 0 else 0
        mean2 = sum(vec2) / len(vec2) if sum(vec2) > 0 else 0
        
        num = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(vec1, vec2))
        den1 = sum((v1 - mean1) ** 2 for v1 in vec1)
        den2 = sum((v2 - mean2) ** 2 for v2 in vec2)
        
        if den1 == 0 or den2 == 0:
            return 0
            
        return num / ((den1 * den2) ** 0.5)

    def _get_position_boundaries(self, position_bias: str) -> Tuple[int, int]:
        """
        Get fret range boundaries based on position bias
        
        Args:
            position_bias: Position preference
            
        Returns:
            Tuple of (min_fret, max_fret)
        """
        if position_bias == 'low':
            return (0, 5)
        elif position_bias == 'mid':
            return (3, 9)
        elif position_bias == 'high':
            return (7, 15)
        elif position_bias == 'extended':
            return (7, self.MAX_FRET)
        elif position_bias == 'open':
            return (0, 3)
        else:  # 'optimal'
            return (0, 12)
    
    def _get_scale_positions(self, key: str) -> Dict[str, List[Tuple[int, int]]]:
        """
        Get preferred fretboard positions for notes in the given key
        
        Args:
            key: Key name (e.g., 'C', 'Am')
            
        Returns:
            Dictionary mapping scale degree to preferred positions
        """
        if not key or not self.include_theory:
            return {}
            
        try:
            # Parse key
            is_minor = key.endswith('m')
            root = key[:-1] if is_minor else key
            
            # Get scale pattern
            pattern = self.MINOR_SCALE_PATTERN if is_minor else self.MAJOR_SCALE_PATTERN
            
            # Get root note number
            root_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 
                      'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 
                      'A#': 10, 'Bb': 10, 'B': 11}
                      
            if root not in root_map:
                return {}
                
            root_num = root_map[root]
            
            # Generate scale positions
            scale_positions = {}
            
            # For each scale degree
            for i, interval in enumerate(pattern):
                # Calculate note
                note_num = (root_num + interval) % 12
                
                # Find good positions for this note in each octave
                good_positions = []
                
                # Check positions in the comfortable range
                min_midi = librosa.note_to_midi(f'C2')  # Low C
                max_midi = librosa.note_to_midi(f'C6')  # High C
                
                for octave in range(2, 6):
                    midi_note = note_num + (octave * 12)
                    
                    if min_midi <= midi_note <= max_midi and midi_note in self.note_to_fret_map:
                        # Get all positions for this note
                        positions = self.note_to_fret_map[midi_note]
                        
                        # Filter to positions in the comfortable range
                        good_pos = [pos for pos in positions 
                                   if 0 <= pos[1] <= 12]  # 0-12 fret range
                        
                        good_positions.extend(good_pos)
                
                # Store the good positions for this scale degree
                scale_degree = i + 1  # 1-based scale degree
                scale_positions[scale_degree] = good_positions
            
            return scale_positions
            
        except Exception as e:
            logger.warning(f"Error generating scale positions: {str(e)}")
            return {}

    def _get_ideal_position_for_single_note(self, note: Dict[str, Any], 
                                         position_bias: str) -> List[Dict[str, Any]]:
        """
        Get the ideal position for a single note based on position bias
        
        Args:
            note: Note dictionary with midi, time, and duration
            position_bias: Position preference
            
        Returns:
            List containing a single note with position
        """
        midi = note['midi']
        
        if midi not in self.note_to_fret_map:
            return []
            
        # Get all possible positions
        positions = self.note_to_fret_map[midi]
        
        # Get position boundaries
        min_fret, max_fret = self._get_position_boundaries(position_bias)
        
        # Filter positions by boundaries
        valid_positions = [pos for pos in positions if min_fret <= pos[1] <= max_fret]
        
        # If no valid positions within boundaries, use any position
        if not valid_positions:
            valid_positions = positions
        
        if valid_positions:
            # For low positions, prefer open strings
            if position_bias == 'open' or position_bias == 'low':
                open_positions = [pos for pos in valid_positions if pos[1] == 0]
                if open_positions:
                    string_idx, fret = open_positions[0]
                else:
                    # Sort by fret number
                    valid_positions.sort(key=lambda x: x[1])
                    string_idx, fret = valid_positions[0]
            elif position_bias == 'mid':
                # For mid positions, sort by closeness to fret 5
                valid_positions.sort(key=lambda x: abs(x[1] - 5))
                string_idx, fret = valid_positions[0]
            elif position_bias == 'high':
                # For high positions, prefer higher frets
                valid_positions.sort(key=lambda x: -x[1])
                string_idx, fret = valid_positions[0]
            else:
                # For optimal, prefer lower strings and lower frets
                valid_positions.sort(key=lambda x: (x[0], x[1]))
                string_idx, fret = valid_positions[0]
                
            return [{
                'midi': midi,
                'string': string_idx,
                'fret': fret,
                'time': note['time'],
                'duration': note['duration'],
                'velocity': note['velocity'],
                'note': note.get('original_note', '')
            }]
        else:
            return []
            
    def _optimize_single_note_position(self, note: Dict[str, Any],
                                    current_position: Tuple[int, int] = None,
                                    position_boundaries: Tuple[int, int] = (0, 12),
                                    scale_positions: Dict[str, List[Tuple[int, int]]] = None,
                                    use_ml_model: bool = False) -> Dict[str, Any]:
        """
        Optimize position for a single note with context awareness
        
        Args:
            note: Note dictionary
            current_position: Current hand position (string, fret)
            position_boundaries: Allowed fret range
            scale_positions: Scale position map if available
            use_ml_model: Whether to use ML model for position inference
            
        Returns:
            Optimized note dictionary with string and fret position
        """
        midi = note['midi']
        pitch_name = note.get('note', f"MIDI:{midi}")
        logger.info(f"Optimizing position for single note: {pitch_name} (MIDI: {midi})")
        
        if midi not in self.note_to_fret_map:
            # Cannot place this note, return placeholder
            logger.warning(f"Note {pitch_name} (MIDI: {midi}) cannot be placed on the fretboard")
            return {
                'midi': midi,
                'string': 0,
                'fret': 0,
                'time': note['time'],
                'duration': note['duration'],
                'velocity': note['velocity'],
                'note': note.get('original_note', '')
            }
        
        # Get all possible positions
        positions = self.note_to_fret_map[midi]
        logger.info(f"Found {len(positions)} possible positions for note {pitch_name}")
        
        # Filter by position boundaries
        min_fret, max_fret = position_boundaries
        valid_positions = [pos for pos in positions if min_fret <= pos[1] <= max_fret]
        logger.info(f"After filtering by position boundaries ({min_fret}-{max_fret}): {len(valid_positions)} valid positions")
        
        # If no valid positions within boundaries, use any position
        if not valid_positions:
            logger.warning(f"No positions for {pitch_name} within boundaries, using all available positions")
            valid_positions = positions
        
        # Check if this note is part of a scale (if we have scale context)
        scale_position_used = False
        if scale_positions:
            # Check which scale degree this note might belong to
            for degree, preferred_positions in scale_positions.items():
                # Check if this note belongs to this scale degree (compare pitch classes)
                scale_pc = int(degree) % 12
                note_pc = midi % 12
                
                if scale_pc == note_pc:
                    logger.info(f"Note {pitch_name} is scale degree {degree} in the current key")
                    
                    # Try to use one of the preferred positions for this scale degree
                    for pref_string, pref_fret in preferred_positions:
                        # Find matching position with the same pitch class (might be different octave)
                        for string_idx, fret in valid_positions:
                            if string_idx == pref_string and (fret % 12) == (pref_fret % 12):
                                string_idx, fret = string_idx, fret
                                logger.info(f"Using scale-based position: string {string_idx}, fret {fret}")
                                scale_position_used = True
                                break
                        
                        if scale_position_used:
                            break
                    
                    if scale_position_used:
                        break
        
        # If we have current position context, use it
        if not scale_position_used and current_position is not None:
            last_string, last_fret = current_position
            logger.info(f"Using current hand position context: string {last_string}, fret {last_fret}")
            
            # Calculate distance to each potential position
            distances = []
            for string_idx, fret in valid_positions:
                # Distance is a combination of string and fret distance
                string_distance = abs(string_idx - last_string)
                fret_distance = abs(fret - last_fret)
                
                # Weight fret distance more heavily to prefer same position
                distance = fret_distance * 1.5 + string_distance
                distances.append((distance, string_idx, fret))
            
            # Sort by distance
            distances.sort()
            
            # Use the closest position
            closest_distance, string_idx, fret = distances[0]
            logger.info(f"Selected position closest to current hand: string {string_idx}, fret {fret} (distance: {closest_distance:.2f})")
        
        # If we have a neural network model, use it for position prediction
        elif not scale_position_used and use_ml_model and self.position_model:
            try:
                logger.info(f"Using machine learning model for position inference: note {pitch_name}")
                
                # Convert note to feature vector
                # Simple feature: normalized MIDI value, octave, and position in measure
                normalized_midi = note['midi'] / 127.0
                normalized_octave = (note['midi'] // 12) / 10.0
                normalized_position = (note['time'] % 4.0) / 4.0
                
                feature = np.array([[
                    normalized_midi,      # Normalize MIDI value
                    normalized_octave,    # Normalize octave
                    normalized_position   # Normalize position in measure
                ]])
                
                logger.debug(f"ML input features: MIDI={normalized_midi:.4f}, octave={normalized_octave:.4f}, position={normalized_position:.4f}")
                
                # Get model prediction
                prediction = self.position_model.predict(feature, verbose=0)
                
                # Extract predicted string and fret
                pred_string = int(prediction[0][0] * len(self.tuning_midi))
                pred_fret = int(prediction[0][1] * self.MAX_FRET)
                
                logger.info(f"ML model predicted position: string={pred_string}, fret={pred_fret}")
                
                # Find the closest valid position
                closest_dist = float('inf')
                closest_pos = None
                
                for string_idx, fret in valid_positions:
                    dist = abs(string_idx - pred_string) + abs(fret - pred_fret)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_pos = (string_idx, fret)
                
                if closest_pos:
                    string_idx, fret = closest_pos
                    logger.info(f"Using ML-suggested position: string {string_idx}, fret {fret} (distance from prediction: {closest_dist})")
                else:
                    # Fallback to the first valid position
                    string_idx, fret = valid_positions[0]
                    logger.warning("No close match to ML prediction, using default position")
                    
            except Exception as e:
                logger.warning(f"Error using ML model for position inference: {str(e)}")
                # Fallback to first valid position
                string_idx, fret = valid_positions[0]
                logger.info(f"Using default position due to ML error: string {string_idx}, fret {fret}")
        
        # Otherwise, use optimal positioning rules based on fretboard ergonomics
        elif not scale_position_used:
            # Log the strategy
            logger.info("Using ergonomic positioning rules for note placement")
            
            # First, check for open string possibilities (often preferred)
            open_positions = [pos for pos in valid_positions if pos[1] == 0]
            if open_positions:
                logger.info(f"Found {len(open_positions)} open string positions")
                # Sort by string thickness (prefer lower-numbered strings)
                open_positions.sort(key=lambda x: x[0])
                string_idx, fret = open_positions[0]
                logger.info(f"Using open string position: string {string_idx}")
            else:
                logger.info("No open string positions available")
                # Sort by string (preference for lower strings) and fret (lower frets first)
                valid_positions.sort(key=lambda x: (x[0], x[1]))
                string_idx, fret = valid_positions[0]
                logger.info(f"Using optimal fretted position: string {string_idx}, fret {fret}")
        
        # Create the optimized note dictionary
        optimized_note = {
            'midi': midi,
            'string': string_idx,
            'fret': fret,
            'time': note['time'],
            'duration': note['duration'],
            'velocity': note['velocity'],
            'note': note.get('original_note', '')
        }
        
        return optimized_note

    def _optimize_chord_positions_advanced(self, chord_notes: List[Dict[str, Any]],
                                        current_position: Tuple[int, int] = None,
                                        position_boundaries: Tuple[int, int] = (0, 12),
                                        key_hint: str = None,
                                        use_ml: bool = False) -> List[Dict[str, Any]]:
        """
        Find optimal positions for chord notes using advanced algorithms
        
        Args:
            chord_notes: List of notes in the chord
            current_position: Current hand position (string, fret) if available
            position_boundaries: Allowed fret range
            key_hint: Optional key hint for better position selection
            use_ml: Whether to use ML for position inference
            
        Returns:
            List of notes with optimized positions
        """
        # Special case for one note "chord"
        if len(chord_notes) == 1:
            print("  • Processing single note...")
            return self._get_ideal_position_for_single_note(chord_notes[0], 'optimal')
            
        # Log chord processing
        print(f"  • Processing chord with {len(chord_notes)} notes...")
        logger.info(f"Optimizing chord with {len(chord_notes)} notes")
            
        # Extract MIDI notes
        midi_notes = [note['midi'] for note in chord_notes]
        logger.info(f"Chord MIDI values: {midi_notes}")
        
        # Try to identify common chord shapes for recognized chords
        chord_shape = None
        if key_hint and self.include_theory:
            print("  • Analyzing chord structure...")
            chord_shape = self._identify_chord_shape(midi_notes, key_hint)
            if chord_shape:
                chord_name = chord_shape.get('full_name', 'Unknown')
                print(f"  ✓ Detected chord: {chord_name}")
                logger.info(f"Identified chord: {chord_name}")
        
        # Get min and max fret boundaries
        min_fret, max_fret = position_boundaries
        print(f"  • Using fret range: {min_fret}-{max_fret}")
        
        # Create a list of possible positions for each note
        position_options = []
        for note in chord_notes:
            midi = note['midi']
            if midi in self.note_to_fret_map:
                # Get positions within boundaries
                positions = [(string, fret) for string, fret in self.note_to_fret_map[midi]
                           if min_fret <= fret <= max_fret]
                
                # If no positions in boundaries, use any position
                if not positions and self.note_to_fret_map[midi]:
                    positions = self.note_to_fret_map[midi]
                
                position_options.append({
                    'midi': midi,
                    'positions': positions,
                    'time': note['time'],
                    'duration': note['duration'],
                    'velocity': note['velocity'],
                    'note': note.get('note', '')
                })
        
        if not position_options:
            print("  ✗ Could not map notes to fretboard positions")
            logger.warning("No valid positions found for notes in chord")
            return chord_notes
            
        # Log the number of possible positions
        total_positions = sum(len(note_data['positions']) for note_data in position_options)
        print(f"  • Found {total_positions} potential positions across all notes")
        
        # If we identified a specific chord shape, prioritize those positions
        if chord_shape:
            print("  • Checking for common chord shapes...")
            solution = self._apply_chord_shape(position_options, chord_shape)
            if solution:
                # Calculate average position for reporting
                frets = [note['fret'] for note in solution if note.get('fret', 0) > 0]
                avg_position = round(sum(frets) / len(frets)) if frets else 0
                
                print(f"  ✓ Applied standard chord shape at position {avg_position}")
                logger.info(f"Applied common chord shape for {chord_shape['full_name']} at position {avg_position}")
                return solution
            else:
                print("  ⚠ Could not apply standard shape, using optimization algorithm")
        
        # No specific shape, use general optimization
        print("  • Using position optimization algorithm...")
        
        # Try different combinations of positions
        best_score = float('inf')
        best_solution = None
        
        # Limit search space if necessary
        max_combinations = 1000
        
        # Calculate total combinations
        total_combinations = 1
        for note_data in position_options:
            total_combinations *= len(note_data['positions'])
        
        # If too many combinations, limit each note to fewer positions
        if total_combinations > max_combinations:
            logger.info(f"Limiting chord position search space: {total_combinations} combinations")
            print(f"  • Limiting search space ({total_combinations} combinations is too many)")
            
            for note_data in position_options:
                positions = note_data['positions']
                if len(positions) > 4:
                    # Prioritize positions based on context
                    if current_position:
                        # Sort by distance to current position
                        last_string, last_fret = current_position
                        positions.sort(key=lambda pos: abs(pos[0] - last_string) + abs(pos[1] - last_fret))
                        print(f"  • Prioritizing positions near current hand position (fret {last_fret})")
                    else:
                        # Sort by fret number (prefer lower frets)
                        positions.sort(key=lambda pos: pos[1])
                    
                    # Keep only the top 4 positions
                    note_data['positions'] = positions[:4]
        
        # Get all possible combinations (limited to reasonable size)
        combinations = list(itertools.product(*[note_data['positions'] for note_data in position_options]))
        logger.info(f"Evaluating {len(combinations)} chord position combinations")
        print(f"  • Evaluating {len(combinations)} potential chord positions")
        
        # Try each combination and find the best one
        valid_combinations = 0
        for positions in combinations:
            # Check if this combination is playable
            strings_used = set()
            max_fret = 0
            min_fret = float('inf')
            
            is_valid = True
            for (string_idx, fret) in positions:
                # Check if string is already used
                if string_idx in strings_used:
                    is_valid = False
                    break
                
                strings_used.add(string_idx)
                if fret > 0:  # Ignore open strings for min/max calculation
                    max_fret = max(max_fret, fret)
                    min_fret = min(min_fret, fret)
            
            # If all strings are open, set min_fret to 0
            if min_fret == float('inf'):
                min_fret = 0
                max_fret = 0
            
            # Check fret span (is it playable by human hand?)
            fret_span = max_fret - min_fret
            if fret_span > self.STRETCH_LIMIT and min_fret > 0:  # Allow wider spans with open strings
                is_valid = False
            
            if is_valid:
                valid_combinations += 1
                # Calculate score based on various factors
                # Lower score is better
                
                # Base score is the maximum fret position plus the span
                score = max_fret + fret_span
                
                # Prefer positions close to current position if available
                if current_position:
                    last_string, last_fret = current_position
                    # Get average distance to current position
                    distances = []
                    for string_idx, fret in positions:
                        if fret > 0:  # Only consider fretted notes
                            dist = abs(string_idx - last_string) + abs(fret - last_fret)
                            distances.append(dist)
                    
                    if distances:
                        avg_distance = sum(distances) / len(distances)
                        # Add distance penalty to score
                        score += avg_distance * 0.5
                
                # Prefer open strings for certain styles
                open_string_count = sum(1 for _, fret in positions if fret == 0)
                if self.position_bias in ['open', 'low']:
                    # Reduce score for each open string
                    score -= open_string_count * 2
                
                # Prefer middle positions for 'mid' bias
                if self.position_bias == 'mid' and min_fret > 0:
                    # Add penalty for positions far from the middle (fret 5-7)
                    middle_fret = 6
                    middle_distance = abs((min_fret + max_fret) / 2 - middle_fret)
                    score += middle_distance
                
                if score < best_score:
                    best_score = score
                    best_solution = positions
        
        print(f"  • Found {valid_combinations} playable combinations")
        logger.info(f"Found {valid_combinations} valid chord positions")
        
        # If no valid solution found, try relaxing constraints
        if best_solution is None:
            logger.warning("No valid chord position found, relaxing constraints")
            print("  ⚠ No valid combinations found, using individual best positions")
            
            # Fallback to simpler algorithm - just pick lowest fret for each note
            solution = []
            for note_data in position_options:
                if note_data['positions']:
                    # Sort by fret number
                    positions = sorted(note_data['positions'], key=lambda x: x[1])
                    string_idx, fret = positions[0]
                    solution.append({
                        'midi': note_data['midi'],
                        'string': string_idx,
                        'fret': fret,
                        'time': note_data['time'],
                        'duration': note_data['duration'],
                        'velocity': note_data['velocity'],
                        'note': note_data.get('note', '')
                    })
            return solution
        
        # Convert best solution to the standard format
        solution = []
        for i, (string_idx, fret) in enumerate(best_solution):
            note_data = position_options[i]
            solution.append({
                'midi': note_data['midi'],
                'string': string_idx,
                'fret': fret,
                'time': note_data['time'],
                'duration': note_data['duration'],
                'velocity': note_data['velocity'],
                'note': note_data.get('note', '')
            })
        
        # Calculate average position for reporting
        frets = [note['fret'] for note in solution if note['fret'] > 0]
        avg_position = round(sum(frets) / len(frets)) if frets else 0
        
        logger.info(f"Optimized chord position: around fret {avg_position} (score: {best_score:.2f})")
        print(f"  ✓ Found optimal chord position: around fret {avg_position}")
        
        return solution
    
    def _identify_chord_shape(self, midi_notes: List[int], key_hint: str = None) -> Dict[str, Any]:
        """
        Identify a common chord shape from a set of MIDI notes
        
        Args:
            midi_notes: List of MIDI note numbers
            key_hint: Optional key hint
            
        Returns:
            Dictionary with chord information or None
        """
        if not midi_notes or not self.include_theory:
            return None
            
        try:
            # Convert MIDI notes to pitch classes (0-11)
            pitch_classes = set(note % 12 for note in midi_notes)
            
            # Skip if less than 3 unique pitch classes
            if len(pitch_classes) < 2:
                return None
                
            # Common chord types and their interval structures
            chord_types = {
                'major': [0, 4, 7],
                'minor': [0, 3, 7],
                'dominant7': [0, 4, 7, 10],
                'major7': [0, 4, 7, 11],
                'minor7': [0, 3, 7, 10],
                'diminished': [0, 3, 6],
                'augmented': [0, 4, 8],
                'sus4': [0, 5, 7],
                'sus2': [0, 2, 7],
                '6': [0, 4, 7, 9],
                'm6': [0, 3, 7, 9],
                '9': [0, 4, 7, 10, 14],
                'add9': [0, 4, 7, 14]
            }
            
            # Try to find a match for each possible root note
            best_match = None
            best_score = 0
            
            for root in range(12):
                for chord_type, intervals in chord_types.items():
                    # Generate chord pitch classes
                    chord_pcs = set((root + interval) % 12 for interval in intervals)
                    
                    # Calculate match score (how many notes match vs. how many notes differ)
                    matches = len(pitch_classes.intersection(chord_pcs))
                    extras = len(pitch_classes - chord_pcs)
                    missing = len(chord_pcs - pitch_classes)
                    
                    # Score formula: matches - extras - missing
                    score = matches - 0.5 * extras - 0.5 * missing
                    
                    if score > best_score:
                        best_score = score
                        root_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][root]
                        best_match = {
                            'root': root,
                            'root_name': root_name,
                            'type': chord_type,
                            'intervals': intervals,
                            'score': score,
                            'full_name': f"{root_name}{self._format_chord_type(chord_type)}"
                        }
            
            # Only return if we have a good match
            if best_match and best_match['score'] >= len(pitch_classes) * 0.75:
                logger.info(f"Identified chord: {best_match['full_name']}")
                print(f"  ✓ Detected chord: {best_match['full_name']}")
                return best_match
                
            return None
            
        except Exception as e:
            logger.warning(f"Error identifying chord shape: {str(e)}")
            return None
    
    def _format_chord_type(self, chord_type: str) -> str:
        """Format chord type for display"""
        if chord_type == 'major':
            return ''
        elif chord_type == 'minor':
            return 'm'
        elif chord_type == 'dominant7':
            return '7'
        elif chord_type == 'major7':
            return 'maj7'
        elif chord_type == 'minor7':
            return 'm7'
        else:
            return chord_type
    
    def _apply_chord_shape(self, position_options: List[Dict[str, Any]], 
                         chord_shape: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply a known chord shape to position the notes
        
        Args:
            position_options: List of note data with position options
            chord_shape: Identified chord shape
            
        Returns:
            List of notes with positions or None if not applicable
        """
        if not chord_shape or not position_options:
            return None
            
        # Common guitar chord shapes and their fret positions
        # Format: {string: fret, ...} where string 0 is the thickest string (E for standard tuning)
        # -1 means string not played
        common_shapes = {
            # Major chords
            'C': {0: -1, 1: 3, 2: 2, 3: 0, 4: 1, 5: 0},
            'A': {0: -1, 1: 0, 2: 2, 3: 2, 4: 2, 5: 0},
            'G': {0: 3, 1: 2, 2: 0, 3: 0, 4: 0, 5: 3},
            'E': {0: 0, 1: 2, 2: 2, 3: 1, 4: 0, 5: 0},
            'D': {0: -1, 1: -1, 2: 0, 3: 2, 4: 3, 5: 2},
            'F': {0: 1, 1: 3, 2: 3, 3: 2, 4: 1, 5: 1},  # Barre chord
            
            # Minor chords
            'Cm': {0: -1, 1: 3, 2: 5, 3: 5, 4: 4, 5: 3},  # Barre chord
            'Am': {0: -1, 1: 0, 2: 2, 3: 2, 4: 1, 5: 0},
            'Gm': {0: 3, 1: 5, 2: 5, 3: 3, 4: 3, 5: 3},  # Barre chord
            'Em': {0: 0, 1: 2, 2: 2, 3: 0, 4: 0, 5: 0},
            'Dm': {0: -1, 1: -1, 2: 0, 3: 2, 4: 3, 5: 1},
            
            # 7th chords
            'C7': {0: -1, 1: 3, 2: 2, 3: 3, 4: 1, 5: 0},
            'A7': {0: -1, 1: 0, 2: 2, 3: 0, 4: 2, 5: 0},
            'G7': {0: 3, 1: 2, 2: 0, 3: 0, 4: 0, 5: 1},
            'E7': {0: 0, 1: 2, 2: 0, 3: 1, 4: 0, 5: 0},
            'D7': {0: -1, 1: -1, 2: 0, 3: 2, 4: 1, 5: 2},
        }
        
        # Choose shape based on the chord type
        shape_key = None
        
        # Try to match with common shapes
        for key in common_shapes.keys():
            if key == chord_shape['full_name'] or (key[0] == chord_shape['root_name'] and 
                                   key[1:] == self._format_chord_type(chord_shape['type'])):
                shape_key = key
                break
        
        if shape_key:
            shape = common_shapes[shape_key]
            logger.info(f"Using common shape for {chord_shape['full_name']}: {shape_key}")
            
            # Calculate transposition if needed
            base_root = shape_key[0]
            target_root = chord_shape['root_name']
            
            # Convert roots to numbers
            root_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 
                     'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 
                     'A#': 10, 'Bb': 10, 'B': 11}
                     
            base_num = root_map.get(base_root, 0)
            target_num = root_map.get(target_root, 0)
            
            # Calculate transposition offset
            offset = target_num - base_num
            
            # Apply shape with transposition
            solution = []
            for note_data in position_options:
                midi = note_data['midi']
                
                # Find which string this note would be on in the shape
                assigned = False
                for string, fret in shape.items():
                    if fret == -1:
                        continue
                    
                    # Calculate expected MIDI note for this string+fret
                    open_string_midi = self.tuning_midi[string]
                    shape_midi = open_string_midi + fret + offset
                    
                    # Allow octave differences
                    if (midi % 12) == (shape_midi % 12):
                        # Find the closest octave
                        octave_diff = (midi - shape_midi) // 12
                        actual_fret = fret + offset + (octave_diff * 12)
                        
                        # Check if fret is valid
                        if 0 <= actual_fret <= self.MAX_FRET:
                            solution.append({
                                'midi': midi,
                                'string': string,
                                'fret': actual_fret,
                                'time': note_data['time'],
                                'duration': note_data['duration'],
                                'velocity': note_data['velocity'],
                                'note': note_data.get('note', '')
                            })
                            assigned = True
                            break
                        
                if not assigned and note_data['positions']:
                    # Sort by fret number
                    positions = sorted(note_data['positions'], key=lambda x: x[1])
                    string_idx, fret = positions[0]
                    solution.append({
                        'midi': midi,
                        'string': string_idx,
                        'fret': fret,
                        'time': note_data['time'],
                        'duration': note_data['duration'],
                        'velocity': note_data['velocity'],
                        'note': note_data.get('note', '')
                    })
            
            # Only return solution if we placed all notes
            if len(solution) == len(position_options):
                return solution
        
        return None
        
    def _add_chord_symbols(self, measures: List[List[Dict[str, Any]]], 
                         key_hint: str = None) -> List[List[Dict[str, Any]]]:
        """
        Analyze measures to detect and add chord symbols
        
        Args:
            measures: List of measures, each containing a list of notes
            key_hint: Optional key hint for chord analysis
            
        Returns:
            Measures with chord symbols added
        """
        if not measures or not self.include_theory:
            return measures
            
        # Process each measure
        for measure_idx, measure in enumerate(measures):
            if not measure:
                continue
                
            # Group notes by time to identify simultaneous notes (chords)
            notes_by_time = defaultdict(list)
            for note in measure:
                notes_by_time[note['time']].append(note)
            
            # Sort times
            times = sorted(notes_by_time.keys())
            
            # Find chord at beginning of measure
            if times:
                first_time = times[0]
                chord_notes = notes_by_time[first_time]
                
                # Need at least 2 notes for a chord
                if len(chord_notes) >= 2:
                    midi_notes = [note['midi'] for note in chord_notes]
                    chord_shape = self._identify_chord_shape(midi_notes, key_hint)
                    
                    if chord_shape:
                        # Add chord symbol to measure
                        for note in measure:
                            note['chord'] = chord_shape['full_name']
        
        return measures

    def _organize_into_measures(self, tab_positions: List[Dict[str, Any]], 
                              beats_per_measure: int = 4) -> List[List[Dict[str, Any]]]:
        """
        Organize tab positions into measures
        
        Args:
            tab_positions: List of note dictionaries with string and fret positions
            beats_per_measure: Number of beats per measure
            
        Returns:
            List of measures, each containing a list of notes
        """
        if not tab_positions:
            return []
        
        # Sort by time
        sorted_positions = sorted(tab_positions, key=lambda x: x['time'])
        
        # Find the first and last times
        first_time = sorted_positions[0]['time']
        last_time = sorted_positions[-1]['time']
        
        # Estimate beat duration based on note durations
        durations = [note['duration'] for note in sorted_positions if note['duration'] > 0]
        if durations:
            beat_duration = min(durations) * 4  # Assume shortest note is a 16th note
        else:
            beat_duration = 0.5  # Default
        
        # Calculate measure duration
        measure_duration = beat_duration * beats_per_measure
        
        # Calculate number of measures
        total_duration = last_time - first_time
        num_measures = max(1, int(np.ceil(total_duration / measure_duration)))
        
        # Initialize measures
        measures = [[] for _ in range(num_measures)]
        
        # Assign notes to measures
        for note in sorted_positions:
            measure_idx = int((note['time'] - first_time) / measure_duration)
            
            # Handle potential rounding errors
            if measure_idx >= num_measures:
                measure_idx = num_measures - 1
            
            # Add note to measure
            measures[measure_idx].append(note)
        
        return measures
        
    def _create_ascii_tab(self, measures: List[List[Dict[str, Any]]]) -> str:
        """
        Create ASCII tablature representation
        
        Args:
            measures: List of measures, each containing a list of notes
            
        Returns:
            ASCII tablature as a string
        """
        if not measures:
            return "No notes to display"
        
        # Create header with tuning
        header = "Tuning: " + " ".join(self.tuning) + "\n\n"
        
        # Initialize tab with empty strings
        tab_lines = [[] for _ in range(len(self.tuning))]
        
        # Process each measure
        for measure_idx, measure in enumerate(measures):
            if not measure:
                continue
                
            # Group notes by time to identify simultaneous notes (chords)
            notes_by_time = defaultdict(list)
            for note in measure:
                notes_by_time[note['time']].append(note)
            
            # Sort times
            times = sorted(notes_by_time.keys())
            
            # Create a mapping of (string, position) -> fret number
            # This handles multiple notes at the same position
            string_positions = defaultdict(list)
            max_position = 0
            
            # First pass: calculate positions
            last_time = None
            position = 0
            
            for time in times:
                chord_notes = notes_by_time[time]
                
                # Need at least 2 notes for a chord
                if len(chord_notes) >= 2:
                    midi_notes = [note['midi'] for note in chord_notes]
                    chord_shape = self._identify_chord_shape(midi_notes)
                    
                    if chord_shape:
                        # Add chord symbol to measure
                        for note in measure:
                            note['chord_symbol'] = chord_shape['full_name']
                
                for note in chord_notes:
                    time = note['time']
                    
                    # If time changed, update position
                    if last_time is not None and time > last_time:
                        position += 2  # Add space between different times
                    
                    # Update position mapping
                    string_positions[(note['string'], position)].append(note['fret'])
                    
                    # Update tracking variables
                    last_time = time
                    max_position = max(max_position, position)
            
            # Add measure number and chord symbol if present
            measure_header = f"Measure {measure_idx + 1}"
            if measure and measure[0].get('chord_symbol'):
                measure_header += f" | Chord: {measure[0]['chord_symbol']}"
            header += measure_header + "\n"
            
            # Create measure separator
            for i in range(len(self.tuning)):
                tab_lines[i].append("|")
            
            # Second pass: fill in the tab
            for pos in range(max_position + 1):
                # For each string
                for string_idx in range(len(self.tuning)):
                    # Check if there's a note at this position
                    if (string_idx, pos) in string_positions:
                        frets = string_positions[(string_idx, pos)]
                        
                        # Format fret number (handle double-digit frets)
                        if len(frets) == 1:
                            fret_str = str(frets[0]).rjust(2)
                        else:
                            # Multiple notes at same position (shouldn't happen often)
                            fret_str = "XX"
                    else:
                        # No note, add a spacer
                        fret_str = "--"
                    
                    tab_lines[string_idx].append(fret_str)
            
            # Add measure end
            for i in range(len(self.tuning)):
                tab_lines[i].append("|")
                
            # Add a newline after each measure
            for i in range(len(self.tuning)):
                tab_lines[i].append("\n")
        
        # Combine all lines
        tab_text = header + "\n"
        
        # Add string labels
        for i, tuning_note in enumerate(self.tuning):
            line = tuning_note[0] + "|" + "".join(tab_lines[i])
            tab_text += line
        
        return tab_text
        
    def optimize_fingering(self, tab_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assign left-hand fingerings to tab positions
        
        Args:
            tab_positions: List of note dictionaries with string and fret positions
            
        Returns:
            Same list with finger assignments added
        """
        if not tab_positions:
            return []
        
        # Sort by time
        sorted_positions = sorted(tab_positions, key=lambda x: x['time'])
        
        # Find notes that are played at the same time (chords)
        notes_by_time = defaultdict(list)
        for note in sorted_positions:
            notes_by_time[note['time']].append(note)
        
        # Process each time slice
        for time, notes in notes_by_time.items():
            # Is this a chord (multiple simultaneous notes)?
            if len(notes) > 1:
                # Sort by string (low to high)
                notes.sort(key=lambda x: x['string'], reverse=True)
                
                # Find the fret span
                min_fret = min([note['fret'] for note in notes if note['fret'] > 0])
                max_fret = max([note['fret'] for note in notes])
                
                # Skip open strings when assigning fingers
                fretted_notes = [note for note in notes if note['fret'] > 0]
                
                # Assign fingers based on fret span
                if fretted_notes:
                    fret_span = max_fret - min_fret
                    
                    if fret_span <= 3:
                        # Can use one finger per fret
                        for note in fretted_notes:
                            # Assign finger (1=index, 2=middle, 3=ring, 4=pinky)
                            finger = note['fret'] - min_fret + 1
                            note['finger'] = min(finger, 4)  # Limit to 4 fingers
                    else:
                        # Needs stretching or position shift
                        # Simple algorithm: assign lowest frets to index finger,
                        # highest to pinky, and distribute middle and ring
                        fretted_notes.sort(key=lambda x: x['fret'])
                        
                        if len(fretted_notes) <= 4:
                            # One finger per note
                            for i, note in enumerate(fretted_notes):
                                note['finger'] = i + 1
                        else:
                            # More notes than fingers, need to use barre or multiple notes per finger
                            for i, note in enumerate(fretted_notes):
                                if i < 3:
                                    note['finger'] = i + 1
                                else:
                                    note['finger'] = 4  # Assign rest to pinky
                
                # Mark open strings
                for note in notes:
                    if note['fret'] == 0:
                        note['finger'] = 0  # 0 = open string
            
            else:
                # Single note
                note = notes[0]
                if note['fret'] == 0:
                    note['finger'] = 0  # Open string
                else:
                    # For single notes, default to first finger for low frets,
                    # middle finger for mid frets, etc.
                    if note['fret'] <= 2:
                        note['finger'] = 1
                    elif note['fret'] <= 4:
                        note['finger'] = 2
                    elif note['fret'] <= 6:
                        note['finger'] = 3
                    else:
                        note['finger'] = 4
        
        return sorted_positions

    def _optimize_fret_positions_advanced(self, notes: List[Dict[str, Any]], 
                                        position_bias: str, 
                                        key_hint: str = None) -> List[Dict[str, Any]]:
        """
        Optimize fretboard positions for a sequence of notes
        
        Args:
            notes: List of note dictionaries
            position_bias: Position preference on the fretboard
            key_hint: Optional key hint for better position selection
            
        Returns:
            List of notes with optimized positions
        """
        if not notes:
            logger.warning("No notes provided for position optimization")
            return []
            
        logger.info(f"Starting advanced fret position optimization for {len(notes)} notes with bias '{position_bias}'")
        print(f"• Processing {len(notes)} notes for fretboard positioning...")
        
        # Get position boundaries based on bias
        position_boundaries = self._get_position_boundaries(position_bias)
        min_fret, max_fret = position_boundaries
        logger.info(f"Using position boundaries: frets {min_fret}-{max_fret}")
        print(f"• Working within fret range: {min_fret}-{max_fret}")
        
        # If machine learning is enabled, try to use it
        use_ml = self.use_ml and self.position_model is not None
        if use_ml:
            logger.info("ML-enhanced position optimization enabled")
            print("• Using machine learning to enhance position selection")
        
        # Get scale positions if we have key information
        scale_positions = {}
        if key_hint and self.include_theory:
            logger.info(f"Using key {key_hint} for scale-based position optimization")
            print(f"• Applying music theory analysis with key: {key_hint}")
            scale_positions = self._get_scale_positions(key_hint)
        
        # Initialize progress tracking
        start_time = time.time()
        processed_count = 0
        total_notes = len(notes)
        progress_interval = max(1, total_notes // 10)  # Show progress every 10%
        
        # Group notes by time to identify chords (notes that play simultaneously)
        notes_by_time = defaultdict(list)
        for note in notes:
            notes_by_time[note.get('time', 0)].append(note)
        
        logger.info(f"Identified {len(notes_by_time)} distinct time points in the composition")
        print(f"• Detected {len(notes_by_time)} distinct time positions (chord points)")
        
        if len(notes_by_time) > 1:
            # Calculate average notes per time position
            avg_notes_per_time = total_notes / len(notes_by_time)
            logger.info(f"Average notes per time position: {avg_notes_per_time:.2f}")
            print(f"• Average notes per chord: {avg_notes_per_time:.1f}")
            
            # Log chord complexity stats
            chord_sizes = [len(chord_notes) for chord_notes in notes_by_time.values()]
            max_chord_size = max(chord_sizes)
            logger.info(f"Largest chord has {max_chord_size} notes")
            if max_chord_size > 6:
                logger.warning(f"Some chords have more notes ({max_chord_size}) than strings ({len(self.tuning)})")
                print(f"⚠ Note: Some chords have more notes ({max_chord_size}) than available strings ({len(self.tuning)})")
        
        # Process each time position and optimize positions
        optimized_positions = []
        last_position = None  # Track the last played position for context
        processing_times = []
        
        # Sort times to process notes in chronological order
        sorted_times = sorted(notes_by_time.keys())
        
        print(f"• Starting chord-by-chord optimization process...")
        for i, time in enumerate(sorted_times):
            # Group of notes at this time position (could be a single note or chord)
            chord_notes = notes_by_time[time]
            
            # Time the processing of this chord
            chord_start_time = time.time()
            
            # Log progress occasionally
            processed_count += len(chord_notes)
            if i % progress_interval == 0 or i == len(sorted_times) - 1:
                progress_pct = (i / len(sorted_times)) * 100
                elapsed = time.time() - start_time
                if i > 0:  # Avoid division by zero
                    est_total = elapsed * (len(sorted_times) / i)
                    remaining = est_total - elapsed
                    print(f"  → {progress_pct:.1f}% complete ({i}/{len(sorted_times)} positions, ~{remaining:.1f}s remaining)")
                    logger.info(f"Position optimization progress: {progress_pct:.1f}%, {i}/{len(sorted_times)} positions processed")
            
            # Single note case
            if len(chord_notes) == 1:
                logger.info(f"Optimizing single note at time {time}")
                optimized_note = self._optimize_single_note_position(
                    chord_notes[0], 
                    current_position=last_position,
                    position_boundaries=position_boundaries,
                    scale_positions=scale_positions,
                    use_ml_model=use_ml
                )
                optimized_positions.append(optimized_note)
                
                # Update last position
                if 'string' in optimized_note and 'fret' in optimized_note:
                    last_position = (optimized_note['string'], optimized_note['fret'])
            
            # Chord case (multiple simultaneous notes)
            else:
                logger.info(f"Optimizing chord with {len(chord_notes)} notes at time {time}")
                optimized_chord = self._optimize_chord_positions_advanced(
                    chord_notes,
                    current_position=last_position,
                    position_boundaries=position_boundaries,
                    key_hint=key_hint,
                    use_ml=use_ml
                )
                optimized_positions.extend(optimized_chord)
                
                # Update last position based on average position in the chord
                fretted_notes = [note for note in optimized_chord if note.get('fret', 0) > 0]
                if fretted_notes:
                    # Use the middle note as reference for hand position
                    mid_idx = len(fretted_notes) // 2
                    mid_note = fretted_notes[mid_idx]
                    last_position = (mid_note['string'], mid_note['fret'])
            
            # Record processing time for this chord
            chord_processing_time = time.time() - chord_start_time
            processing_times.append(chord_processing_time)
        
        # Calculate and log performance metrics
        total_time = time.time() - start_time
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        max_processing_time = max(processing_times) if processing_times else 0
        
        logger.info(f"Position optimization completed in {total_time:.2f}s")
        logger.info(f"Average processing time per chord: {avg_processing_time:.4f}s")
        logger.info(f"Maximum processing time for a single chord: {max_processing_time:.4f}s")
        
        print(f"✓ Fretboard position optimization complete ({total_time:.2f}s)")
        print(f"• Optimized {len(notes_by_time)} chord positions across the fretboard")
        
        return optimized_positions

    def _position_inference_model(self, note_midi, previous_pos=None, string_hint=None, key_context=None, style=None):
        """
        Use machine learning model to predict optimal fretboard positions
        
        Args:
            note_midi: MIDI note number to place
            previous_pos: Optional tuple of (string, fret) for previous position
            string_hint: Optional string number to prefer
            key_context: Optional key context (0-11 for root note)
            style: Optional playing style hint
            
        Returns:
            List of (string, fret) tuples ordered by model prediction score
        """
        if self.position_model is None:
            # If model not available, fall back to traditional algorithm
            return self._position_inference_traditional(note_midi, previous_pos, string_hint)
            
        try:
            # Create input vector for model
            input_vector = np.zeros(176)
            
            # Set MIDI note in one-hot encoding (first 128 elements)
            if 0 <= note_midi < 128:
                input_vector[note_midi] = 1.0
            
            # Set previous position (next 25 elements)
            if previous_pos:
                prev_string, prev_fret = previous_pos
                if 0 <= prev_fret <= 24:
                    input_vector[128 + prev_fret] = 1.0
            
            # Set string hint (next 6 elements)
            if string_hint is not None and 0 <= string_hint < len(self.tuning):
                input_vector[153 + string_hint] = 1.0
                
            # Set key context (next 12 elements)
            if key_context is not None and 0 <= key_context < 12:
                input_vector[159 + key_context] = 1.0
                
            # Set style (last 5 elements)
            if style is not None:
                style_idx = 0  # Default style
                if style == 'blues':
                    style_idx = 0
                elif style == 'rock':
                    style_idx = 1
                elif style == 'jazz':
                    style_idx = 2
                elif style == 'folk':
                    style_idx = 3
                elif style == 'metal':
                    style_idx = 4
                input_vector[171 + style_idx] = 1.0
            
            # Reshape for batch prediction
            input_batch = np.reshape(input_vector, (1, 176))
            
            # Get model prediction
            prediction = self.position_model.predict(input_batch, verbose=0)[0]
            
            # Map prediction scores to string/fret combinations and sort
            positions = []
            for i, score in enumerate(prediction):
                string_idx = i // 25  # 25 frets per string (0-24)
                fret = i % 25
                
                # Only consider valid positions for this note
                target_midi = self.tuning_midi[string_idx] + fret
                if target_midi == note_midi:
                    positions.append((string_idx, fret, float(score)))
            
            # Sort by descending score
            positions.sort(key=lambda x: x[2], reverse=True)
            
            # Return string/fret pairs without the scores
            return [(p[0], p[1]) for p in positions]
            
        except Exception as e:
            # Log error and fall back to traditional algorithm
            logger.error(f"Error in model inference: {str(e)}. Using fallback.")
            return self._position_inference_traditional(note_midi, previous_pos, string_hint)
