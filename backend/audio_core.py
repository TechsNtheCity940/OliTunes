from audio_imports import *

class AudioAnalyzer:
    """
    Core audio analysis functionality for OliTunes.
    This class handles basic audio loading, preprocessing, and
    serves as the foundation for more specialized analysis.
    """
    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        self.y, self.sr = librosa.load(audio_path)
        
        # Cache directory for analysis results
        self.cache_dir = os.path.join(os.path.dirname(self.audio_path), "analysis_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize essentia audio loader if available
        if ESSENTIA_AVAILABLE:
            try:
                self.audio_essentia = es.MonoLoader(filename=audio_path)()
                logger.info("Essentia audio loader initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Essentia audio loader: {e}")
                self.audio_essentia = None

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
                    # Check for NaN values before processing
                    if not np.all(np.isfinite(chunk)):
                        chunk = np.nan_to_num(chunk)
                    
                    # Process with lower margin for better stability
                    y_harmonic = librosa.effects.harmonic(chunk, margin=3.0)
                    
                    # Validate output
                    if not np.all(np.isfinite(y_harmonic)):
                        logger.warning(f"Harmonic separation produced NaN values for chunk {i}")
                        y_harmonic_chunks.append(chunk)  # Use original chunk as fallback
                    elif len(y_harmonic) != len(chunk):
                        logger.warning(f"Harmonic separation produced mismatched dimensions for chunk {i}. Expected {len(chunk)}, got {len(y_harmonic)}")
                        
                        # Resize to match (duplicate ending or truncate)
                        if len(y_harmonic) < len(chunk):
                            # Pad the harmonic array to match the original size
                            padding = len(chunk) - len(y_harmonic)
                            y_harmonic = np.pad(y_harmonic, (0, padding), mode='edge')
                        else:
                            # Truncate the harmonic array to the original size
                            y_harmonic = y_harmonic[:len(chunk)]
                        
                        y_harmonic_chunks.append(y_harmonic)
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
    
    def _post_process_notes(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process detected notes for better quality
        
        Applies multiple filtering steps:
        1. Remove very short notes (likely artifacts)
        2. Join adjacent similar notes (fix fragmentation)
        3. Filter outliers (far from the distribution of all detected notes)
        4. Apply velocity normalization
        """
        if not notes:
            return []
        
        # Remove very short notes (likely detection artifacts)
        min_duration = 0.05  # seconds
        notes = [note for note in notes if note.get('duration', 0) >= min_duration]
        
        # Join adjacent notes of same pitch
        # This helps with fragmented detections of sustained notes
        if len(notes) > 1:
            joined_notes = []
            current_note = notes[0].copy()
            
            for next_note in notes[1:]:
                # Check if notes are of the same pitch and close in time
                same_pitch = current_note['note'] == next_note['note']
                time_gap = next_note['time'] - (current_note['time'] + current_note['duration'])
                
                if same_pitch and time_gap < 0.1:  # Less than 100ms gap
                    # Extend current note
                    current_note['duration'] = (
                        next_note['time'] + next_note['duration'] - current_note['time']
                    )
                    # Use max of confidences
                    current_note['confidence'] = max(current_note['confidence'], next_note.get('confidence', 0))
                    current_note['velocity'] = max(current_note['velocity'], next_note.get('velocity', 64))
                else:
                    # Add current note and start a new one
                    joined_notes.append(current_note)
                    current_note = next_note.copy()
            
            # Add the last note
            joined_notes.append(current_note)
            notes = joined_notes
            
        # Ensure velocities are normalized
        velocities = [note.get('velocity', 0) for note in notes]
        if velocities:
            max_velocity = max(velocities)
            if max_velocity > 0:
                for note in notes:
                    if 'velocity' in note:
                        # Scale to MIDI velocity range (0-127)
                        note['velocity'] = min(127, int((note['velocity'] / max_velocity) * 127))
        
        return notes
        
    def analyze_tempo(self) -> Dict[str, Any]:
        """Analyze tempo and beat information"""
        # Check if results are cached
        cache_file = os.path.join(self.cache_dir, f"{os.path.basename(self.audio_path)}_tempo.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    logger.info(f"Using cached tempo analysis results")
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cached tempo analysis: {e}")
        
        result = {}
        
        # Try different methods for tempo detection with graceful degradation
        if MADMOM_AVAILABLE:
            try:
                logger.info("Using madmom for advanced tempo detection...")
                from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
                from madmom.features.tempo import TempoEstimationProcessor
                
                # Process audio with neural network beat tracker
                rnn_processor = RNNBeatProcessor()(self.y)
                beats = BeatTrackingProcessor(fps=100)(rnn_processor)
                tempo = TempoEstimationProcessor(fps=100)(rnn_processor)
                
                if len(tempo) > 0:
                    result['tempo'] = float(tempo[0][0])
                    result['confidence'] = float(tempo[0][1])
                    result['beat_times'] = beats.tolist()
                    logger.info(f"Madmom detected tempo: {result['tempo']} BPM")
            except Exception as e:
                logger.warning(f"Madmom tempo detection failed: {e}")
        
        # Fallback to librosa if madmom failed or isn't available
        if 'tempo' not in result:
            try:
                logger.info("Using librosa for tempo detection...")
                # Get tempo and beat frames
                onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
                tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
                
                # Convert beat frames to time
                beat_times = librosa.frames_to_time(beats, sr=self.sr)
                
                result['tempo'] = float(tempo)
                result['beat_times'] = beat_times.tolist()
                result['confidence'] = 0.8  # Default confidence
                logger.info(f"Librosa detected tempo: {result['tempo']} BPM")
            except Exception as e:
                logger.warning(f"Librosa tempo detection failed: {e}")
                result['tempo'] = 120.0  # Default fallback
                result['beat_times'] = []
                result['confidence'] = 0.0
        
        # Try to cache the results
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to cache tempo analysis: {e}")
            
        return result
