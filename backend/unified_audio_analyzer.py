"""
Unified Audio Analyzer for OliTunes
This module combines functionality from various audio analyzer implementations
into a single coherent interface with graceful degradation for missing dependencies.
"""

from audio_imports import *

class UnifiedAudioAnalyzer:
    """
    Comprehensive audio analysis for OliTunes with support for multiple libraries
    and graceful degradation when dependencies are unavailable.
    """
    
    # Standard guitar tuning
    STANDARD_TUNING = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']  # Low to high
    TUNING_MIDI = [40, 45, 50, 55, 59, 64]  # MIDI note numbers
    
    # Pitch classes for note conversion
    PITCH_CLASSES = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    
    def __init__(self, audio_path: str, lazy_load: bool = True):
        """
        Initialize the audio analyzer with an audio file path.
        
        Args:
            audio_path: Path to the audio file for analysis
            lazy_load: If True, only load basic metadata and defer heavy processing
        """
        self.audio_path = audio_path
        self.lazy_load = lazy_load
        
        # Cache directory for analysis results
        self.cache_dir = os.path.join(os.path.dirname(self.audio_path), "analysis_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize with None values for lazy loading
        self.y = None
        self.sr = None
        self.audio_essentia = None
        self.madmom_audio = None
        self.aubio_source = None
        
        # Load basic audio data if not in lazy mode
        if not lazy_load:
            self._load_audio()
            self._init_specialized_libraries()
    
    def _load_audio(self):
        """Load audio data on demand."""
        if self.y is None:
            logger.info(f"Loading audio file: {self.audio_path}")
            try:
                self.y, self.sr = librosa.load(self.audio_path)
                
                # Validate loaded audio
                if self.y.size == 0:
                    raise ValueError("Empty audio data loaded")
            except Exception as e:
                logger.error(f"Error loading audio file {self.audio_path}: {str(e)}")
                # Return empty array as fallback
                self.y = np.zeros(1000)
                self.sr = 44100
    
    def _ensure_audio_loaded(self):
        """Ensure audio is loaded before operations that need it."""
        if self.y is None:
            self._load_audio()
    
    def _init_specialized_libraries(self):
        """Initialize specialized music libraries if available."""
        # Initialize essentia
        if ESSENTIA_AVAILABLE:
            try:
                self.audio_essentia = es.MonoLoader(filename=self.audio_path)()
                logger.info("Essentia audio loader initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Essentia audio loader: {e}")
                self.audio_essentia = None
        
        # Initialize madmom
        if MADMOM_AVAILABLE:
            try:
                self.madmom_audio = mm.audio.Signal(self.audio_path)
                logger.info("Madmom audio loader initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Madmom audio loader: {e}")
                self.madmom_audio = None
        
        # Initialize aubio
        if AUBIO_AVAILABLE:
            try:
                self.aubio_source = aubio.source(self.audio_path)
                logger.info("Aubio audio loader initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Aubio audio loader: {e}")
                self.aubio_source = None
    
    def _ensure_libraries_initialized(self):
        """Ensure specialized libraries are initialized before using them."""
        if self.audio_essentia is None and ESSENTIA_AVAILABLE:
            self._init_specialized_libraries()
    
    def detect_tempo(self) -> Dict[str, Any]:
        """
        Detect tempo of the audio using the best available method.
        
        Returns:
            Dictionary with tempo information
        """
        result = {'tempo': 0.0, 'confidence': 0.0}
        
        # First try with madmom if available (most accurate)
        if MADMOM_AVAILABLE and self.madmom_audio is not None:
            try:
                logger.info("Detecting tempo with madmom...")
                tempo_processor = mm.features.tempo.TempoEstimationProcessor(fps=100)
                act = mm.features.tempo.TempoEstimation(self.audio_path)
                tempo = tempo_processor(act)
                if len(tempo) > 0:
                    result['tempo'] = float(tempo[0][0])
                    result['confidence'] = float(tempo[0][1])
                    return result
            except Exception as e:
                logger.warning(f"Madmom tempo detection failed: {e}")
        
        # Try with essentia if available
        if ESSENTIA_AVAILABLE and self.audio_essentia is not None:
            try:
                logger.info("Detecting tempo with essentia...")
                rhythm_extractor = es.RhythmExtractor2013()
                rhythm = rhythm_extractor(self.audio_essentia)
                result['tempo'] = float(rhythm[0])
                result['confidence'] = float(rhythm[3])
                return result
            except Exception as e:
                logger.warning(f"Essentia tempo detection failed: {e}")
        
        # Fallback to librosa
        try:
            logger.info("Detecting tempo with librosa...")
            self._ensure_audio_loaded()
            onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sr)
            if len(tempo) > 0:
                result['tempo'] = float(tempo[0])
                result['confidence'] = 0.7  # Default confidence for librosa
            return result
        except Exception as e:
            logger.error(f"Librosa tempo detection failed: {e}")
            
        return result
    
    def detect_key(self) -> Dict[str, Any]:
        """
        Detect musical key of the audio using the best available method.
        
        Returns:
            Dictionary with key information
        """
        result = {'key': 'C', 'mode': 'major', 'confidence': 0.0}
        
        # Try with essentia if available
        if ESSENTIA_AVAILABLE and self.audio_essentia is not None:
            try:
                logger.info("Detecting key with essentia...")
                key_extractor = es.KeyExtractor()
                key, scale, strength = key_extractor(self.audio_essentia)
                result['key'] = key
                result['mode'] = scale
                result['confidence'] = float(strength)
                return result
            except Exception as e:
                logger.warning(f"Essentia key detection failed: {e}")
        
        # Fallback to librosa
        try:
            logger.info("Detecting key with librosa...")
            self._ensure_audio_loaded()
            # Compute the chromagram
            chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
            
            # Compute key profile correlation
            key_weights = self._correlate_key_profiles(chroma)
            max_key_idx = np.argmax(key_weights)
            
            # Map the index to key and mode
            is_major = (max_key_idx < 12)
            key_idx = max_key_idx % 12
            
            # Convert key index to key name
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_name = key_names[key_idx]
            
            result['key'] = key_name
            result['mode'] = 'major' if is_major else 'minor'
            result['confidence'] = float(key_weights[max_key_idx])
            
            return result
        except Exception as e:
            logger.error(f"Librosa key detection failed: {e}")
            
        return result
    
    def _correlate_key_profiles(self, chroma: np.ndarray) -> np.ndarray:
        """
        Correlate chroma features with key profiles to determine the key.
        
        Args:
            chroma: Chromagram from librosa
            
        Returns:
            Array of correlation values for each key
        """
        # Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Normalize profiles
        major_profile = major_profile / major_profile.sum()
        minor_profile = minor_profile / minor_profile.sum()
        
        # Calculate the mean chroma vector
        mean_chroma = np.mean(chroma, axis=1)
        
        # Normalize the mean chroma vector
        if np.sum(mean_chroma) > 0:
            mean_chroma = mean_chroma / np.sum(mean_chroma)
        
        # Calculate correlation for all possible keys
        correlations = np.zeros(24)
        
        # Major keys (0-11)
        for i in range(12):
            rotated_profile = np.roll(major_profile, i)
            correlations[i] = np.corrcoef(rotated_profile, mean_chroma)[0, 1]
            
        # Minor keys (12-23)
        for i in range(12):
            rotated_profile = np.roll(minor_profile, i)
            correlations[i + 12] = np.corrcoef(rotated_profile, mean_chroma)[0, 1]
            
        # Handle potential NaN values
        correlations = np.nan_to_num(correlations)
        
        return correlations
    
    def detect_time_signature(self) -> Dict[str, Any]:
        """
        Detect time signature of the audio.
        
        Returns:
            Dictionary with time signature information
        """
        # Default result
        result = {'numerator': 4, 'denominator': 4, 'confidence': 0.6}
        
        # Try with madmom if available
        if MADMOM_AVAILABLE and self.madmom_audio is not None:
            try:
                logger.info("Detecting time signature with madmom...")
                downbeat_processor = mm.features.downbeats.DBNDownBeatTrackingProcessor(fps=100)
                act = mm.features.downbeats.RNNDownBeatProcessor()(self.audio_path)
                beats = downbeat_processor(act)
                
                if len(beats) > 0:
                    # Count time between downbeats
                    downbeats = beats[beats[:, 1] == 1]
                    if len(downbeats) > 3:
                        beat_counts = []
                        for i in range(len(downbeats) - 1):
                            # Find how many beats between consecutive downbeats
                            start_time = downbeats[i][0]
                            end_time = downbeats[i+1][0]
                            measure_beats = sum(1 for b in beats if start_time <= b[0] < end_time)
                            beat_counts.append(measure_beats)
                        
                        # Get most common number of beats per measure
                        from collections import Counter
                        count = Counter(beat_counts)
                        common_beats = count.most_common(1)[0][0]
                        
                        if common_beats in [2, 3, 4, 6, 9, 12]:
                            result['numerator'] = common_beats
                            result['confidence'] = 0.85
                            
                            # Simple heuristic for denominator
                            if self.detect_tempo()['tempo'] > 100:
                                result['denominator'] = 8 if common_beats in [6, 9, 12] else 4
                            else:
                                result['denominator'] = 4
                                
                            return result
            except Exception as e:
                logger.warning(f"Madmom time signature detection failed: {e}")
        
        # Fallback to tempo-based heuristic
        try:
            tempo = self.detect_tempo()['tempo']
            
            # Adjust confidence based on tempo
            confidence_boost = min(0.2, max(0, (tempo - 80) / 100))
            result['confidence'] += confidence_boost
            
            # Perform rhythmic analysis to guess if it's triple meter
            self._ensure_audio_loaded()
            onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
            _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
            
            if len(beats) > 0:
                # Analyze time between onsets around beats
                accentuation = onset_env[beats]
                if len(accentuation) > 0:
                    # Look for patterns of strong-weak-weak beats (3/4 or 6/8)
                    accentuation = accentuation[:len(accentuation) - (len(accentuation) % 3)]
                    if len(accentuation) > 0:
                        reshaped = accentuation.reshape(-1, 3)
                        mean_pattern = np.mean(reshaped, axis=0)
                        
                        # If first beat is significantly stronger, might be triple meter
                        if mean_pattern[0] > 1.5 * mean_pattern[1] and mean_pattern[0] > 1.5 * mean_pattern[2]:
                            if tempo > 120:
                                result['numerator'] = 6
                                result['denominator'] = 8
                            else:
                                result['numerator'] = 3
                                result['denominator'] = 4
                            result['confidence'] = 0.7
        except Exception as e:
            logger.error(f"Librosa-based time signature detection failed: {e}")
        
        return result
