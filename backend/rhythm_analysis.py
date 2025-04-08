from audio_imports import *

class RhythmAnalyzer:
    """
    Analyzes rhythmic content of audio, including beat detection, tempo estimation,
    and rhythmic pattern identification. Uses madmom when available with graceful 
    fallback to librosa.
    """
    
    def __init__(self):
        self.has_madmom = MADMOM_AVAILABLE
        
    def detect_tempo(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Detect the tempo of the audio
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Dictionary with tempo information
        """
        # Try madmom for advanced tempo detection if available
        if self.has_madmom:
            try:
                return self._detect_tempo_madmom(audio, sr)
            except Exception as e:
                logger.warning(f"Madmom tempo detection failed: {str(e)}")
        
        # Fallback to librosa
        return self._detect_tempo_librosa(audio, sr)
    
    def _detect_tempo_madmom(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Advanced tempo detection using madmom"""
        import madmom
        
        # Create a copy of the audio with proper format for madmom
        if audio.dtype != np.float32:
            audio_madmom = audio.astype(np.float32)
        else:
            audio_madmom = audio.copy()
        
        # Normalize audio
        if np.max(np.abs(audio_madmom)) > 0:
            audio_madmom = audio_madmom / np.max(np.abs(audio_madmom))
        
        # If sample rate is different from 44100, we need to resample
        if sr != 44100:
            import scipy.signal
            audio_madmom = scipy.signal.resample_poly(audio_madmom, 44100, sr)
            sr_madmom = 44100
        else:
            sr_madmom = sr
        
        # Get beat tracking processor
        proc = madmom.features.tempo.TempoEstimationProcessor(fps=100)
        act = madmom.features.tempo.tempogram(audio_madmom, sr_madmom)
        
        # Get tempo estimates
        tempi = proc(act)
        
        # Extract the most likely tempo
        if len(tempi) > 0:
            # tempi is a list of tuples (tempo, strength)
            main_tempo = float(tempi[0][0])
            confidence = float(tempi[0][1])
            
            # Get secondary tempo if available
            secondary_tempo = float(tempi[1][0]) if len(tempi) > 1 else None
            secondary_confidence = float(tempi[1][1]) if len(tempi) > 1 else None
            
            # Calculate tempo ratio (useful for determining time signature)
            tempo_ratio = secondary_tempo / main_tempo if secondary_tempo else None
            
            # Infer time signature based on tempo ratio
            time_signature = self._infer_time_signature(tempo_ratio)
            
            return {
                'tempo': main_tempo,
                'confidence': confidence,
                'secondary_tempo': secondary_tempo,
                'secondary_confidence': secondary_confidence,
                'tempo_ratio': tempo_ratio,
                'time_signature': time_signature
            }
        else:
            return {
                'tempo': None,
                'confidence': 0,
                'error': 'Could not detect tempo'
            }
    
    def _detect_tempo_librosa(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Basic tempo detection using librosa"""
        try:
            # Extract onset envelope
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            
            # Dynamic tempo detection
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            # Calculate beat intervals and convert to BPM
            if len(beat_times) >= 2:
                intervals = np.diff(beat_times)
                beat_intervals = 60 / intervals
                
                # Filter out extreme values
                valid_intervals = beat_intervals[(beat_intervals > 40) & (beat_intervals < 240)]
                
                if len(valid_intervals) > 0:
                    # Calculate alternate tempo estimate and confidence
                    alt_tempo = np.median(valid_intervals)
                    
                    # Estimate confidence based on consistency of intervals
                    consistency = 1.0 - min(1.0, np.std(valid_intervals) / np.mean(valid_intervals))
                    
                    # Infer time signature from beat pattern
                    time_signature = self._analyze_beat_pattern(beat_times, alt_tempo)
                else:
                    alt_tempo = tempo
                    consistency = 0.5
                    time_signature = "4/4"  # Default
            else:
                alt_tempo = tempo
                consistency = 0.5
                time_signature = "4/4"  # Default
            
            return {
                'tempo': float(tempo),
                'confidence': float(consistency),
                'detected_beats': len(beats),
                'alt_tempo': float(alt_tempo),
                'time_signature': time_signature
            }
        except Exception as e:
            logger.error(f"Error in tempo detection: {str(e)}")
            return {
                'tempo': 120.0,  # Default
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _infer_time_signature(self, tempo_ratio):
        """Infer time signature from tempo ratio"""
        if tempo_ratio is None:
            return "4/4"  # Default
        
        # Check common tempo ratios
        if 0.65 <= tempo_ratio <= 0.75:
            return "3/4"  # Waltz
        elif 0.45 <= tempo_ratio <= 0.55:
            return "2/4"  # March
        elif 0.8 <= tempo_ratio <= 0.95:
            return "4/4"  # Common time
        elif 1.45 <= tempo_ratio <= 1.55:
            return "6/8"  # Compound duple
        else:
            return "4/4"  # Default
    
    def _analyze_beat_pattern(self, beat_times, tempo):
        """Analyze beat pattern to infer time signature"""
        if len(beat_times) < 8:
            return "4/4"  # Not enough data
            
        # Calculate the normalized weight/emphasis of each beat
        weights = []
        
        for i in range(1, len(beat_times) - 1):
            prev_interval = beat_times[i] - beat_times[i-1]
            next_interval = beat_times[i+1] - beat_times[i]
            
            # A longer interval before or after a beat often indicates a measure boundary
            # or emphasized beat
            weight = max(prev_interval, next_interval) / min(prev_interval, next_interval)
            weights.append(weight)
        
        # Look for patterns in the weights
        pattern_length = self._find_pattern_length(weights)
        
        if pattern_length == 3:
            return "3/4"
        elif pattern_length == 2:
            return "2/4"
        elif pattern_length == 6:
            return "6/8"
        else:
            return "4/4"  # Default
    
    def _find_pattern_length(self, weights, max_length=8):
        """Find the most likely pattern length in the beat weights"""
        if len(weights) < max_length * 2:
            # Not enough data to be confident
            return 4  # Default pattern length
            
        best_score = 0
        best_length = 4
        
        # Try different pattern lengths
        for length in [2, 3, 4, 6, 8]:
            if len(weights) < length * 2:
                continue
                
            # Calculate correlation between consecutive patterns
            corr_sum = 0
            count = 0
            
            for i in range(len(weights) - length):
                if i + length < len(weights):
                    pattern1 = weights[i:i+length]
                    pattern2 = weights[i+length:i+length*2]
                    
                    # Calculate correlation between patterns
                    if len(pattern1) == len(pattern2) and len(pattern1) > 0:
                        correlation = np.corrcoef(pattern1, pattern2)[0, 1]
                        if not np.isnan(correlation):
                            corr_sum += correlation
                            count += 1
            
            score = corr_sum / count if count > 0 else 0
            
            # Adjust score based on musical preference
            # We prefer patterns that fit common time signatures
            if length in [2, 3, 4, 6, 8]:
                score *= 1.2
            
            if score > best_score:
                best_score = score
                best_length = length
        
        return best_length
    
    def detect_beats(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Detect beat positions in the audio
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Dictionary with beat information
        """
        # Use madmom for beat detection if available
        if self.has_madmom:
            try:
                return self._detect_beats_madmom(audio, sr)
            except Exception as e:
                logger.warning(f"Madmom beat detection failed: {str(e)}")
        
        # Fallback to librosa
        return self._detect_beats_librosa(audio, sr)
    
    def _detect_beats_madmom(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Advanced beat detection using madmom"""
        import madmom
        
        # Create a copy of the audio with proper format for madmom
        if audio.dtype != np.float32:
            audio_madmom = audio.astype(np.float32)
        else:
            audio_madmom = audio.copy()
        
        # Normalize audio
        if np.max(np.abs(audio_madmom)) > 0:
            audio_madmom = audio_madmom / np.max(np.abs(audio_madmom))
        
        # If sample rate is different from 44100, we need to resample
        if sr != 44100:
            import scipy.signal
            audio_madmom = scipy.signal.resample_poly(audio_madmom, 44100, sr)
            sr_madmom = 44100
        else:
            sr_madmom = sr
        
        # Get beat tracking processor
        proc = madmom.features.beats.BeatTrackingProcessor(fps=100)
        act = madmom.features.beats.RNNBeatProcessor()(audio_madmom)
        
        # Get beat positions
        beat_times = proc(act)
        
        if len(beat_times) > 0:
            # Convert beat times to frames
            beat_frames = librosa.time_to_frames(beat_times, sr=sr)
            
            # Get tempo information
            tempo_info = self.detect_tempo(audio, sr)
            time_signature = tempo_info.get('time_signature', '4/4')
            
            # Estimate downbeats (first beat of each measure)
            if len(beat_times) >= 4:
                try:
                    # Try to use madmom's downbeat tracking
                    db_proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
                    db_act = madmom.features.downbeats.RNNDownBeatProcessor()(audio_madmom)
                    downbeats = db_proc(db_act)
                    
                    # Extract just the downbeat times
                    downbeat_times = np.array([db[0] for db in downbeats if db[1] == 1])
                    
                    # Convert to frames
                    downbeat_frames = librosa.time_to_frames(downbeat_times, sr=sr)
                    
                    # Infer time signature from beats per bar
                    if len(beat_times) > 0 and len(downbeat_times) > 1:
                        avg_beats_per_bar = len(beat_times) / len(downbeat_times)
                        time_signature = self._infer_time_signature_from_beats_per_bar(avg_beats_per_bar)
                    else:
                        time_signature = tempo_info.get('time_signature', '4/4')
                        
                except Exception as e:
                    logger.warning(f"Downbeat detection failed: {str(e)}")
                    # Fallback to basic downbeat estimation
                    downbeat_times, downbeat_frames, time_signature = self._estimate_downbeats(
                        beat_times, beat_frames, tempo_info.get('time_signature', '4/4'))
            else:
                # Not enough beats for downbeat detection
                downbeat_times = np.array([beat_times[0]]) if len(beat_times) > 0 else np.array([])
                downbeat_frames = np.array([beat_frames[0]]) if len(beat_frames) > 0 else np.array([])
                time_signature = tempo_info.get('time_signature', '4/4')
            
            return {
                'beat_times': beat_times.tolist(),
                'beat_frames': beat_frames.tolist(),
                'downbeat_times': downbeat_times.tolist(),
                'downbeat_frames': downbeat_frames.tolist(),
                'count': len(beat_times),
                'tempo': tempo_info.get('tempo', 120.0),
                'time_signature': time_signature
            }
        else:
            return {
                'beat_times': [],
                'beat_frames': [],
                'downbeat_times': [],
                'downbeat_frames': [],
                'count': 0,
                'tempo': 120.0,
                'error': 'Could not detect beats'
            }
    
    def _detect_beats_librosa(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Basic beat detection using librosa"""
        try:
            # Extract onset envelope
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            
            # Dynamic tempo detection with beat tracking
            tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            
            # Get tempo information
            tempo_info = self._detect_tempo_librosa(audio, sr)
            time_signature = tempo_info.get('time_signature', '4/4')
            
            # Estimate downbeats
            downbeat_times, downbeat_frames, time_signature = self._estimate_downbeats(
                beat_times, beat_frames, time_signature)
            
            return {
                'beat_times': beat_times.tolist(),
                'beat_frames': beat_frames.tolist(),
                'downbeat_times': downbeat_times.tolist(),
                'downbeat_frames': downbeat_frames.tolist(),
                'count': len(beat_frames),
                'tempo': float(tempo),
                'time_signature': time_signature
            }
        except Exception as e:
            logger.error(f"Error in beat detection: {str(e)}")
            return {
                'beat_times': [],
                'beat_frames': [],
                'downbeat_times': [],
                'downbeat_frames': [],
                'count': 0,
                'tempo': 120.0,
                'error': str(e)
            }
    
    def _estimate_downbeats(self, beat_times, beat_frames, time_signature):
        """Estimate downbeat positions from beats"""
        # Parse time signature
        if '/' in time_signature:
            numerator = int(time_signature.split('/')[0])
        else:
            numerator = 4  # Default
        
        # No beats detected
        if len(beat_times) == 0:
            return np.array([]), np.array([]), time_signature
        
        # Determine beats per bar based on time signature
        beats_per_bar = numerator
        
        # Get onset strength at beat positions to find likely downbeats
        # Downbeats often have stronger onset
        
        # Simple approach: assume first beat is a downbeat and
        # every 'beats_per_bar' beat after that
        downbeat_indices = np.arange(0, len(beat_times), beats_per_bar)
        
        # Extract the downbeat times and frames
        downbeat_times = np.array([beat_times[i] for i in downbeat_indices if i < len(beat_times)])
        downbeat_frames = np.array([beat_frames[i] for i in downbeat_indices if i < len(beat_frames)])
        
        return downbeat_times, downbeat_frames, time_signature
    
    def _infer_time_signature_from_beats_per_bar(self, avg_beats_per_bar):
        """Infer time signature from average beats per bar"""
        if 2.5 <= avg_beats_per_bar <= 3.5:
            return "3/4"
        elif avg_beats_per_bar < 2.5:
            return "2/4"
        elif 5.5 <= avg_beats_per_bar <= 6.5:
            return "6/8"
        else:
            return "4/4"  # Default
    
    def detect_rhythm_patterns(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Detect rhythmic patterns in the audio
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Dictionary with rhythm pattern information
        """
        # First detect beats
        beats_info = self.detect_beats(audio, sr)
        beat_times = beats_info['beat_times']
        
        # No beats detected
        if len(beat_times) < 4:
            return {
                'pattern_found': False,
                'error': 'Not enough beats detected for rhythm analysis'
            }
        
        # Calculate inter-beat intervals
        intervals = np.diff(beat_times)
        
        # Normalize intervals relative to median
        median_interval = np.median(intervals)
        norm_intervals = intervals / median_interval
        
        # Quantize intervals to common values
        quantized = []
        for interval in norm_intervals:
            if interval < 0.6:
                quantized.append(0.5)  # Eighth note
            elif interval < 0.85:
                quantized.append(0.75)  # Dotted eighth
            elif interval < 1.3:
                quantized.append(1.0)  # Quarter note
            elif interval < 1.8:
                quantized.append(1.5)  # Dotted quarter
            else:
                quantized.append(2.0)  # Half note
        
        # Look for repeating patterns
        pattern_length = self._find_rhythm_pattern_length(quantized)
        
        # Extract the most common pattern
        if pattern_length > 0 and pattern_length < len(quantized) / 2:
            # Get all potential patterns of this length
            potential_patterns = []
            for i in range(0, len(quantized) - pattern_length + 1, pattern_length):
                if i + pattern_length <= len(quantized):
                    potential_patterns.append(quantized[i:i+pattern_length])
            
            # Find the most common pattern
            if potential_patterns:
                # Convert patterns to strings for comparison
                pattern_strs = ['|'.join([str(x) for x in p]) for p in potential_patterns]
                pattern_counts = {}
                for p in pattern_strs:
                    if p in pattern_counts:
                        pattern_counts[p] += 1
                    else:
                        pattern_counts[p] = 1
                
                # Find the most common pattern
                most_common = max(pattern_counts.items(), key=lambda x: x[1])
                common_pattern_str = most_common[0]
                common_pattern = [float(x) for x in common_pattern_str.split('|')]
                
                # Calculate pattern coverage (how much of the song follows this pattern)
                coverage = most_common[1] * pattern_length / len(quantized)
                
                # Translate pattern to musical notation
                notation = self._rhythm_pattern_to_notation(common_pattern)
                
                return {
                    'pattern_found': True,
                    'pattern_length': pattern_length,
                    'pattern': common_pattern,
                    'pattern_notation': notation,
                    'pattern_coverage': float(coverage),
                    'time_signature': beats_info['time_signature']
                }
        
        # No clear pattern found
        return {
            'pattern_found': False,
            'intervals': quantized,
            'time_signature': beats_info['time_signature']
        }
    
    def _find_rhythm_pattern_length(self, intervals, max_length=8):
        """Find the most likely rhythm pattern length"""
        if len(intervals) < max_length * 2:
            # Not enough data to be confident
            return min(4, len(intervals))
            
        best_score = 0
        best_length = 4  # Default
        
        # Try pattern lengths from 2 to max_length
        for length in range(2, min(max_length + 1, len(intervals) // 2 + 1)):
            # Calculate auto-correlation for this pattern length
            score = 0
            count = 0
            
            for offset in range(0, len(intervals) - length, length):
                if offset + length < len(intervals):
                    pattern1 = intervals[offset:offset+length]
                    pattern2 = intervals[offset+length:offset+length*2]
                    
                    # Calculate similarity between patterns
                    # (inverse of mean absolute difference)
                    diff = np.mean(np.abs(np.array(pattern1) - np.array(pattern2)))
                    similarity = 1.0 / (1.0 + diff)
                    
                    score += similarity
                    count += 1
            
            avg_score = score / count if count > 0 else 0
            
            # Adjust score based on musical preference
            # We prefer patterns that fit common time signatures
            if length in [2, 3, 4, 6, 8]:
                avg_score *= 1.2
            
            if avg_score > best_score:
                best_score = avg_score
                best_length = length
        
        return best_length
    
    def _rhythm_pattern_to_notation(self, pattern):
        """Convert a rhythm pattern to musical notation"""
        notation = []
        
        for interval in pattern:
            if interval == 0.5:
                notation.append('e')  # Eighth note
            elif interval == 0.75:
                notation.append('e.')  # Dotted eighth
            elif interval == 1.0:
                notation.append('q')  # Quarter note
            elif interval == 1.5:
                notation.append('q.')  # Dotted quarter
            elif interval == 2.0:
                notation.append('h')  # Half note
            else:
                notation.append(f"({interval})")  # Unknown duration
        
        return ' '.join(notation)
    
    def detect_swing(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Detect if the rhythm has swing feel, and quantify the swing ratio
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Dictionary with swing information
        """
        # First get onset information for fine-grained timing analysis
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)
        onset_times = librosa.times_like(onset_env, sr=sr, hop_length=512)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, 
                                         backtrack=True, hop_length=512)
        onset_times = onset_times[onsets]
        
        # Get beat information
        beat_info = self.detect_beats(audio, sr)
        beat_times = np.array(beat_info['beat_times'])
        
        # Not enough beats or onsets for swing analysis
        if len(beat_times) < 4 or len(onset_times) < 8:
            return {
                'has_swing': False,
                'error': 'Not enough beats/onsets for swing analysis'
            }
        
        # Examine the placement of eighth notes between beats
        swing_ratios = []
        
        for i in range(len(beat_times) - 1):
            beat_start = beat_times[i]
            beat_end = beat_times[i+1]
            beat_duration = beat_end - beat_start
            
            # Get onsets that fall within this beat
            beat_onsets = onset_times[(onset_times >= beat_start) & (onset_times < beat_end)]
            
            if len(beat_onsets) >= 2:
                # Calculate position of first onset after the beat
                # (normalized position between 0 and 1)
                # Note: We're looking for 2nd eighth note of the beat
                for onset in beat_onsets:
                    pos = (onset - beat_start) / beat_duration
                    if 0.25 <= pos <= 0.75:  # Rough position of second eighth note
                        swing_ratios.append(pos)
                        break
        
        # Calculate average swing ratio if we have enough data
        if len(swing_ratios) >= 3:
            avg_ratio = np.mean(swing_ratios)
            
            # Determine if the music has swing
            # In straight rhythm, the second eighth note falls at 0.5
            # In swing, it falls later (typically around 0.66 or 0.75)
            is_swing = avg_ratio >= 0.54
            
            # Calculate swing intensity
            if is_swing:
                # Map from range [0.5, 0.75] to [0, 1] intensity
                intensity = min(1.0, max(0.0, (avg_ratio - 0.5) / 0.25))
            else:
                intensity = 0.0
            
            # Calculate consistency of swing (inverse of standard deviation)
            consistency = 1.0 - min(1.0, np.std(swing_ratios) * 4.0)
            
            return {
                'has_swing': is_swing,
                'swing_ratio': float(avg_ratio),
                'swing_intensity': float(intensity),
                'swing_consistency': float(consistency),
                'swing_type': self._classify_swing_type(avg_ratio)
            }
        else:
            return {
                'has_swing': False,
                'error': 'Could not determine swing ratio'
            }
    
    def _classify_swing_type(self, ratio):
        """Classify swing type based on ratio"""
        if ratio < 0.54:
            return "straight"
        elif ratio < 0.58:
            return "light swing"
        elif ratio < 0.64:
            return "medium swing"
        elif ratio < 0.7:
            return "heavy swing"
        else:
            return "shuffle"

    def analyze(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive rhythm analysis of audio data
        
        Args:
            audio: Audio data (waveform)
            
        Returns:
            Dictionary with rhythm analysis results
        """
        if not isinstance(audio, np.ndarray):
            logger.error("Input audio must be a numpy array")
            return {
                "error": "Invalid audio data format"
            }
        
        # Initialize result dictionary
        result = {}
        
        try:
            # Step 1: Detect tempo and time signature
            tempo_info = self.detect_tempo(audio, self.sr if hasattr(self, 'sr') else 22050)
            result.update(tempo_info)
            
            # Step 2: Detect beats and downbeats
            beats_info = self.detect_beats(audio, self.sr if hasattr(self, 'sr') else 22050)
            result.update(beats_info)
            
            # Step 3: Analyze rhythmic patterns
            patterns = self.analyze_rhythmic_patterns(
                audio, 
                self.sr if hasattr(self, 'sr') else 22050,
                beats_info.get('beat_times', [])
            )
            result['patterns'] = patterns
            
            # Step 4: Compute groove features
            groove = self.compute_groove_features(
                audio, 
                self.sr if hasattr(self, 'sr') else 22050,
                beats_info.get('beat_times', [])
            )
            result['groove'] = groove
            
            return result
            
        except Exception as e:
            logger.error(f"Error in rhythm analysis: {str(e)}")
            return {
                "error": f"Rhythm analysis failed: {str(e)}",
                "tempo": 120,  # Default values
                "time_signature": "4/4",
                "confidence": 0.5
            }
