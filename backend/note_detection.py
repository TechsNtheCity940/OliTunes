from audio_imports import *

class NoteDetector:
    """
    Module for detecting musical notes using various pitch detection algorithms.
    Provides multiple detection strategies with fallback mechanisms.
    """
    
    def __init__(self, sr: int):
        """Initialize with sample rate"""
        self.sr = sr
    
    def detect_notes_with_crepe(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """Use CREPE neural network for high-accuracy pitch detection if available"""
        try:
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
        except ImportError:
            logger.warning("CREPE not available")
            return []
        except Exception as e:
            logger.error(f"Error in CREPE detection: {str(e)}")
            return []
    
    def detect_notes_with_cqt(self, audio: np.ndarray) -> List[Dict[str, Any]]:
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
            
            # Find frames corresponding to this segment
            start_frame = np.max([0, np.argmin(np.abs(times - onset_time))])
            end_frame = np.min([len(times) - 1, np.argmin(np.abs(times - end_time))])
            
            if end_frame <= start_frame:
                continue
            
            # Extract the segment
            segment = C[:, start_frame:end_frame+1]
            
            # Find the strongest frequency bin for this segment
            if segment.size > 0:
                avg_energy = np.mean(segment, axis=1)
                peak_bin = np.argmax(avg_energy)
                
                # Convert bin index to frequency and note
                peak_freq = librosa.cqt_frequencies(
                    n_bins=n_bins,
                    fmin=fmin,
                    bins_per_octave=bins_per_octave
                )[peak_bin]
                
                # Convert frequency to MIDI note
                midi_note = librosa.hz_to_midi(peak_freq)
                note_name = librosa.midi_to_note(int(round(midi_note)), unicode=False)
                
                # Calculate confidence as normalized peak energy
                confidence = float(avg_energy[peak_bin] / np.max(avg_energy) if np.max(avg_energy) > 0 else 0)
                
                # Only add if confidence is above threshold
                if confidence > 0.3:
                    # Add note to list
                    notes.append({
                        'note': note_name,
                        'time': float(onset_time),
                        'duration': float(end_time - onset_time),
                        'frequency': float(peak_freq),
                        'confidence': confidence,
                        'velocity': float(min(127, confidence * 127))
                    })
        
        return notes
    
    def detect_notes_basic(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """Simple note detection for fallback when other methods fail"""
        # Basic configuration
        hop_length = 512
        frame_length = 2048
        
        # Compute chroma feature for pitch content
        chroma = librosa.feature.chroma_cqt(
            y=audio, 
            sr=self.sr,
            hop_length=hop_length
        )
        
        # Get times for each frame
        times = librosa.times_like(chroma, sr=self.sr, hop_length=hop_length)
        
        # Detect onsets for segmentation
        onsets = librosa.onset.onset_detect(
            y=audio, 
            sr=self.sr,
            hop_length=hop_length
        )
        
        onset_times = librosa.frames_to_time(onsets, sr=self.sr, hop_length=hop_length)
        
        # Ensure we have some onsets to work with
        if len(onset_times) == 0:
            # Create artificial segments every 0.5 seconds
            onset_times = np.arange(0, times[-1], 0.5)
        
        # Add end time
        onset_times = np.append(onset_times, times[-1])
        
        # Process each segment to find dominant note
        notes = []
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for i in range(len(onset_times) - 1):
            start_time = onset_times[i]
            end_time = onset_times[i+1]
            
            # Find frames in this segment
            start_frame = np.max([0, np.argmin(np.abs(times - start_time))])
            end_frame = np.min([chroma.shape[1] - 1, np.argmin(np.abs(times - end_time))])
            
            if end_frame <= start_frame:
                continue
                
            # Get average chroma for this segment
            segment_chroma = np.mean(chroma[:, start_frame:end_frame+1], axis=1)
            
            # Find the strongest pitch class
            pitch_class = np.argmax(segment_chroma)
            confidence = float(segment_chroma[pitch_class] / np.sum(segment_chroma) if np.sum(segment_chroma) > 0 else 0)
            
            # Assume middle octave (C4-B4) for basic detection
            # This is a simplified approach without precise octave detection
            note_name = f"{note_names[pitch_class]}4"
            
            # Only add if confidence is reasonable
            if confidence > 0.2:
                notes.append({
                    'note': note_name,
                    'time': float(start_time),
                    'duration': float(end_time - start_time),
                    'frequency': float(librosa.note_to_hz(note_name)),
                    'confidence': confidence,
                    'velocity': float(min(127, confidence * 127))
                })
        
        return notes
    
    def detect_with_aubio(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """Use aubio pitch detection if available"""
        if not AUBIO_AVAILABLE:
            logger.warning("Aubio not available for pitch detection")
            return []
            
        try:
            # Initialize aubio pitch detector
            win_s = 2048  # window size
            hop_s = 512   # hop size
            tolerance = 0.8
            
            # Ensure audio is the right format (32-bit float)
            audio_float32 = audio.astype(np.float32)
            
            # Create pitch detector
            pitch_detector = aubio.pitch("yin", win_s, hop_s, self.sr)
            pitch_detector.set_unit("Hz")
            pitch_detector.set_tolerance(tolerance)
            
            # Process audio in chunks
            total_frames = len(audio_float32)
            notes = []
            
            current_note = None
            current_start = None
            current_pitches = []
            current_confidences = []
            
            min_note_duration = 0.1  # seconds
            confidence_threshold = 0.7
            
            # Process the audio in frame-by-frame
            for i in range(0, total_frames - hop_s, hop_s):
                # Get the current audio chunk
                chunk = audio_float32[i:i+win_s]
                if len(chunk) < win_s:
                    chunk = np.pad(chunk, (0, win_s - len(chunk)), 'constant')
                
                # Get the pitch and confidence
                pitch = pitch_detector(chunk)[0]
                confidence = pitch_detector.get_confidence()
                
                # Current time in seconds
                current_time = i / self.sr
                
                # Skip low confidence or silence
                if confidence < confidence_threshold or pitch <= 0:
                    # If we were tracking a note, end it
                    if current_note is not None and len(current_pitches) > 0:
                        # Process the collected note data
                        avg_pitch = np.median(current_pitches)
                        avg_confidence = np.mean(current_confidences)
                        
                        # Convert to note name
                        midi_note = librosa.hz_to_midi(avg_pitch)
                        note_name = librosa.midi_to_note(int(round(midi_note)), unicode=False)
                        
                        # Calculate duration
                        duration = current_time - current_start
                        
                        # Add if duration is sufficient
                        if duration >= min_note_duration:
                            notes.append({
                                'note': note_name,
                                'time': float(current_start),
                                'duration': float(duration),
                                'frequency': float(avg_pitch),
                                'confidence': float(avg_confidence),
                                'velocity': float(min(127, avg_confidence * 127))
                            })
                        
                        # Reset tracking
                        current_note = None
                        current_pitches = []
                        current_confidences = []
                    
                    continue
                
                # Convert pitch to note
                midi_note = librosa.hz_to_midi(pitch)
                note_name = librosa.midi_to_note(int(round(midi_note)), unicode=False)
                
                # Either start a new note or continue current note
                if current_note is None:
                    # Start new note
                    current_note = note_name
                    current_start = current_time
                    current_pitches = [pitch]
                    current_confidences = [confidence]
                elif current_note == note_name:
                    # Continue current note
                    current_pitches.append(pitch)
                    current_confidences.append(confidence)
                else:
                    # Note changed, finish the current note and start a new one
                    # Process the collected note data
                    avg_pitch = np.median(current_pitches)
                    avg_confidence = np.mean(current_confidences)
                    
                    # Calculate duration
                    duration = current_time - current_start
                    
                    # Add if duration is sufficient
                    if duration >= min_note_duration:
                        notes.append({
                            'note': current_note,
                            'time': float(current_start),
                            'duration': float(duration),
                            'frequency': float(avg_pitch),
                            'confidence': float(avg_confidence),
                            'velocity': float(min(127, avg_confidence * 127))
                        })
                    
                    # Start new note
                    current_note = note_name
                    current_start = current_time
                    current_pitches = [pitch]
                    current_confidences = [confidence]
            
            # Add the last note if we were tracking one
            if current_note is not None and len(current_pitches) > 0:
                avg_pitch = np.median(current_pitches)
                avg_confidence = np.mean(current_confidences)
                
                # Calculate duration
                duration = (total_frames / self.sr) - current_start
                
                # Add if duration is sufficient
                if duration >= min_note_duration:
                    notes.append({
                        'note': current_note,
                        'time': float(current_start),
                        'duration': float(duration),
                        'frequency': float(avg_pitch),
                        'confidence': float(avg_confidence),
                        'velocity': float(min(127, avg_confidence * 127))
                    })
            
            return notes
            
        except Exception as e:
            logger.error(f"Error in aubio pitch detection: {str(e)}")
            return []
