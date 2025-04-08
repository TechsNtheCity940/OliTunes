from audio_imports import *

class InstrumentSeparator:
    """
    Handles separation of mixed audio into individual instrument tracks.
    Supports multiple separation algorithms with graceful degradation.
    """
    
    def __init__(self, cache_dir: str = None):
        self.has_demucs = DEMUCS_AVAILABLE
        self.has_spleeter = SPLEETER_AVAILABLE
        self.has_openunmix = OPENUNMIX_AVAILABLE
        
        # Check if at least one separation library is available
        self.has_separation = self.has_demucs or self.has_spleeter or self.has_openunmix
        
        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = os.path.join(tempfile.gettempdir(), 'olitunes_stems')
        else:
            self.cache_dir = cache_dir
            
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Initialize models
        self.demucs_model = None
        self.spleeter_model = None
        self.openunmix_model = None
    
    def separate_instruments(self, audio_path: str, 
                           method: str = 'auto',
                           output_dir: str = None) -> Dict[str, Any]:
        """
        Separate a mixed audio file into individual instrument stems
        
        Args:
            audio_path: Path to audio file to separate
            method: Separation method ('demucs', 'spleeter', 'openunmix', or 'auto')
            output_dir: Directory to save separated stems (if None, uses cache_dir)
            
        Returns:
            Dictionary with paths to separated stems and metadata
        """
        if not self.has_separation:
            return {
                'success': False,
                'error': 'No separation libraries available. Install demucs, spleeter, or openunmix.',
                'stems': {}
            }
        
        # Validate audio file
        if not os.path.exists(audio_path):
            return {
                'success': False,
                'error': f'Audio file not found: {audio_path}',
                'stems': {}
            }
        
        # Determine output directory
        if output_dir is None:
            output_dir = self.cache_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate a unique ID for this separation
        audio_hash = self._generate_audio_hash(audio_path)
        stem_dir = os.path.join(output_dir, audio_hash)
        
        # Check if stems are already cached
        if os.path.exists(stem_dir) and os.path.isdir(stem_dir):
            cached_stems = self._get_cached_stems(stem_dir)
            if cached_stems and len(cached_stems) > 0:
                logger.info(f"Found cached stems for {audio_path}")
                return {
                    'success': True,
                    'method': 'cached',
                    'stems': cached_stems,
                    'stem_dir': stem_dir
                }
        
        # Determine which method to use
        if method == 'auto':
            if self.has_demucs:
                method = 'demucs'
            elif self.has_spleeter:
                method = 'spleeter'
            elif self.has_openunmix:
                method = 'openunmix'
            else:
                return {
                    'success': False,
                    'error': 'No separation libraries available',
                    'stems': {}
                }
        
        # Call the appropriate separation method
        if method == 'demucs' and self.has_demucs:
            result = self._separate_demucs(audio_path, stem_dir)
        elif method == 'spleeter' and self.has_spleeter:
            result = self._separate_spleeter(audio_path, stem_dir)
        elif method == 'openunmix' and self.has_openunmix:
            result = self._separate_openunmix(audio_path, stem_dir)
        else:
            return {
                'success': False,
                'error': f'Method {method} not available',
                'stems': {}
            }
        
        # Add additional metadata
        result['method'] = method
        result['stem_dir'] = stem_dir
        
        return result
    
    def _generate_audio_hash(self, audio_path: str) -> str:
        """Generate a unique hash for an audio file"""
        try:
            with open(audio_path, 'rb') as f:
                # Read the first 1MB of the file for hashing
                data = f.read(1024 * 1024)
                
            # Calculate MD5 hash
            hash_obj = hashlib.md5(data)
            file_hash = hash_obj.hexdigest()
            
            # Add file size to hash
            file_size = os.path.getsize(audio_path)
            combined_hash = f"{file_hash}_{file_size}"
            
            return combined_hash
        except Exception as e:
            logger.error(f"Error generating audio hash: {str(e)}")
            # Fallback to filename and timestamp
            filename = os.path.basename(audio_path)
            timestamp = int(time.time())
            return f"{filename}_{timestamp}"
    
    def _get_cached_stems(self, stem_dir: str) -> Dict[str, str]:
        """Get cached stem files from a directory"""
        stems = {}
        
        if not os.path.exists(stem_dir):
            return stems
        
        # Look for common stem names
        stem_names = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']
        
        for stem_name in stem_names:
            # Check for different file extensions
            for ext in ['.wav', '.mp3', '.ogg']:
                stem_path = os.path.join(stem_dir, f"{stem_name}{ext}")
                if os.path.exists(stem_path):
                    stems[stem_name] = stem_path
                    break
        
        return stems
    
    def _separate_demucs(self, audio_path: str, stem_dir: str) -> Dict[str, Any]:
        """Separate using Demucs"""
        if not self.has_demucs:
            return {'success': False, 'error': 'Demucs not available', 'stems': {}}
        
        try:
            import demucs.separate
            
            # Create output directory
            if not os.path.exists(stem_dir):
                os.makedirs(stem_dir)
            
            # Prepare arguments for Demucs
            # We use subprocess to avoid modifying sys.argv
            cmd = [
                'python', '-m', 'demucs.separate',
                '-o', os.path.dirname(stem_dir),
                '--filename', os.path.basename(stem_dir) + '/{stem}.{ext}',
                audio_path
            ]
            
            # Run Demucs
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"Demucs error: {process.stderr}")
                return {
                    'success': False, 
                    'error': f'Demucs failed: {process.stderr}',
                    'stems': {}
                }
            
            # Find separated stems
            stems = self._get_cached_stems(stem_dir)
            
            if not stems:
                return {
                    'success': False,
                    'error': 'No stems were generated by Demucs',
                    'stems': {}
                }
            
            return {
                'success': True,
                'stems': stems
            }
            
        except Exception as e:
            logger.error(f"Error using Demucs: {str(e)}")
            return {
                'success': False,
                'error': f'Demucs error: {str(e)}',
                'stems': {}
            }
    
    def _separate_spleeter(self, audio_path: str, stem_dir: str) -> Dict[str, Any]:
        """Separate using Spleeter"""
        if not self.has_spleeter:
            return {'success': False, 'error': 'Spleeter not available', 'stems': {}}
        
        try:
            from spleeter.separator import Separator
            
            # Create output directory
            if not os.path.exists(stem_dir):
                os.makedirs(stem_dir)
            
            # Initialize Spleeter with 4-stem model if not already initialized
            if self.spleeter_model is None:
                self.spleeter_model = Separator('spleeter:4stems')
            
            # Process the audio file
            self.spleeter_model.separate_to_file(
                audio_path,
                os.path.dirname(stem_dir),
                filename_format=os.path.basename(stem_dir) + '/{instrument}.{codec}'
            )
            
            # Map Spleeter's output to our standard stem names
            mapping = {
                'vocals': 'vocals',
                'drums': 'drums',
                'bass': 'bass',
                'other': 'other'
            }
            
            # Find separated stems
            stems = {}
            for spleeter_name, our_name in mapping.items():
                stem_path = os.path.join(stem_dir, f"{spleeter_name}.wav")
                if os.path.exists(stem_path):
                    stems[our_name] = stem_path
            
            if not stems:
                return {
                    'success': False,
                    'error': 'No stems were generated by Spleeter',
                    'stems': {}
                }
            
            return {
                'success': True,
                'stems': stems
            }
            
        except Exception as e:
            logger.error(f"Error using Spleeter: {str(e)}")
            return {
                'success': False,
                'error': f'Spleeter error: {str(e)}',
                'stems': {}
            }
    
    def _separate_openunmix(self, audio_path: str, stem_dir: str) -> Dict[str, Any]:
        """Separate using Open-Unmix"""
        if not self.has_openunmix:
            return {'success': False, 'error': 'Open-Unmix not available', 'stems': {}}
        
        try:
            import torch
            import torchaudio
            import openunmix
            
            # Create output directory
            if not os.path.exists(stem_dir):
                os.makedirs(stem_dir)
            
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Check if GPU is available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize model
            separator = openunmix.pretrained.umxse(device=device)
            
            # Separate
            stems = separator(waveform.to(device))
            
            # Save separated stems
            stem_paths = {}
            for name, audio in stems.items():
                if name == 'vocals':
                    output_name = 'vocals'
                else:
                    output_name = 'accompaniment'
                
                # Convert from torch tensor to numpy
                audio_np = audio.cpu().detach().numpy()
                
                # Save as wav
                output_path = os.path.join(stem_dir, f"{output_name}.wav")
                sf.write(output_path, audio_np[0].T, sample_rate)
                
                stem_paths[output_name] = output_path
            
            if not stem_paths:
                return {
                    'success': False,
                    'error': 'No stems were generated by Open-Unmix',
                    'stems': {}
                }
            
            return {
                'success': True,
                'stems': stem_paths
            }
            
        except Exception as e:
            logger.error(f"Error using Open-Unmix: {str(e)}")
            return {
                'success': False,
                'error': f'Open-Unmix error: {str(e)}',
                'stems': {}
            }
    
    def apply_effects_to_stems(self, stems: Dict[str, str], 
                             effects: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """
        Apply audio effects to separated stems
        
        Args:
            stems: Dictionary mapping stem names to file paths
            effects: Dictionary mapping stem names to lists of effects to apply
            
        Returns:
            Dictionary with paths to processed stems
        """
        # Import audio effects processor
        from audio_effects import AudioEffectsProcessor
        
        processor = AudioEffectsProcessor()
        
        processed_stems = {}
        
        for stem_name, stem_path in stems.items():
            # Check if effects are specified for this stem
            if stem_name in effects and effects[stem_name]:
                try:
                    # Load audio
                    audio, sr = librosa.load(stem_path, sr=None)
                    
                    # Apply effects
                    processed_audio = processor.apply_effects_chain(audio, sr, effects[stem_name])
                    
                    # Save processed audio
                    output_path = stem_path.replace('.wav', '_processed.wav')
                    sf.write(output_path, processed_audio, sr)
                    
                    processed_stems[stem_name] = output_path
                except Exception as e:
                    logger.error(f"Error processing {stem_name}: {str(e)}")
                    processed_stems[stem_name] = stem_path
            else:
                # No effects to apply, use original
                processed_stems[stem_name] = stem_path
        
        return processed_stems
    
    def mix_stems(self, stems: Dict[str, str], 
                mix_levels: Dict[str, float] = None, 
                output_path: str = None) -> str:
        """
        Mix stems into a single audio file with adjustable levels
        
        Args:
            stems: Dictionary mapping stem names to file paths
            mix_levels: Dictionary mapping stem names to volume levels (1.0 = original)
            output_path: Path to save mixed audio
            
        Returns:
            Path to mixed audio file
        """
        if not stems:
            return None
            
        # Set default mix levels if not provided
        if mix_levels is None:
            mix_levels = {name: 1.0 for name in stems}
        
        # Set default output path if not provided
        if output_path is None:
            # Use directory of the first stem
            first_stem = list(stems.values())[0]
            stem_dir = os.path.dirname(first_stem)
            output_path = os.path.join(stem_dir, 'mixed.wav')
        
        try:
            # Load stems and determine max length
            audio_data = {}
            max_length = 0
            sample_rate = None
            
            for name, path in stems.items():
                audio, sr = librosa.load(path, sr=None)
                audio_data[name] = audio
                
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    # Resample if sample rates differ
                    audio_data[name] = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                
                max_length = max(max_length, len(audio))
            
            # Initialize mixed audio
            mixed = np.zeros(max_length)
            
            # Mix stems with levels
            for name, audio in audio_data.items():
                level = mix_levels.get(name, 1.0)
                
                # Pad if necessary
                if len(audio) < max_length:
                    audio = np.pad(audio, (0, max_length - len(audio)))
                
                # Apply level and add to mix
                mixed += audio * level
            
            # Normalize to prevent clipping
            if np.max(np.abs(mixed)) > 1.0:
                mixed = mixed / np.max(np.abs(mixed))
            
            # Save mixed audio
            sf.write(output_path, mixed, sample_rate)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error mixing stems: {str(e)}")
            return None
    
    def analyze_stem(self, stem_path: str, stem_type: str = None) -> Dict[str, Any]:
        """
        Analyze a separated stem
        
        Args:
            stem_path: Path to the stem audio file
            stem_type: Type of stem ('vocals', 'drums', 'bass', etc.)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Load the stem
            audio, sr = librosa.load(stem_path, sr=None)
            
            # Basic analysis
            duration = librosa.get_duration(y=audio, sr=sr)
            rms = float(np.sqrt(np.mean(audio**2)))
            
            # Compute spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
            
            # Create analysis result
            result = {
                'duration': float(duration),
                'rms': float(rms),
                'spectral_centroid': float(spectral_centroid),
                'spectral_bandwidth': float(spectral_bandwidth)
            }
            
            # Stem-specific analysis
            if stem_type == 'vocals':
                # Vocal-specific analysis
                from note_detection import NoteDetector
                
                # Create note detector
                detector = NoteDetector()
                
                # Detect pitch
                pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
                
                # Select pitches with highest magnitude at each frame
                pitch_values = []
                for i in range(pitches.shape[1]):
                    index = magnitudes[:, i].argmax()
                    pitch = pitches[index, i]
                    if pitch > 0:  # Filter out zero frequencies
                        pitch_values.append(float(pitch))
                
                # Calculate pitch statistics
                if pitch_values:
                    result['pitch_mean'] = float(np.mean(pitch_values))
                    result['pitch_std'] = float(np.std(pitch_values))
                    result['pitch_min'] = float(np.min(pitch_values))
                    result['pitch_max'] = float(np.max(pitch_values))
                
                # Detect vocal range
                result['vocal_range'] = detector.estimate_vocal_range(audio, sr)
                
            elif stem_type == 'drums':
                # Drum-specific analysis
                onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
                tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                result['tempo'] = float(tempo)
                
                # Count drum hits
                onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
                result['num_drum_hits'] = len(onsets)
                
            elif stem_type == 'bass':
                # Bass-specific analysis
                # Analyze low frequency content
                spec = np.abs(librosa.stft(audio))
                low_freq_energy = np.sum(spec[:10, :]) / np.sum(spec)
                result['low_freq_energy_ratio'] = float(low_freq_energy)
                
                # Detect bass notes
                from note_detection import NoteDetector
                detector = NoteDetector()
                bass_notes = detector.detect_bass_notes(audio, sr)
                if bass_notes:
                    result['bass_notes'] = bass_notes
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing stem: {str(e)}")
            return {'error': str(e)}
    
    def clean_cache(self, max_age_days: int = 7) -> int:
        """
        Remove old cached stems
        
        Args:
            max_age_days: Maximum age of cache files in days
            
        Returns:
            Number of directories removed
        """
        if not os.path.exists(self.cache_dir):
            return 0
            
        count = 0
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        for item in os.listdir(self.cache_dir):
            item_path = os.path.join(self.cache_dir, item)
            
            if os.path.isdir(item_path):
                # Check age of directory
                mtime = os.path.getmtime(item_path)
                age = current_time - mtime
                
                if age > max_age_seconds:
                    try:
                        shutil.rmtree(item_path)
                        count += 1
                    except Exception as e:
                        logger.error(f"Error removing {item_path}: {str(e)}")
        
        return count
