from audio_imports import *

class AudioEffectsProcessor:
    """
    Audio effects processing module using Pedalboard library when available.
    Provides effects like reverb, delay, chorus, and distortion with graceful fallback.
    """
    
    def __init__(self, sr: int = 44100):
        """Initialize with sample rate"""
        self.sr = sr
        self.has_pedalboard = PEDALBOARD_AVAILABLE
        
    def apply_effects(self, audio: np.ndarray, effects_config: Dict[str, Any]) -> np.ndarray:
        """Apply audio effects to the input audio based on provided configuration"""
        if not self.has_pedalboard:
            logger.warning("Pedalboard library not available. Cannot apply effects.")
            return audio
            
        try:
            # Create a pedalboard with specified effects
            board = pedalboard.Pedalboard()
            
            # Add requested effects based on config
            if effects_config.get('reverb', {}).get('enabled', False):
                reverb_config = effects_config.get('reverb', {})
                board.append(pedalboard.Reverb(
                    room_size=reverb_config.get('room_size', 0.5),
                    damping=reverb_config.get('damping', 0.5),
                    wet_level=reverb_config.get('wet_level', 0.33),
                    dry_level=reverb_config.get('dry_level', 0.4),
                    width=reverb_config.get('width', 1.0)
                ))
                
            if effects_config.get('delay', {}).get('enabled', False):
                delay_config = effects_config.get('delay', {})
                board.append(pedalboard.Delay(
                    delay_seconds=delay_config.get('delay_seconds', 0.5),
                    feedback=delay_config.get('feedback', 0.3),
                    mix=delay_config.get('mix', 0.3)
                ))
                
            if effects_config.get('chorus', {}).get('enabled', False):
                chorus_config = effects_config.get('chorus', {})
                board.append(pedalboard.Chorus(
                    rate_hz=chorus_config.get('rate_hz', 1.0),
                    depth=chorus_config.get('depth', 0.25),
                    centre_delay_ms=chorus_config.get('centre_delay_ms', 7.0),
                    feedback=chorus_config.get('feedback', 0.0),
                    mix=chorus_config.get('mix', 0.5)
                ))
                
            if effects_config.get('distortion', {}).get('enabled', False):
                dist_config = effects_config.get('distortion', {})
                board.append(pedalboard.Distortion(
                    drive_db=dist_config.get('drive_db', 25.0)
                ))
                
            if effects_config.get('compressor', {}).get('enabled', False):
                comp_config = effects_config.get('compressor', {})
                board.append(pedalboard.Compressor(
                    threshold_db=comp_config.get('threshold_db', -10.0),
                    ratio=comp_config.get('ratio', 4.0),
                    attack_ms=comp_config.get('attack_ms', 1.0),
                    release_ms=comp_config.get('release_ms', 100.0)
                ))
                
            if effects_config.get('phaser', {}).get('enabled', False):
                phaser_config = effects_config.get('phaser', {})
                board.append(pedalboard.Phaser(
                    rate_hz=phaser_config.get('rate_hz', 1.0),
                    depth=phaser_config.get('depth', 0.5),
                    centre_frequency_hz=phaser_config.get('centre_frequency_hz', 1300.0),
                    feedback=phaser_config.get('feedback', 0.0),
                    mix=phaser_config.get('mix', 0.5)
                ))
                
            # Skip processing if no effects were added
            if len(board) == 0:
                return audio
                
            # Process audio through the pedalboard
            processed_audio = board.process(audio, self.sr)
            
            # Ensure output is normalized
            if np.max(np.abs(processed_audio)) > 0:
                processed_audio = processed_audio / np.max(np.abs(processed_audio))
                
            return processed_audio
            
        except Exception as e:
            logger.error(f"Error applying audio effects: {str(e)}")
            return audio  # Return original audio on error
            
    def apply_preset(self, audio: np.ndarray, preset_name: str) -> np.ndarray:
        """Apply a predefined effect preset"""
        presets = {
            'clean': {
                'reverb': {'enabled': True, 'room_size': 0.2, 'wet_level': 0.2},
                'compressor': {'enabled': True, 'threshold_db': -12.0, 'ratio': 2.5}
            },
            'warm': {
                'reverb': {'enabled': True, 'room_size': 0.3, 'damping': 0.6, 'wet_level': 0.25},
                'chorus': {'enabled': True, 'depth': 0.15, 'mix': 0.2},
                'compressor': {'enabled': True, 'threshold_db': -12.0, 'ratio': 3.0}
            },
            'ambient': {
                'reverb': {'enabled': True, 'room_size': 0.8, 'wet_level': 0.5, 'width': 1.0},
                'delay': {'enabled': True, 'delay_seconds': 0.4, 'feedback': 0.4, 'mix': 0.2}
            },
            'rock': {
                'distortion': {'enabled': True, 'drive_db': 15.0},
                'reverb': {'enabled': True, 'room_size': 0.3, 'wet_level': 0.15},
                'compressor': {'enabled': True, 'threshold_db': -15.0, 'ratio': 4.0}
            },
            'metal': {
                'distortion': {'enabled': True, 'drive_db': 30.0},
                'compressor': {'enabled': True, 'threshold_db': -20.0, 'ratio': 6.0}
            },
            'blues': {
                'distortion': {'enabled': True, 'drive_db': 8.0},
                'reverb': {'enabled': True, 'room_size': 0.4, 'wet_level': 0.2}
            },
            'psychedelic': {
                'phaser': {'enabled': True, 'depth': 0.8, 'rate_hz': 0.5, 'mix': 0.7},
                'delay': {'enabled': True, 'delay_seconds': 0.3, 'feedback': 0.5, 'mix': 0.3},
                'reverb': {'enabled': True, 'room_size': 0.6, 'wet_level': 0.4}
            }
        }
        
        # Get preset or use 'clean' as default
        preset = presets.get(preset_name.lower(), presets['clean'])
        
        return self.apply_effects(audio, preset)
        
    def separate_frequency_bands(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Separate audio into frequency bands:
        - Low (bass): 20Hz - 250Hz
        - Mid: 250Hz - 4kHz
        - High (treble): 4kHz - 20kHz
        """
        bands = {}
        
        try:
            # Low frequency band (20Hz - 250Hz)
            low_sos = signal.butter(8, 250, 'lowpass', fs=self.sr, output='sos')
            bands['low'] = signal.sosfilt(low_sos, audio)
            
            # High frequency band (4kHz - 20kHz)
            high_sos = signal.butter(8, 4000, 'highpass', fs=self.sr, output='sos')
            bands['high'] = signal.sosfilt(high_sos, audio)
            
            # Mid frequency band (250Hz - 4kHz) - use bandpass
            mid_sos = signal.butter(8, [250, 4000], 'bandpass', fs=self.sr, output='sos')
            bands['mid'] = signal.sosfilt(mid_sos, audio)
            
            # Normalize each band
            for band in bands:
                if np.max(np.abs(bands[band])) > 0:
                    bands[band] = bands[band] / np.max(np.abs(bands[band]))
            
        except Exception as e:
            logger.error(f"Error separating frequency bands: {str(e)}")
            # Return original audio in all bands on error
            bands = {'low': audio.copy(), 'mid': audio.copy(), 'high': audio.copy()}
            
        return bands
        
    def harmonic_percussive_separation(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Separate audio into harmonic and percussive components"""
        result = {}
        
        try:
            # Use librosa's HPSS implementation
            harmonic, percussive = librosa.effects.hpss(
                audio,
                kernel_size=31,
                power=2.0,
                mask=False
            )
            
            result['harmonic'] = harmonic
            result['percussive'] = percussive
            
            # Normalize components
            for component in result:
                if np.max(np.abs(result[component])) > 0:
                    result[component] = result[component] / np.max(np.abs(result[component]))
                    
        except Exception as e:
            logger.error(f"Error in harmonic-percussive separation: {str(e)}")
            # Return original audio in both components on error
            result = {'harmonic': audio.copy(), 'percussive': audio.copy()}
            
        return result
