from audio_imports import *

class AudioVisualizer:
    """
    Generates various visualizations of audio data, including spectrograms,
    waveforms, and chord diagrams.
    """
    
    def __init__(self):
        # Set default visualization parameters
        self.figsize = (10, 6)
        self.dpi = 100
        self.cmap = 'magma'
        
    def create_spectrogram(self, audio: np.ndarray, sr: int, output_path: str = None,
                         hop_length: int = 512, n_fft: int = 2048,
                         fmin: int = 20, fmax: int = 8000) -> Dict[str, Any]:
        """
        Generate a spectrogram visualization of audio data
        
        Args:
            audio: Audio data
            sr: Sample rate
            output_path: Path to save the image (if None, returns base64 encoded image)
            hop_length: Hop length for STFT
            n_fft: FFT window size
            fmin: Minimum frequency to display
            fmax: Maximum frequency to display
            
        Returns:
            Dictionary with image data and parameters
        """
        try:
            plt.figure(figsize=self.figsize, dpi=self.dpi)
            
            # Generate spectrogram
            S = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=hop_length, n_fft=n_fft)), ref=np.max)
            
            # Plot spectrogram
            librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', 
                                  cmap=self.cmap, fmin=fmin, fmax=fmax)
            
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            plt.tight_layout()
            
            # Save or encode
            if output_path:
                plt.savefig(output_path)
                plt.close()
                return {'path': output_path}
            else:
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                return {'image_data': img_str, 'format': 'base64'}
                
        except Exception as e:
            logger.error(f"Error creating spectrogram: {str(e)}")
            return {'error': str(e)}
    
    def create_mel_spectrogram(self, audio: np.ndarray, sr: int, output_path: str = None,
                            hop_length: int = 512, n_mels: int = 128,
                            fmin: int = 20, fmax: int = 8000) -> Dict[str, Any]:
        """
        Generate a mel spectrogram visualization
        
        Args:
            audio: Audio data
            sr: Sample rate
            output_path: Path to save the image (if None, returns base64 encoded image)
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            fmin: Minimum frequency to display
            fmax: Maximum frequency to display
            
        Returns:
            Dictionary with image data and parameters
        """
        try:
            plt.figure(figsize=self.figsize, dpi=self.dpi)
            
            # Generate mel spectrogram
            S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, 
                                            fmin=fmin, fmax=fmax, hop_length=hop_length)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Plot mel spectrogram
            librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', 
                                  cmap=self.cmap, fmin=fmin, fmax=fmax)
            
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            plt.tight_layout()
            
            # Save or encode
            if output_path:
                plt.savefig(output_path)
                plt.close()
                return {'path': output_path}
            else:
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                return {'image_data': img_str, 'format': 'base64'}
                
        except Exception as e:
            logger.error(f"Error creating mel spectrogram: {str(e)}")
            return {'error': str(e)}
    
    def create_waveform(self, audio: np.ndarray, sr: int, output_path: str = None) -> Dict[str, Any]:
        """
        Generate a waveform visualization
        
        Args:
            audio: Audio data
            sr: Sample rate
            output_path: Path to save the image (if None, returns base64 encoded image)
            
        Returns:
            Dictionary with image data and parameters
        """
        try:
            plt.figure(figsize=self.figsize, dpi=self.dpi)
            
            # Plot waveform
            librosa.display.waveshow(audio, sr=sr)
            
            plt.title('Waveform')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.tight_layout()
            
            # Save or encode
            if output_path:
                plt.savefig(output_path)
                plt.close()
                return {'path': output_path}
            else:
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                return {'image_data': img_str, 'format': 'base64'}
                
        except Exception as e:
            logger.error(f"Error creating waveform: {str(e)}")
            return {'error': str(e)}
    
    def create_chromagram(self, audio: np.ndarray, sr: int, output_path: str = None,
                       hop_length: int = 512) -> Dict[str, Any]:
        """
        Generate a chromagram visualization
        
        Args:
            audio: Audio data
            sr: Sample rate
            output_path: Path to save the image (if None, returns base64 encoded image)
            hop_length: Hop length for chroma calculation
            
        Returns:
            Dictionary with image data and parameters
        """
        try:
            plt.figure(figsize=self.figsize, dpi=self.dpi)
            
            # Generate chromagram
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
            
            # Plot chromagram
            librosa.display.specshow(chroma, sr=sr, hop_length=hop_length, x_axis='time', y_axis='chroma', cmap=self.cmap)
            
            plt.colorbar()
            plt.title('Chromagram')
            plt.tight_layout()
            
            # Save or encode
            if output_path:
                plt.savefig(output_path)
                plt.close()
                return {'path': output_path}
            else:
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                return {'image_data': img_str, 'format': 'base64'}
                
        except Exception as e:
            logger.error(f"Error creating chromagram: {str(e)}")
            return {'error': str(e)}
    
    def create_chord_diagram(self, chord_name: str, output_path: str = None) -> Dict[str, Any]:
        """
        Generate a chord diagram for guitar or piano
        
        Args:
            chord_name: Name of the chord (e.g., "C", "Dm", "G7")
            output_path: Path to save the image (if None, returns base64 encoded image)
            
        Returns:
            Dictionary with image data and parameters
        """
        try:
            # Import music theory analyzer for chord data
            from music_theory import MusicTheoryAnalyzer
            
            # Parse chord name
            chord_data = self._parse_chord_name(chord_name)
            
            # Generate chord diagram based on type
            plt.figure(figsize=(8, 8), dpi=self.dpi)
            
            # Draw guitar chord diagram
            self._draw_guitar_chord(chord_data)
            
            plt.title(f'Chord Diagram: {chord_name}')
            plt.tight_layout()
            
            # Save or encode
            if output_path:
                plt.savefig(output_path)
                plt.close()
                return {'path': output_path}
            else:
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                return {'image_data': img_str, 'format': 'base64'}
                
        except Exception as e:
            logger.error(f"Error creating chord diagram: {str(e)}")
            return {'error': str(e)}
    
    def _parse_chord_name(self, chord_name: str) -> Dict[str, Any]:
        """Parse a chord name into its components"""
        # Import music theory analyzer for chord data
        from music_theory import MusicTheoryAnalyzer
        
        # Basic chord parsing
        # This is simplified - a full implementation would use music21
        
        # Extract root note
        if len(chord_name) > 1 and chord_name[1] in ['#', 'b']:
            root = chord_name[:2]
            quality = chord_name[2:]
        else:
            root = chord_name[:1]
            quality = chord_name[1:]
        
        # Default to major if no quality specified
        if not quality:
            quality = 'maj'
        
        # Map quality to full name and intervals
        quality_map = {
            '': 'major',
            'maj': 'major',
            'm': 'minor',
            'min': 'minor',
            'dim': 'diminished',
            'aug': 'augmented',
            '7': 'dominant 7th',
            'maj7': 'major 7th',
            'm7': 'minor 7th',
            'min7': 'minor 7th',
            'dim7': 'diminished 7th',
            'sus4': 'suspended 4th',
            'sus2': 'suspended 2nd',
            '6': 'major 6th',
            'm6': 'minor 6th',
            '9': 'dominant 9th',
            'add9': 'add 9th'
        }
        
        # Get the full quality name
        full_quality = quality_map.get(quality, quality)
        
        # Get chord notes using a simplified mapping
        # In a full implementation, we would use music21 to get accurate notes
        intervals = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'diminished': [0, 3, 6],
            'augmented': [0, 4, 8],
            'dominant 7th': [0, 4, 7, 10],
            'major 7th': [0, 4, 7, 11],
            'minor 7th': [0, 3, 7, 10],
            'diminished 7th': [0, 3, 6, 9],
            'suspended 4th': [0, 5, 7],
            'suspended 2nd': [0, 2, 7],
            'major 6th': [0, 4, 7, 9],
            'minor 6th': [0, 3, 7, 9],
            'dominant 9th': [0, 4, 7, 10, 14],
            'add 9th': [0, 4, 7, 14]
        }
        
        # Convert root to MIDI note number for C0
        note_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 
                   'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 
                   'A#': 10, 'Bb': 10, 'B': 11}
        
        root_num = note_map.get(root, 0)
        
        # Get intervals for this quality
        quality_intervals = intervals.get(full_quality, [0, 4, 7])  # Default to major
        
        # Calculate notes
        notes = []
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for interval in quality_intervals:
            note_idx = (root_num + interval) % 12
            notes.append(note_names[note_idx])
        
        return {
            'root': root,
            'quality': quality,
            'full_quality': full_quality,
            'notes': notes
        }
    
    def _draw_guitar_chord(self, chord_data: Dict[str, Any]) -> None:
        """Draw a guitar chord diagram"""
        # Guitar parameters
        num_strings = 6
        num_frets = 5
        start_fret = 0
        
        # Get chord fingering
        # For full implementation, use a chord database or music21
        # Here we use a simplified approach with some common chords
        chord_map = {
            # Format: [start_fret, [frets for each string]]
            # -1 means don't play that string
            # 0 means play open string
            'C': [0, [0, 1, 0, 2, 3, -1]],
            'Cm': [3, [3, 4, 5, 5, 3, 3]],
            'C7': [0, [0, 1, 3, 2, 3, -1]],
            'D': [0, [-1, -1, 0, 2, 3, 2]],
            'Dm': [0, [-1, -1, 0, 2, 3, 1]],
            'D7': [0, [-1, -1, 0, 2, 1, 2]],
            'E': [0, [0, 2, 2, 1, 0, 0]],
            'Em': [0, [0, 2, 2, 0, 0, 0]],
            'E7': [0, [0, 2, 0, 1, 0, 0]],
            'F': [0, [1, 3, 3, 2, 1, 1]],
            'Fm': [0, [1, 3, 3, 1, 1, 1]],
            'F7': [0, [1, 3, 1, 2, 1, 1]],
            'G': [0, [3, 2, 0, 0, 0, 3]],
            'Gm': [3, [3, 5, 5, 3, 3, 3]],
            'G7': [0, [3, 2, 0, 0, 0, 1]],
            'A': [0, [-1, 0, 2, 2, 2, 0]],
            'Am': [0, [-1, 0, 2, 2, 1, 0]],
            'A7': [0, [-1, 0, 2, 0, 2, 0]],
            'B': [0, [-1, 2, 4, 4, 4, 2]],
            'Bm': [0, [-1, 2, 4, 4, 3, 2]],
            'B7': [0, [-1, 2, 1, 2, 0, 2]]
        }
        
        # Get chord root
        root = chord_data['root']
        quality = chord_data['quality']
        
        # Simplify quality for mapping
        if quality == 'min':
            quality = 'm'
        
        # Check if we have this chord in our map
        chord_key = root + quality
        if chord_key in chord_map:
            start_fret, fingering = chord_map[chord_key]
        else:
            # Default to a C chord if not found
            start_fret, fingering = chord_map['C']
        
        # Find highest fret for display
        max_fret = max([f for f in fingering if f > 0]) if any(f > 0 for f in fingering) else 0
        
        # Adjust start fret for barred chords
        if max_fret > 4:
            start_fret = max(1, max_fret - 4)
        
        # Set up the plot
        plt.axis('off')
        
        # Draw fretboard
        # Strings (vertical lines)
        for i in range(num_strings):
            plt.plot([i, i], [0, num_frets], 'k-', lw=1)
        
        # Frets (horizontal lines)
        for i in range(num_frets + 1):
            plt.plot([0, num_strings - 1], [i, i], 'k-', lw=1)
        
        # Mark the nut or starting fret
        if start_fret == 0:
            plt.plot([0, num_strings - 1], [0, 0], 'k-', lw=3)
        else:
            plt.text(-0.5, 0.5, str(start_fret), fontsize=12)
        
        # Draw finger positions
        for i, fret in enumerate(fingering):
            if fret == -1:
                # Don't play this string
                plt.plot([i - 0.25, i + 0.25], [-0.5, -0.5], 'kx', markersize=8)
            elif fret == 0:
                # Open string
                plt.plot([i], [-0.5], 'ko', markersize=8, mfc='none')
            else:
                # Fretted note
                pos = fret - start_fret
                if 0 <= pos <= num_frets:
                    plt.plot([i], [pos - 0.5], 'ko', markersize=10, mfc='black')
        
        # Set limits and aspect
        plt.xlim(-1, num_strings)
        plt.ylim(-1, num_frets + 0.5)
        plt.gca().set_aspect('equal')
    
    def create_multi_visualization(self, audio: np.ndarray, sr: int, output_path: str = None) -> Dict[str, Any]:
        """
        Generate a combination visualization with waveform, spectrogram, and chromagram
        
        Args:
            audio: Audio data
            sr: Sample rate
            output_path: Path to save the image (if None, returns base64 encoded image)
            
        Returns:
            Dictionary with image data and parameters
        """
        try:
            plt.figure(figsize=(12, 12), dpi=self.dpi)
            
            # Plot waveform
            plt.subplot(3, 1, 1)
            librosa.display.waveshow(audio, sr=sr)
            plt.title('Waveform')
            
            # Plot spectrogram
            plt.subplot(3, 1, 2)
            S = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', cmap=self.cmap)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            
            # Plot chromagram
            plt.subplot(3, 1, 3)
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', cmap=self.cmap)
            plt.colorbar()
            plt.title('Chromagram')
            
            plt.tight_layout()
            
            # Save or encode
            if output_path:
                plt.savefig(output_path)
                plt.close()
                return {'path': output_path}
            else:
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                return {'image_data': img_str, 'format': 'base64'}
                
        except Exception as e:
            logger.error(f"Error creating multi-visualization: {str(e)}")
            return {'error': str(e)}
