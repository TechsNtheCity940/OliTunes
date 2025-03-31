import os
import logging
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from PIL import Image
import io
import base64
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoteVisualizer:
    """
    Creates piano roll and spectral visualizations of audio similar to 
    commercial tools like RipX and LALAL.AI.
    """
    
    # MIDI note range and labels
    MIDI_MIN = 36  # C2
    MIDI_MAX = 96  # C7
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Colors for visualization
    # Using a blue-to-red colormap similar to RipX
    CMAP = LinearSegmentedColormap.from_list('RipX', 
                                             [(0, '#0080FF'),   # Blue for low confidence
                                              (0.5, '#FFFFFF'), # White for medium
                                              (1, '#FF4040')])  # Red for high confidence/amplitude
    
    def __init__(self, y: np.ndarray, sr: int, notes: List[Dict[str, Any]] = None):
        """
        Initialize visualizer with audio data and optional note list
        
        Args:
            y: Audio signal
            sr: Sample rate
            notes: Detected notes (optional)
        """
        self.y = y
        self.sr = sr
        self.notes = notes
        
        # Create output directory
        self.output_dir = os.path.join(os.getcwd(), "frontend", "public", "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_piano_roll(self, filename: str, show_grid: bool = True) -> str:
        """
        Generate a piano roll visualization of detected notes
        Similar to how RipX displays notes on a digital piano
        
        Args:
            filename: Base filename to save the image
            show_grid: Whether to show piano key grid
            
        Returns:
            Path to generated image
        """
        if not self.notes:
            logger.warning("No notes provided for piano roll generation")
            return None
        
        # Set up the figure
        plt.figure(figsize=(14, 8), dpi=100)
        ax = plt.gca()
        
        # Calculate duration
        duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        # Configure piano roll axes
        ax.set_xlim(0, duration)
        ax.set_ylim(self.MIDI_MIN - 0.5, self.MIDI_MAX + 0.5)
        
        # Add piano key labels
        yticks = []
        yticklabels = []
        
        for midi_note in range(self.MIDI_MIN, self.MIDI_MAX + 1):
            note_name = self.NOTE_NAMES[midi_note % 12]
            octave = midi_note // 12 - 1
            
            # Only show C notes and A4 (440Hz)
            if note_name == 'C' or (note_name == 'A' and octave == 4):
                yticks.append(midi_note)
                yticklabels.append(f"{note_name}{octave}")
        
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        
        # Draw keyboard background (white and black keys)
        if show_grid:
            # Draw horizontal lines for piano keys
            for midi_note in range(self.MIDI_MIN, self.MIDI_MAX + 1):
                color = '#DDDDDD' if midi_note % 12 in [0, 2, 4, 5, 7, 9, 11] else '#AAAAAA'
                ax.axhline(y=midi_note, color=color, linestyle='-', alpha=0.5)
            
            # Draw vertical beat grid (assuming 4/4 at 120 BPM by default)
            tempo = 120
            beat_duration = 60.0 / tempo
            for beat in range(int(duration / beat_duration) + 1):
                beat_time = beat * beat_duration
                if beat % 4 == 0:  # Measure line
                    ax.axvline(x=beat_time, color='#888888', linestyle='-', alpha=0.8)
                else:  # Regular beat line
                    ax.axvline(x=beat_time, color='#AAAAAA', linestyle='--', alpha=0.4)
        
        # Plot detected notes as rectangles
        for note in self.notes:
            midi = note.get('midi')
            if midi is None:
                continue
                
            # Skip notes outside our range
            if not (self.MIDI_MIN <= midi <= self.MIDI_MAX):
                continue
                
            start_time = note.get('time', 0)
            duration = note.get('duration', 0.25)
            confidence = note.get('confidence', 0.8)
            
            # Create rectangle for the note
            rect = patches.Rectangle(
                (start_time, midi - 0.4),
                duration,
                0.8,
                linewidth=1,
                edgecolor='black',
                facecolor=self.CMAP(confidence),
                alpha=0.8
            )
            
            # Add rectangle to the plot
            ax.add_patch(rect)
            
        # Add labels and title
        plt.xlabel('Time (seconds)')
        plt.ylabel('MIDI Note')
        plt.title('Piano Roll Visualization (RipX-style)')
        
        # Save figure
        output_path = os.path.join(self.output_dir, f"{filename}_piano_roll.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def create_spectral_visualization(self, filename: str) -> str:
        """
        Generate an enhanced spectral visualization of the audio,
        highlighting detected notes
        
        Args:
            filename: Base filename to save the image
            
        Returns:
            Path to generated image
        """
        # Set up the figure
        plt.figure(figsize=(14, 8), dpi=100)
        
        # Calculate spectrograms
        
        # 1. Regular STFT
        D = librosa.stft(self.y, n_fft=2048, hop_length=512)
        
        # 2. Log-frequency spectrogram (similar to CQT but with better resolution)
        C = np.abs(librosa.cqt(
            self.y, 
            sr=self.sr,
            hop_length=512,
            bins_per_octave=36,
            n_bins=144,
            fmin=librosa.note_to_hz('C2')
        ))
        
        # Convert to log amplitude
        log_C = librosa.amplitude_to_db(C, ref=np.max)
        
        # Calculate time base
        times = librosa.times_like(log_C, sr=self.sr, hop_length=512)
        
        # Plot spectrogram
        plt.subplot(2, 1, 1)
        librosa.display.specshow(
            log_C,
            sr=self.sr,
            x_axis='time',
            y_axis='cqt_note',
            bins_per_octave=36,
            fmin=librosa.note_to_hz('C2'),
            cmap=self.CMAP
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('CQT Spectrogram with Note Detection')
        
        # Highlight detected notes if available
        if self.notes:
            for note in self.notes:
                midi = note.get('midi')
                if midi is None:
                    continue
                    
                start_time = note.get('time', 0)
                duration = note.get('duration', 0.25)
                note_name = note.get('note', '')
                
                # Find corresponding frequency
                freq = librosa.midi_to_hz(midi)
                
                # Draw rectangles around detected notes
                plt.annotate(
                    note_name,
                    (start_time + duration/2, note_name),
                    color='white',
                    weight='bold',
                    fontsize=8,
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc='black', ec='white', alpha=0.6)
                )
        
        # Plot harmonic-percussion separation (LALAL.AI style)
        plt.subplot(2, 1, 2)
        
        # Perform harmonic-percussion separation
        y_harmonic, y_percussive = librosa.effects.hpss(self.y)
        
        # Calculate STFT for both components
        S_harmonic = librosa.feature.melspectrogram(y=y_harmonic, sr=self.sr, n_mels=128)
        S_percussive = librosa.feature.melspectrogram(y=y_percussive, sr=self.sr, n_mels=128)
        
        # Convert to log amplitude
        log_Sh = librosa.power_to_db(S_harmonic, ref=np.max)
        log_Sp = librosa.power_to_db(S_percussive, ref=np.max)
        
        # Create RGB image by assigning harmonic to one channel and percussive to another
        spec_img = np.zeros((log_Sh.shape[0], log_Sh.shape[1], 3))
        spec_img[:, :, 0] = (log_Sh - log_Sh.min()) / (log_Sh.max() - log_Sh.min())  # Red = Harmonic
        spec_img[:, :, 2] = (log_Sp - log_Sp.min()) / (log_Sp.max() - log_Sp.min())  # Blue = Percussive
        
        # Display the colored spectrogram
        librosa.display.specshow(
            spec_img,
            sr=self.sr,
            x_axis='time',
            y_axis='mel',
            cmap=None
        )
        plt.title('Harmonic (Red) vs Percussive (Blue) Content')
        
        # Save figure
        output_path = os.path.join(self.output_dir, f"{filename}_spectral.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def generate_visualizations(self, filename: str) -> Dict[str, str]:
        """
        Generate all visualizations for the audio and notes
        
        Args:
            filename: Base filename to save images
            
        Returns:
            Dictionary with paths to generated images
        """
        results = {}
        
        # Generate piano roll
        piano_roll_path = self.create_piano_roll(filename)
        if piano_roll_path:
            results['piano_roll'] = piano_roll_path
        
        # Generate spectral visualization
        spectral_path = self.create_spectral_visualization(filename)
        if spectral_path:
            results['spectral'] = spectral_path
            
        return results
    
    @staticmethod
    def base64_encode_image(image_path: str) -> str:
        """Convert an image to base64 string for embedding in HTML/JSON"""
        if not os.path.exists(image_path):
            return None
            
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
