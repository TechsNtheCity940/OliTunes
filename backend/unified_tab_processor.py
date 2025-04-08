"""
Unified Tablature Processing Pipeline

1. Audio Separation (Demucs)
2. Tab Prediction (TabCNN)
3. Tab Refinement (LSTM)
4. Visualization
"""
import os
import logging
from typing import Dict, Optional, List
import numpy as np
import json
import soundfile as sf

# Import existing modules
from interpreter import DemucsProcessor, TabCNNProcessor
from librosa import load as load_audio_file
from librosa import effects, util

def preprocess_audio(audio_path: str) -> np.ndarray:
    """Basic audio preprocessing"""
    y, sr = load_audio_file(audio_path)
    
    # Apply basic preprocessing
    y = effects.preemphasis(y)
    y = util.normalize(y)
    
    return y

def enhance_guitar_audio(audio_path: str, output_path: str = None) -> str:
    """
    Apply advanced audio processing to enhance the guitar stem for better tablature prediction
    
    Args:
        audio_path: Path to the guitar audio file
        output_path: Path to save the enhanced audio (if None, will modify the original path)
        
    Returns:
        Path to the enhanced audio file
    """
    import librosa
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Create output path if not provided
    if output_path is None:
        dirname = os.path.dirname(audio_path)
        basename = os.path.basename(audio_path)
        output_path = os.path.join(dirname, f"enhanced_{basename}")
    
    logger.info(f"Enhancing guitar audio from {audio_path}")
    logger.info(f"Output will be saved to {output_path}")
    
    try:
        # Load the audio file
        logger.info("Loading audio file...")
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        logger.info(f"Loaded audio with shape {y.shape} and sample rate {sr}")
        
        # Apply a series of audio enhancements
        logger.info("Starting audio enhancement process...")
        
        # 1. Noise reduction using spectral gating
        try:
            import noisereduce as nr
            logger.info("Applying noise reduction...")
            y = nr.reduce_noise(y=y, sr=sr)
            logger.info("Noise reduction completed")
        except ImportError:
            logger.warning("Noisereduce library not available, skipping noise reduction")
        except Exception as e:
            logger.warning(f"Error during noise reduction: {str(e)}")
        
        # 2. Apply high-pass filter to remove low frequency rumble
        try:
            from scipy import signal
            logger.info("Applying high-pass filter...")
            # Create high-pass filter
            b, a = signal.butter(4, 80/(sr/2), btype='highpass')
            y = signal.filtfilt(b, a, y)
            logger.info("High-pass filtering completed")
        except ImportError:
            logger.warning("SciPy not available, skipping high-pass filtering")
        except Exception as e:
            logger.warning(f"Error during high-pass filtering: {str(e)}")
        
        # 3. Apply EQ to enhance mid frequencies where guitar is prominent
        try:
            from scipy import signal
            logger.info("Applying mid-frequency boost...")
            # Boost frequencies between 500Hz and 5000Hz
            b, a = signal.butter(4, [500/(sr/2), 5000/(sr/2)], btype='bandpass')
            y_filtered = signal.filtfilt(b, a, y)
            # Mix with original
            y = y * 0.5 + y_filtered * 0.8
            logger.info("Mid-frequency boost completed")
        except ImportError:
            logger.warning("SciPy not available, skipping EQ enhancement")
        except Exception as e:
            logger.warning(f"Error during EQ enhancement: {str(e)}")
        
        # 4. Apply compression to even out dynamics
        try:
            logger.info("Applying dynamic compression...")
            # Simple compression algorithm
            threshold = 0.5
            ratio = 4.0
            makeup_gain = 1.5
            
            # Apply compression
            mask = np.abs(y) > threshold
            y[mask] = threshold + (np.abs(y[mask]) - threshold) / ratio * np.sign(y[mask])
            
            # Apply makeup gain
            y = y * makeup_gain
            logger.info("Dynamic compression completed")
        except Exception as e:
            logger.warning(f"Error applying compression: {str(e)}")
        
        # 5. Normalize the audio
        try:
            logger.info("Normalizing audio...")
            y = librosa.util.normalize(y)
            logger.info("Normalization completed")
        except Exception as e:
            logger.warning(f"Error during normalization: {str(e)}")
        
        # Save the processed audio
        logger.info(f"Saving enhanced audio to {output_path}...")
        sf.write(output_path, y, sr)
        
        logger.info(f"Enhanced guitar audio successfully saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error enhancing guitar audio: {str(e)}")
        # Return the original file as fallback
        logger.warning(f"Using original audio file as fallback: {audio_path}")
        return audio_path

from tab_text_generator import TabTextGenerator
from models.lstm_model.predictor import LSTMPredictor  # We'll need to create this

class UnifiedTabProcessor:
    """Handles complete audio-to-tab pipeline"""
    
    def __init__(self):
        self.demucs = DemucsProcessor()
        self.tabcnn = TabCNNProcessor()
        self.lstm = LSTMPredictor()
        self.text_gen = TabTextGenerator()
        
        # Load style configurations
        with open('backend/data/tab_data/style_configs.json') as f:
            self.style_configs = json.load(f)
    
    def process_audio(self, audio_path: str, style: str = 'metal') -> Dict[str, Dict]:
        """Full processing pipeline"""
        try:
            # Create a dedicated directory for stems and processed files
            base_dir = os.path.dirname(os.path.abspath(audio_path))
            song_name = os.path.splitext(os.path.basename(audio_path))[0]
            stems_dir = os.path.join(base_dir, "olitunes_stems", song_name)
            os.makedirs(stems_dir, exist_ok=True)
            
            # 1. Separate guitar tracks
            stems = self.demucs.separate_audio(audio_path, output_dir=stems_dir)
            
            if not stems:
                logging.error("Failed to separate stems")
                return {}
            
            # Save the paths to a file for reference
            stems_file = os.path.join(stems_dir, "separated_stems.txt")
            with open(stems_file, 'w') as f:
                for instrument, path in stems.items():
                    f.write(f"{instrument}: {path}\n")
            logging.info(f"Stem paths saved to: {stems_file}")
            
            # 2. Process the guitar stem (which is labeled as "other" in Demucs)
            guitar_track = None
            if 'other' in stems:
                logging.info("Found guitar stem (labeled as 'other')")
                guitar_track = stems['other']
                
                # Apply audio enhancement to the guitar stem
                enhanced_guitar_path = os.path.join(stems_dir, "enhanced_guitar.wav")
                enhanced_guitar_track = enhance_guitar_audio(guitar_track, enhanced_guitar_path)
                
                # Update the stems dictionary with the enhanced guitar track
                stems['guitar'] = enhanced_guitar_track
            elif 'guitar' in stems:
                logging.info("Found guitar stem")
                guitar_track = stems['guitar']
                
                # Apply audio enhancement to the guitar stem
                enhanced_guitar_path = os.path.join(stems_dir, "enhanced_guitar.wav")
                enhanced_guitar_track = enhance_guitar_audio(guitar_track, enhanced_guitar_path)
                
                # Update the stems dictionary
                stems['guitar'] = enhanced_guitar_track
            else:
                logging.warning("No guitar stem found, using full audio")
                guitar_track = audio_path
                
                # Process the full audio as a fallback
                enhanced_guitar_path = os.path.join(stems_dir, "enhanced_full_audio.wav")
                enhanced_guitar_track = enhance_guitar_audio(guitar_track, enhanced_guitar_path)
                stems['guitar'] = enhanced_guitar_track
            
            results = {}
            for track_type in ['guitar', 'rhythm']:
                if track_type in stems:
                    # Preprocess the audio for TabCNN
                    audio_representation = self.tabcnn.preprocess_audio(stems[track_type])
                    
                    # 3. Predict raw tablature
                    raw_tab = self.tabcnn.predict_tablature(audio_representation)
                    
                    if raw_tab is not None:
                        # 4. Refine with LSTM
                        refined_tab = self.lstm.predict(raw_tab)
                        
                        # Get style-specific formatting
                        config = self.style_configs.get(style, {})
                        
                        results[track_type] = {
                            'text_tab': self.text_gen.generate(refined_tab, config.get('bpm', 120)),
                            'fretboard_data': self._create_fretboard_data(refined_tab),
                            'stem_path': stems[track_type]  # Include the path to the stem
                        }
                    else:
                        logging.error(f"Failed to predict tablature for {track_type}")
            
            # Save the results to a file for reference
            results_file = os.path.join(stems_dir, "tab_results.json")
            with open(results_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = {}
                for track_type, data in results.items():
                    serializable_results[track_type] = {
                        'text_tab': data['text_tab'],
                        'stem_path': data['stem_path'],
                        'fretboard_data': data['fretboard_data']
                    }
                json.dump(serializable_results, f, indent=2)
            
            return results
            
        except Exception as e:
            logging.error(f"Processing failed: {str(e)}")
            raise
    
    def _create_fretboard_data(self, predictions: np.ndarray) -> List[Dict]:
        """Generate data for interactive fretboard visualization"""
        # Load fretboard position mappings
        positions = np.load('backend/data/tab_data/X_tab_positions.npy')
        
        frames = []
        for frame_idx in range(predictions.shape[0]):
            frame_data = {
                'time': frame_idx * 0.1,  # 100ms per frame
                'notes': []
            }
            
            for string in range(6):
                fret = np.argmax(predictions[frame_idx, string])
                if fret > 0:
                    frame_data['notes'].append({
                        'string': string,
                        'fret': fret,
                        'position': positions[string][fret]
                    })
            
            frames.append(frame_data)
        
        return frames

# Helper function for app.py integration
def create_tablature(audio_file: str, style: str = 'rock') -> Optional[Dict[str, Dict]]:
    """Main entry point for app.py"""
    processor = UnifiedTabProcessor()
    return processor.process_audio(audio_file, style)

def test_with_audio_file(audio_path: str):
    """
    Test the unified tab processor with a specific audio file
    
    Args:
        audio_path: Path to the audio file to process
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create a dedicated directory for stems and processed files
        base_dir = os.path.dirname(os.path.abspath(audio_path))
        song_name = os.path.splitext(os.path.basename(audio_path))[0]
        stems_dir = os.path.join(base_dir, "olitunes_stems", song_name)
        os.makedirs(stems_dir, exist_ok=True)
        
        # Load and preprocess the audio
        processed_audio = preprocess_audio(audio_path)
        logger.info(f"Processed audio shape: {processed_audio.shape}")
        
        # Initialize processors
        demucs_processor = DemucsProcessor()
        tabcnn_processor = TabCNNProcessor()
        
        # Process audio
        separated_tracks = demucs_processor.separate_audio(audio_path, output_dir=stems_dir)
        logger.info("Instrument separation complete")
        
        # Display the separated stems
        if separated_tracks:
            print("\nSeparated stems:")
            for instrument, path in separated_tracks.items():
                print(f"{instrument}: {path}")
            
            # Save the paths to a file for reference
            stems_file = os.path.join(stems_dir, "separated_stems.txt")
            with open(stems_file, 'w') as f:
                for instrument, path in separated_tracks.items():
                    f.write(f"{instrument}: {path}\n")
            print(f"\nStem paths saved to: {stems_file}")
        else:
            logger.error("No stems were separated")
            return None
        
        # Process guitar track (labeled as "other" in Demucs)
        guitar_track = None
        if 'other' in separated_tracks:
            logger.info("Found guitar stem (labeled as 'other')")
            guitar_track = separated_tracks['other']
            
            # Apply audio enhancement to the guitar stem
            enhanced_guitar_path = os.path.join(stems_dir, "enhanced_guitar.wav")
            enhanced_guitar_track = enhance_guitar_audio(guitar_track, enhanced_guitar_path)
            
            # Update the separated_tracks dictionary
            separated_tracks['guitar'] = enhanced_guitar_track
        elif 'guitar' in separated_tracks:
            logger.info("Found guitar stem")
            guitar_track = separated_tracks['guitar']
            
            # Apply audio enhancement to the guitar stem
            enhanced_guitar_path = os.path.join(stems_dir, "enhanced_guitar.wav")
            enhanced_guitar_track = enhance_guitar_audio(guitar_track, enhanced_guitar_path)
            
            # Update the separated_tracks dictionary
            separated_tracks['guitar'] = enhanced_guitar_track
        else:
            logger.warning("No guitar track found, using full audio")
            guitar_track = audio_path
            
            # Process the full audio as a fallback
            enhanced_guitar_path = os.path.join(stems_dir, "enhanced_full_audio.wav")
            enhanced_guitar_track = enhance_guitar_audio(guitar_track, enhanced_guitar_path)
            separated_tracks['guitar'] = enhanced_guitar_track
            
        # Load and preprocess guitar track for TabCNN
        guitar_audio, sr = load_audio_file(separated_tracks['guitar'], sr=22050, mono=True)
        logger.info(f"Loaded guitar audio with shape: {guitar_audio.shape}")
        
        # Save a copy of the processed audio for debugging
        processed_audio_path = os.path.join(stems_dir, "processed_guitar.wav")
        sf.write(processed_audio_path, guitar_audio, sr)
        logger.info(f"Saved processed guitar audio to: {processed_audio_path}")
        
        audio_representation = tabcnn_processor.preprocess_audio(processed_audio_path)
        
        # Get predictions
        predictions = tabcnn_processor.predict_tablature(audio_representation)
        if predictions is not None:
            tab_positions = tabcnn_processor.convert_predictions_to_tab(predictions)
            logger.info(f"Generated {len(tab_positions)} tab positions")
            
            # Print some results
            print("\nFirst few tab positions:")
            for i, pos in enumerate(tab_positions[:5]):
                print(f"Position {i+1}: {pos}")
                
            # Save tab positions to a file
            tab_positions_file = os.path.join(stems_dir, "tab_positions.npy")
            np.save(tab_positions_file, tab_positions)
            logger.info(f"Saved tab positions to: {tab_positions_file}")
        else:
            logger.error("Failed to get predictions from TabCNN")
            
        return tab_positions if predictions is not None else None
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python unified_tab_processor.py <path_to_audio_file>")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    print(f"\nProcessing audio file: {audio_path}")
    test_with_audio_file(audio_path)
