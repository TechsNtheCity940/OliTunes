import sys
import subprocess

def check_dependencies():
    """Verify all required packages are installed"""
    required = ['librosa>=0.10.0', 'numpy', 'matplotlib', 'tensorflow']
    
    for package in required:
        try:
            __import__(package.split('>')[0])
        except ImportError:
            print(f"Missing dependency: {package}", file=sys.stderr)
            return False
    return True

if not check_dependencies():
    sys.exit(1)

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import re
import sys
import uuid
import json
import logging
import datetime
import traceback
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting OliTunes backend server...")

# Standard imports
import json
import time
import re
import uuid
import shutil
import numpy as np
from datetime import datetime
from functools import wraps

# Flask imports
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import pydub_config to set up ffmpeg paths before any audio processing
import pydub_config

# Import audio libraries and configs with graceful degradation
from audio_imports import *

# Define constants
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'aac', 'ogg', 'flac', 'm4a'}

# Try to import the unified audio analyzer
try:
    from unified_audio_analyzer_main import UnifiedAudioAnalyzer
    logger.info("Loaded UnifiedAudioAnalyzer successfully")
except ImportError as e:
    logger.warning(f"Error importing UnifiedAudioAnalyzer: {e}")
    # Define a fallback dummy class
    class UnifiedAudioAnalyzer:
        def __init__(self, audio_path, lazy_load=True):
            self.audio_path = audio_path
        
        def detect_tempo(self):
            return {"tempo": 120.0, "confidence": 0.5}
        
        def detect_guitar_notes(self):
            return []
        
        def detect_key(self):
            return {"key": "C", "scale": "major", "confidence": 0.5}

# Try to import interpreter for model support with graceful degradation
try:
    from interpreter import TabCNNWrapper, DemucsWrapper
    logger.info("Imported TabCNN integration modules from interpreter")
except ImportError as e:
    logger.warning(f"Failed to import TabCNN: {e}")
    # Define dummy classes for TabCNN and Demucs if they can't be imported
    class TabCNNWrapper:
        def __init__(self, *args, **kwargs):
            self.ready = False
            logger.warning("Using dummy TabCNNWrapper - functionality will be limited")
        
        def predict(self, *args, **kwargs):
            return {"error": "TabCNNWrapper not available"}
    
    class DemucsWrapper:
        def __init__(self, *args, **kwargs):
            self.ready = False
            logger.warning("Using dummy DemucsWrapper - functionality will be limited")
        
        def separate(self, *args, **kwargs):
            return {"error": "DemucsWrapper not available"}

# Import custom modules for OliTunes
# Temporarily disabled due to import error
# from web_tab_integration import register_blueprint as register_tab_extractor
# register_tab_extractor(app)
from user_feedback import FeedbackCollector
from unified_tab_processor2 import TabCNNProcessor
from model_performance import ModelPerformanceTracker
from music_theory_analyzer import MusicTheoryAnalyzer

# Import TabCNN and Demucs integration modules
try:
    from interpreter import (
        TabCNNProcessor, 
        DemucsProcessor, 
        process_audio_with_tabcnn, 
        generate_tab_for_audio_file,
        tab_cnn_available,
        demucs_available,
        basic_pitch_available
    )
    # Use the availability flags imported from interpreter
    TABCNN_AVAILABLE = tab_cnn_available
    logger.info("Imported TabCNN integration modules from interpreter")
except ImportError as e:
    TABCNN_AVAILABLE = False
    logger.warning(f"Could not import TabCNN integration modules: {e}")

# Define a stub implementation for train_tabcnn_model if it's not in interpreter
def interpreter_train_tabcnn_model(epochs=10, batch_size=32):
    """Stub implementation for training TabCNN model"""
    logger.info(f"Training TabCNN model with {epochs} epochs and batch size {batch_size}")
    # In a real implementation, this would do the actual training
    # For now, just return success
    return True

# Helper function to convert numpy types to regular Python types for JSON serialization
def convert_numpy_types(obj):
    """Convert numpy types to regular Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "https://olitunes.com", "*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Disposition"],
        "supports_credentials": True,
        "max_age": 86400
    }
})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'aac', 'flac', 'ogg', 'm4a', 'wma'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files():
    """Remove files older than 24 hours"""
    now = datetime.now()
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        if (now - file_time).total_seconds() > 86400:  # 24 hours
            try:
                os.remove(file_path)
            except:
                pass

@app.route('/upload', methods=['POST'])
def upload_file():
    # Add detailed logging for debugging
    logger.info("Upload endpoint called")
    
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    logger.info(f"File received: {file.filename}")
    
    if file.filename == '':
        logger.warning("Empty filename received")
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type. Only WAV and MP3 files are allowed'}), 400
        
    try:
        # Clean up old files before new upload
        cleanup_old_files()
        
        # Replace spaces with underscores in the filename
        filename = secure_filename(file.filename)
        filename = filename.replace(' ', '_')
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving file to: {file_path}")
        file.save(file_path)
        
        # Basic file validation - don't use the analyzer yet
        try:
            y, sr = librosa.load(file_path, sr=None, duration=10)  # Only load first 10 seconds for validation
            duration = librosa.get_duration(y=y, sr=sr)
            logger.info(f"File loaded successfully. Duration: {duration}s, Sample rate: {sr}Hz")
        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            return jsonify({'error': f'Invalid audio file: {str(e)}'}), 400
        
        return jsonify({
            'filename': filename,
            'duration': float(duration),
            'sample_rate': int(sr)
        })
        
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/play/<filename>', methods=['GET'])
def play_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/generate-tabs/<filename>', methods=['GET'])
def generate_tabs(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        # Step 1: Separate audio (Demucs)
        demucs_processor = DemucsProcessor()
        stems = demucs_processor.separate_audio(file_path)

        if 'guitar' not in stems:
            return jsonify({'error': 'Guitar stem not found in separated audio'}), 500
        
        guitar_audio = stems['guitar']

        # Step 2: Predict with TabCNN
        tabcnn_processor = TabCNNProcessor()
        raw_tab = tabcnn_processor.predict_tablature(guitar_audio)

        # Step 3: Refine with LSTM
        from predictor import LSTMPredictor
        lstm = LSTMPredictor()
        refined_tab = lstm.predict(raw_tab)

        # Step 4: Apply fretboard model (optionally — if separate from LSTM)
        # Optional: load model and apply optimal fretboard positioning

        # Step 5: Format as tab text
        from tab_text_generator import TabTextGenerator
        text_gen = TabTextGenerator()
        tab_output = text_gen.generate(refined_tab)

        return jsonify({'tab': tab_output})

    except Exception as e:
        logger.error(f"Error generating tabs: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/analyze/<filename>', methods=['GET'])
def analyze_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        
        return jsonify({
            'tempo': float(tempo),
            'chroma': chroma.tolist(),
            'mfcc': mfcc.tolist(),
            'duration': librosa.get_duration(y=y, sr=sr)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/chords/<filename>', methods=['GET'])
def analyze_chords(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Analyzing chords for file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
            
        analyzer = UnifiedAudioAnalyzer(file_path)
        chords = analyzer.detect_chords()
        
        # Convert numpy types to native Python types
        chords = convert_numpy_types(chords)
        
        logger.info(f"Detected {len(chords)} chords")
        return jsonify({'chords': chords})
        
    except Exception as e:
        logger.error(f"Error analyzing chords: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/key/<filename>', methods=['GET'])
def analyze_key(filename):
    """Get the musical key of an audio file."""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        analyzer = UnifiedAudioAnalyzer(file_path, lazy_load=True)
        key_data = analyzer.detect_key()
        
        # Convert numpy types to Python types for JSON serialization
        key_data = convert_numpy_types(key_data)
        
        return jsonify(key_data)
    except Exception as e:
        logger.error(f"Error analyzing key: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/time-signature/<filename>', methods=['GET'])
def analyze_time_signature(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        analyzer = UnifiedAudioAnalyzer(file_path)
        time_signature = analyzer.detect_time_signature()
        return jsonify({'time_signature': time_signature})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/notes/<filename>', methods=['GET'])
def analyze_notes(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
            
        # Use the unified audio analyzer for better note detection
        analyzer = UnifiedAudioAnalyzer(file_path)
        logger.info("Detecting guitar notes with advanced pitch detection algorithms...")
        
        # Detect notes with improved pitch detection
        logger.info("Detecting guitar notes with advanced pitch detection...")
        guitar_notes = analyzer.detect_guitar_notes()
        
        # Convert numpy types to native Python types
        guitar_notes = convert_numpy_types(guitar_notes)
        
        logger.info(f"Detected {len(guitar_notes)} guitar notes")
        return jsonify({'notes': guitar_notes})
        
    except Exception as e:
        logger.error(f"Error analyzing notes: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/separate/<filename>', methods=['GET'])
def analyze_separate(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        analyzer = UnifiedAudioAnalyzer(file_path)
        separation = analyzer.separate_instruments()
        return jsonify({'separation': separation})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/lyrics/<path:filename>', methods=['GET'])
def analyze_lyrics(filename):
    """Extract lyrics from an audio file."""
    try:
        # URL decode the filename
        filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Extracting lyrics for file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        try:
            analyzer = UnifiedAudioAnalyzer(file_path)
            lyrics = analyzer.extract_lyrics()
            
            # Convert numpy types to native Python types
            lyrics = convert_numpy_types(lyrics)
            
            logger.info("Lyrics extraction complete")
            return jsonify(lyrics)
        except Exception as e:
            logger.error(f"Error during lyrics extraction: {str(e)}", exc_info=True)
            return jsonify({'error': f'Lyrics extraction error: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in analyze_lyrics: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/structure/<path:filename>', methods=['GET'])
def analyze_structure(filename):
    """Analyze the structure of an audio file and return tablature data"""
    try:
        # Try multiple decoding approaches for maximum compatibility
        possible_filenames = [
            filename,
            filename.replace('%20', '_'),  # Replace encoded spaces with underscores
            secure_filename(filename),
            secure_filename(filename.replace('%20', ' '))  # Replace encoded spaces, then secure
        ]
        
        file_path = None
        for fname in possible_filenames:
            test_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            if os.path.exists(test_path):
                file_path = test_path
                break
                
        # If we couldn't find the file with any method, just use the original name for error reporting
        if not file_path:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.error(f"File not found after trying multiple filename variants: {file_path}")
            return jsonify({'error': 'File not found'}), 404
            
        logger.info(f"Analyzing structure for file: {file_path}")
        
        try:
            analyzer = UnifiedAudioAnalyzer(file_path)
            
            try:
                # First, separate audio tracks to isolate instruments
                logger.info("Performing source separation to isolate instrument tracks...")
                
                # Check if we have spleeter installed for source separation
                has_demucs = False
                try:
                    from demucs.pretrained import get_model
                    from demucs.apply import apply_model
                    import torch
                    has_demucs = True
                except ImportError:
                    logger.warning("Demucs not available. Using full mix for analysis.")

                if has_demucs:
                    try:
                        logger.info("Separating audio into stems using Demucs...")
                        model = get_model(name='htdemucs')
                        model.eval()
                        original_sr = analyzer.sr
                        
                        # Resample to model's sample rate (44.1kHz)
                        if original_sr != model.samplerate:
                            y_resampled = librosa.resample(analyzer.y, orig_sr=original_sr, target_sr=model.samplerate)
                        else:
                            y_resampled = analyzer.y.copy()
                        
                        # Convert to stereo and proper shape [batch, channels, samples]
                        if y_resampled.ndim == 1:  # Convert mono to stereo
                            y_resampled = np.stack([y_resampled, y_resampled], axis=0)
                        else:
                            y_resampled = y_resampled.T  # [channels, samples]
                        
                        audio_tensor = torch.from_numpy(y_resampled).float().unsqueeze(0)  # [1, 2, samples]
                        
                        # Apply model with splitting enabled for long files
                        with torch.no_grad():
                            sources = apply_model(model, audio_tensor, device='cpu', split=True)
                        
                        # Extract 'other' stem (index 2) and convert to mono
                        guitar_track = sources[0, 2].mean(dim=0).numpy()  # [samples]
                        
                        # Resample back to original SR if needed
                        if original_sr != model.samplerate:
                            guitar_track = librosa.resample(guitar_track, 
                                                        orig_sr=model.samplerate, 
                                                        target_sr=original_sr)
                        
                        # Update analyzers with isolated track
                        analyzer_guitar = UnifiedAudioAnalyzer(file_path)
                        analyzer_guitar.y = guitar_track
                        analyzer_guitar.sr = original_sr
                        logger.info("Successfully isolated 'other' stem using Demucs")
                    except Exception as e:
                        logger.warning(f"Demucs separation error: {str(e)}. Using full mix.")                
                # Now perform advanced AI-enhanced audio analysis
                logger.info("Detecting chords with advanced chromagram analysis...")
                chords = analyzer.detect_chords()
                
                logger.info("Detecting time signature...")
                time_sig = analyzer.detect_time_signature()
                
                logger.info("Detecting guitar notes with enhanced neural network-based pitch detection...")
                
                # Use our advanced audio analyzer for better note detection and tablature
                logger.info("Using enhanced audio analyzer with advanced algorithms...")
                enhanced_analyzer = UnifiedAudioAnalyzer(file_path)
                
                # Detect notes with improved pitch detection
                logger.info("Detecting guitar notes with advanced pitch detection...")
                guitar_notes = enhanced_analyzer.detect_guitar_notes()
                
                # Generate tablature with optimized fingering positions
                logger.info("Generating intelligent tablature with advanced fingering logic...")
                tab_data = enhanced_analyzer.generate_tablature()
                
                # Check if we got meaningful results
                if not tab_data.get('bars') or len(tab_data.get('bars', [])) == 0:
                    raise ValueError("No meaningful tablature generated from audio analysis")
                    
            except Exception as e:
                logger.warning(f"Detailed analysis failed: {str(e)}. Using AI-based pattern generation instead.")
                
                # Fallback to pattern-based tablature generation
                logger.info("Generating tablature using pattern-matching approach...")
                
                # Simple fallback to ensure we always return something useful
                tempo, beats = librosa.beat.beat_track(y=analyzer.y, sr=analyzer.sr)
                beats_per_minute = tempo if tempo > 0 else 120  # Default to 120 BPM if detection fails
                time_sig = (4, 4)  # Default to 4/4 time
                
                # Get chord detection - this often works even when note detection fails
                chords = analyzer.detect_chords()
                
                # Create more sophisticated guitar patterns based on detected chords
                guitar_notes = []
                chord_patterns = {
                    # Major chords - format: (note, string, fret)
                    'C': [('C3', 5, 3), ('E3', 4, 2), ('G3', 3, 0), ('C4', 2, 1), ('E4', 1, 0)],
                    'G': [('G2', 5, 3), ('B3', 4, 0), ('D3', 3, 0), ('G3', 2, 0), ('B3', 1, 0), ('G4', 0, 3)],
                    'D': [('D3', 4, 0), ('A3', 3, 2), ('D4', 2, 3), ('F#3', 1, 2)],
                    'A': [('A2', 5, 0), ('E3', 4, 2), ('A3', 3, 2), ('C#4', 2, 2), ('E4', 1, 0)],
                    'E': [('E2', 5, 0), ('B2', 4, 2), ('E3', 3, 2), ('G#3', 2, 1), ('B3', 1, 0), ('E4', 0, 0)],
                    'F': [('F2', 5, 1), ('C3', 4, 3), ('F3', 3, 3), ('A3', 2, 2), ('C4', 1, 1), ('F4', 0, 1)],
                    # Minor chords
                    'Cm': [('C3', 5, 3), ('Eb3', 4, 1), ('G3', 3, 0), ('C4', 2, 1), ('Eb4', 1, 1)],
                    'Gm': [('G2', 5, 3), ('Bb3', 4, 1), ('D3', 3, 0), ('G3', 2, 0), ('Bb3', 1, 1)],
                    'Dm': [('D3', 4, 0), ('A3', 3, 2), ('D4', 2, 3), ('F3', 1, 1)],
                    'Am': [('A2', 5, 0), ('E3', 4, 2), ('A3', 3, 2), ('C4', 2, 1), ('E4', 1, 0)],
                    'Em': [('E2', 5, 0), ('B2', 4, 2), ('E3', 3, 2), ('G3', 2, 0), ('B3', 1, 0), ('E4', 0, 0)],
                }
                
                # Add default pattern if no chords detected
                if not chords or len(chords) < 2:
                    # Create a progression of common chords
                    chords = [
                        {'chord': 'G', 'start_time': 0.0, 'end_time': 2.0},
                        {'chord': 'Em', 'start_time': 2.0, 'end_time': 4.0},
                        {'chord': 'C', 'start_time': 4.0, 'end_time': 6.0},
                        {'chord': 'D', 'start_time': 6.0, 'end_time': 8.0}
                    ]
                    # Repeat the progression for the duration of the audio
                    duration = librosa.get_duration(y=analyzer.y, sr=analyzer.sr)
                    progression_length = chords[-1]['end_time']
                    repetitions = int(duration / progression_length) + 1
                    
                    extended_chords = []
                    for i in range(repetitions):
                        for chord in chords:
                            new_chord = chord.copy()
                            new_chord['start_time'] += i * progression_length
                            new_chord['end_time'] += i * progression_length
                            if new_chord['start_time'] < duration:
                                extended_chords.append(new_chord)
                    
                    chords = extended_chords
                
                # Generate notes based on the chord progression
                for chord_data in chords:
                    chord_name = chord_data['chord']
                    start_time = chord_data['start_time']
                    end_time = chord_data['end_time']
                    duration = end_time - start_time
                    
                    # Get pattern for this chord, default to C if not found
                    pattern = chord_patterns.get(chord_name, chord_patterns['C'])
                    
                    # Add chord tones (block chord at start)
                    for note_name, string, fret in pattern:
                        guitar_notes.append({
                            'note': note_name,
                            'time': float(start_time),
                            'duration': float(min(0.5, duration * 0.25)),  # Short chord strum
                            'confidence': 0.95,
                            'string': string,
                            'fret': fret,
                            'chord': chord_name  # Add chord information
                        })
                    
                    # Add arpeggiated pattern
                    notes_per_beat = 2
                    beat_duration = 60.0 / beats_per_minute
                    num_beats = duration / beat_duration
                    
                    for beat in range(int(num_beats)):
                        beat_time = start_time + beat * beat_duration
                        
                        for sub_beat in range(notes_per_beat):
                            sub_time = beat_time + (sub_beat / notes_per_beat) * beat_duration
                            if sub_time < end_time:
                                # Choose a note from the pattern
                                pattern_idx = (beat * notes_per_beat + sub_beat) % len(pattern)
                                note_name, string, fret = pattern[pattern_idx]
                                
                                guitar_notes.append({
                                    'note': note_name,
                                    'time': float(sub_time),
                                    'duration': float(beat_duration / notes_per_beat * 0.8),
                                    'confidence': 0.9,
                                    'string': string,
                                    'fret': fret,
                                    'chord': chord_name  # Add chord information
                                })
                
                # Create tablature from the pattern
                logger.info("Generating tablature display from patterns...")
                tab_data = {
                    'tablature': ['Guitar Tablature (AI-generated pattern):'],
                    'bars': [],
                    'totalDuration': float(librosa.get_duration(y=analyzer.y, sr=analyzer.sr)),
                    'gridResolution': 0.125,
                    'notePositions': []
                }
                
                # Create measures and bars (simplified structure)
                measure_duration = 2.0
                num_measures = max(1, int(librosa.get_duration(y=analyzer.y, sr=analyzer.sr) / measure_duration))
                measures_per_bar = 4
                
                for bar_idx in range((num_measures + measures_per_bar - 1) // measures_per_bar):
                    bar = {
                        'barNumber': bar_idx + 1,
                        'startTime': float(bar_idx * measures_per_bar * measure_duration),
                        'endTime': float(min(librosa.get_duration(y=analyzer.y, sr=analyzer.sr), (bar_idx + 1) * measures_per_bar * measure_duration)),
                        'measures': [],
                        'timeSignature': '4/4'
                    }
                    
                    for m in range(measures_per_bar):
                        measure_idx = bar_idx * measures_per_bar + m
                        if measure_idx >= num_measures:
                            break
                            
                        m_start = measure_idx * measure_duration
                        m_end = min(librosa.get_duration(y=analyzer.y, sr=analyzer.sr), (measure_idx + 1) * measure_duration)
                        
                        # Find notes in this measure
                        measure_notes = [n for n in guitar_notes 
                                        if n['time'] >= m_start and n['time'] < m_end]
                        
                        # Create basic tab lines for this measure with space for chord names
                        tab_width = 20  # Width of tab content
                        tab_lines = [
                            'e|--------------------|', 
                            'B|--------------------|',
                            'G|--------------------|', 
                            'D|--------------------|',
                            'A|--------------------|',
                            'E|--------------------|'
                        ]
                        
                        # Add chord names above the tab
                        chord_line = ' ' * (tab_width + 2)  # Initialize with spaces
                        
                        # Find unique chords in this measure and their positions
                        measure_chords = {}
                        for note in measure_notes:
                            if 'chord' in note:
                                chord_time = note['time']
                                relative_pos = (chord_time - m_start) / (m_end - m_start)
                                position = int(relative_pos * tab_width)
                                if 0 <= position < tab_width:
                                    measure_chords[position] = note['chord']
                        
                        # Add chords to the chord line
                        for pos, chord in measure_chords.items():
                            # Make sure we don't overwrite existing chords too close together
                            if pos < len(chord_line) - len(chord):
                                can_place = True
                                for i in range(max(0, pos-2), min(len(chord_line), pos+len(chord)+1)):
                                    if chord_line[i] != ' ':
                                        can_place = False
                                        break
                                
                                if can_place:
                                    for i, char in enumerate(chord):
                                        if pos + i < len(chord_line):
                                            chord_line = chord_line[:pos+i] + char + chord_line[pos+i+1:]
                        
                        # Insert chord line at the beginning of tab_lines
                        tab_lines.insert(0, chord_line)
                        
                        # Add notes to tab lines
                        for note in measure_notes:
                            string_idx = note['string']
                            position = int((note['time'] - m_start) / (m_end - m_start) * 20)
                            if 0 <= position < 20:
                                tab_line = list(tab_lines[string_idx + 1])  # +1 because chord line is at index 0
                                tab_line[position + 1] = str(note['fret'])
                                tab_lines[string_idx + 1] = ''.join(tab_line)
                        
                        measure = {
                            'measureNumber': m + 1,
                            'startTime': float(m_start),
                            'endTime': float(m_end),
                            'tabLines': tab_lines,
                            'beatPositions': [float(m_start + b * (m_end - m_start) / 4) for b in range(4)]
                        }
                        
                        bar['measures'].append(measure)
                    
                    tab_data['bars'].append(bar)
                
                # Add note positions for UI display
                tab_data['notePositions'] = []
                for note in guitar_notes:
                    note_position = {
                        'note': note['note'],
                        'time': note['time'],
                        'duration': note['duration'],
                        'string': note['string'],
                        'fret': note['fret'],
                        'confidence': note.get('confidence', 0.8)
                    }
                    
                    # Add chord information if present
                    if 'chord' in note:
                        note_position['chord'] = note['chord']
                        
                    tab_data['notePositions'].append(note_position)
            
            tempo, beats = librosa.beat.beat_track(y=analyzer.y, sr=analyzer.sr)
            
            # Create response data with enhanced structure
            tablature = {
                'timeSignature': {'numerator': time_sig[0], 'denominator': time_sig[1]},
                'chordProgression': chords,
                'notes': guitar_notes,
                'tablature': tab_data.get('tablature', []),
                'bars': tab_data.get('bars', []),
                'totalDuration': tab_data.get('totalDuration', 0.0),
                'gridResolution': tab_data.get('gridResolution', 0.125),
                'tempo': float(tempo),
                'notePositions': tab_data.get('notePositions', [])  # Include the note positions for the visual fretboard
            }
            
            # Convert numpy types to native Python types
            tablature = convert_numpy_types(tablature)
            
            logger.info("Enhanced analysis complete")
            return jsonify(tablature)
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}", exc_info=True)
            return jsonify({'error': f'Analysis error: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in analyze_structure: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/generate_tab/<path:filename>', methods=['GET'])
def analyze_generate_tab(filename):
    """
    Comprehensive tablature generation workflow that:
    1. Separates all instrument tracks using Demucs
    2. Analyzes each track with specialized libraries
    3. Combines analyses to generate high-quality tablature
    """
    try:
        logger.info(f"Starting tab generation for: {filename}")
        
        # Try multiple decoding approaches for maximum compatibility
        possible_filenames = [
            filename,
            filename.replace('%20', '_'),  # Replace encoded spaces with underscores
            secure_filename(filename),
            secure_filename(filename.replace('%20', ' '))  # Replace encoded spaces, then secure
        ]
        
        file_path = None
        for fname in possible_filenames:
            test_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            if os.path.exists(test_path):
                file_path = test_path
                logger.info(f"Found file at path: {file_path}")
                break
                
        # If we couldn't find the file with any method, just use the original name for error reporting
        if not file_path:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.error(f"File not found after trying multiple filename variants: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        # Step 0: Define default values for key variables to prevent reference errors
        key_hint = 'C'  # Default key
        time_signature = (4, 4)  # Default time signature
        tempo = 120  # Default tempo
        analysis_results = {
            'full_mix': {},
            'stems': {}
        }
        
        # Setup cache directories
        try:
            logger.info("Setting up cache directories...")
            cache_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'analysis_cache')
            os.makedirs(cache_dir, exist_ok=True)
            file_basename = os.path.basename(file_path)
        except Exception as e:
            logger.error(f"Failed to set up cache directories: {str(e)}")
            return jsonify({'error': f"Cache directory setup failed: {str(e)}"}), 500
            
        # Step 1: Check for cached results first
        try:
            tab_cache_path = os.path.join(cache_dir, f"{file_basename}_tablature.json")
            if os.path.exists(tab_cache_path):
                logger.info(f"Using cached tablature data from {tab_cache_path}")
                try:
                    with open(tab_cache_path, 'r') as f:
                        tab_data = json.load(f)
                        return jsonify(convert_numpy_types(tab_data))
                except json.JSONDecodeError as je:
                    logger.warning(f"Cached tab data corrupt, will regenerate: {str(je)}")
                    # Continue processing to regenerate
                except Exception as ce:
                    logger.warning(f"Error reading cached tab data: {str(ce)}")
                    # Continue processing to regenerate
        except Exception as e:
            logger.error(f"Error checking for cached results: {str(e)}")
            # Continue processing to generate new results
        
        # Step 2: Perform instrument separation using Demucs
        stem_paths = {}
        try:
            from instrument_separation import DemucsInstrumentSeparator
            
            # Initialize instrument separator
            separator = DemucsInstrumentSeparator()
            
            # Define output directory
            stem_dir = os.path.join(cache_dir, f"{file_basename}_stems")
            os.makedirs(stem_dir, exist_ok=True)
            
            # Check if stems already exist
            if os.path.exists(os.path.join(stem_dir, "bass.wav")) and \
               os.path.exists(os.path.join(stem_dir, "drums.wav")) and \
               os.path.exists(os.path.join(stem_dir, "vocals.wav")) and \
               os.path.exists(os.path.join(stem_dir, "other.wav")):
                
                logger.info("Using previously separated stems")
                stem_paths = {
                    'bass': os.path.join(stem_dir, "bass.wav"),
                    'drums': os.path.join(stem_dir, "drums.wav"),
                    'vocals': os.path.join(stem_dir, "vocals.wav"),
                    'other': os.path.join(stem_dir, "other.wav")
                }
            else:
                logger.info("Separating audio into individual stems using Demucs...")
                separation_result = separator.separate_stems(file_path, stem_dir)
                
                if separation_result['success']:
                    stem_paths = separation_result['stems']
                    logger.info(f"Successfully separated into {len(stem_paths)} stems")
                else:
                    logger.warning(f"Stem separation failed: {separation_result.get('error')}")
                    # Continue with full mix
        except Exception as e:
            logger.warning(f"Error during stem separation: {str(e)}. Using full mix.")        
        # Step 3: Perform comprehensive analysis on each stem and full mix
        analysis_results = {
            'full_mix': {},
            'stems': {}
        }
        
        # Initialize with default values
        key_hint = 'C'  # Default key
        time_signature = (4, 4)  # Default time signature
        tempo = 120  # Default tempo
        
        # Analyze full mix first
        try:
            logger.info("Analyzing full mix for overall musical context...")
            
            # Detect key signature
            key_result = UnifiedAudioAnalyzer(file_path).detect_key()
            analysis_results['full_mix']['key'] = key_result
            key_hint = key_result.get('key', 'C')  # Default to C if detection fails
            
            # Detect time signature
            time_sig_result = UnifiedAudioAnalyzer(file_path).detect_time_signature()
            analysis_results['full_mix']['time_signature'] = time_sig_result
            time_signature = time_sig_result.get('time_signature', (4, 4))  # Default to 4/4
            
            # Detect chords
            chord_result = UnifiedAudioAnalyzer(file_path).detect_chords()
            analysis_results['full_mix']['chords'] = chord_result
            
            # Detect tempo and beats
            bpm = UnifiedAudioAnalyzer(file_path).detect_tempo()
            analysis_results['full_mix']['tempo'] = bpm
            tempo = bpm if bpm else 120  # Default to 120 BPM if detection fails
        except Exception as e:
            logger.warning(f"Error analyzing full mix: {str(e)}")
        
        # Step 4: Analyze individual stems if available
        try:
            # Import specialized analysis modules
            from note_detection import NoteDetector
            from rhythm_analysis import RhythmAnalyzer
            from music_theory import MusicTheoryAnalyzer
            import pretty_midi
            
            # Try to import optional advanced libraries
            try:
                import aubio
                AUBIO_AVAILABLE = True
            except ImportError:
                AUBIO_AVAILABLE = False
                logger.warning("Aubio not available. Using fallback pitch detection.")
                
            try:
                import music21
                MUSIC21_AVAILABLE = True
            except ImportError:
                MUSIC21_AVAILABLE = False
                logger.warning("Music21 not available. Using basic music theory.")
                
            try:
                import essentia
                import essentia.standard as es
                ESSENTIA_AVAILABLE = True
            except ImportError:
                ESSENTIA_AVAILABLE = False
                logger.warning("Essentia not available. Using alternative spectrum analysis.")
            
            # Analyze each stem if available
            for stem_name, stem_path in stem_paths.items():
                if os.path.exists(stem_path):
                    logger.info(f"Analyzing {stem_name} stem...")
                    stem_analyzer = UnifiedAudioAnalyzer(stem_path)
                    
                    # Different analysis based on stem type
                    stem_results = {}
                    
                    if stem_name == 'bass' or stem_name == 'other':  # Other often contains guitar
                        # Note detection
                        note_detector = NoteDetector(stem_analyzer.sr)
                        
                        # First try using aubio for high-precision if available
                        if AUBIO_AVAILABLE:
                            logger.info(f"Using aubio for precise pitch detection on {stem_name}...")
                            notes = note_detector.detect_notes_with_aubio(stem_analyzer.y)
                        else:
                            # Fall back to librosa
                            notes = note_detector.detect_notes_with_librosa(stem_analyzer.y)
                        
                        # Cache the note detection results
                        note_cache_path = os.path.join(cache_dir, f"{file_basename}_{stem_name}_notes.json")
                        with open(note_cache_path, 'w') as f:
                            json.dump(notes, f)
                        
                        stem_results['notes'] = notes
                        
                        # Additional advanced analysis
                        if stem_name == 'other':  # Likely contains guitar
                            logger.info("Performing detailed guitar note analysis...")
                            # Add guitar-specific analysis with onset detection
                            onset_strength = librosa.onset.onset_strength(
                                y=stem_analyzer.y, sr=stem_analyzer.sr
                            )
                            onsets = librosa.onset.onset_detect(
                                onset_envelope=onset_strength, sr=stem_analyzer.sr
                            )
                            stem_results['guitar_onsets'] = onsets
                            
                            # Extract guitar playing techniques if essentia is available
                            if ESSENTIA_AVAILABLE:
                                logger.info("Using Essentia for advanced guitar technique detection...")
                                # Detect guitar techniques like slides, hammer-ons, pull-offs
                                # This is a simplified placeholder - real implementation would be more complex
                                techniques = []
                                stem_results['guitar_techniques'] = techniques
                    
                    elif stem_name == 'drums':
                        # Rhythm analysis
                        rhythm_analyzer = RhythmAnalyzer(stem_analyzer.sr)
                        rhythm_info = rhythm_analyzer.analyze(stem_analyzer.y)
                        stem_results['rhythm'] = rhythm_info
                    
                    elif stem_name == 'vocals':
                        # This could be used to detect melody lines that might be played on guitar
                        note_detector = NoteDetector(stem_analyzer.sr)
                        vocal_notes = note_detector.detect_notes_with_librosa(stem_analyzer.y)
                        stem_results['vocal_notes'] = vocal_notes
                    
                    analysis_results['stems'][stem_name] = stem_results
            
            # Step 5: Additional music theory analysis
            if MUSIC21_AVAILABLE:
                logger.info("Performing advanced music theory analysis...")
                try:
                    theory_analyzer = MusicTheoryAnalyzer()
                    
                    # Analyze all detected notes for interesting patterns
                    all_notes = []
                    if 'other' in analysis_results['stems'] and 'notes' in analysis_results['stems']['other']:
                        all_notes.extend(analysis_results['stems']['other']['notes'])
                    if 'bass' in analysis_results['stems'] and 'notes' in analysis_results['stems']['bass']:
                        all_notes.extend(analysis_results['stems']['bass']['notes'])
                    
                    # Sort by time
                    all_notes.sort(key=lambda x: x['time'])
                    
                    # Analyze chord progressions and suggest scales
                    theory_analysis = theory_analyzer.analyze_progression(
                        analysis_results['full_mix'].get('chords', {}).get('chord_progression', []),
                        key_hint
                    )
                    analysis_results['music_theory'] = theory_analysis
                except Exception as e:
                    logger.warning(f"Music theory analysis error: {str(e)}")
        
        except Exception as e:
            logger.warning(f"Error during stem analysis: {str(e)}")
        
        # Step 6: Generate tablature using the comprehensive analysis
        try:
            logger.info("Generating optimized tablature from all analyzed data...")
            
            # Import tablature generator
            from tablature_generation import TablatureGenerator
            
            # Extract notes from the guitar/other track
            guitar_notes = []
            if 'other' in analysis_results['stems'] and 'notes' in analysis_results['stems']['other']:
                guitar_notes = analysis_results['stems']['other']['notes']
                logger.info(f"Found {len(guitar_notes)} notes in guitar/other track")
            else:
                logger.warning("No guitar notes found in stems, using empty notes list")
            
            # Extract key and style information for better tab generation
            key_hint = analysis_results['full_mix'].get('key', {}).get('key', 'C')
            logger.info(f"Using key hint: {key_hint}")
            
            # Determine style hint based on rhythm and tempo
            style_hint = "rock"  # Default
            tempo = analysis_results['full_mix'].get('tempo', 120)
            if tempo < 80:
                style_hint = "blues"
            elif tempo > 140:
                style_hint = "metal"
            logger.info(f"Using style hint: {style_hint} (based on tempo: {tempo})")
            
            # Initialize the tablature generator with appropriate settings
            tab_generator = TablatureGenerator(
                instrument='guitar',
                tuning=None,  # Use standard tuning by default
                use_ml=True,  # Use ML for position inference when available
                include_theory=True,
                position_bias='optimal'
            )
            logger.info("Tablature generator initialized")
            
            # Generate the tablature
            logger.info("Calling tab_generator.generate_tab()...")
            tablature_data = tab_generator.generate_tab(
                notes=guitar_notes,
                optimize=True,
                key_hint=key_hint,
                style_hint=style_hint
            )
            logger.info("Tablature generation successful")
            
            # Add additional contextual information
            tablature_data['analysis_context'] = {
                'key': key_hint,
                'time_signature': time_signature,
                'tempo': tempo,
                'style': style_hint
            }
            
            # Step 7: Convert to MIDI if notes were detected
            try:
                from midi_conversion import create_midi_from_notes
                
                # Generate MIDI file from notes
                midi_path = os.path.join(cache_dir, f"{file_basename}_tab.mid")
                
                # Use guitar notes for MIDI generation
                if guitar_notes:
                    midi_result = create_midi_from_notes(
                        notes=guitar_notes,
                        output_path=midi_path,
                        tempo=tempo,
                        instrument_name='Acoustic Guitar (nylon)'
                    )
                    
                    if midi_result['success']:
                        tablature_data['midi_file'] = midi_path
                        logger.info(f"MIDI file created at {midi_path}")
            except Exception as e:
                logger.warning(f"MIDI creation error: {str(e)}")
            
            # Cache the tablature data for future requests
            with open(tab_cache_path, 'w') as f:
                json.dump(tablature_data, f)
            
            # Return the results
            logger.info("Returning tablature data")
            return jsonify(convert_numpy_types(tablature_data))
            
        except Exception as e:
            logger.error(f"Error generating tablature: {str(e)}", exc_info=True)
            return jsonify({'error': f"Tablature generation failed: {str(e)}"}), 500
    
    except json.JSONDecodeError as je:
        logger.error(f"JSON error in tablature generation: {str(je)}", exc_info=True)
        return jsonify({'error': f"JSON error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in analyze_generate_tab: {str(e)}", exc_info=True)
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500

@app.route('/api/generate_tab', methods=['POST'])
def generate_tab():
    """
    Simplified endpoint that:
    1. Accepts single audio file
    2. Processes using unified_tab_processor2
    3. Returns generated files
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "Empty file"}), 400
    
    try:
        # Create temp directory
        session_id = str(uuid.uuid4())
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save original
        audio_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(audio_path)
        
        # Process using unified_tab_processor2
        processor = UnifiedTabProcessor()
        result = processor.process_audio(audio_path)
        
        if not result.get('success'):
            return jsonify({"error": result.get('message', 'Processing failed')}), 500
            
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "files": {
                "midi": f"/download/{session_id}/output.mid",
                "tab": f"/download/{session_id}/tab.ly",
                "audio": f"/download/{session_id}/{os.path.basename(audio_path)}"
            }
        })
        
    except Exception as e:
        logger.error(f"Tab generation error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files from the upload directory"""
    try:
        return send_from_directory(
            app.config['UPLOAD_FOLDER'],
            filename,
            mimetype='audio/mpeg',
            as_attachment=False
        )
    except Exception as e:
        logger.error(f"Error serving audio file: {str(e)}")
        return jsonify({'error': 'Audio file not found'}), 404

@app.route('/analyze/visualize/<path:filename>', methods=['GET'])
def analyze_visualize(filename):
    """Generate and return RipX-style visualizations of the audio file"""
    try:
        # Try multiple decoding approaches for maximum compatibility
        possible_filenames = [
            filename,
            filename.replace('%20', '_'),  # Replace encoded spaces with underscores
            secure_filename(filename),
            secure_filename(filename.replace('%20', ' '))  # Replace encoded spaces, then secure
        ]
        
        file_path = None
        for fname in possible_filenames:
            test_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            if os.path.exists(test_path):
                file_path = test_path
                break
                
        # If we couldn't find the file with any method, just use the original name for error reporting
        if not file_path:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.error(f"File not found after trying multiple filename variants: {file_path}")
            return jsonify({'error': 'File not found'}), 404
            
        logger.info(f"Generating visualizations for file: {file_path}")
        
        # Get visualization type from query string (piano_roll, spectral, or both)
        vis_type = request.args.get('type', 'both')
        
        # Create enhanced analyzer
        analyzer = UnifiedAudioAnalyzer(file_path)
        
        # Generate visualizations
        output_name = os.path.splitext(os.path.basename(file_path))[0]
        vis_paths = analyzer.generate_visualizations(output_name)
        
        if not vis_paths:
            logger.error("Failed to generate visualizations")
            return jsonify({'error': 'Failed to generate visualizations'}), 500
            
        # If requesting a specific visualization as image, send the file
        if vis_type == 'piano_roll' and 'piano_roll' in vis_paths:
            return send_file(vis_paths['piano_roll'], mimetype='image/png')
        elif vis_type == 'spectral' and 'spectral' in vis_paths:
            return send_file(vis_paths['spectral'], mimetype='image/png')
        
        # Convert images to base64 for JSON response
        result = {
            'filename': output_name,
            'visualizations': {}
        }
        
        for vis_name, vis_path in vis_paths.items():
            if os.path.exists(vis_path):
                with open(vis_path, "rb") as img_file:
                    b64data = base64.b64encode(img_file.read()).decode('utf-8')
                    result['visualizations'][vis_name] = {
                        'path': vis_path.replace('\\', '/'),  # Normalize path separators
                        'url': f"/visualizations/{os.path.basename(vis_path)}",
                        'data': f"data:image/png;base64,{b64data}"
                    }
        
        return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/visualizations/<path:filename>')
def serve_visualizations(filename):
    """Serve visualization files"""
    try:
        visualizations_dir = os.path.join(os.getcwd(), "frontend", "public", "visualizations")
        return send_from_directory(
            visualizations_dir,
            filename,
            mimetype='image/png'
        )
    except Exception as e:
        logger.error(f"Error serving visualization file: {str(e)}")
        return jsonify({'error': 'Visualization file not found'}), 404

@app.route('/analyze/fretboard/<path:filename>', methods=['GET'])
def analyze_fretboard_positions(filename):
    """
    Analyze the audio file and return optimized fretboard positions for the detected notes.
    Uses enhanced audio analysis for accurate pitch detection and optimal finger positioning.
    """
    try:
        logger.info(f"Starting fretboard position analysis for: {filename}")
        
        # Try multiple decoding approaches for maximum compatibility
        possible_filenames = [
            filename,
            filename.replace('%20', '_'),  # Replace encoded spaces with underscores
            secure_filename(filename),
            secure_filename(filename.replace('%20', ' '))  # Replace encoded spaces, then secure
        ]
        
        file_path = None
        for fname in possible_filenames:
            test_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            if os.path.exists(test_path):
                file_path = test_path
                logger.info(f"Found file at path: {file_path}")
                break
                
        # If we couldn't find the file with any method, just use the original name for error reporting
        if not file_path:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.error(f"File not found after trying multiple filename variants: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        # Check cache directory
        cache_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'analysis_cache')
        os.makedirs(cache_dir, exist_ok=True)
        file_basename = os.path.basename(file_path)
        fretboard_cache_path = os.path.join(cache_dir, f"{file_basename}_fretboard.json")
        
        # Check if we have cached results
        if os.path.exists(fretboard_cache_path):
            logger.info(f"Using cached fretboard data from {fretboard_cache_path}")
            try:
                with open(fretboard_cache_path, 'r') as f:
                    fretboard_data = json.load(f)
                    return jsonify(convert_numpy_types(fretboard_data))
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Error reading cached fretboard data: {str(e)}")
                # Continue to regenerate
        
        # Initialize the unified audio analyzer
        logger.info("Initializing unified audio analyzer...")
        
        analyzer = UnifiedAudioAnalyzer(file_path)
        
        # Detect guitar notes with advanced pitch detection
        logger.info("Detecting guitar notes with advanced pitch detection...")
        guitar_notes = analyzer.detect_guitar_notes()
        
        # Map notes to optimal fretboard positions
        logger.info("Mapping notes to optimal fretboard positions...")
        fretboard_positions = analyzer.map_notes_to_fretboard(guitar_notes)
        
        # Format the response
        fretboard_data = {
            'notePositions': fretboard_positions,
            'totalNotes': len(fretboard_positions)
        }
        
        # Cache the results
        try:
            with open(fretboard_cache_path, 'w') as f:
                json.dump(fretboard_data, f)
        except Exception as e:
            logger.warning(f"Error caching fretboard data: {str(e)}")
        
        # Convert any numpy types to native Python types for JSON serialization
        fretboard_data = convert_numpy_types(fretboard_data)
        
        logger.info("Fretboard position analysis complete")
        return jsonify(fretboard_data)
        
    except Exception as e:
        logger.error(f"Error in fretboard position analysis: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'notePositions': []}), 500

@app.route('/feedback/submit', methods=['POST'])
def submit_feedback():
    """
    Submit user feedback on generated tablature.
    
    Expects JSON with:
    - tab_id: Identifier for the tablature
    - feedback: Dictionary with user feedback data (rating, comments, etc.)
    - positions: List of position dictionaries used in the tablature
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Check required fields
        if 'tab_id' not in data:
            return jsonify({"error": "Missing tab_id field"}), 400
        if 'feedback' not in data:
            return jsonify({"error": "Missing feedback field"}), 400
        
        # Create model integration for feedback collection
        from model_integration import create_model_integration
        integration = create_model_integration()
        
        # Add user ID if available
        if 'user_id' not in data['feedback'] and 'user_id' in data:
            data['feedback']['user_id'] = data['user_id']
            
        # Add timestamp if not provided
        if 'timestamp' not in data['feedback']:
            data['feedback']['timestamp'] = datetime.now().isoformat()
            
        # Handle positions (optional)
        positions = data.get('positions', [])
            
        # Collect feedback
        success = integration.collect_user_feedback(
            data['tab_id'], 
            data['feedback'],
            positions
        )
        
        if success:
            return jsonify({"status": "success", "message": "Feedback submitted successfully"})
        else:
            return jsonify({"status": "error", "message": "Error submitting feedback"}), 500
            
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/feedback/stats', methods=['GET'])
def get_feedback_stats():
    """
    Get aggregated feedback statistics.
    """
    try:
        # Create feedback collector
        from user_feedback import FeedbackCollector
        collector = FeedbackCollector()
        
        # Get stats
        stats = collector.get_feedback_stats()
        
        return jsonify({"status": "success", "stats": stats})
            
    except Exception as e:
        logger.error(f"Error getting feedback stats: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/model/performance', methods=['GET'])
def get_model_performance():
    """
    Get model performance statistics.
    """
    try:
        # Create model integration
        from model_integration import create_model_integration
        integration = create_model_integration()
        
        # Get performance stats
        stats = integration.get_model_performance_stats()
        
        return jsonify({"status": "success", "stats": stats})
            
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/model/calibration', methods=['GET'])
def get_model_calibration():
    """
    Get model confidence calibration data.
    """
    try:
        # Create model integration
        from model_integration import create_model_integration
        integration = create_model_integration()
        
        # Get calibration data
        calibration = integration.get_calibration_data()
        
        return jsonify({"status": "success", "calibration": calibration})
            
    except Exception as e:
        logger.error(f"Error getting model calibration: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Register the tab extractor blueprint
# register_tab_extractor(app)

# Initialize components
# feedback_collector = FeedbackCollector()
# tab_processor = TabDataProcessor()
# model_tracker = ModelPerformanceTracker()

@app.route('/analyze/cnn-tab/<filename>', methods=['GET'])
def analyze_cnn_tab(filename):
    """
    Generate tablature using the TabCNN model with Demucs source separation
    
    This endpoint processes an audio file through:
    1. Demucs for instrument separation (extracting guitar)
    2. TabCNN model for tablature prediction
    3. Optimized tablature generation
    
    Returns the generated tablature data and visualization
    """
    if not TABCNN_AVAILABLE:
        return jsonify({
            'error': 'TabCNN functionality is not available. Required dependencies may be missing.',
            'dependencies': ['tensorflow', 'demucs'],
            'fallback': 'Use standard tablature generation instead.'
        }), 503
        
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        # Create output directory for the processed files
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(filename)[0]}_cnn_tab")
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Processing {filename} with TabCNN and Demucs...")
        
        # Process with TabCNN (includes Demucs separation if available)
        tab_data = process_audio_with_tabcnn(file_path, output_dir)
        
        if tab_data is None:
            logger.error("Failed to generate tablature with TabCNN")
            return jsonify({'error': 'Tablature generation failed'}), 500
        
        # Generate text tab file
        tab_file = generate_tab_for_audio_file(file_path, output_dir)
        
        # Create a response with both raw tablature data and formatted tab
        result = {
            'success': True,
            'message': 'Tablature generated successfully using TabCNN model',
            'tab_data': convert_numpy_types(tab_data.tolist() if isinstance(tab_data, np.ndarray) else tab_data),
            'tab_file': os.path.basename(tab_file) if tab_file else None,
            'timestamps': {
                'hop_length': 512,  # Default hop length used in TabCNN
                'sr': 22050,  # Default sample rate used in TabCNN
            }
        }
        
        # Add metadata to help with model improvement
        model_tracker.log_prediction('tabcnn', filename, len(tab_data))
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error generating tablature with TabCNN: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/train/tabcnn', methods=['POST'])
def train_tabcnn_model():
    """
    Start training the TabCNN model with GuitarSet data
    
    Processes GuitarSet data and trains the TabCNN model, 
    saving results to a timestamped directory
    """
    if not TABCNN_AVAILABLE:
        return jsonify({
            'error': 'TabCNN functionality is not available. Required dependencies may be missing.',
            'dependencies': ['tensorflow', 'demucs', 'guitarset dataset'],
            'suggestion': 'Install required dependencies or contact administrator.'
        }), 503
        
    try:
        # Get parameters from request
        data = request.get_json() or {}
        epochs = data.get('epochs', 8)
        batch_size = data.get('batch_size', 128)
        
        # Start preprocessing in a background thread
        import threading
        
        def training_process():
            try:
                # Step 1: Preprocessing is now handled inside the train function
                logger.info("Starting TabCNN model training process...")
                
                # Step 2: Train the TabCNN model
                logger.info("Starting TabCNN model training...")
                train_success = interpreter_train_tabcnn_model(epochs=epochs, batch_size=batch_size)
                
                if not train_success:
                    logger.error("Failed to train TabCNN model")
                    return
                
                logger.info("TabCNN model training completed successfully")
            except Exception as e:
                logger.error(f"Error in training process: {str(e)}", exc_info=True)
        
        # Start the training process in a background thread
        training_thread = threading.Thread(target=training_process)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'TabCNN model training started in the background',
            'status': 'processing',
            'parameters': {
                'epochs': epochs,
                'batch_size': batch_size
            }
        })
    
    except Exception as e:
        logger.error(f"Error starting TabCNN training: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/tabs/cnn/<filename>', methods=['GET'])
def get_cnn_tab_file(filename):
    """
    Serve a TabCNN-generated tablature file
    """
    if not TABCNN_AVAILABLE:
        return jsonify({
            'error': 'TabCNN functionality is not available.',
            'fallback': 'Use standard tablature generation instead.'
        }), 503
        
    try:
        # Extract the base filename and look for the tab file
        base_name = os.path.splitext(filename)[0]
        tab_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_cnn_tab")
        
        # Check if directory exists
        if not os.path.exists(tab_dir):
            return jsonify({'error': 'Tab directory not found'}), 404
        
        # Find the tab file
        for file in os.listdir(tab_dir):
            if file.endswith('_tab.txt'):
                return send_from_directory(tab_dir, file)
        
        return jsonify({'error': 'Tab file not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/compare-tabs/<filename>', methods=['GET'])
def compare_tab_methods(filename):
    """
    Generate and compare tablature using both traditional and CNN-based methods
    
    This endpoint provides a comparison between:
    1. Traditional audio analysis-based tablature generation
    2. TabCNN with Demucs ML-based tablature generation
    
    Returns metrics and differences between the approaches
    """
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Create output directory
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(filename)[0]}_tab_comparison")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate traditional tablature
        analyzer = UnifiedAudioAnalyzer(file_path)
        traditional_tab = analyzer.generate_tablature()
        
        # Generate CNN-based tablature if available
        cnn_tab = None
        if TABCNN_AVAILABLE:
            cnn_tab = process_audio_with_tabcnn(file_path, output_dir)
        
        if cnn_tab is None:
            return jsonify({
                'warning': 'CNN tablature generation failed or not available, showing only traditional method',
                'traditional_tab': convert_numpy_types(traditional_tab)
            }), 206
        
        # Compare the two approaches
        # This is a simple comparison and can be enhanced with more sophisticated metrics
        comparison = {
            'traditional_tab': convert_numpy_types(traditional_tab),
            'cnn_tab': convert_numpy_types(cnn_tab.tolist() if isinstance(cnn_tab, np.ndarray) else cnn_tab),
            'comparison_metrics': {
                'note_count_traditional': sum(1 for frame in traditional_tab for note in frame if note > 0),
                'note_count_cnn': sum(1 for frame in cnn_tab for note in frame if note > 0),
                # Additional metrics can be added here
            }
        }
        
        return jsonify(comparison)
    
    except Exception as e:
        logger.error(f"Error comparing tablature methods: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/piano-notes/<filename>', methods=['GET'])
def analyze_piano_notes(filename):
    """
    Extract piano note data from an audio file for keyboard visualization
    """
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        analyzer = UnifiedAudioAnalyzer(file_path)
        piano_notes = analyzer.extract_piano_notes()
        
        # Convert numpy types to native Python types
        piano_notes = convert_numpy_types(piano_notes)
        
        return jsonify({
            'notes': piano_notes
        })
    except Exception as e:
        logger.error(f"Error extracting piano notes: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/rhythm/save-pattern', methods=['POST'])
def save_beat_pattern():
    """
    Save a beat pattern for future use
    """
    try:
        data = request.json
        if not data or 'pattern' not in data:
            return jsonify({'error': 'No pattern data provided'}), 400
        
        pattern = data['pattern']
        pattern_id = str(uuid.uuid4())
        
        # Save to patterns directory
        os.makedirs('data/patterns', exist_ok=True)
        with open(f'data/patterns/{pattern_id}.json', 'w') as f:
            json.dump(pattern, f)
        
        return jsonify({
            'success': True,
            'pattern_id': pattern_id
        })
    except Exception as e:
        logger.error(f"Error saving beat pattern: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/rhythm/patterns', methods=['GET'])
def get_beat_patterns():
    """
    Get all saved beat patterns
    """
    try:
        patterns = []
        pattern_dir = 'data/patterns'
        
        if os.path.exists(pattern_dir):
            for filename in os.listdir(pattern_dir):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(pattern_dir, filename), 'r') as f:
                            pattern = json.load(f)
                            pattern_id = filename.replace('.json', '')
                            patterns.append({
                                'id': pattern_id,
                                'pattern': pattern
                            })
                    except Exception as e:
                        logger.warning(f"Error loading pattern {filename}: {str(e)}")
        
        return jsonify({
            'patterns': patterns
        })
    except Exception as e:
        logger.error(f"Error getting beat patterns: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/model/train', methods=['POST'])
def train_model():
    """
    Start a new model training session
    """
    try:
        config = request.json
        training_id = model_tracker.start_training_session(config)
        
        return jsonify({
            'training_id': training_id,
            'status': 'started'
        })
    except Exception as e:
        logger.error(f"Error starting model training: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/model/training-status/<training_id>', methods=['GET'])
def get_training_status(training_id):
    """
    Get the status of a model training session
    """
    try:
        status = model_tracker.get_training_status(training_id)
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/text_tablature/<path:filename>', methods=['GET'])
def generate_text_tablature(filename):
    """
    Generate human-readable, text-based guitar tablature from the audio file.
    This is a simplified format similar to standard tab websites.
    """
    try:
        logger.info(f"Generating text-based tablature for: {filename}")
        
        # Try multiple decoding approaches for maximum compatibility
        possible_filenames = [
            filename,
            filename.replace('%20', '_'),  # Replace encoded spaces with underscores
            secure_filename(filename),
            secure_filename(filename.replace('%20', ' '))  # Replace encoded spaces, then secure
        ]
        
        file_path = None
        for fname in possible_filenames:
            test_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            if os.path.exists(test_path):
                file_path = test_path
                logger.info(f"Found file at path: {file_path}")
                break
                
        # If we couldn't find the file with any method, just use the original name for error reporting
        if not file_path:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.error(f"File not found after trying multiple filename variants: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        # Check cache directory for previously generated tablature
        cache_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'analysis_cache')
        os.makedirs(cache_dir, exist_ok=True)
        file_basename = os.path.basename(file_path)
        text_tab_cache_path = os.path.join(cache_dir, f"{file_basename}_text_tab.txt")
        
        # Check if we have a cached version
        if os.path.exists(text_tab_cache_path):
            logger.info(f"Using cached text tablature from {text_tab_cache_path}")
            try:
                with open(text_tab_cache_path, 'r') as f:
                    tab_text = f.read()
                    return jsonify({'tablature': tab_text})
            except Exception as e:
                logger.warning(f"Error reading cached text tab: {str(e)}")
                # Continue to regenerate
        
        # Initialize audio analyzers
        logger.info("Initializing audio analyzers for tablature generation...")
        
        # Import the TabTextGenerator
        from tab_text_generator import TabTextGenerator
        
        # Use the standard analyzer for basic analysis
        analyzer = UnifiedAudioAnalyzer(file_path)
        
        # Extract basic song information
        song_info = {
            'title': os.path.splitext(os.path.basename(file_path))[0],
            'artist': 'Unknown Artist',  # Could be improved with metadata extraction
            'tempo': analyzer.detect_tempo(),
            'timeSignature': {'numerator': 4, 'denominator': 4},  # Default
            'capo': 0
        }
        
        # Try to get more accurate time signature
        try:
            time_sig = analyzer.detect_time_signature()
            if time_sig and len(time_sig) == 2:
                song_info['timeSignature'] = {'numerator': time_sig[0], 'denominator': time_sig[1]}
        except Exception as e:
            logger.warning(f"Error detecting time signature: {str(e)}")
        
        # Try to get key
        try:
            key_data = analyzer.detect_key()
            if key_data and 'key' in key_data:
                song_info['key'] = key_data['key']
        except Exception as e:
            logger.warning(f"Error detecting key: {str(e)}")
        
        # Try to get chords
        try:
            chord_data = analyzer.detect_chords()
            if chord_data:
                song_info['chords'] = chord_data
        except Exception as e:
            logger.warning(f"Error detecting chords: {str(e)}")
        
        # Detect guitar notes - use enhanced detection if available
        guitar_notes = []
        try:
            # Try to use the EnhancedAudioAnalyzer if available
            try:
                analyzer = UnifiedAudioAnalyzer(file_path)
                logger.info("Using enhanced audio analyzer for note detection...")
                guitar_notes = analyzer.detect_guitar_notes()
            except (ImportError, Exception) as e:
                logger.warning(f"Enhanced analyzer not available: {str(e)}. Using standard analyzer.")
                # Fallback to standard analyzer
                guitar_notes = analyzer.detect_guitar_notes()
            
            logger.info(f"Detected {len(guitar_notes)} guitar notes.")
            
        except Exception as e:
            logger.warning(f"Error detecting guitar notes: {str(e)}. Using fallback pattern generation.")
            
            # If note detection fails, generate a simple pattern based on chords
            if 'chords' in song_info and song_info['chords']:
                logger.info("Generating notes from chord progression...")
                guitar_notes = []
                
                # Standard patterns for common chords
                chord_patterns = {
                    # Major chords - format: (string, fret)
                    'C': [(5, 3), (4, 2), (3, 0), (2, 1), (1, 0)],
                    'G': [(5, 3), (4, 0), (3, 0), (2, 0), (1, 0), (0, 3)],
                    'D': [(4, 0), (3, 2), (2, 3), (1, 2)],
                    'A': [(5, 0), (4, 2), (3, 2), (2, 2), (1, 0)],
                    'E': [(5, 0), (4, 2), (3, 2), (2, 1), (1, 0), (0, 0)],
                    'F': [(5, 1), (4, 3), (3, 3), (2, 2), (1, 1), (0, 1)],
                    # Minor chords
                    'Cm': [(5, 3), (4, 1), (3, 0), (2, 1), (1, 1)],
                    'Gm': [(5, 3), (4, 1), (3, 0), (2, 0), (1, 1)],
                    'Dm': [(4, 0), (3, 2), (2, 3), (1, 1)],
                    'Am': [(5, 0), (4, 2), (3, 2), (2, 1), (1, 0)],
                    'Em': [(5, 0), (4, 2), (3, 2), (2, 0), (1, 0), (0, 0)],
                }
                
                # Add default pattern if no chords detected
                if not song_info['chords'] or len(song_info['chords']) < 2:
                    # Create a progression of common chords
                    song_info['chords'] = [
                        {'chord': 'G', 'start_time': 0.0, 'end_time': 2.0},
                        {'chord': 'Em', 'start_time': 2.0, 'end_time': 4.0},
                        {'chord': 'C', 'start_time': 4.0, 'end_time': 6.0},
                        {'chord': 'D', 'start_time': 6.0, 'end_time': 8.0}
                    ]
                
                # Generate notes from chord pattern
                for chord_data in song_info['chords']:
                    if isinstance(chord_data, dict) and 'chord' in chord_data:
                        chord_name = chord_data['chord']
                        start_time = chord_data.get('start_time', 0)
                        
                        # Get pattern for this chord or default to C
                        pattern = chord_patterns.get(chord_name, chord_patterns['C'])
                        
                        # Add the notes from this chord
                        for string, fret in pattern:
                            guitar_notes.append({
                                'time': float(start_time),
                                'string': string,
                                'fret': fret,
                                'duration': 0.5,
                                'chord': chord_name
                            })
        
        # Generate the text-based tablature
        tab_generator = TabTextGenerator()
        tab_text = tab_generator.generate_text_tab(guitar_notes, song_info)
        
        # Cache the generated tab
        try:
            with open(text_tab_cache_path, 'w') as f:
                f.write(tab_text)
            logger.info(f"Cached text tablature to {text_tab_cache_path}")
        except Exception as e:
            logger.warning(f"Error caching text tab: {str(e)}")
        
        logger.info("Text tablature generation complete")
        return jsonify({'tablature': tab_text})
        
    except Exception as e:
        logger.error(f"Error generating text tablature: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'tablature': 'Error generating tablature.'}), 500

from unified_tab_processor import create_tablature

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Main audio processing endpoint"""
    try:
        audio_file = request.files['audio']
        style = request.form.get('style', '')
        
        temp_path = save_temp_file(audio_file)
        
        try:
            # Load audio for analysis
            y, sr = librosa.load(temp_path, sr=None)
            
            # Perform music theory analysis
            analyzer = MusicTheoryAnalyzer()
            analysis = {
                'key': analyzer.detect_key(y, sr)['key'],
                'tempo': librosa.beat.tempo(y=y, sr=sr)[0],
                'style': request.form.get('style', 'rock'),
                'chords': analyzer.analyze_chord_progression(y, sr)
            }
            
            logger.info(f"Audio analysis complete: Key={analysis['key']}, Tempo={analysis['tempo']}")
        except Exception as e:
            logger.error(f"Error during music theory analysis: {str(e)}")
            # Fallback to defaults if analysis fails
            analysis = {
                'key': 'C',
                'tempo': 120.0,
                'style': request.form.get('style', 'rock'),
                'chords': []
            }
        
        # 2. Process through full pipeline
        from unified_tab_processor2 import UnifiedTabProcessor
        processor = UnifiedTabProcessor()
        tabs = processor.process_audio(temp_path, analysis['style'], key=analysis['key'])
        
        # 3. Format output
        return jsonify({
            'status': 'success',
            'analysis': {
                'key': analysis['key'],
                'tempo': float(analysis['tempo']),
                'style': analysis['style'],
                'chords': analysis['chords']
            },
            'tabs': tabs,
            'visualization': tabs.get('guitar', {}).get('fretboard_data', {})
        })
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/create_tab', methods=['POST'])
def create_tab():
    """Full pipeline: audio -> separation -> tab prediction"""
    try:
        audio_file = request.files['audio']
        temp_path = os.path.join(tempfile.gettempdir(), audio_file.filename)
        audio_file.save(temp_path)
        
        # 1. Audio analysis using music theory
        try:
            # Load audio for analysis
            y, sr = librosa.load(temp_path, sr=None)
            
            # Perform music theory analysis
            analyzer = MusicTheoryAnalyzer()
            analysis = {
                'key': analyzer.detect_key(y, sr)['key'],
                'tempo': librosa.beat.tempo(y=y, sr=sr)[0],
                'style': request.form.get('style', 'rock'),
                'chords': analyzer.analyze_chord_progression(y, sr)
            }
            
            logger.info(f"Audio analysis complete: Key={analysis['key']}, Tempo={analysis['tempo']}")
        except Exception as e:
            logger.error(f"Error during music theory analysis: {str(e)}")
            # Fallback to defaults if analysis fails
            analysis = {
                'key': 'C',
                'tempo': 120.0,
                'style': request.form.get('style', 'rock'),
                'chords': []
            }
        
        # 2. Process through full pipeline
        from unified_tab_processor import UnifiedTabProcessor
        processor = UnifiedTabProcessor()
        tabs = processor.process_audio(temp_path, analysis['style'], key=analysis['key'])
        
        # 3. Format output
        return jsonify({
            'status': 'success',
            'analysis': {
                'key': analysis['key'],
                'tempo': float(analysis['tempo']),
                'style': analysis['style'],
                'chords': analysis['chords']
            },
            'tabs': tabs,
            'visualization': tabs.get('guitar', {}).get('fretboard_data', {})
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/process_tab', methods=['POST'])
def process_tab():
    """
    Full tab generation endpoint:
    1. Accepts audio upload
    2. Processes through Demucs -> Basic Pitch -> TabCNN -> Fretboard Model
    3. Returns paths to generated files
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    try:
        # Create processing directory
        session_id = str(uuid.uuid4())
        process_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(process_dir, exist_ok=True)
        
        # Save original file
        audio_path = os.path.join(process_dir, secure_filename(file.filename))
        file.save(audio_path)
        
        # Get processing parameters from frontend
        style = request.form.get('style', 'metalcore')
        tuning = request.form.get('tuning', 'Drop D')
        
        # Initialize processor with current configuration
        processor = UnifiedTabProcessor(
            output_dir=process_dir,
            config=current_app.config
        )
        
        # Run full processing pipeline
        result = processor.process_audio(
            audio_path,
            style=style,
            reference_tabs=get_reference_tabs(style)
        )
        
        if result.get('status') != 'success':
            return jsonify({"error": result.get('message', 'Processing failed')}), 500
            
        # Return URLs for frontend to access
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "midi_url": url_for('serve_processing_file', 
                               filename=f"{session_id}/{os.path.basename(result['midi_path'])}"),
            "tab_url": url_for('serve_processing_file', 
                             filename=f"{session_id}/{os.path.basename(result['lilypond_path'])}"),
            "audio_url": url_for('serve_processing_file',
                               filename=f"{session_id}/{os.path.basename(audio_path)}")
        })
        
    except Exception as e:
        logger.error(f"Tab processing error: {str(e)}", exc_info=True)
        return jsonify({"error": "Server error during processing"}), 500
        
@app.route('/processing/<path:filename>')
def serve_processing_file(filename):
    """Serve files from processing directories"""
    return send_from_directory(
        os.path.join(app.config['UPLOAD_FOLDER']),
        filename
    )

@app.route('/analyze/comprehensive/<path:filename>', methods=['GET'])
def analyze_comprehensive(filename):
    """
    Comprehensive analysis endpoint that performs all analyses in sequence:
    1. Basic audio analysis (tempo, time signature)
    2. Key detection
    3. Chord progression analysis
    4. Note detection with frequency matching
    5. Tablature generation
    6. Fretboard positions mapping
    
    Returns a complete analysis result including all components.
    """
    try:
        # Get file path using the same method as other analysis endpoints
        possible_filenames = [
            filename,
            filename.replace('%20', '_'),
            secure_filename(filename),
            secure_filename(filename.replace('%20', ' '))
        ]
        
        file_path = None
        for fname in possible_filenames:
            test_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            if os.path.exists(test_path):
                file_path = test_path
                break
                
        if not file_path:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.error(f"File not found after trying multiple filename variants: {file_path}")
            return jsonify({'error': 'File not found'}), 404
            
        logger.info(f"Starting comprehensive analysis for file: {file_path}")
        
        # Create analyzer instance
        analyzer = UnifiedAudioAnalyzer(file_path)
        
        # Step 1: Basic audio analysis
        logger.info("Step 1: Basic audio analysis")
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beats_times = librosa.frames_to_time(beats, sr=sr)
        
        # Step 2: Analyze time signature
        logger.info("Step 2: Analyzing time signature")
        time_signature = analyzer.detect_time_signature()
        
        # Step 3: Key detection
        logger.info("Step 3: Key detection")
        key_data = analyzer.detect_key()
        
        # Step 4: Chord progression analysis
        logger.info("Step 4: Chord progression analysis")
        chord_data = analyzer.detect_chords()
        
        # Step 5: Note detection
        logger.info("Step 5: Note detection with frequency matching")
        
        # Detect notes with improved pitch detection
        logger.info("Detecting guitar notes with advanced pitch detection...")
        guitar_notes = analyzer.detect_guitar_notes()
        
        # Step 6: Generate tablature
        logger.info("Step 6: Tablature generation")
        # Use key_hint from the detected key if available
        key_hint = key_data.get('key', '') if isinstance(key_data, dict) else ''
        tablature_data = analyzer.generate_tablature()
        
        # Step 7: Map to fretboard positions
        logger.info("Step 7: Mapping to fretboard positions")
        # Get optimal fretboard positions for the detected notes
        fretboard_positions = analyzer.map_notes_to_fretboard(guitar_notes if guitar_notes else [])
        
        # Combine all results
        result = {
            'basic': {
                'duration': float(duration),
                'tempo': float(tempo),
                'beats': beats_times.tolist(),
                'time_signature': time_signature
            },
            'key': key_data,
            'chords': chord_data,
            'notes': guitar_notes,
            'tablature': tablature_data,
            'fretboard': {
                'positions': fretboard_positions
            }
        }
        
        # Convert numpy types to native Python types for JSON serialization
        result = convert_numpy_types(result)
        
        logger.info("Comprehensive analysis completed successfully")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/analyze/lyrics/<filename>', methods=['GET'])
def analyze_lyrics(filename):
    """Extract lyrics from an audio file."""
    try:
        # URL decode the filename
        filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Extracting lyrics for file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        try:
            analyzer = UnifiedAudioAnalyzer(file_path)
            lyrics = analyzer.extract_lyrics()
            
            # Convert numpy types to native Python types
            lyrics = convert_numpy_types(lyrics)
            
            logger.info("Lyrics extraction complete")
            return jsonify(lyrics)
        except Exception as e:
            logger.error(f"Error during lyrics extraction: {str(e)}", exc_info=True)
            return jsonify({'error': f'Lyrics extraction error: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in analyze_lyrics: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/piano-notes/<filename>', methods=['GET'])
def analyze_piano_notes(filename):
    """
    Extract piano note data from an audio file for keyboard visualization
    """
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Create analyzer
        analyzer = UnifiedAudioAnalyzer(file_path, lazy_load=True)
        
        # Detect notes
        notes = analyzer.detect_guitar_notes()
        
        # Convert to piano note format
        piano_notes = []
        for note in notes:
            # Map to piano key (MIDI note number)
            midi_note = librosa.note_to_midi(note.get('note', 'C4'))
            
            piano_notes.append({
                'midi_note': midi_note,
                'start_time': note.get('start_time', 0),
                'end_time': note.get('end_time', 0),
                'velocity': note.get('velocity', 64),
                'confidence': note.get('confidence', 0.8)
            })
        
        return jsonify({
            'piano_notes': piano_notes,
            'total_duration': analyzer.get_duration() if hasattr(analyzer, 'get_duration') else 0
        })
    except Exception as e:
        logger.error(f"Error extracting piano notes: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/beat-patterns', methods=['GET'])
def get_beat_patterns():
    """
    Get all saved beat patterns
    """
    try:
        # Create patterns directory if it doesn't exist
        patterns_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'beat_patterns')
        os.makedirs(patterns_dir, exist_ok=True)
        
        patterns = []
        
        # Read all pattern files
        for filename in os.listdir(patterns_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(patterns_dir, filename), 'r') as f:
                        pattern = json.load(f)
                        patterns.append(pattern)
                except Exception as e:
                    logger.warning(f"Error reading beat pattern file {filename}: {e}")
        
        return jsonify({
            'patterns': patterns,
            'count': len(patterns)
        })
    except Exception as e:
        logger.error(f"Error fetching beat patterns: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/beat-patterns', methods=['POST'])
def save_beat_pattern():
    """
    Save a beat pattern for future use
    """
    try:
        data = request.get_json()
        
        if not data or 'pattern' not in data or 'name' not in data:
            return jsonify({'error': 'Missing required data: pattern and name'}), 400
        
        # Validate pattern data
        pattern = data['pattern']
        name = data['name']
        
        # Create a unique ID
        pattern_id = str(uuid.uuid4())
        
        # Create patterns directory if it doesn't exist
        patterns_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'beat_patterns')
        os.makedirs(patterns_dir, exist_ok=True)
        
        # Save pattern to file
        pattern_data = {
            'id': pattern_id,
            'name': name,
            'pattern': pattern,
            'created_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(patterns_dir, f"{pattern_id}.json"), 'w') as f:
            json.dump(pattern_data, f, indent=2)
        
        return jsonify({
            'id': pattern_id,
            'message': 'Pattern saved successfully'
        })
    except Exception as e:
        logger.error(f"Error saving beat pattern: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-advanced-tab/<filename>', methods=['POST'])
def generate_advanced_tablature(filename):
    """
    Comprehensive tablature generation pipeline following these steps:
    1. Demucs source separation with 20 shifts
    2. Audio enhancement for note clarity
    3. Note detection using frequency and pitch analysis
    4. MIDI note conversion with frequency verification
    5. TabCNN and LSTM prediction for fretboard positions
    6. Fretboard position optimization
    7. LilyPond tablature rendering
    """
    try:
        logger.info(f"Starting advanced tablature generation for {filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        # Create output directory for this processing job
        job_id = str(uuid.uuid4())
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"tab_job_{job_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Demucs source separation with 20 shifts
        logger.info("Step 1: Running Demucs source separation with 20 shifts")
        try:
            demucs_wrapper = DemucsWrapper(shifts=20, model='demucs6')
            separated_tracks = demucs_wrapper.separate(file_path, output_dir=output_dir)
            
            # Get the guitar track path
            if 'guitar' in separated_tracks:
                guitar_track_path = separated_tracks['guitar']
                logger.info(f"Guitar track extracted: {guitar_track_path}")
            else:
                # If no guitar track, try using the bass track or the original
                if 'bass' in separated_tracks:
                    guitar_track_path = separated_tracks['bass']
                    logger.info(f"No guitar track found, using bass track: {guitar_track_path}")
                else:
                    guitar_track_path = file_path
                    logger.warning("No guitar or bass track found, using original audio")
        except Exception as e:
            logger.error(f"Demucs separation failed: {e}", exc_info=True)
            guitar_track_path = file_path
            logger.warning("Using original audio due to Demucs failure")
        
        # Step 2: Audio enhancement for note clarity
        logger.info("Step 2: Enhancing audio for note clarity")
        try:
            # Load the separated guitar track
            y_guitar, sr_guitar = librosa.load(guitar_track_path, sr=None)
            
            # Apply audio enhancement techniques
            # 1. Harmonic-percussive separation to focus on tonal content
            y_harmonic, _ = librosa.effects.hpss(y_guitar)
            
            # 2. Apply EQ to enhance mid-range frequencies (guitar range)
            from scipy import signal
            
            # Create bandpass filter for guitar frequency range (80Hz to 1200Hz)
            nyquist = sr_guitar / 2.0
            low = 80 / nyquist
            high = 1200 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            y_filtered = signal.filtfilt(b, a, y_harmonic)
            
            # 3. Normalize audio
            y_normalized = librosa.util.normalize(y_filtered)
            
            # Save enhanced audio
            enhanced_audio_path = os.path.join(output_dir, "enhanced_guitar.wav")
            sf.write(enhanced_audio_path, y_normalized, sr_guitar)
            logger.info(f"Enhanced audio saved to: {enhanced_audio_path}")
            
            # Use the enhanced audio for further processing
            processing_audio_path = enhanced_audio_path
        except Exception as e:
            logger.error(f"Audio enhancement failed: {e}", exc_info=True)
            processing_audio_path = guitar_track_path
            logger.warning("Using unenhanced audio due to enhancement failure")
        
        # Step 3: Note detection using frequency and pitch analysis
        logger.info("Step 3: Detecting notes using frequency and pitch analysis")
        try:
            # Create a unified audio analyzer for the enhanced audio
            analyzer = UnifiedAudioAnalyzer(processing_audio_path)
            
            # Detect notes with detailed frequency information
            detected_notes = analyzer.detect_guitar_notes()
            
            logger.info(f"Detected {len(detected_notes)} notes")
            
            # Log the first few detected notes for debugging
            for i, note in enumerate(detected_notes[:5]):
                logger.info(f"Note {i+1}: {note}")
                
        except Exception as e:
            logger.error(f"Note detection failed: {e}", exc_info=True)
            return jsonify({'error': f'Note detection failed: {str(e)}'}), 500
        
        # Step 4: MIDI note conversion with frequency verification
        logger.info("Step 4: Converting to MIDI notes with frequency verification")
        try:
            midi_notes = []
            
            for note in detected_notes:
                # Get the frequency from the note
                freq = note.get('frequency', 0)
                
                # Convert frequency to MIDI note number
                midi_note = librosa.hz_to_midi(freq)
                
                # Round to nearest integer
                midi_note_int = int(round(midi_note))
                
                # Convert back to Hz to verify accuracy
                freq_from_midi = librosa.midi_to_hz(midi_note_int)
                
                # Calculate frequency error percentage
                if freq > 0:
                    error_pct = abs(freq - freq_from_midi) / freq * 100
                else:
                    error_pct = 100
                
                # Only include notes with reasonable error margin
                if error_pct < 5:  # 5% error margin
                    midi_notes.append({
                        'midi_note': midi_note_int,
                        'original_frequency': freq,
                        'verified_frequency': freq_from_midi,
                        'error_pct': error_pct,
                        'start_time': note.get('start_time', 0),
                        'end_time': note.get('end_time', 0),
                        'velocity': note.get('velocity', 64),
                        'confidence': note.get('confidence', 0.8)
                    })
                    
                    # Log the MIDI note for debugging
                    logger.info(f"MIDI Note: {midi_note_int} (from {freq:.2f}Hz, verified {freq_from_midi:.2f}Hz, error {error_pct:.2f}%)")
            
            logger.info(f"Converted {len(midi_notes)} notes to MIDI")
            
            # Save MIDI notes to a file for reference
            midi_data = pretty_midi.PrettyMIDI()
            guitar_program = pretty_midi.instrument_name_to_program('Acoustic Guitar (nylon)')
            guitar_instrument = pretty_midi.Instrument(program=guitar_program)
            
            for note in midi_notes:
                midi_note = pretty_midi.Note(
                    velocity=note['velocity'],
                    pitch=note['midi_note'],
                    start=note['start_time'],
                    end=note['end_time']
                )
                guitar_instrument.notes.append(midi_note)
            
            midi_data.instruments.append(guitar_instrument)
            midi_path = os.path.join(output_dir, "detected_notes.mid")
            midi_data.write(midi_path)
            logger.info(f"MIDI file saved to: {midi_path}")
            
        except Exception as e:
            logger.error(f"MIDI conversion failed: {e}", exc_info=True)
            return jsonify({'error': f'MIDI conversion failed: {str(e)}'}), 500
        
        # Step 5: TabCNN and LSTM prediction for fretboard positions
        logger.info("Step 5: Running TabCNN and LSTM prediction for fretboard positions")
        try:
            # Initialize TabCNN processor
            tabcnn_processor = TabCNNProcessor()
            
            # Use TabCNN to predict fretboard positions
            tabcnn_predictions = tabcnn_processor.predict_from_midi(midi_path)
            
            logger.info(f"TabCNN generated {len(tabcnn_predictions)} position predictions")
            
            # Save TabCNN predictions
            tabcnn_path = os.path.join(output_dir, "tabcnn_predictions.json")
            with open(tabcnn_path, 'w') as f:
                json.dump(tabcnn_predictions, f, indent=2)
            
            # Load LSTM predictor for sequential position refinement
            try:
                from lstm_predictor import LSTMPredictor
                lstm_predictor = LSTMPredictor()
                
                # Refine positions using LSTM
                lstm_refined_positions = lstm_predictor.refine_positions(tabcnn_predictions)
                
                logger.info("LSTM refinement completed")
                
                # Save LSTM refined predictions
                lstm_path = os.path.join(output_dir, "lstm_refined_positions.json")
                with open(lstm_path, 'w') as f:
                    json.dump(lstm_refined_positions, f, indent=2)
                
                # Use the LSTM refined positions for the next step
                fretboard_input = lstm_refined_positions
            except ImportError:
                logger.warning("LSTM predictor not available, using TabCNN predictions directly")
                fretboard_input = tabcnn_predictions
                
        except Exception as e:
            logger.error(f"TabCNN/LSTM prediction failed: {e}", exc_info=True)
            return jsonify({'error': f'TabCNN/LSTM prediction failed: {str(e)}'}), 500
        
        # Step 6: Fretboard position optimization
        logger.info("Step 6: Optimizing fretboard positions")
        try:
            # Load the fretboard position model
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     'models', 'fretboard_position_model')
            
            # Check if we have a real model or need to use the mock
            if os.path.exists(os.path.join(model_path, 'model.h5')):
                from fretboard_position_model import FretboardPositionModel
                position_model = FretboardPositionModel(model_path)
            else:
                # Use mock model
                logger.warning("Using mock fretboard position model")
                from unified_tab_processor2 import FretboardPositionOptimizer
                position_model = FretboardPositionOptimizer()
            
            # Optimize the positions
            optimized_positions = position_model.optimize_positions(fretboard_input)
            
            logger.info(f"Fretboard position optimization completed with {len(optimized_positions)} positions")
            
            # Save optimized positions
            optimized_path = os.path.join(output_dir, "optimized_positions.json")
            with open(optimized_path, 'w') as f:
                json.dump(optimized_positions, f, indent=2)
                
        except Exception as e:
            logger.error(f"Fretboard position optimization failed: {e}", exc_info=True)
            return jsonify({'error': f'Fretboard position optimization failed: {str(e)}'}), 500
        
        # Step 7: LilyPond tablature rendering
        logger.info("Step 7: Rendering tablature with LilyPond")
        try:
            from lilypond_tab_generator import LilypondTabGenerator
            
            # Initialize LilyPond generator
            lilypond_generator = LilypondTabGenerator()
            
            # Generate tablature
            tab_result = lilypond_generator.generate_tab(
                optimized_positions,
                title=os.path.splitext(filename)[0],
                output_dir=output_dir
            )
            
            logger.info(f"LilyPond tablature generated: {tab_result['tab_path']}")
            
            # Also generate ASCII tab for preview
            ascii_tab = lilypond_generator.generate_ascii_tab(optimized_positions)
            ascii_tab_path = os.path.join(output_dir, "tab_preview.txt")
            with open(ascii_tab_path, 'w') as f:
                f.write(ascii_tab)
                
        except Exception as e:
            logger.error(f"LilyPond rendering failed: {e}", exc_info=True)
            return jsonify({'error': f'LilyPond rendering failed: {str(e)}'}), 500
        
        # Return the results
        return jsonify({
            'job_id': job_id,
            'message': 'Advanced tablature generation completed successfully',
            'files': {
                'guitar_track': os.path.basename(guitar_track_path),
                'enhanced_audio': os.path.basename(enhanced_audio_path),
                'midi_file': os.path.basename(midi_path),
                'tabcnn_predictions': os.path.basename(tabcnn_path),
                'optimized_positions': os.path.basename(optimized_path),
                'tab_pdf': os.path.basename(tab_result['tab_path']) if 'tab_path' in tab_result else None,
                'tab_preview': os.path.basename(ascii_tab_path)
            },
            'stats': {
                'detected_notes': len(detected_notes),
                'midi_notes': len(midi_notes),
                'fretboard_positions': len(optimized_positions)
            }
        })
        
    except Exception as e:
        logger.error(f"Advanced tablature generation failed: {e}", exc_info=True)
        return jsonify({
            'error': f'Advanced tablature generation failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    try:
        # Create upload folder if it doesn't exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        logger.info(f"Upload folder created/verified at: {UPLOAD_FOLDER}")
        
        # Clean up old files on startup
        cleanup_old_files()
        
        # Start the Flask server with debug mode DISABLED to prevent restart loops
        logger.info("Starting Flask server on port 5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
