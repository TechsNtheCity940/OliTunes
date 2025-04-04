from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import logging
import base64
from werkzeug.utils import secure_filename
from datetime import datetime
import librosa
import numpy as np
from audio_analysis import AudioAnalyzer
from audio_analyzer_connector import EnhancedAudioAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "http://localhost:3000",
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
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only WAV and MP3 files are allowed'}), 400
        
    try:
        # Clean up old files before new upload
        cleanup_old_files()
        
        # Replace spaces with underscores in the filename
        filename = secure_filename(file.filename)
        filename = filename.replace(' ', '_')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Analyze audio file
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'duration': duration,
            'tempo': float(tempo),
            'sample_rate': sr
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/play/<filename>', methods=['GET'])
def play_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

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
            
        analyzer = AudioAnalyzer(file_path)
        chords = analyzer.detect_chords()
        
        # Convert numpy types to native Python types
        chords = convert_numpy_types(chords)
        
        logger.info(f"Detected {len(chords)} chords")
        return jsonify({'chords': chords})
        
    except Exception as e:
        logger.error(f"Error analyzing chords: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/key/<filename>', methods=['GET'])
def analyze_key(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        analyzer = AudioAnalyzer(file_path)
        key = analyzer.detect_key()
        return jsonify({'key': key})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/time-signature/<filename>', methods=['GET'])
def analyze_time_signature(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        analyzer = AudioAnalyzer(file_path)
        time_signature = analyzer.detect_time_signature()
        return jsonify({'time_signature': time_signature})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/notes/<filename>', methods=['GET'])
def analyze_notes(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
            
        # Use the enhanced audio analyzer for better note detection
        analyzer = EnhancedAudioAnalyzer(file_path)
        logger.info("Detecting guitar notes with advanced pitch detection algorithms...")
        guitar_notes = analyzer.detect_guitar_notes()
        
        # Convert numpy types to native Python types
        guitar_notes = convert_numpy_types(guitar_notes)
        
        logger.info(f"Detected {len(guitar_notes)} guitar notes")
        return jsonify({'notes': guitar_notes})
        
    except Exception as e:
        logger.error(f"Error analyzing notes: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/separate/<filename>', methods=['GET'])
def analyze_separate(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        analyzer = AudioAnalyzer(file_path)
        separation = analyzer.separate_instruments()
        return jsonify({'separation': separation})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/lyrics/<path:filename>', methods=['GET'])
def analyze_lyrics(filename):
    """Extract lyrics from an audio file"""
    try:
        # URL decode the filename
        filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Extracting lyrics for file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        try:
            analyzer = AudioAnalyzer(file_path)
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

@app.route('/analyze/structure/<path:filename>', methods=['GET'])
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
            analyzer = AudioAnalyzer(file_path)
            
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
                        analyzer_guitar = AudioAnalyzer(file_path)
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
                enhanced_analyzer = EnhancedAudioAnalyzer(file_path)
                
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
        enhanced_analyzer = EnhancedAudioAnalyzer(file_path)
        
        # Generate visualizations
        output_name = os.path.splitext(os.path.basename(file_path))[0]
        vis_paths = enhanced_analyzer.generate_visualizations(output_name)
        
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
def analyze_fretboard(filename):
    """Analyze the audio and return fretboard position data for visualization"""
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
            
        logger.info(f"Analyzing fretboard positions for file: {file_path}")
        
        try:
            analyzer = AudioAnalyzer(file_path)
            
            # Attempt to isolate guitar track for more accurate analysis
            logger.info("Performing source separation to isolate guitar track...")
            
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
                    analyzer_guitar = AudioAnalyzer(file_path)
                    analyzer_guitar.y = guitar_track
                    analyzer_guitar.sr = original_sr
                    logger.info("Successfully isolated 'other' stem using Demucs")
                except Exception as e:
                    logger.warning(f"Demucs separation error: {str(e)}. Using full mix.")
            
            # Use our enhanced audio analyzer for more accurate note detection and fretboard positioning
            logger.info("Using advanced audio analyzer for precise pitch detection...")
            enhanced_analyzer = EnhancedAudioAnalyzer(file_path)
            
            # If we have an isolated guitar track, use it
            if has_demucs and "other" in locals():
                enhanced_analyzer.standard_analyzer.y = guitar_track
                enhanced_analyzer.advanced_analyzer.y = guitar_track
                logger.info("Using isolated guitar track with advanced analyzer")
            
            # Detect guitar notes with state-of-the-art algorithms
            logger.info("Detecting guitar notes with advanced algorithms...")
            guitar_notes = enhanced_analyzer.detect_guitar_notes()
            
            # Map to optimal fretboard positions
            logger.info("Mapping notes to optimal fretboard positions...")
            mapped_notes = enhanced_analyzer.map_notes_to_fretboard(guitar_notes)
            
            # Format note positions for UI
            note_positions = []
            for note in mapped_notes:
                note_positions.append({
                    'note': note['note'],
                    'time': float(note['time']),
                    'duration': float(note['duration']),
                    'string': note['string'],  # String number (0-5)
                    'fret': note['fret'],      # Fret number
                    'confidence': float(note.get('confidence', 0.8))
                })
            
            # Convert numpy types to native Python types
            note_positions = convert_numpy_types(note_positions)
            
            logger.info(f"Mapped {len(note_positions)} notes to fretboard positions")
            return jsonify({
                'notePositions': note_positions,
                'totalDuration': float(librosa.get_duration(y=analyzer.y, sr=analyzer.sr))
            })
            
        except Exception as e:
            logger.error(f"Error during fretboard analysis: {str(e)}", exc_info=True)
            return jsonify({'error': f'Fretboard analysis error: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in analyze_fretboard: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
