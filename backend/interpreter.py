"""interpreter
"""
import numpy as np
import pretty_midi
from matplotlib import lines as mlines, pyplot as plt
import tempfile
import librosa
import soundfile as sf
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Set the backend before importing pyplot


def save_small_wav(out_path, y, fs):
    try:
        sf.write(out_path, y, fs, subtype='PCM_16')
    except Exception as e:
        print(f"Error saving audio: {str(e)}")
        raise

def jams_to_midi(jam, q=1):
    # q = 1: with pitch bend. q = 0: without pitch bend.
    midi = pretty_midi.PrettyMIDI()
    annos = jam.search(namespace='note_midi')
    if len(annos) == 0:
        annos = jam.search(namespace='pitch_midi')
    for anno in annos:
        midi_ch = pretty_midi.Instrument(program=25)
        for note in anno:
            pitch = int(round(note.value))
            bend_amount = int(round((note.value - pitch) * 4096))
            st = note.time
            dur = note.duration
            n = pretty_midi.Note(
                velocity=100 + np.random.choice(range(-5, 5)),
                pitch=pitch, start=st,
                end=st + dur
            )
            pb = pretty_midi.PitchBend(pitch=bend_amount * q, time=st)
            midi_ch.notes.append(n)
            midi_ch.pitch_bends.append(pb)
        if len(midi_ch.notes) != 0:
            midi.instruments.append(midi_ch)
    return midi


def sonify_jams(jam, fpath=None, q=1):
    midi = jams_to_midi(jam, q) # q=1 : with pitchbend
    signal_out = midi.fluidsynth()
    if fpath != None:
        save_small_wav(fpath, signal_out, 44100)
    return signal_out, 44100


def visualize_jams_note(jam, save_path=None):
    style_dict = {0 : 'r', 1 : 'y', 2 : 'b', 3 : '#FF7F50', 4 : 'g', 5 : '#800080'}
    string_dict = {0: 'E', 1: 'A', 2: 'D', 3: 'G', 4: 'B', 5: 'e' }
    s = 0
    handle_list = []
    fig = plt.figure()
    annos = jam.search(namespace='note_midi')
    if len(annos) == 0:
        annos = jam.search(namespace='pitch_midi')
    for string_tran in annos:
        handle_list.append(mlines.Line2D([], [], color=style_dict[s],
                                         label=string_dict[s]))
        for note in string_tran:
            start_time = note[0]
            midi_note = note[2]
            dur = note[1]
            plt.plot([start_time, start_time + dur],
                     [midi_note, midi_note],
                     style_dict[s], label=string_dict[s])
        s += 1
    plt.xlabel('Time (sec)')
    plt.ylabel('Pitch (midi note number)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=handle_list)
    plt.title(jam.file_metadata.title)
    plt.xlim(-0.5, jam.file_metadata.duration)
    fig.set_size_inches(6,3)
    if save_path:
        plt.savefig(save_path)

def visualize_jams_pt(jam, save_path=None):
    style_dict = {0 : 'r', 1 : 'y', 2 : 'b', 3 : '#FF7F50', 4 : 'g', 5 : '#800080'}
    string_dict = {0: 'E', 1: 'A', 2: 'D', 3: 'G', 4: 'B', 5: 'e' }
    s = 0
    handle_list = []
    # fig = plt.figure()
    annos_pt = jam.search(namespace='pitch_contour')
    # plot pitch
    for string_tran in annos_pt:
        handle_list.append(mlines.Line2D([], [], color=style_dict[s],
                                         label=string_dict[s]))
        df = string_tran.to_dataframe()
        pitch_s = df.value.apply(
            lambda x: librosa.hz_to_midi(float(x['frequency'])))
        pitch_s.name = 'pitch'
        df = pd.concat([df, pitch_s], axis=1)
        plt.scatter(df.time, df.pitch, s=0.1, color=style_dict[s],
                        label=string_dict[s])

        s += 1

    # plot Beat
    anno_b = jam.search(namespace='beat_position')[0]
    handle_list.append(mlines.Line2D([], [], color='k',
                                         label='downbeat'))
    handle_list.append(mlines.Line2D([], [], color='k', linestyle='dotted',
                                     label='beat'))
    for b in anno_b.data:
        t = b.time
        plt.axvline(t, linestyle='dotted', color='k', alpha=0.5)
        if int(b.value['position']) == 1:
            plt.axvline(t, linestyle='-', color='k', alpha=0.8)


    # plt.xlabel('Time (sec)')
    plt.ylabel('Pitch Contour (midi note number)')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=handle_list)
    plt.title(jam.file_metadata.title)
    plt.xlim(-0.06, jam.file_metadata.duration)
    # fig.set_size_inches(6, 3)
    if save_path:
        plt.savefig(save_path)

def visualize_jams_onset(jam, save_path=None, low=None, high=None):
    style_dict = {0 : 'r', 1 : 'y', 2 : 'b', 3 : '#FF7F50', 4 : 'g', 5 : '#800080'}
    string_dict = {0: 'E', 1: 'A', 2: 'D', 3: 'G', 4: 'B', 5: 'e' }
    s = 0
    handle_list = []
    # fig = plt.figure()
    annos = jam.search(namespace='note_midi')
    if len(annos) == 0:
        annos = jam.search(namespace='pitch_midi')
    for string_tran in annos:
        handle_list.append(mlines.Line2D([], [], color=style_dict[s],
                                         label=string_dict[s]))
        for note in string_tran:
            
            start_time = note[0]
            if low and start_time < low:
                continue
            if high and start_time > high:
                continue
            plt.vlines(start_time,s, s+2,style_dict[s], label=string_dict[s])
        s += 1
    
    plt.xlabel('sec')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=handle_list)
    plt.ylabel('String Number')

    if not low:
        low = -0.1
    if not high:
        high = jam.file_metadata.duration
    plt.xlim(low, high)
    # fig.set_size_inches(jam.file_metadata.duration / 2.5, 6)
#    plt.title('Onsets of Individual Strings for excerpt of 00_Rock2-142-D_comp')
    if save_path:
        plt.savefig(save_path)


def tablaturize_jams(jam, save_path=None):
    str_midi_dict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
    string_dict = {0: 'E', 1: 'A', 2: 'D', 3: 'G', 4: 'B', 5: 'e'}
    style_dict = {0 : 'r', 1 : 'y', 2 : 'b', 3 : '#FF7F50', 4 : 'g', 5 : '#800080'}
    s = 0

    handle_list = []

    annos = jam.search(namespace='note_midi')
    if len(annos) == 0:
        annos = jam.search(namespace='pitch_midi')

    for string_tran in annos:
        handle_list.append(mlines.Line2D([], [], color=style_dict[s],
                                         label=string_dict[s]))
        for note in string_tran:
            start_time = note[0]
            midi_note = note[2]
            fret = int(round(midi_note - str_midi_dict[s]))
            plt.scatter(start_time, s+1, marker="${}$".format(fret), color =
            style_dict[s])
        s += 1

    # plot Beat
    anno_b = jam.search(namespace='beat_position')[0]
    for b in anno_b.data:
        t = b.time
        plt.axvline(t, linestyle='dotted', color='k', alpha=0.5)
        if int(b.value['position']) == 1:
            plt.axvline(t, linestyle='-', color='k', alpha=0.8)

    handle_list.append(mlines.Line2D([], [], color='k',
                                     label='downbeat'))
    handle_list.append(mlines.Line2D([], [], color='k', linestyle='dotted',
                                     label='beat'))
    plt.xlabel('Time (sec)')
    plt.ylabel('String Number')
    # plt.title(jam.file_metadata.title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
               handles=handle_list, ncol=8)
    plt.xlim(-0.5, jam.file_metadata.duration)
    # fig.set_size_inches(6, 3)
    if save_path:
        plt.savefig(save_path)

def visualize_chords(jam, save_path=None):

    chord_ann = jam.search(namespace='chord')[1]


    for chord in chord_ann.data:
        t = chord.time


    # for string_tran in annos:
    #     for note in string_tran:
    #         start_time = note[0]
    #         midi_note = note[2]
    #         fret = int(round(midi_note - str_midi_dict[s]))
    #         plt.scatter(start_time, s+1, marker="${}$".format(fret), color =
    #         style_dict[s])
    #     s += 1
    #
    # # plot Beat
    # anno_b = jam.search(namespace='beat_position')[0]
    # for b in anno_b.data:
    #     t = b.time
    #     plt.axvline(t, linestyle='dotted', color='k', alpha=0.5)
    #     if int(b.value['position']) == 1:
    #         plt.axvline(t, linestyle='-', color='k', alpha=0.8)


    plt.xlabel('Time (sec)')
    plt.ylabel('String Number')
    # plt.title(jam.file_metadata.title)
    plt.xlim(-0.5, jam.file_metadata.duration)
    # fig.set_size_inches(6, 3)
    if save_path:
        plt.savefig(save_path)

def add_annotations_to_barline(ax, annotations, beat_annotations,
                               ygrow_ratio=1, label_xoffset=0, label_yoffset=0):
    """
    Add annotation values at each barline of a given pair of axes.
    
    Display at the top of the given Matplotlib Axes the values of a JAMS annotation as text
    labels at each barline specified by a JAMS beat_position annotation array. If there are
    multiple annotations per bar, only the first one will be displayed and if an annotation
    lasts multiple bars, its label will be repeated at every bar line.
    
    Keyword arguments:
    ax -- Matplotlib Axes
    annotations -- the JAMS AnnotationArray whose values to display
    beat_annotations -- the JAMS beat_position AnnotationArray specifying the barlines
    ygrow_ratio -- amount to increase the y-axis in order to accomodate the text fields,
                   specified as the ratio of the current y-range
    label_xoffset -- amount to offset the text labels in order to avoid overlap with barlines,
                     specified in the units of the x-axis
    label_yoffset -- amount to offset the text labels in order to avoid overlap with barlines,
                     specified in the units of the y-axis
    
    Note: the optimal values for ygrow_ratio, label_xoffset and label_yoffset are plot-dependent,
          so will have to be determined experimentally
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max+(y_max-y_min)*ygrow_ratio) # make room for label
    for beat in beat_annotations:
        if beat.value['position'] == 1 and beat.time >= x_min and beat.time < x_max:
            ax.text(beat.time+label_xoffset, y_max-label_yoffset,
                    annotations.to_samples([beat.time+0.001])[0][0],
                    fontdict={'backgroundcolor': ax.get_facecolor()})

def add_annotations(ax, annotations,
                    ygrow_ratio=0, label_xoffset=0, label_yoffset=0):
    """
        Add annotation values to a given pair of axes.
        
        Display at the top of the given Matplotlib Axes the values of a JAMS annotation as text
        labels.
        
        Keyword arguments:
        ax -- Matplotlib Axes
        annotations -- the JAMS AnnotationArray whose values to display
        ygrow_ratio -- amount to increase the y-axis in order to accomodate the text fields,
        specified as the ratio of the current y-range
        label_xoffset -- amount to offset the text labels in order to avoid overlap with barlines,
        specified in the units of the x-axis
        label_yoffset -- amount to offset the text labels in order to avoid overlap with barlines,
        specified in the units of the y-axis
        
        Note: the optimal values for ygrow_ratio, label_xoffset and label_yoffset are plot-dependent,
        so will have to be determined experimentally
        """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max+(y_max-y_min)*ygrow_ratio) # make room for label
    for segment in annotations:
        if segment.time >= x_min and segment.time < x_max:
            ax.text(segment.time+label_xoffset, y_max-label_yoffset,
                    annotations.to_samples([segment.time+0.001])[0][0],
                    fontdict={'backgroundcolor': ax.get_facecolor()})

# TabCNN and Demucs Integration

import sys
import tensorflow as tf
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add TabCNN model path to sys.path if needed
backend_dir = os.path.dirname(os.path.abspath(__file__))
tab_cnn_dir = os.path.join(backend_dir, 'models', 'tab-cnn')
sys.path.append(tab_cnn_dir)
sys.path.append(os.path.join(backend_dir, 'data'))

# Import required modules with error handling
try:
    from TabCNN import TabCNNProcessor
    tab_cnn_available = True
    logger.info("TabCNN model loaded successfully")
except ImportError as e:
    tab_cnn_available = False
    logger.error(f"Error importing TabCNN modules: {e}")

try:
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    demucs_available = True
    logger.info("Demucs loaded successfully")
except ImportError as e:
    demucs_available = False
    logger.error(f"Error importing Demucs: {e}")

try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    basic_pitch_available = True
    logger.info("Basic Pitch loaded successfully")
except ImportError as e:
    basic_pitch_available = False
    logger.error(f"Error importing Basic Pitch: {e}")

class TabCNNProcessor:
    """Handles the TabCNN model for tablature prediction from audio"""
    
    def __init__(self, model_path=None):
        """Initialize the TabCNN processor
        
        Args:
            model_path: Path to saved TabCNN model weights (if None, will use default)
        """
        self.model_path = model_path
        self.model = None
        self.initialized = False
        
        # Default paths
        if self.model_path is None:
            # Use the specific weights file path
            self.model_path = os.path.join(backend_dir, "models", "tab-cnn", "model", "saved", "c 2025-04-06 115653", "1", "model.weights.h5")
        
        # Initialize model if available
        if tab_cnn_available:
            try:
                self.initialize_model()
            except Exception as e:
                logger.error(f"Failed to initialize TabCNN model: {e}")
    
    def initialize_model(self):
        """Initialize the TabCNN model"""
        try:
            # Load the model using the build_model method
            self.model = self.build_model()
            
            # Load weights if available
            if os.path.exists(self.model_path):
                self.model.load_weights(self.model_path)
                logger.info(f"Loaded TabCNN weights from {self.model_path}")
                self.initialized = True
            else:
                logger.warning(f"Weights file not found at {self.model_path}")
                self.initialized = False
        except Exception as e:
            logger.error(f"Error initializing TabCNN model: {e}")
            raise
    
    def build_model(self):
        """Build the enhanced TabCNN model with residual connections"""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Add
        
        inputs = Input(shape=(128, 9, 1))  # Mel-spectrogram input
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        residual = x
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Add()([x, residual])  # Residual connection
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        note_output = Dense(6 * 21, activation='sigmoid')(x)  # Note presence
        fret_output = Dense(6 * 21, activation='softmax')(x)  # Fret positions
        model = Model(inputs, [note_output, fret_output])
        model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
                    optimizer='adam', metrics=['accuracy'])
        return model
    
    def preprocess_audio(self, audio_file, temp_dir=None):
        """Preprocess audio for TabCNN model
        
        Args:
            audio_file: Path to audio file
            temp_dir: Directory for temporary files
            
        Returns:
            Preprocessed audio representation
        """
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        try:
            # Load audio and convert to the right format
            y, sr = librosa.load(audio_file, sr=22050, mono=True)
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            # Generate mel-spectrogram
            spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=128,
                fmin=80,
                fmax=4000,
                hop_length=512
            )
            
            # Convert to log scale
            spec = librosa.power_to_db(spec, ref=np.max)
            
            # Normalize
            spec = (spec - spec.mean()) / (spec.std() + 1e-8)
            
            # Create context windows
            frames_with_context = []
            for i in range(4, spec.shape[1] - 4):
                window = spec[:, i-4:i+5]  # 9-frame context window
                frames_with_context.append(window)
            
            if not frames_with_context:
                # Handle short audio files
                window = np.pad(spec, ((0, 0), (4, 4)), mode='constant')
                frames_with_context.append(window[:, :9])
            
            # Stack frames
            spec_data = np.stack(frames_with_context)
            
            # Add channel dimension for CNN
            spec_data = np.expand_dims(spec_data, axis=-1)
            
            return spec_data
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise
    
    def predict(self, mel_spectrogram):
        """Make predictions using the TabCNN model
        
        Args:
            mel_spectrogram: Mel spectrogram of the audio (shape should match model input)
            
        Returns:
            Predictions matrix or empty list if model not available
        """
        if self.model is None:
            logger.warning("TabCNN model not loaded, cannot make predictions")
            return []
            
        try:
            # Ensure mel_spectrogram is in the right format for the model
            if len(mel_spectrogram.shape) == 2:  # Need to add batch and channel dimensions
                # Add batch dimension and channel dimension for CNN input
                # Model expects [batch_size, time_steps, mel_bins, channels]
                mel_spectrogram = np.expand_dims(np.expand_dims(mel_spectrogram, axis=0), axis=-1)
            
            # Make predictions
            predictions = self.model.predict(mel_spectrogram)
            
            # Process predictions if needed
            if isinstance(predictions, list):
                # If model outputs multiple arrays, take the first one
                predictions = predictions[0]
                
            # Remove batch dimension if present
            if len(predictions.shape) > 2 and predictions.shape[0] == 1:
                predictions = predictions[0]
                
            logger.info(f"TabCNN predictions shape: {predictions.shape}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making TabCNN predictions: {str(e)}")
            return []
    
    def convert_predictions_to_tab(self, predictions):
        """Convert model predictions to tablature format
        
        Args:
            predictions: Model predictions
            
        Returns:
            Tablature data
        """
        num_frames = predictions.shape[0]
        num_strings = 6
        
        # Convert from one-hot to tablature notation
        tab_data = []
        for frame_idx in range(num_frames):
            frame_data = []
            for string_idx in range(num_strings):
                # Get the predicted fret number (argmax of the string's predictions)
                fret = np.argmax(predictions[frame_idx, string_idx])
                # Adjust numbering (0 = not played, 1-20 = frets 0-19)
                if fret > 0:
                    fret -= 1  # Convert back to 0-indexed
                else:
                    fret = -1  # Not played
                frame_data.append(fret)
            tab_data.append(frame_data)
        
        return np.array(tab_data)

def visualize_tab(predictions):
    """Display guitar tab visualization"""
    strings = ['E', 'A', 'D', 'G', 'B', 'e']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Display predictions as a tab
    for i, (string, pred) in enumerate(zip(strings, predictions)):
        fret = np.argmax(pred)
        ax.text(0, i, f"{string} |--{fret if fret > 0 else '-'}--", 
                fontfamily='monospace', fontsize=12)
    
    ax.axis('off')
    plt.title('Predicted Tablature')
    plt.tight_layout()
    plt.show()

class DemucsProcessor:
    """Handles audio source separation using Demucs"""
    
    def __init__(self, model_name="htdemucs_6s"):
        """Initialize Demucs processor
        
        Args:
            model_name: Demucs model to use (default: htdemucs_6s for 6-stem separation)
        """
        self.model_name = model_name
        self.model = None
        self.initialized = False
        
        # Initialize if available
        if demucs_available:
            try:
                self.initialize_model()
            except Exception as e:
                logger.error(f"Failed to initialize Demucs: {e}")
                # Fallback to standard model if 6s fails
                if model_name == "htdemucs_6s":
                    logger.info("Falling back to standard htdemucs model")
                    self.model_name = "htdemucs"
                    try:
                        self.initialize_model()
                    except Exception as e2:
                        logger.error(f"Failed to initialize fallback Demucs model: {e2}")
    
    def initialize_model(self):
        """Initialize Demucs model"""
        try:
            # Load pretrained model
            self.model = get_model(self.model_name)
            self.initialized = True
            logger.info(f"Loaded Demucs model: {self.model_name}")
            logger.info(f"Available stems: {self.model.sources}")
        except Exception as e:
            logger.error(f"Error initializing Demucs model: {e}")
            raise
    
    def separate_audio(self, audio_file, output_dir=None):
        """Separate audio into stems
        
        Args:
            audio_file: Path to audio file
            output_dir: Directory to save stems
            
        Returns:
            Dictionary of separated stems {instrument: path_to_audio}
        """
        # Use the enhanced method with shifts for better quality
        return self.separate_audio_with_model(audio_file, output_dir=output_dir, shifts=10)
            
    def separate_audio_with_model(self, audio_file, output_dir=None, model_name=None, shifts=10):
        """Separate audio into stems with enhanced quality using shifts
        
        Args:
            audio_file: Path to audio file or numpy array with sample rate
            output_dir: Directory to save stems
            model_name: Override model name (if None, uses self.model_name)
            shifts: Number of shifts for better extraction quality (default: 10)
            
        Returns:
            Dictionary of separated stems {instrument: path_to_audio}
        """
        if model_name is None:
            model_name = self.model_name
            
        # If model name is different from current, initialize it
        if model_name != self.model_name or not self.initialized or self.model is None:
            original_model_name = self.model_name
            try:
                self.model_name = model_name
                self.initialize_model()
            except Exception as e:
                logger.error(f"Failed to initialize model {model_name}: {e}")
                # Restore original model if possible
                if original_model_name != model_name:
                    try:
                        self.model_name = original_model_name
                        self.initialize_model()
                    except:
                        pass
                return None
            
        if not self.initialized or self.model is None:
            logger.error("Demucs not initialized")
            return None
            
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
            
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Handle different input types
            if isinstance(audio_file, str):
                # Load audio file with librosa
                audio, sr = librosa.load(audio_file, sr=44100, mono=False)
            elif isinstance(audio_file, tuple) and len(audio_file) == 2:
                # Assume (audio, sr) tuple
                audio, sr = audio_file
            else:
                raise ValueError("audio_file must be a path or (audio, sr) tuple")
            
            # If audio is mono, convert to stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            
            # Convert to torch tensor
            audio_tensor = torch.tensor(audio).float()
            
            # Apply model with shifts for better quality
            logger.info(f"Applying Demucs model with {shifts} shifts for enhanced quality")
            with torch.no_grad():
                sources = apply_model(
                    self.model, 
                    audio_tensor.unsqueeze(0), 
                    device="cpu", 
                    shifts=shifts,
                    split=True,
                    overlap=0.25
                )[0]
            
            # Get instrument names
            instrument_names = self.model.sources
            
            # Save separated stems
            stem_paths = {}
            for i, instrument in enumerate(instrument_names):
                source_audio = sources[i].numpy()
                output_path = os.path.join(output_dir, f"{instrument}.wav")
                
                # Normalize and convert to mono
                source_audio = np.mean(source_audio, axis=0)
                source_audio = librosa.util.normalize(source_audio)
                
                # Save the audio file
                sf.write(output_path, source_audio, sr)
                stem_paths[instrument] = output_path
            
            logger.info(f"Separated audio into {len(stem_paths)} stems with {shifts} shifts: {', '.join(instrument_names)}")
            return stem_paths
            
        except Exception as e:
            logger.error(f"Error separating audio with shifts: {str(e)}")
            return None


class BasicPitchProcessor:
    """Handles audio to MIDI conversion using Basic Pitch"""
    
    def __init__(self):
        """Initialize Basic Pitch processor"""
        self.initialized = basic_pitch_available
        if not self.initialized:
            logger.warning("Basic Pitch is not available. MIDI conversion will be limited.")
    
    def audio_to_midi(self, audio_file, output_dir=None, save_midi=True, save_notes=True):
        """Convert audio to MIDI using Basic Pitch
        
        Args:
            audio_file: Path to audio file
            output_dir: Directory to save MIDI and notes files
            save_midi: Whether to save MIDI file
            save_notes: Whether to save notes file
            
        Returns:
            Dictionary containing:
                - midi_path: Path to saved MIDI file
                - notes: List of note events with pitch, start_time, duration, etc.
                - confidence: Confidence scores for note predictions
        """
        if not self.initialized:
            logger.error("Basic Pitch not initialized")
            return self._fallback_audio_to_midi(audio_file, output_dir)
        
        if output_dir is None:
            output_dir = os.path.dirname(audio_file)
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate base filename from input file
            base_filename = os.path.splitext(os.path.basename(audio_file))[0]
            midi_path = os.path.join(output_dir, f"{base_filename}_basic_pitch.mid")
            notes_path = os.path.join(output_dir, f"{base_filename}_notes.json")
            
            # Run Basic Pitch prediction
            model_output = predict(
                audio_file,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                onset_threshold=0.5,  # Adjust sensitivity as needed
                frame_threshold=0.3,
                minimum_note_length=0.05,  # 50ms minimum note length
                minimum_frequency=librosa.note_to_hz('E2'),  # Lowest guitar string
                maximum_frequency=librosa.note_to_hz('E6'),  # Reasonable upper limit for guitar
                multiple_pitch_bends=False,  # Simpler MIDI output
            )
            
            # Extract data from model output
            midi_data = model_output['midi_data']
            notes_data = model_output['note_events']
            confidence = model_output.get('confidence', 0.8)  # Default if not available
            
            # Save MIDI file if requested
            if save_midi:
                midi_data.write(midi_path)
                logger.info(f"Saved MIDI file to {midi_path}")
            else:
                midi_path = None
            
            # Save notes data if requested
            if save_notes:
                with open(notes_path, 'w') as f:
                    json.dump(notes_data, f, indent=2)
                logger.info(f"Saved notes data to {notes_path}")
            
            # Format notes for further processing
            formatted_notes = []
            for note in notes_data:
                formatted_notes.append({
                    'pitch': note['pitch'],
                    'start_time': note['start_time'],
                    'end_time': note['end_time'],
                    'duration': note['end_time'] - note['start_time'],
                    'velocity': int(note['amplitude'] * 127),
                    'confidence': note.get('confidence', confidence)
                })
            
            return {
                'midi_path': midi_path,
                'notes': formatted_notes,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error converting audio to MIDI with Basic Pitch: {str(e)}")
            return self._fallback_audio_to_midi(audio_file, output_dir)
    
    def _fallback_audio_to_midi(self, audio_file, output_dir=None):
        """Fallback method for audio to MIDI conversion using librosa
        
        This is used when Basic Pitch is not available.
        
        Args:
            audio_file: Path to audio file
            output_dir: Directory to save MIDI file
            
        Returns:
            Dictionary with MIDI conversion results
        """
        if output_dir is None:
            output_dir = os.path.dirname(audio_file)
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate base filename from input file
            base_filename = os.path.splitext(os.path.basename(audio_file))[0]
            midi_path = os.path.join(output_dir, f"{base_filename}_librosa.mid")
            
            # Load audio
            y, sr = librosa.load(audio_file, sr=None)
            
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Create MIDI file
            midi_data = pretty_midi.PrettyMIDI()
            guitar = pretty_midi.Instrument(program=24)  # 24 is acoustic guitar
            
            # Process pitch data
            notes = []
            current_note = None
            
            # Simple note extraction from pitch data
            for i, time in enumerate(librosa.times_like(pitches)):
                index = magnitudes[:, i].argmax()
                pitch = pitches[index, i]
                
                if pitch > 0 and magnitudes[index, i] > 0.1:  # Threshold for note detection
                    midi_pitch = int(round(librosa.hz_to_midi(pitch)))
                    
                    if current_note is None or current_note['pitch'] != midi_pitch:
                        # End previous note if exists
                        if current_note is not None:
                            end_time = time
                            note = pretty_midi.Note(
                                velocity=100,
                                pitch=current_note['pitch'],
                                start=current_note['start_time'],
                                end=end_time
                            )
                            guitar.notes.append(note)
                            
                            notes.append({
                                'pitch': current_note['pitch'],
                                'start_time': current_note['start_time'],
                                'end_time': end_time,
                                'duration': end_time - current_note['start_time'],
                                'velocity': 100,
                                'confidence': 0.6  # Lower confidence for fallback method
                            })
                        
                        # Start new note
                        current_note = {
                            'pitch': midi_pitch,
                            'start_time': time
                        }
            
            # Add the last note if exists
            if current_note is not None:
                end_time = len(y) / sr
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=current_note['pitch'],
                    start=current_note['start_time'],
                    end=end_time
                )
                guitar.notes.append(note)
                
                notes.append({
                    'pitch': current_note['pitch'],
                    'start_time': current_note['start_time'],
                    'end_time': end_time,
                    'duration': end_time - current_note['start_time'],
                    'velocity': 100,
                    'confidence': 0.6
                })
            
            midi_data.instruments.append(guitar)
            midi_data.write(midi_path)
            
            logger.info(f"Saved fallback MIDI file to {midi_path}")
            
            return {
                'midi_path': midi_path,
                'notes': notes,
                'confidence': 0.6  # Lower confidence for fallback method
            }
            
        except Exception as e:
            logger.error(f"Error in fallback audio to MIDI conversion: {str(e)}")
            return {
                'midi_path': None,
                'notes': [],
                'confidence': 0
            }

def generate_tab_from_midi_notes(notes, output_file=None):
    """Generate guitar tablature from MIDI notes
    
    Args:
        notes: List of MIDI notes (can be pretty_midi.Note objects or dicts with note, start, end keys)
        output_file: Path to save tablature file
        
    Returns:
        Tablature as string
    """
    if not notes:
        return "No notes found for tablature generation"
    
    # Standard guitar tuning (E2, A2, D3, G3, B3, E4)
    guitar_tuning = [40, 45, 50, 55, 59, 64]
    
    # Sort notes by start time
    if hasattr(notes[0], 'start'):
        # pretty_midi.Note objects
        sorted_notes = sorted(notes, key=lambda n: n.start)
        midi_notes = [(note.pitch, note.start, note.end) for note in sorted_notes]
    else:
        # Dict style notes
        sorted_notes = sorted(notes, key=lambda n: n.get('start', 0))
        midi_notes = [(note.get('note', 0), note.get('start', 0), note.get('end', 0)) for note in sorted_notes]
    
    # Generate ASCII tablature
    tab_lines = [[] for _ in range(6)]  # 6 strings
    current_time = 0
    max_time = max([end for _, _, end in midi_notes]) if midi_notes else 0
    time_resolution = 0.25  # Each position represents a 16th note
    
    # Calculate how many positions we need in the tab
    tab_length = int(max_time / time_resolution) + 1
    
    # Initialize tab with dashes
    for i in range(6):
        tab_lines[i] = ['-' for _ in range(tab_length)]
    
    # Place notes in tab
    for pitch, start, end in midi_notes:
        # Find best string and fret position for this note
        string, fret = find_best_position(pitch, guitar_tuning)
        if string is not None:
            # Convert start time to position
            position = int(start / time_resolution)
            if 0 <= position < tab_length and 0 <= string < 6:
                # Place note in tab
                tab_lines[string][position] = str(fret)
    
    # Construct tab string
    tab_str = ""
    for i, string_line in enumerate(tab_lines):
        tab_str += f"String {i+1} ({chr(69-i)}): |" + ''.join(string_line) + "|\n"
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(tab_str)
    
    return tab_str

def find_best_position(pitch, tuning):
    """Find best string and fret position for a given pitch
    
    Args:
        pitch: MIDI pitch
        tuning: List of open string pitches
        
    Returns:
        (string_index, fret) tuple
    """
    best_string = None
    best_fret = None
    
    # Prefer lower positions
    for string_idx, open_pitch in enumerate(tuning):
        if pitch >= open_pitch:
            fret = pitch - open_pitch
            if fret <= 24:  # Most guitars have 24 frets or less
                if best_string is None or fret < best_fret:
                    best_string = string_idx
                    best_fret = fret
    
    # If no suitable position found, try again with higher positions
    if best_string is None:
        for string_idx, open_pitch in enumerate(tuning):
            fret = pitch - open_pitch
            if fret >= 0 and fret <= 24:
                if best_string is None or fret < best_fret:
                    best_string = string_idx
                    best_fret = fret
    
    return best_string, best_fret

def process_audio_with_tabcnn(audio_file, output_dir=None, use_source_separation=True):
    """Process audio file with TabCNN to generate tablature
    
    Args:
        audio_file: Path to audio file
        output_dir: Directory to save output files
        use_source_separation: Whether to use Demucs for source separation
        
    Returns:
        Tablature data
    """
    if output_dir is None:
        output_dir = os.path.dirname(audio_file)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processors
    tabcnn = TabCNNProcessor()
    
    # Step 1: Source separation if enabled
    guitar_audio_path = audio_file
    if use_source_separation and demucs_available:
        try:
            logger.info("Performing source separation with Demucs...")
            demucs = DemucsProcessor(model_name="htdemucs_6s")  # Use 6-stem model by default
            stems = demucs.separate_audio(audio_file, output_dir)
            
            # Try to use guitar stem if available
            if stems and 'guitar' in stems:
                guitar_audio_path = stems['guitar']
                logger.info(f"Using separated guitar stem: {guitar_audio_path}")
            elif stems and 'other' in stems:
                guitar_audio_path = stems['other']
                logger.info(f"Using 'other' stem as fallback: {guitar_audio_path}")
            else:
                logger.warning("No suitable stem found, using original audio")
        except Exception as e:
            logger.error(f"Source separation failed: {e}")
            logger.warning("Using original audio file")
    
    # Step 2: Convert audio to MIDI using Basic Pitch
    midi_data = None
    notes_data = None
    if basic_pitch_available:
        try:
            logger.info("Converting audio to MIDI with Basic Pitch...")
            basic_pitch = BasicPitchProcessor()
            midi_result = basic_pitch.audio_to_midi(
                guitar_audio_path, 
                output_dir=output_dir,
                save_midi=True,
                save_notes=True
            )
            
            if midi_result and midi_result['midi_path']:
                logger.info(f"MIDI conversion successful: {midi_result['midi_path']}")
                midi_data = midi_result['midi_path']
                notes_data = midi_result['notes']
        except Exception as e:
            logger.error(f"MIDI conversion failed: {e}")
    
    # Step 3: Process with TabCNN
    try:
        logger.info("Processing with TabCNN...")
        spec = tabcnn.preprocess_audio(guitar_audio_path)
        tab_results = tabcnn.predict(spec)
        
        # If we have MIDI data from Basic Pitch, enhance the TabCNN results
        if notes_data:
            logger.info("Enhancing TabCNN results with Basic Pitch MIDI data...")
            # Convert MIDI notes to fretboard positions
            fretboard_positions = []
            for note in notes_data:
                midi_pitch = note['pitch']
                # Simple mapping to guitar fretboard (can be improved)
                string, fret = map_midi_to_guitar(midi_pitch)
                if string is not None and fret is not None:
                    fretboard_positions.append({
                        'string': string,
                        'fret': fret,
                        'time': note['start_time'],
                        'duration': note['duration'],
                        'confidence': note.get('confidence', 0.8)
                    })
            
            # Blend TabCNN predictions with Basic Pitch data
            if 'predictions' in tab_results and len(fretboard_positions) > 0:
                tab_results['enhanced_predictions'] = blend_predictions(
                    tab_results['predictions'],
                    fretboard_positions
                )
                tab_results['midi_path'] = midi_data
                tab_results['notes_data'] = notes_data
        
        # Save results
        tab_path = os.path.join(output_dir, "tablature.txt")
        with open(tab_path, 'w') as f:
            f.write(tab_results.get('tab_text', ''))
        
        logger.info(f"Saved tablature to {tab_path}")
        return tab_results
        
    except Exception as e:
        logger.error(f"TabCNN processing failed: {e}")
        return None

def map_midi_to_guitar(midi_pitch):
    """Map MIDI pitch to guitar string and fret
    
    Args:
        midi_pitch: MIDI pitch value
        
    Returns:
        Tuple of (string, fret) or (None, None) if mapping not possible
    """
    # Standard guitar tuning (from low to high): E2(40), A2(45), D3(50), G3(55), B3(59), E4(64)
    string_tunings = [40, 45, 50, 55, 59, 64]
    
    # Try to find the best string/fret combination
    best_string = None
    best_fret = None
    best_distance = float('inf')
    
    for string_idx, base_pitch in enumerate(string_tunings):
        # Calculate fret number
        fret = midi_pitch - base_pitch
        
        # Check if note is playable on this string
        if 0 <= fret <= 24:  # Assuming 24 frets max
            # Prefer lower frets and lower strings
            distance = fret + (string_idx * 0.5)  # Slight preference for lower strings
            if distance < best_distance:
                best_string = string_idx
                best_fret = fret
                best_distance = distance
    
    return best_string, best_fret

def blend_predictions(tabcnn_predictions, basic_pitch_positions):
    """Blend TabCNN predictions with Basic Pitch positions
    
    Args:
        tabcnn_predictions: Predictions from TabCNN
        basic_pitch_positions: Positions from Basic Pitch
        
    Returns:
        Blended predictions
    """
    # If TabCNN predictions are empty, just use Basic Pitch
    if tabcnn_predictions is None or len(tabcnn_predictions) == 0:
        return basic_pitch_positions
    
    # Create a copy of TabCNN predictions
    blended = np.copy(tabcnn_predictions)
    
    # For each Basic Pitch position, find the closest time in TabCNN predictions
    # and enhance that prediction
    for position in basic_pitch_positions:
        time_idx = int(position['time'] / 0.0232)  # Convert time to frame index
        if time_idx < len(blended):
            string = position['string']
            fret = position['fret']
            confidence = position.get('confidence', 0.8)
            
            # Enhance the prediction if confidence is high enough
            if confidence > 0.6:
                # Reset the string's predictions
                blended[time_idx, string, :] = 0
                # Set the predicted fret
                blended[time_idx, string, fret] = 1.0
    
    return blended

def generate_tab_for_audio_file(audio_file, output_dir=None):
    """Generate tablature for an audio file
    
    Args:
        audio_file: Path to audio file
        output_dir: Directory to save output files
        
    Returns:
        Path to generated tablature file
    """
    if output_dir is None:
        output_dir = os.path.dirname(audio_file)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Source separation with Demucs (htdemucs_6s model)
        guitar_audio_path = audio_file
        if demucs_available:
            try:
                logger.info("Performing source separation with Demucs htdemucs_6s model...")
                demucs = DemucsProcessor(model_name="htdemucs_6s")
                stems = demucs.separate_audio(audio_file, output_dir)
                
                # Try to use guitar stem if available
                if stems and 'guitar' in stems:
                    guitar_audio_path = stems['guitar']
                    logger.info(f"Using separated guitar stem: {guitar_audio_path}")
                elif stems and 'other' in stems:
                    guitar_audio_path = stems['other']
                    logger.info(f"Using 'other' stem as fallback: {guitar_audio_path}")
                else:
                    logger.warning("No suitable stem found, using original audio")
            except Exception as e:
                logger.error(f"Source separation failed: {e}")
                logger.warning("Using original audio file")
        
        # Step 2: Convert audio to MIDI using Basic Pitch
        midi_data = None
        notes_data = None
        
        if basic_pitch_available:
            try:
                logger.info("Converting audio to MIDI with Basic Pitch...")
                basic_pitch = BasicPitchProcessor()
                midi_result = basic_pitch.audio_to_midi(
                    guitar_audio_path, 
                    output_dir=output_dir,
                    save_midi=True,
                    save_notes=True
                )
                
                if midi_result and midi_result['midi_path']:
                    logger.info(f"MIDI conversion successful: {midi_result['midi_path']}")
                    midi_data = midi_result['midi_path']
                    notes_data = midi_result['notes']
            except Exception as e:
                logger.error(f"Basic Pitch MIDI conversion failed: {e}")
        
        # If Basic Pitch failed or is not available, try using MidiConverter
        if midi_data is None:
            try:
                logger.info("Using MidiConverter for audio to MIDI conversion...")
                from midi_converter import MidiConverter
                converter = MidiConverter()
                
                # Generate output file path
                base_filename = os.path.splitext(os.path.basename(guitar_audio_path))[0]
                midi_path = os.path.join(output_dir, f"{base_filename}.mid")
                
                midi_result = converter.audio_to_midi(
                    audio_file=guitar_audio_path,
                    output_file=midi_path
                )
                
                if midi_result['success']:
                    logger.info(f"MidiConverter conversion successful: {midi_result['path']}")
                    midi_data = midi_result['path']
                    notes_data = midi_result.get('notes', [])
                else:
                    logger.error(f"MidiConverter failed: {midi_result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"MidiConverter error: {e}")
        
        # Step 3: Generate tablature using available methods
        tab_path = None
        
        # Try LilyPond first if available
        try:
            # Check if LilyPond is available
            lilypond_available = False
            try:
                import lilypond_tab_generator
                lilypond_available = hasattr(lilypond_tab_generator, 'lilypond_available') and lilypond_tab_generator.lilypond_available
            except ImportError:
                logger.warning("LilyPond tab generator module not available")
            
            if lilypond_available and midi_data:
                logger.info("Generating tablature with LilyPond...")
                tab_result = lilypond_tab_generator.generate_tab_from_midi(
                    midi_file=midi_data,
                    output_dir=output_dir,
                    output_format='pdf',
                    include_standard_notation=True
                )
                
                if tab_result['success'] and tab_result['output_files']:
                    logger.info(f"LilyPond tablature generation successful: {tab_result['output_files']}")
                    # Also generate a text tab file for compatibility
                    tab_path = os.path.join(output_dir, "tablature.txt")
                    with open(tab_path, 'w') as f:
                        f.write(f"Tablature generated with LilyPond\n")
                        f.write(f"PDF file: {os.path.basename(tab_result['output_files'][0])}\n\n")
                        
                        # Add a simple ASCII representation
                        if notes_data:
                            ascii_tab = generate_tab_from_midi_notes(notes_data)
                            f.write(ascii_tab)
                    
                    return tab_result['output_files'][0]  # Return the PDF file path
                else:
                    logger.warning(f"LilyPond tablature generation failed: {tab_result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error generating tablature with LilyPond: {e}")
        
        # If LilyPond failed or is not available, try TabCNN
        if tab_path is None and tab_cnn_available:
            try:
                logger.info("Generating tablature with TabCNN...")
                tab_results = process_audio_with_tabcnn(
                    audio_file=guitar_audio_path,
                    output_dir=output_dir,
                    use_source_separation=False  # Already did source separation
                )
                
                if tab_results:
                    tab_path = os.path.join(output_dir, "tablature.txt")
                    logger.info(f"TabCNN tablature generation successful: {tab_path}")
                    return tab_path
            except Exception as e:
                logger.error(f"Error generating tablature with TabCNN: {e}")
        
        # If all else fails, generate simple tablature from MIDI notes
        if tab_path is None and notes_data:
            try:
                logger.info("Generating simple tablature from MIDI notes...")
                tab_text = generate_tab_from_midi_notes(notes_data)
                
                tab_path = os.path.join(output_dir, "tablature.txt")
                with open(tab_path, 'w') as f:
                    f.write(tab_text)
                
                logger.info(f"Simple tablature generation successful: {tab_path}")
                return tab_path
            except Exception as e:
                logger.error(f"Error generating simple tablature: {e}")
        
        # If we got here, all methods failed
        if tab_path is None:
            logger.error("Failed to generate tablature: all methods failed")
            return None
        
        return tab_path
        
    except Exception as e:
        logger.error(f"Error generating tablature: {e}")
        return None