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
    from TabCNN import TabCNN
    from TabDataReprGen import TabDataReprGen
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
            # Create TabCNN model
            self.model = TabCNN(
                batch_size=1,  # For inference, batch size of 1
                epochs=1,      # Not used for inference
                con_win_size=9,
                spec_repr="c", # Using CQT representation
                data_path=os.path.join(backend_dir, "models", "tab-cnn", "data", "spec_repr", "c"),
                save_path=self.model_path
            )
            
            # Build the model architecture
            self.model.build_model()
            
            # Load weights directly from the specified path
            if os.path.exists(self.model_path):
                self.model.model.load_weights(self.model_path)
                logger.info(f"Loaded TabCNN weights from {self.model_path}")
                self.initialized = True
            else:
                logger.warning(f"Weights file not found at {self.model_path}")
                self.initialized = False
        except Exception as e:
            logger.error(f"Error initializing TabCNN model: {e}")
            raise
    
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
            # Create preprocessor
            preprocessor = TabDataReprGen(mode="c")  # Use CQT representation
            
            # Load audio and convert to the right format
            y, sr = librosa.load(audio_file, sr=22050, mono=True)
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            # Generate CQT representation
            repr_data = np.abs(librosa.cqt(
                y,
                hop_length=preprocessor.hop_length, 
                sr=sr, 
                n_bins=preprocessor.cqt_n_bins, 
                bins_per_octave=preprocessor.cqt_bins_per_octave
            ))
            
            # Ensure shape is compatible with model
            repr_data = np.swapaxes(repr_data, 0, 1)
            
            return repr_data
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise
    
    def predict_tablature(self, audio_representation):
        """Run inference with TabCNN model
        
        Args:
            audio_representation: Preprocessed audio representation
            
        Returns:
            Predicted tablature
        """
        if not self.initialized or self.model is None:
            logger.error("TabCNN model not initialized")
            return None
        
        try:
            # Prepare batch data with context window
            con_win_size = self.model.con_win_size
            half_win = con_win_size // 2
            padded_repr = np.pad(audio_representation, ((half_win, half_win), (0, 0)), mode='constant')
            
            frames_with_context = []
            for i in range(half_win, padded_repr.shape[0] - half_win):
                window = padded_repr[i - half_win:i + half_win + 1]
                frames_with_context.append(window)
            
            frames_with_context = np.array(frames_with_context)
            
            # Reshape for CNN input: (batch, height, width, channels)
            X = np.expand_dims(frames_with_context, axis=3)
            
            # Run prediction
            predictions = self.model.model.predict(X)
            
            return predictions
        except Exception as e:
            logger.error(f"Error predicting tablature: {e}")
            return None
    
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
    
    def __init__(self, model_name="htdemucs"):
        """Initialize Demucs processor
        
        Args:
            model_name: Demucs model to use
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
    
    def initialize_model(self):
        """Initialize Demucs model"""
        try:
            # Load pretrained model
            self.model = get_model(self.model_name)
            self.initialized = True
            logger.info(f"Loaded Demucs model: {self.model_name}")
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
        if not self.initialized or self.model is None:
            logger.error("Demucs not initialized")
            return None
            
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
            
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Load audio file with librosa
            audio, sr = librosa.load(audio_file, sr=44100, mono=False)
            
            # If audio is mono, convert to stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            
            # Convert to torch tensor
            audio_tensor = torch.tensor(audio).float()
            
            # Apply model
            with torch.no_grad():
                sources = apply_model(self.model, audio_tensor.unsqueeze(0), device="cpu")[0]
            
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
            
            logger.info(f"Separated audio into {len(stem_paths)} stems")
            return stem_paths
            
        except Exception as e:
            logger.error(f"Error separating audio: {str(e)}")
            return None

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
        output_dir = tempfile.mkdtemp()
    
    try:
        # Create processors
        demucs_processor = DemucsProcessor() if demucs_available and use_source_separation else None
        tabcnn_processor = TabCNNProcessor()
        
        # Check if TabCNN is available
        if not tab_cnn_available or not tabcnn_processor.initialized:
            logger.error("TabCNN model not available")
            return None
        
        # Perform source separation if available
        guitar_file = audio_file
        if demucs_processor and demucs_processor.initialized and use_source_separation:
            logger.info("Separating instruments using Demucs...")
            stems = demucs_processor.separate_audio(audio_file, output_dir)
            
            if stems and 'guitar' in stems:
                guitar_file = stems['guitar']
                logger.info(f"Using separated guitar track: {guitar_file}")
            else:
                logger.warning("Guitar stem not found, using original audio")
        
        # Preprocess audio
        logger.info("Preprocessing audio for TabCNN...")
        audio_repr = tabcnn_processor.preprocess_audio(guitar_file)
        
        # Predict tablature
        logger.info("Predicting tablature with TabCNN...")
        predictions = tabcnn_processor.predict_tablature(audio_repr)
        
        if predictions is None:
            logger.error("Failed to generate predictions")
            return None
        
        # Convert to tablature format
        tab_data = tabcnn_processor.convert_predictions_to_tab(predictions)
        
        visualize_tab(predictions)
        
        return tab_data
    except Exception as e:
        logger.error(f"Error processing audio with TabCNN: {e}")
        return None


def generate_tab_for_audio_file(audio_file, output_dir=None):
    """Generate tablature for an audio file
    
    Args:
        audio_file: Path to audio file
        output_dir: Directory to save output files
        
    Returns:
        Path to generated tablature file
    """
    # Process with TabCNN
    tab_data = process_audio_with_tabcnn(audio_file, output_dir)
    
    if tab_data is None:
        logger.error("Failed to generate tablature")
        return None
    
    # Create output path
    if output_dir is None:
        output_dir = os.path.dirname(audio_file)
    
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_tab.txt")
    
    # Generate ASCII tablature representation
    with open(output_file, 'w') as f:
        # Write header
        f.write("Guitar Tablature generated by OliTunes TabCNN\n\n")
        f.write("e |")
        f.write("B |")
        f.write("G |")
        f.write("D |")
        f.write("A |")
        f.write("E |")
        
        # Write tablature data
        # Group frames into measures (assuming 4/4 time)
        frames_per_measure = 16  # Adjust based on your audio's tempo and hop length
        for i in range(0, len(tab_data), frames_per_measure):
            measure_data = tab_data[i:i+frames_per_measure]
            
            # Create measure lines for each string
            measure_lines = ["e |", "B |", "G |", "D |", "A |", "E |"]
            
            # Fill in fret numbers
            for frame in measure_data:
                for string_idx, fret in enumerate(frame):
                    # Adjust string order (high to low)
                    tab_string_idx = 5 - string_idx
                    
                    if fret == -1:
                        # Not played
                        measure_lines[tab_string_idx] += "-"
                    else:
                        # Add fret number (handle double digits)
                        if fret >= 10:
                            measure_lines[tab_string_idx] += str(fret)
                        else:
                            measure_lines[tab_string_idx] += str(fret) + "-"
            
            # Add measure ending
            for idx in range(6):
                measure_lines[idx] += "|"
            
            # Write measure to file
            for line in measure_lines:
                f.write(line + "\n")
            
            # Add empty line between measures
            f.write("\n")
    
    logger.info(f"Tablature saved to {output_file}")
    return output_file


def preprocess_guitarset_data():
    """Preprocess GuitarSet data for TabCNN training"""
    try:
        # Run the parallel preprocessing script
        import subprocess
        
        # Path to the parallel preprocessing script
        script_path = os.path.join(backend_dir, 'data', 'Parallel_TabDataReprGen.py')
        
        # Run the script
        subprocess.run([sys.executable, script_path], check=True)
        
        logger.info("Successfully preprocessed GuitarSet data")
        return True
    except Exception as e:
        logger.error(f"Error preprocessing GuitarSet data: {e}")
        return False


def train_tabcnn_model():
    """Train the TabCNN model"""
    try:
        # Path to the TabCNN script
        script_path = os.path.join(backend_dir, 'models', 'tab-cnn', 'TabCNN.py')
        
        # Run the script
        subprocess.run([sys.executable, script_path], check=True)
        
        logger.info("Successfully trained TabCNN model")
        return True
    except Exception as e:
        logger.error(f"Error training TabCNN model: {e}")
        return False