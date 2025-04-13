import os
import sys
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json
import re
import traceback

# Get the backend directory path
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(backend_dir)

# Import custom modules
try:
    from TAB_trainer import build_tabcnn_model, CustomAccuracy, CustomPrecision, CustomRecall, CustomF1Score, CustomAUC
except ImportError:
    # Try relative import
    sys.path.append(script_dir)
    from TAB_trainer import build_tabcnn_model, CustomAccuracy, CustomPrecision, CustomRecall, CustomF1Score, CustomAUC

def load_model(model_path):
    """
    Load a trained TabCNN model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        model: Loaded model
    """
    print(f"Loading model from {model_path}...")
    
    # Define custom objects for loading
    custom_objects = {
        'CustomAccuracy': CustomAccuracy,
        'CustomPrecision': CustomPrecision,
        'CustomRecall': CustomRecall,
        'CustomF1Score': CustomF1Score,
        'CustomAUC': CustomAUC
    }
    
    try:
        model = tf.keras.models.load_model(os.path.abspath(model_path), custom_objects=custom_objects, compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def process_audio(audio_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128, time_steps=50):
    """
    Process audio file to extract features for the model.
    
    Args:
        audio_path: Path to the audio file
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bands
        time_steps: Number of time steps
        
    Returns:
        features: Extracted features
    """
    print(f"Processing audio file: {audio_path}")
    
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
        
        # Transpose to (time_steps, n_mels)
        log_mel_spec = log_mel_spec.T
        
        # Segment the spectrogram into chunks of time_steps
        segments = []
        for i in range(0, len(log_mel_spec) - time_steps + 1, time_steps // 2):  # 50% overlap
            segment = log_mel_spec[i:i+time_steps]
            if len(segment) == time_steps:
                segments.append(segment)
        
        # Convert to numpy array
        features = np.array(segments)
        
        print(f"Extracted {len(features)} segments with shape {features[0].shape}")
        return features
    
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def predict_tablature(model, features):
    """
    Predict tablature using the model.
    
    Args:
        model: Trained model
        features: Input features
        
    Returns:
        predictions: Predicted tablature
    """
    print(f"Predicting tablature for {len(features)} segments...")
    
    try:
        # Make predictions
        predictions = model.predict(features)
        
        # Convert predictions to binary (threshold = 0.5)
        binary_predictions = (predictions > 0.5).astype(int)
        
        print(f"Predictions shape: {predictions.shape}")
        return binary_predictions
    
    except Exception as e:
        print(f"Error predicting tablature: {e}")
        return None

def convert_predictions_to_tab(predictions, time_steps=50):
    """
    Convert model predictions to tablature format.
    
    Args:
        predictions: Model predictions
        time_steps: Number of time steps
        
    Returns:
        tab: Tablature in string format
    """
    print("Converting predictions to tablature...")
    
    try:
        # Reshape predictions to (time_steps, 6, 25)
        tab_data = predictions.reshape(predictions.shape[0], time_steps, 6, 25)
        
        # String names
        string_names = ['e', 'B', 'G', 'D', 'A', 'E']
        
        # Initialize tab
        tab_lines = [[] for _ in range(6)]
        
        # Process each segment
        for segment_idx, segment in enumerate(tab_data):
            # Process each time step
            for time_idx, time_step in enumerate(segment):
                # Process each string
                for string_idx, string in enumerate(time_step):
                    # Find the fret position (argmax)
                    fret_pos = np.argmax(string)
                    
                    # If the position is 0, it means no note (represented by '-')
                    if fret_pos == 0 or string[fret_pos] <= 0.5:
                        tab_lines[string_idx].append('-')
                    else:
                        # Convert to fret number (subtract 1 because index 0 is "no note")
                        fret_num = fret_pos - 1
                        tab_lines[string_idx].append(str(fret_num))
        
        # Combine tab lines
        tab = []
        for i, string_name in enumerate(string_names):
            tab_line = f"{string_name}|{''.join(tab_lines[i])}|"
            tab.append(tab_line)
        
        return '\n'.join(tab)
    
    except Exception as e:
        print(f"Error converting predictions to tab: {e}")
        return None

def parse_tab_file(tab_file_path):
    """
    Parse a tab file to extract the tablature.
    
    Args:
        tab_file_path: Path to the tab file
        
    Returns:
        parsed_tab: Parsed tablature
    """
    print(f"Parsing tab file: {tab_file_path}")
    
    try:
        with open(tab_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract tab sections (between [tab] and [/tab])
        tab_sections = re.findall(r'\[tab\](.*?)\[/tab\]', content, re.DOTALL)
        
        if not tab_sections:
            # Try alternative format with [/tab] ending
            tab_sections = re.findall(r'(D\|.*?G\|.*?C\|.*?)\[/tab\]', content, re.DOTALL)
        
        if not tab_sections:
            # Try to find sections with standard tab notation (6 strings)
            pattern = r'([EADGBC]\|[-0-9hpb/\\~xsrto\s]+\|[\r\n]+){3,6}'
            tab_sections = re.findall(pattern, content, re.MULTILINE)
        
        if not tab_sections:
            # Try to find any section with tab-like notation
            lines = content.split('\n')
            tab_sections = []
            current_section = []
            in_tab_section = False
            
            for line in lines:
                # Check if line contains tab notation
                if re.search(r'[EADGBC]\|[-0-9hpb/\\~xsrto\s]+\|', line):
                    in_tab_section = True
                    current_section.append(line)
                elif in_tab_section and line.strip() == '':
                    # Empty line might indicate end of tab section
                    if current_section:
                        tab_sections.append('\n'.join(current_section))
                        current_section = []
                        in_tab_section = False
                elif in_tab_section:
                    current_section.append(line)
            
            # Add the last section if it exists
            if current_section:
                tab_sections.append('\n'.join(current_section))
        
        if not tab_sections:
            print("No tab sections found in the file.")
            # As a fallback, just use the entire file content
            tab_sections = [content]
        
        print(f"Found {len(tab_sections)} tab sections.")
        return tab_sections
    
    except Exception as e:
        print(f"Error parsing tab file: {e}")
        traceback.print_exc()
        return None

def compare_tabs(generated_tab, original_tab_sections):
    """
    Compare generated tab with original tab.
    
    Args:
        generated_tab: Generated tablature
        original_tab_sections: Original tablature sections
        
    Returns:
        similarity_score: Similarity score between 0 and 1
    """
    print("Comparing generated tab with original tab...")
    
    try:
        # Extract lines from generated tab
        generated_lines = generated_tab.strip().split('\n')
        
        # Initialize similarity scores for each section
        section_scores = []
        
        for section in original_tab_sections:
            # Extract lines from original tab section
            original_lines = [line.strip() for line in section.strip().split('\n') if line.strip()]
            
            # Filter out lines that don't contain tablature
            original_tab_lines = [line for line in original_lines if re.match(r'^[eEADGBb]\|', line)]
            
            if not original_tab_lines:
                continue
            
            # Calculate similarity for each string
            string_similarities = []
            
            for gen_line in generated_lines:
                gen_string = gen_line[0]  # e, B, G, D, A, or E
                
                # Find corresponding line in original tab
                for orig_line in original_tab_lines:
                    orig_string = orig_line[0]
                    
                    # If strings match, compare the content
                    if gen_string.upper() == orig_string.upper():
                        # Extract fret numbers
                        gen_frets = re.findall(r'[0-9]+', gen_line)
                        orig_frets = re.findall(r'[0-9]+', orig_line)
                        
                        # Calculate Jaccard similarity
                        if not gen_frets and not orig_frets:
                            similarity = 1.0
                        elif not gen_frets or not orig_frets:
                            similarity = 0.0
                        else:
                            gen_set = set(gen_frets)
                            orig_set = set(orig_frets)
                            intersection = len(gen_set.intersection(orig_set))
                            union = len(gen_set.union(orig_set))
                            similarity = intersection / union if union > 0 else 0.0
                        
                        string_similarities.append(similarity)
                        break
            
            # Calculate average similarity for this section
            if string_similarities:
                section_scores.append(sum(string_similarities) / len(string_similarities))
        
        # Calculate overall similarity
        if section_scores:
            overall_similarity = sum(section_scores) / len(section_scores)
            print(f"Overall similarity score: {overall_similarity:.4f}")
            return overall_similarity
        else:
            print("Could not calculate similarity score.")
            return 0.0
    
    except Exception as e:
        print(f"Error comparing tabs: {e}")
        return 0.0

def visualize_comparison(audio_path, generated_tab, original_tab_sections, output_dir):
    """
    Visualize the comparison between generated and original tabs.
    
    Args:
        audio_path: Path to the audio file
        generated_tab: Generated tablature
        original_tab_sections: Original tablature sections
        output_dir: Output directory for visualizations
    """
    print("Visualizing comparison...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot waveform
        plt.subplot(3, 1, 1)
        plt.title('Audio Waveform')
        plt.plot(np.linspace(0, len(y) / sr, len(y)), y)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot mel spectrogram
        plt.subplot(3, 1, 2)
        plt.title('Mel Spectrogram')
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        
        # Display tabs
        plt.subplot(3, 1, 3)
        plt.title('Tab Comparison')
        plt.axis('off')
        
        # Format tabs for display
        display_text = "Generated Tab:\n" + generated_tab + "\n\n"
        display_text += "Original Tab (first section):\n" + original_tab_sections[0]
        
        plt.text(0.05, 0.95, display_text, fontsize=10, family='monospace', 
                 verticalalignment='top', horizontalalignment='left',
                 transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"tab_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(output_path)
        plt.close()
        
        print(f"Visualization saved to {output_path}")
    
    except Exception as e:
        print(f"Error visualizing comparison: {e}")

def main():
    """Main function to test the model on a song."""
    parser = argparse.ArgumentParser(description='Test TabCNN model on a song')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='Path to the audio file')
    parser.add_argument('--tab_path', type=str, required=True,
                        help='Path to the tab file for comparison')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(script_dir, "test_results", 
                                      f"test_{os.path.basename(args.audio_path).split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return 1
    
    # Process audio
    features = process_audio(args.audio_path)
    if features is None:
        print("Failed to process audio. Exiting.")
        return 1
    
    # Predict tablature
    predictions = predict_tablature(model, features)
    if predictions is None:
        print("Failed to predict tablature. Exiting.")
        return 1
    
    # Convert predictions to tab
    generated_tab = convert_predictions_to_tab(predictions)
    if generated_tab is None:
        print("Failed to convert predictions to tab. Exiting.")
        return 1
    
    # Save generated tab
    tab_output_path = os.path.join(args.output_dir, "generated_tab.txt")
    with open(tab_output_path, 'w') as f:
        f.write(generated_tab)
    print(f"Generated tab saved to {tab_output_path}")
    
    # Parse original tab
    original_tab_sections = parse_tab_file(args.tab_path)
    if original_tab_sections is None:
        print("Failed to parse original tab. Exiting.")
        return 1
    
    # Compare tabs
    similarity_score = compare_tabs(generated_tab, original_tab_sections)
    
    # Save comparison results
    results = {
        'model_path': args.model_path,
        'audio_path': args.audio_path,
        'tab_path': args.tab_path,
        'similarity_score': similarity_score,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Comparison results saved to {results_path}")
    
    # Visualize comparison
    visualize_comparison(args.audio_path, generated_tab, original_tab_sections, args.output_dir)
    
    print("Test completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
