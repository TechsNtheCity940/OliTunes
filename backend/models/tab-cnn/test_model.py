import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# Get the backend directory path
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(script_dir, "../.."))

def load_latest_model():
    """Load the most recently trained model"""
    # Look in the new trained_models directory structure
    models_dir = os.path.join(backend_dir, "models", "tab-cnn", "trained_models")
    
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return None
    
    # Find all training directories
    training_dirs = [d for d in os.listdir(models_dir) 
                     if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('tabcnn_')]
    
    if not training_dirs:
        print("No trained models found.")
        return None
    
    # Sort by creation time (newest first)
    training_dirs.sort(reverse=True)
    latest_dir = os.path.join(models_dir, training_dirs[0])
    
    print(f"Loading model from {latest_dir}")
    
    # Try to load the model
    try:
        # First try to load the full model
        model_path = os.path.join(latest_dir, "tabcnn_model.keras")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"Loaded full model from {model_path}")
            return model
        
        # If that fails, try to load from checkpoints
        checkpoint_dir = os.path.join(latest_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.keras')]
            if checkpoint_files:
                # Sort by validation loss (best first)
                checkpoint_files.sort(key=lambda x: float(x.split('_')[-1].replace('.keras', '')))
                best_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
                model = tf.keras.models.load_model(best_checkpoint, compile=False)
                print(f"Loaded model from checkpoint: {best_checkpoint}")
                return model
        
        print("No model files found in the latest training directory.")
        return None
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def load_test_data(data_path=None, num_samples=5):
    """Load real test data from the training directory"""
    if data_path is None:
        data_path = os.path.join(backend_dir, "data", "guitarset_training_data")
    
    # Look for synthetic data batches
    batch_dirs = []
    for i in range(20):  # Check batches 0-19
        batch_dir = os.path.join(data_path, f"batch_{i}")
        if os.path.exists(batch_dir):
            batch_dirs.append(batch_dir)
    
    # Also check for Ultimate Guitar data
    ug_dir = os.path.join(data_path, "ultimate_guitar")
    if os.path.exists(ug_dir):
        batch_dirs.append(ug_dir)
    
    if not batch_dirs:
        print("No data batches found. Using random test data instead.")
        return generate_test_input(num_samples), None
    
    # Load some samples from each batch
    X_samples = []
    y_samples = []
    samples_per_batch = max(1, num_samples // len(batch_dirs))
    
    for batch_dir in batch_dirs:
        # Find all .npy files in the batch directory
        x_files = [f for f in os.listdir(batch_dir) if f.startswith('x_') and f.endswith('.npy')]
        y_files = [f for f in os.listdir(batch_dir) if f.startswith('y_') and f.endswith('.npy')]
        
        if not x_files or not y_files:
            continue
        
        # Load a few samples from this batch
        for i in range(min(samples_per_batch, len(x_files))):
            try:
                x_path = os.path.join(batch_dir, x_files[i])
                y_path = os.path.join(batch_dir, y_files[i])
                
                if os.path.exists(x_path) and os.path.exists(y_path):
                    x = np.load(x_path)
                    y = np.load(y_path)
                    
                    # Make sure the shapes are correct and reshape if needed
                    if len(x.shape) == 2:
                        # Add time dimension if missing
                        x = x.reshape(1, x.shape[0], x.shape[1])
                    if len(y.shape) == 2:
                        # Add time dimension if missing
                        y = y.reshape(1, y.shape[0], y.shape[1])
                    
                    X_samples.append(x)
                    y_samples.append(y)
            except Exception as e:
                print(f"Error loading sample: {str(e)}")
    
    if not X_samples:
        print("Failed to load any valid samples. Using random test data instead.")
        return generate_test_input(num_samples), None
    
    # Stack the samples into arrays
    X = np.vstack(X_samples[:num_samples])
    y = np.vstack(y_samples[:num_samples])
    
    print(f"Loaded {X.shape[0]} real data samples with shapes X:{X.shape}, y:{y.shape}")
    
    return X, y

def generate_test_input(num_samples=5, time_steps=50, mel_bins=128):
    """Generate test input data"""
    # Make sure to return data in the correct shape (batch, time_steps, mel_bins)
    return np.random.randn(num_samples, time_steps, mel_bins).astype(np.float32)

def predict_tablature(model, input_data):
    """Use the model to predict tablature positions"""
    predictions = model.predict(input_data)
    return predictions

def threshold_predictions(predictions, threshold=0.5):
    """Apply a threshold to convert probabilities to binary values"""
    return (predictions > threshold).astype(np.float32)

def visualize_tablature(predictions, sample_idx=0, time_start=0, time_end=None, save_path=None):
    """
    Visualize the predicted tablature as a guitar tab diagram
    
    Args:
        predictions: Model predictions with shape (samples, time_steps, 150)
        sample_idx: Which sample to visualize
        time_start: Starting time step
        time_end: Ending time step (if None, use time_start + 16)
        save_path: Path to save the visualization (if None, just display)
    """
    if time_end is None:
        time_end = min(time_start + 16, predictions.shape[1])
    
    # Extract the prediction for the specified sample and time range
    pred = predictions[sample_idx, time_start:time_end]
    
    # Convert from flattened 150-dim to 6x25 grid (6 strings, 25 frets)
    tab_grid = np.zeros((6, time_end - time_start, 25))
    
    for t in range(pred.shape[0]):
        for string in range(6):
            for fret in range(25):
                idx = string * 25 + fret
                if idx < pred.shape[1]:
                    tab_grid[string, t, fret] = pred[t, idx]
    
    # Create a figure with 6 subplots (one for each string)
    fig, axes = plt.subplots(6, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Guitar Tablature Visualization", fontsize=16)
    
    string_names = ["E", "A", "D", "G", "B", "e"]  # Standard tuning
    
    for string in range(6):
        ax = axes[5 - string]  # Invert to match standard tab notation (high E at top)
        
        # For each time step, find the active fret position (if any)
        for t in range(tab_grid.shape[1]):
            active_frets = np.where(tab_grid[string, t] > 0.5)[0]
            
            if len(active_frets) > 0:
                # If multiple frets are active, choose the one with highest probability
                if len(active_frets) > 1:
                    probs = [tab_grid[string, t, f] for f in active_frets]
                    active_fret = active_frets[np.argmax(probs)]
                else:
                    active_fret = active_frets[0]
                
                # Plot the fret number
                ax.text(t, 0.5, str(active_fret), ha='center', va='center', 
                        fontsize=12, fontweight='bold')
            else:
                # No active fret, plot a dash
                ax.text(t, 0.5, "-", ha='center', va='center', fontsize=12)
        
        # Set up the axis
        ax.set_ylabel(f"{string_names[5-string]} string", fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, tab_grid.shape[1] - 0.5)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_yticks([])
    
    axes[-1].set_xlabel("Time Steps", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    return fig

def visualize_as_ascii_tab(predictions, sample_idx=0, time_start=0, time_end=None):
    """
    Visualize the predicted tablature as ASCII guitar tab notation
    
    Args:
        predictions: Model predictions with shape (samples, time_steps, 150)
        sample_idx: Which sample to visualize
        time_start: Starting time step
        time_end: Ending time step (if None, use time_start + 16)
    """
    if time_end is None:
        time_end = min(time_start + 16, predictions.shape[1])
    
    # Extract the prediction for the specified sample and time range
    pred = predictions[sample_idx, time_start:time_end]
    
    # Convert from flattened 150-dim to 6x25 grid (6 strings, 25 frets)
    tab_grid = np.zeros((6, time_end - time_start, 25))
    
    for t in range(pred.shape[0]):
        for string in range(6):
            for fret in range(25):
                idx = string * 25 + fret
                if idx < pred.shape[1]:
                    tab_grid[string, t, fret] = pred[t, idx]
    
    # Create ASCII tab notation
    string_names = ["e", "B", "G", "D", "A", "E"]  # Standard tuning (high to low)
    
    print("\nGuitar Tablature (ASCII notation):")
    print("=================================")
    
    for string in range(6):
        # Print string name
        line = f"{string_names[string]}|"
        
        # For each time step, find the active fret position (if any)
        for t in range(tab_grid.shape[1]):
            active_frets = np.where(tab_grid[string, t] > 0.5)[0]
            
            if len(active_frets) > 0:
                # If multiple frets are active, choose the one with highest probability
                if len(active_frets) > 1:
                    probs = [tab_grid[string, t, f] for f in active_frets]
                    active_fret = active_frets[np.argmax(probs)]
                else:
                    active_fret = active_frets[0]
                
                # Add the fret number to the line
                fret_str = str(active_fret)
                if len(fret_str) == 1:
                    line += f"{fret_str}--"
                else:
                    line += f"{fret_str}-"
            else:
                # No active fret, add dashes
                line += "---"
        
        # Print the completed line
        print(line)
    
    print("=================================")

def main():
    """Main function to test the model"""
    parser = argparse.ArgumentParser(description='Test the TabCNN model')
    parser.add_argument('--model_path', type=str, help='Path to a specific model file to test')
    parser.add_argument('--data_path', type=str, help='Path to test data directory')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to test')
    parser.add_argument('--save_dir', type=str, help='Directory to save visualizations')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--ascii', action='store_true', help='Use ASCII tab visualization')
    args = parser.parse_args()
    
    # Load the model
    if args.model_path:
        try:
            model = tf.keras.models.load_model(args.model_path, compile=False)
            print(f"Loaded model from {args.model_path}")
        except Exception as e:
            print(f"Error loading specified model: {str(e)}")
            return
    else:
        model = load_latest_model()
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Print model summary
    model.summary()
    
    # Load test data
    X_test, y_test = load_test_data(args.data_path, args.num_samples)
    
    # Make predictions
    print("Making predictions...")
    predictions = predict_tablature(model, X_test)
    
    # Apply threshold
    binary_predictions = threshold_predictions(predictions, args.threshold)
    
    # Create save directory if needed
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Visualize each sample
    for i in range(len(X_test)):
        print(f"\nSample {i+1}/{len(X_test)}:")
        
        # Visualize the input (mel spectrogram)
        plt.figure(figsize=(10, 4))
        plt.imshow(X_test[i].T, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Input Mel Spectrogram')
        plt.xlabel('Time Steps')
        plt.ylabel('Mel Bins')
        
        if args.save_dir:
            plt.savefig(os.path.join(args.save_dir, f"sample_{i+1}_input.png"))
            plt.close()
        else:
            plt.show()
        
        # Visualize the prediction
        if args.ascii:
            visualize_as_ascii_tab(binary_predictions, sample_idx=i)
        else:
            save_path = os.path.join(args.save_dir, f"sample_{i+1}_tab.png") if args.save_dir else None
            visualize_tablature(binary_predictions, sample_idx=i, save_path=save_path)
        
        # If we have ground truth, visualize it too
        if y_test is not None:
            print("\nGround Truth:")
            if args.ascii:
                visualize_as_ascii_tab(y_test, sample_idx=i)
            else:
                save_path = os.path.join(args.save_dir, f"sample_{i+1}_ground_truth.png") if args.save_dir else None
                visualize_tablature(y_test, sample_idx=i, save_path=save_path)
        
        # Calculate accuracy if we have ground truth
        if y_test is not None:
            # Apply threshold to ground truth if needed
            binary_y_test = threshold_predictions(y_test, args.threshold)
            
            # Calculate accuracy
            correct = np.sum(binary_predictions[i] == binary_y_test[i])
            total = np.prod(binary_predictions[i].shape)
            accuracy = correct / total
            print(f"Accuracy: {accuracy:.4f} ({correct}/{total} correct)")
            
            # Calculate precision and recall
            true_positives = np.sum((binary_predictions[i] == 1) & (binary_y_test[i] == 1))
            predicted_positives = np.sum(binary_predictions[i] == 1)
            actual_positives = np.sum(binary_y_test[i] == 1)
            
            precision = true_positives / max(1, predicted_positives)
            recall = true_positives / max(1, actual_positives)
            
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
