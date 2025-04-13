import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.animation import FuncAnimation
import json
import time
from datetime import datetime
import argparse
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import glob
import sys

class TrainingVisualizer:
    """Class to visualize training metrics in real-time."""
    
    def __init__(self, log_dir=None, use_tensorboard=False, update_interval=1.0):
        """
        Initialize the training visualizer.
        
        Args:
            log_dir: Directory containing training logs
            use_tensorboard: Whether to use TensorBoard logs
            update_interval: Interval in seconds between updates
        """
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.update_interval = update_interval
        self.history = {
            'loss': [],
            'val_loss': [],
            'custom_accuracy': [],
            'val_custom_accuracy': [],
            'custom_precision': [],
            'val_custom_precision': [],
            'custom_recall': [],
            'val_custom_recall': [],
            'custom_f1': [],
            'val_custom_f1': [],
            'epochs': []
        }
        
        # Set up the figure and subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('TabCNN Training Progress', fontsize=16)
        
        # Configure subplots
        self.loss_line, = self.axes[0, 0].plot([], [], 'b-', label='Training Loss')
        self.val_loss_line, = self.axes[0, 0].plot([], [], 'r-', label='Validation Loss')
        self.axes[0, 0].set_title('Loss')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        self.acc_line, = self.axes[0, 1].plot([], [], 'b-', label='Training Accuracy')
        self.val_acc_line, = self.axes[0, 1].plot([], [], 'r-', label='Validation Accuracy')
        self.axes[0, 1].set_title('Accuracy')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Accuracy')
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        self.prec_line, = self.axes[1, 0].plot([], [], 'b-', label='Training Precision')
        self.val_prec_line, = self.axes[1, 0].plot([], [], 'r-', label='Validation Precision')
        self.axes[1, 0].set_title('Precision')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Precision')
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        self.recall_line, = self.axes[1, 1].plot([], [], 'b-', label='Training Recall')
        self.val_recall_line, = self.axes[1, 1].plot([], [], 'r-', label='Validation Recall')
        self.axes[1, 1].set_title('Recall')
        self.axes[1, 1].set_xlabel('Epoch')
        self.axes[1, 1].set_ylabel('Recall')
        self.axes[1, 1].legend()
        self.axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        # Add timestamp and model info
        self.timestamp_text = self.fig.text(0.01, 0.01, '', fontsize=8)
        self.model_info_text = self.fig.text(0.5, 0.01, '', fontsize=8, ha='center')
        self.status_text = self.fig.text(0.99, 0.01, '', fontsize=8, ha='right')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Initialize animation
        self.ani = None
        
    def _load_csv_logs(self):
        """Load training metrics from CSV logs."""
        csv_path = None
        
        # Find the CSV log file
        if self.log_dir:
            csv_files = glob.glob(os.path.join(self.log_dir, "*.csv"))
            if csv_files:
                csv_path = csv_files[0]
        
        if csv_path and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                for col in df.columns:
                    if col in self.history:
                        self.history[col] = df[col].tolist()
                
                # Add epoch numbers if not present
                if 'epoch' in df.columns:
                    self.history['epochs'] = df['epoch'].tolist()
                else:
                    self.history['epochs'] = list(range(1, len(df) + 1))
                
                return True
            except Exception as e:
                print(f"Error loading CSV logs: {e}")
        
        return False
    
    def _load_tensorboard_logs(self):
        """Load training metrics from TensorBoard logs."""
        if not self.use_tensorboard or not self.log_dir:
            return False
        
        tb_log_dir = os.path.join(self.log_dir, "logs")
        if not os.path.exists(tb_log_dir):
            return False
        
        try:
            # Find the most recent event file
            event_files = glob.glob(os.path.join(tb_log_dir, "events.out.tfevents.*"))
            if not event_files:
                return False
            
            latest_event_file = max(event_files, key=os.path.getctime)
            
            # Load the event file
            ea = event_accumulator.EventAccumulator(latest_event_file)
            ea.Reload()
            
            # Get available tags
            tags = ea.Tags()['scalars']
            
            # Clear history
            for key in self.history:
                if key != 'epochs':
                    self.history[key] = []
            
            # Load metrics
            max_steps = 0
            for tag in tags:
                if tag in self.history:
                    events = ea.Scalars(tag)
                    self.history[tag] = [event.value for event in events]
                    max_steps = max(max_steps, len(events))
            
            # Update epochs
            self.history['epochs'] = list(range(1, max_steps + 1))
            
            return True
        except Exception as e:
            print(f"Error loading TensorBoard logs: {e}")
            return False
    
    def _load_json_history(self):
        """Load training metrics from JSON history file."""
        json_path = None
        
        # Find the JSON history file
        if self.log_dir:
            json_files = glob.glob(os.path.join(self.log_dir, "*history*.json"))
            if json_files:
                json_path = json_files[0]
        
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    loaded_history = json.load(f)
                
                for key, values in loaded_history.items():
                    if key in self.history:
                        self.history[key] = values
                
                # Add epoch numbers if not present
                if 'epochs' not in loaded_history:
                    max_len = max([len(v) for v in loaded_history.values()])
                    self.history['epochs'] = list(range(1, max_len + 1))
                
                return True
            except Exception as e:
                print(f"Error loading JSON history: {e}")
        
        return False
    
    def _update_plot(self, frame):
        """Update the plot with the latest metrics."""
        # Load the latest metrics
        loaded = False
        
        if self.use_tensorboard:
            loaded = self._load_tensorboard_logs()
        
        if not loaded:
            loaded = self._load_csv_logs()
        
        if not loaded:
            loaded = self._load_json_history()
        
        if not loaded and not any(self.history['loss']):
            # No data loaded yet
            self.status_text.set_text("Waiting for training data...")
            return (self.loss_line, self.val_loss_line, self.acc_line, self.val_acc_line,
                    self.prec_line, self.val_prec_line, self.recall_line, self.val_recall_line,
                    self.timestamp_text, self.model_info_text, self.status_text)
        
        # Update the plots
        epochs = self.history['epochs']
        
        # Update loss plot
        if self.history['loss']:
            self.loss_line.set_data(epochs, self.history['loss'])
            if self.history['val_loss']:
                self.val_loss_line.set_data(epochs, self.history['val_loss'])
            
            self.axes[0, 0].relim()
            self.axes[0, 0].autoscale_view()
        
        # Update accuracy plot
        if self.history['custom_accuracy']:
            self.acc_line.set_data(epochs, self.history['custom_accuracy'])
            if self.history['val_custom_accuracy']:
                self.val_acc_line.set_data(epochs, self.history['val_custom_accuracy'])
            
            self.axes[0, 1].relim()
            self.axes[0, 1].autoscale_view()
        
        # Update precision plot
        if self.history['custom_precision']:
            self.prec_line.set_data(epochs, self.history['custom_precision'])
            if self.history['val_custom_precision']:
                self.val_prec_line.set_data(epochs, self.history['val_custom_precision'])
            
            self.axes[1, 0].relim()
            self.axes[1, 0].autoscale_view()
        
        # Update recall plot
        if self.history['custom_recall']:
            self.recall_line.set_data(epochs, self.history['custom_recall'])
            if self.history['val_custom_recall']:
                self.val_recall_line.set_data(epochs, self.history['val_custom_recall'])
            
            self.axes[1, 0].relim()
            self.axes[1, 0].autoscale_view()
        
        # Update timestamp and info
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.timestamp_text.set_text(f"Last updated: {current_time}")
        
        model_type = os.path.basename(self.log_dir).split('_')[1] if self.log_dir else "Unknown"
        self.model_info_text.set_text(f"Model: {model_type} | Current epoch: {max(epochs)}")
        
        # Update status
        if max(epochs) >= 50:  # Assuming 50 is the total number of epochs
            self.status_text.set_text("Training completed")
        else:
            self.status_text.set_text(f"Training in progress ({max(epochs)}/50 epochs)")
        
        return (self.loss_line, self.val_loss_line, self.acc_line, self.val_acc_line,
                self.prec_line, self.val_prec_line, self.recall_line, self.val_recall_line,
                self.timestamp_text, self.model_info_text, self.status_text)
    
    def start_visualization(self):
        """Start the visualization animation."""
        self.ani = FuncAnimation(
            self.fig, self._update_plot, frames=None,
            interval=self.update_interval * 1000,  # Convert to milliseconds
            blit=True, repeat=True
        )
        
        plt.show()

def find_latest_model_dir(base_dir=None):
    """Find the latest model directory based on timestamp."""
    if base_dir is None:
        # Get the backend directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(os.path.dirname(script_dir))
        base_dir = os.path.join(backend_dir, "models", "tab-cnn", "trained_models")
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist.")
        return None
    
    # Find all model directories
    model_dirs = glob.glob(os.path.join(base_dir, "tabcnn_*"))
    
    if not model_dirs:
        print(f"Error: No model directories found in {base_dir}.")
        return None
    
    # Sort by creation time (newest first)
    latest_dir = max(model_dirs, key=os.path.getctime)
    
    return latest_dir

def launch_tensorboard(log_dir):
    """Launch TensorBoard for the specified log directory."""
    try:
        import subprocess
        import threading
        
        def run_tensorboard():
            tensorboard_dir = os.path.join(log_dir, "logs")
            if not os.path.exists(tensorboard_dir):
                print(f"TensorBoard log directory {tensorboard_dir} does not exist.")
                return
            
            print(f"Launching TensorBoard for {tensorboard_dir}...")
            subprocess.run([
                sys.executable, "-m", "tensorboard.main",
                "--logdir", tensorboard_dir,
                "--port", "6006"
            ])
        
        # Launch TensorBoard in a separate thread
        tb_thread = threading.Thread(target=run_tensorboard)
        tb_thread.daemon = True
        tb_thread.start()
        
        print("TensorBoard launched. Open http://localhost:6006/ in your browser.")
        
        return tb_thread
    except Exception as e:
        print(f"Error launching TensorBoard: {e}")
        return None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize TabCNN model training progress')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory containing training logs')
    parser.add_argument('--update_interval', type=float, default=1.0,
                        help='Interval in seconds between updates')
    parser.add_argument('--use_tensorboard', action='store_true',
                        help='Use TensorBoard logs for visualization')
    parser.add_argument('--launch_tensorboard', action='store_true',
                        help='Launch TensorBoard server')
    
    args = parser.parse_args()
    
    # Find the latest model directory if not specified
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = find_latest_model_dir()
        if log_dir is None:
            print("Error: Could not find a model directory.")
            return 1
    
    print(f"Using log directory: {log_dir}")
    
    # Launch TensorBoard if requested
    tb_thread = None
    if args.launch_tensorboard:
        tb_thread = launch_tensorboard(log_dir)
    
    # Start visualization
    visualizer = TrainingVisualizer(
        log_dir=log_dir,
        use_tensorboard=args.use_tensorboard,
        update_interval=args.update_interval
    )
    
    try:
        visualizer.start_visualization()
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
