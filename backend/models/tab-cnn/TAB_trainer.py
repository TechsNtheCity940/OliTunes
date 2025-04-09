import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sys

# Add the backend directory to the path to import from interpreter
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(backend_dir)
from interpreter import TabCNNProcessor

# GPU configuration function
def configure_gpu():
    """Configure TensorFlow to use GPU if available, with memory growth enabled."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU acceleration enabled. Found {len(gpus)} GPU(s).")
            return True
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    print("No GPU found. Using CPU for training (this will be slower).")
    return False

# Configure GPU at module import time
has_gpu = configure_gpu()

# Synthetic data generation (simplified)
def generate_synthetic_data(num_samples: int, frames: int = 9):
    """Generate synthetic data for training when real data is not available"""
    X = np.random.randn(num_samples, 128, frames, 1)  # Mel-spectrogram-like input
    y_note = np.random.randint(0, 2, size=(num_samples, 6 * 21))  # Note presence (binary)
    y_fret = np.zeros((num_samples, 6 * 21))  # Fret positions
    
    for i in range(num_samples):
        for s in range(6):  # 6 strings
            # One-hot encode a random fret position for each string
            fret = np.random.randint(0, 21)
            start_idx = s * 21
            y_fret[i, start_idx + fret] = 1.0
    
    return X, [y_note, y_fret]

# Training script
def train_tabcnn(data_path: str = None, epochs: int = 50, batch_size: int = 32):
    """Train the TabCNN model using preprocessed data or synthetic data"""
    # Set default data path if not provided
    if data_path is None:
        data_path = os.path.join(backend_dir, "data", "guitarset_training_data")
    
    print(f"Looking for training data in: {data_path}")
    
    # Load real data if available
    if os.path.exists(data_path):
        try:
            X_spec_path = os.path.join(data_path, 'X_spec_train.npy')
            y_tab_path = os.path.join(data_path, 'y_lstm_train.npy')
            
            if os.path.exists(X_spec_path) and os.path.exists(y_tab_path):
                print("Loading preprocessed data...")
                X_train = np.load(X_spec_path)
                y_train = np.load(y_tab_path)
                
                # Reshape for TabCNN input
                # We need to create context windows for each frame
                print("Preparing data for TabCNN...")
                X_processed = []
                y_note = []
                y_fret = []
                
                for i in range(len(X_train)):
                    # Create context windows
                    frames_with_context = []
                    padded_spec = np.pad(X_train[i], ((4, 4), (0, 0)), mode='constant')
                    for j in range(4, padded_spec.shape[0] - 4):
                        window = padded_spec[j-4:j+5]  # 9-frame context window
                        frames_with_context.append(window)
                    
                    # Convert to correct shape for TabCNN
                    frames_with_context = np.array(frames_with_context)
                    frames_with_context = np.expand_dims(frames_with_context, axis=-1)  # Add channel dimension
                    X_processed.append(frames_with_context)
                    
                    # Process targets
                    y_frame = y_train[i]
                    # Create note presence (binary) and fret position (one-hot) targets
                    note_presence = (y_frame > 0).astype(np.float32)
                    fret_positions = np.zeros_like(y_frame)
                    
                    # For each frame and string, set the active fret to 1.0
                    for t in range(y_frame.shape[0]):
                        for s in range(6):
                            start_idx = s * 21
                            end_idx = start_idx + 21
                            if np.any(y_frame[t, start_idx:end_idx] > 0):
                                fret = np.argmax(y_frame[t, start_idx:end_idx])
                                fret_positions[t, start_idx + fret] = 1.0
                    
                    y_note.append(note_presence)
                    y_fret.append(fret_positions)
                
                X_train = np.vstack(X_processed)
                y_note = np.vstack(y_note)
                y_fret = np.vstack(y_fret)
                
                print(f"Prepared data shapes: X_train={X_train.shape}, y_note={y_note.shape}, y_fret={y_fret.shape}")
            else:
                print("Required data files not found. Generating synthetic data...")
                X_train, [y_note, y_fret] = generate_synthetic_data(1000)
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Falling back to synthetic data...")
            X_train, [y_note, y_fret] = generate_synthetic_data(1000)
    else:
        print(f"Data path {data_path} not found. Generating synthetic data...")
        X_train, [y_note, y_fret] = generate_synthetic_data(1000)
    
    # Initialize model
    print("Initializing TabCNN model...")
    tabcnn = TabCNNProcessor
    model = tabcnn.build_model()
    
    # Set up TensorFlow mixed precision if GPU is available
    if has_gpu:
        print("Enabling mixed precision training for better GPU performance")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Create save directory
    save_dir = os.path.join(backend_dir, "models", "tab-cnn", "model", "saved")
    os.makedirs(save_dir, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(save_dir, 'tabcnn_best.h5'), 
            save_best_only=True, 
            monitor='val_loss'
        ),
        EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train
    print(f"Starting training for {epochs} epochs with batch size {batch_size}...")
    history = model.fit(
        X_train, [y_note, y_fret],
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'tabcnn_final.h5')
    model.save_weights(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    
    # Save model for use in TabCNNProcessor
    tabcnn_model_dir = os.path.join(backend_dir, "models", "tab-cnn", "model", "saved", 
                                   f"c {tf.timestamp()}", "1")
    os.makedirs(tabcnn_model_dir, exist_ok=True)
    model.save_weights(os.path.join(tabcnn_model_dir, "model.weights.h5"))
    print(f"Model also saved to {tabcnn_model_dir} for use by TabCNNProcessor")
    
    return history

if __name__ == "__main__":
    print("TabCNN Training Script")
    print("=====================")
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        print(f"Using data path from command line: {data_path}")
        history = train_tabcnn(data_path=data_path)
    else:
        history = train_tabcnn()
    
    print("Training complete.")