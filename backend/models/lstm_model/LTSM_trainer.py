import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from datetime import datetime

# Add the parent directory to the Python path to find the unified_tab_processor2 module
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(backend_dir)

# Now import from the backend directory
try:
    from unified_tab_processor2 import UnifiedTabProcessor  # Assuming this is your consolidated file
except ImportError as e:
    print(f"Warning: Could not import UnifiedTabProcessor: {e}")
    print("Will create a standalone LSTM model instead.")

# Check for GPU availability and configure TensorFlow accordingly
def configure_gpu():
    """Configure TensorFlow to use GPU if available."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            return True
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"GPU error: {e}")
            return False
    else:
        print("No GPU found. Using CPU for training (this will be slower).")
        return False

def generate_synthetic_lstm_data(num_samples: int = 1000, timesteps: int = 50, strings: int = 6, frets: int = 21):
    """Generate synthetic data for LSTM training."""
    print(f"Generating {num_samples} synthetic training samples...")
    
    # Initialize arrays
    X = np.zeros((num_samples, timesteps, strings * frets))  # Input: raw predictions
    y = np.zeros((num_samples, timesteps, strings * frets))  # Target: clean positions
    
    for i in range(num_samples):
        for t in range(timesteps):
            # For each string, randomly select a fret
            for s in range(strings):
                if np.random.random() < 0.3:  # 30% chance of a note on this string
                    fret = np.random.randint(0, frets)
                    # Input has noise
                    X[i, t, s * frets:(s + 1) * frets] = np.random.random(frets) * 0.5
                    X[i, t, s * frets + fret] = 1.0  # Peak at correct fret
                    # Ground truth is clean
                    y[i, t, s * frets + fret] = 1.0
            # Add temporal consistency: copy previous frame with small chance of change
            if t > 0 and np.random.random() < 0.8:
                y[i, t] = y[i, t-1]
                X[i, t] = X[i, t-1] * 0.9 + np.random.random(strings * frets) * 0.1
    
    print(f"Generated synthetic data with shape X: {X.shape}, y: {y.shape}")
    return X, y

def train_lstm(data_path: str = None, epochs: int = 50, batch_size: int = 32, model_name: str = None):
    """Train the LSTM model for tablature refinement."""
    # Default data path if none provided
    if data_path is None:
        data_path = os.path.join(backend_dir, "data", "lstm_training_data")
    
    print(f"Looking for training data in: {data_path}")
    
    # Load real data if available
    if os.path.exists(data_path):
        try:
            x_path = os.path.join(data_path, 'X_lstm_train.npy')
            y_path = os.path.join(data_path, 'y_lstm_train.npy')
            
            if os.path.exists(x_path) and os.path.exists(y_path):
                print(f"Loading training data from {data_path}")
                X_train = np.load(x_path)  # Raw TabCNN outputs
                y_train = np.load(y_path)  # Ground truth tabs
                print(f"Loaded {X_train.shape[0]} training samples with shape {X_train.shape}")
            else:
                print(f"Training data files not found in {data_path}")
                print("Generating synthetic data instead...")
                X_train, y_train = generate_synthetic_lstm_data(1000)
        except Exception as e:
            print(f"Error loading training data: {e}")
            print("Falling back to synthetic data generation...")
            X_train, y_train = generate_synthetic_lstm_data(1000)
    else:
        print(f"Data path {data_path} not found")
        print("Generating synthetic data instead...")
        X_train, y_train = generate_synthetic_lstm_data(1000)

    # Create a new LSTM model directly
    print("Creating new LSTM model...")
    
    # Get the dimensions from the data
    print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")
    
    # Create a model that matches the data dimensions
    if len(X_train.shape) > 2:
        # If data is 3D (samples, timesteps, features)
        timesteps = X_train.shape[1]
        features = X_train.shape[2]
        
        lstm_model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(timesteps, features)),
            Dropout(0.3),
            Dense(y_train.shape[-1], activation='sigmoid')
        ])
        print(f"Created LSTM model for 3D input with shape (samples, {timesteps}, {features})")
    else:
        # If data is 2D (samples, features), reshape to 3D
        print("Reshaping 2D data to 3D for LSTM...")
        # Assume each sample is a single timestep
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        timesteps = 1
        features = X_train.shape[2]
        
        lstm_model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(timesteps, features)),
            Dropout(0.3),
            Dense(y_train.shape[-1], activation='sigmoid')
        ])
        print(f"Created LSTM model for reshaped 2D input with shape (samples, 1, {features})")

    # Set up TensorFlow mixed precision if GPU is available
    has_gpu = configure_gpu()
    if has_gpu:
        print("Enabling mixed precision training for better GPU performance")
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        except Exception as e:
            print(f"Could not enable mixed precision: {e}")

    # Create output directory for model
    model_output_dir = os.path.join(backend_dir, "models", "lstm_model", "saved")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Generate timestamp for model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_name is None:
        model_name = f"lstm_model_{timestamp}"
    
    model_path = os.path.join(model_output_dir, f"{model_name}.keras")
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        model_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=1
    )

    # Compile the model
    print("Compiling model...")
    lstm_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Changed from categorical_crossentropy to match sigmoid activation
        metrics=['accuracy']
    )

    # Print model summary
    lstm_model.summary()

    # Train
    print(f"Starting training with {X_train.shape[0]} samples for {epochs} epochs")
    try:
        history = lstm_model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping],
            verbose=2
        )
        
        # Save final model
        final_model_path = os.path.join(model_output_dir, f"{model_name}_final.keras")
        print(f"Saving final model to {final_model_path}")
        lstm_model.save(final_model_path)
        print(f"Training complete. Model saved to {final_model_path}")
        
        return history, lstm_model
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main function to parse arguments and run the training."""
    parser = argparse.ArgumentParser(description='Train LSTM model for tablature refinement')
    parser.add_argument('--data_path', type=str, default=None, 
                        help='Path to the directory containing training data')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Name for the saved model')
    
    args = parser.parse_args()
    
    print("Starting LSTM model training...")
    history, model = train_lstm(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model_name
    )
    
    if history is not None:
        print("LSTM training complete. Model saved successfully.")
    else:
        print("LSTM training failed. See error messages above.")

if __name__ == "__main__":
    main()