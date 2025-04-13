import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import random

def generate_synthetic_tab_data(output_dir, num_batches=10, samples_per_batch=1000, time_steps=50, feature_dim=128, tab_dim=150):
    """
    Generate synthetic tablature data for training the TabCNN model.
    
    Args:
        output_dir: Directory to save the generated data
        num_batches: Number of batch directories to create
        samples_per_batch: Number of samples in each batch
        time_steps: Number of time steps in each sample
        feature_dim: Dimension of the input features
        tab_dim: Dimension of the tablature output (6 strings * 25 frets)
    """
    print(f"Generating {num_batches} batches with {samples_per_batch} samples each...")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create batches directory
    batches_dir = output_dir / "batches"
    os.makedirs(batches_dir, exist_ok=True)
    
    # Generate a small validation set first
    X_spec_val = np.random.randn(100, time_steps, feature_dim).astype(np.float32)
    X_lstm_val = np.random.randn(100, time_steps, feature_dim).astype(np.float32)
    
    # Generate tablature data - sparse representation with occasional non-zero values
    y_tab_val = np.zeros((100, time_steps, tab_dim), dtype=np.float32)
    
    # Add some active fret positions to make it realistic
    for i in range(100):
        for t in range(time_steps):
            # For each time step, randomly activate 0-4 fret positions
            num_active = random.randint(0, 4)
            for _ in range(num_active):
                # Randomly select a string (0-5) and fret (0-24)
                string = random.randint(0, 5)
                fret = random.randint(0, 24)
                # Calculate the index in the flattened representation
                idx = string * 25 + fret
                if idx < tab_dim:  # Safety check
                    y_tab_val[i, t, idx] = 1.0
    
    # Save validation set
    np.save(output_dir / "X_spec_train.npy", X_spec_val)
    np.save(output_dir / "X_lstm_train.npy", X_lstm_val)
    np.save(output_dir / "y_lstm_train.npy", y_tab_val)
    
    # Create a combined data file
    np.savez(output_dir / "guitarset_training_data.npz", 
             X_spec=X_spec_val, 
             X_lstm=X_lstm_val, 
             y_tab=y_tab_val)
    
    # Generate batch data
    for batch_idx in tqdm(range(num_batches)):
        batch_dir = batches_dir / f"batch_{batch_idx}"
        os.makedirs(batch_dir, exist_ok=True)
        
        # Generate input features - random normal distribution
        X_spec = np.random.randn(samples_per_batch, time_steps, feature_dim).astype(np.float32)
        X_lstm = np.random.randn(samples_per_batch, time_steps, feature_dim).astype(np.float32)
        
        # Generate tablature data - sparse representation with occasional non-zero values
        y_tab = np.zeros((samples_per_batch, time_steps, tab_dim), dtype=np.float32)
        
        # Add some active fret positions to make it realistic
        for i in range(samples_per_batch):
            for t in range(time_steps):
                # For each time step, randomly activate 0-4 fret positions
                num_active = random.randint(0, 4)
                for _ in range(num_active):
                    # Randomly select a string (0-5) and fret (0-24)
                    string = random.randint(0, 5)
                    fret = random.randint(0, 24)
                    # Calculate the index in the flattened representation
                    idx = string * 25 + fret
                    if idx < tab_dim:  # Safety check
                        y_tab[i, t, idx] = 1.0
        
        # Save batch data
        np.save(batch_dir / "X_spec.npy", X_spec)
        np.save(batch_dir / "X_lstm.npy", X_lstm)
        np.save(batch_dir / "y_tab.npy", y_tab)
        
        # Print statistics for the first batch
        if batch_idx == 0:
            active_positions = np.count_nonzero(y_tab)
            total_positions = y_tab.size
            active_percentage = 100 * active_positions / total_positions
            print(f"Batch 0 statistics:")
            print(f"  X_spec shape: {X_spec.shape}, dtype: {X_spec.dtype}")
            print(f"  X_lstm shape: {X_lstm.shape}, dtype: {X_lstm.dtype}")
            print(f"  y_tab shape: {y_tab.shape}, dtype: {y_tab.dtype}")
            print(f"  Active fret positions: {active_positions} out of {total_positions} ({active_percentage:.4f}%)")
    
    print(f"Generated {num_batches} batches of synthetic data in {output_dir}")
    print(f"Each batch contains {samples_per_batch} samples with shape:")
    print(f"  X_spec: ({samples_per_batch}, {time_steps}, {feature_dim})")
    print(f"  X_lstm: ({samples_per_batch}, {time_steps}, {feature_dim})")
    print(f"  y_tab: ({samples_per_batch}, {time_steps}, {tab_dim})")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic tablature data for TabCNN training")
    parser.add_argument("--output_dir", type=str, default="data/guitarset_training_data",
                        help="Directory to save the generated data")
    parser.add_argument("--num_batches", type=int, default=20,
                        help="Number of batch directories to create")
    parser.add_argument("--samples_per_batch", type=int, default=1000,
                        help="Number of samples in each batch")
    parser.add_argument("--time_steps", type=int, default=50,
                        help="Number of time steps in each sample")
    parser.add_argument("--feature_dim", type=int, default=128,
                        help="Dimension of the input features")
    parser.add_argument("--tab_dim", type=int, default=150,
                        help="Dimension of the tablature output (6 strings * 25 frets)")
    
    args = parser.parse_args()
    
    # Get absolute path for output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    output_dir = os.path.join(backend_dir, args.output_dir)
    
    generate_synthetic_tab_data(
        output_dir=output_dir,
        num_batches=args.num_batches,
        samples_per_batch=args.samples_per_batch,
        time_steps=args.time_steps,
        feature_dim=args.feature_dim,
        tab_dim=args.tab_dim
    )

if __name__ == "__main__":
    main()
