import os
import numpy as np
import sys

# Set up path
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(script_dir, "../.."))
print(f"Backend directory: {backend_dir}")

# Test data loading from multiple batches
def test_load_batches(num_batches_to_check=5):
    print(f"Testing data from {num_batches_to_check} batches...")
    
    batches_dir = os.path.join(backend_dir, "data", "guitarset_training_data", "batches")
    
    if not os.path.exists(batches_dir):
        print(f"Error: Batches directory {batches_dir} does not exist.")
        return False
    
    # Get list of batch directories
    batch_dirs = [d for d in os.listdir(batches_dir) 
                 if os.path.isdir(os.path.join(batches_dir, d)) and d.startswith('batch_')]
    
    if not batch_dirs:
        print(f"Error: No batch directories found in {batches_dir}")
        return False
    
    print(f"Found {len(batch_dirs)} batch directories.")
    
    # Select a subset of batches to check
    import random
    random.seed(42)  # For reproducible results
    if len(batch_dirs) > num_batches_to_check:
        selected_batches = random.sample(batch_dirs, num_batches_to_check)
    else:
        selected_batches = batch_dirs
    
    print(f"Selected batches: {selected_batches}")
    
    results = []
    
    for batch_dir in selected_batches:
        batch_path = os.path.join(batches_dir, batch_dir)
        print(f"\nChecking batch: {batch_dir}")
        
        x_spec_path = os.path.join(batch_path, 'X_spec.npy')
        y_tab_path = os.path.join(batch_path, 'y_tab.npy')
        
        if not os.path.exists(x_spec_path) or not os.path.exists(y_tab_path):
            print(f"  Error: Missing data files in {batch_dir}")
            continue
        
        try:
            X_spec = np.load(x_spec_path)
            y_tab = np.load(y_tab_path)
            
            # Count non-zero elements in y_tab (active fret positions)
            active_positions = np.count_nonzero(y_tab)
            total_positions = y_tab.size
            active_percentage = 100 * active_positions / total_positions
            
            # Basic statistics
            x_min, x_max = X_spec.min(), X_spec.max()
            y_min, y_max = y_tab.min(), y_tab.max()
            
            print(f"  X_spec.shape = {X_spec.shape}, dtype = {X_spec.dtype}, range = [{x_min:.4f}, {x_max:.4f}]")
            print(f"  y_tab.shape = {y_tab.shape}, dtype = {y_tab.dtype}, range = [{y_min:.4f}, {y_max:.4f}]")
            print(f"  Active fret positions: {active_positions} out of {total_positions} ({active_percentage:.4f}%)")
            
            # Check for NaN values
            x_nan = np.isnan(X_spec).sum()
            y_nan = np.isnan(y_tab).sum()
            
            if x_nan > 0 or y_nan > 0:
                print(f"  WARNING: Found NaN values - X_spec: {x_nan}, y_tab: {y_nan}")
            
            # Check if this is just binary data (0s and 1s) or has other values
            unique_values = np.unique(y_tab)
            if len(unique_values) <= 5:  # Only show if there are a small number of unique values
                print(f"  Unique values in y_tab: {unique_values}")
            else:
                print(f"  y_tab has {len(unique_values)} unique values")
            
            results.append({
                'batch': batch_dir,
                'x_shape': X_spec.shape,
                'y_shape': y_tab.shape,
                'active_percentage': active_percentage,
                'has_nan': (x_nan > 0 or y_nan > 0),
                'unique_values': len(unique_values)
            })
            
        except Exception as e:
            print(f"  Error loading data: {str(e)}")
    
    # Summary of findings
    if results:
        print("\n--- Summary of Findings ---")
        total_active_pct = sum(r['active_percentage'] for r in results) / len(results)
        print(f"Average active positions: {total_active_pct:.4f}%")
        
        batches_with_data = sum(1 for r in results if r['active_percentage'] > 0)
        print(f"Batches with active positions: {batches_with_data} out of {len(results)} ({100*batches_with_data/len(results):.1f}%)")
        
        batches_with_nan = sum(1 for r in results if r['has_nan'])
        if batches_with_nan > 0:
            print(f"WARNING: {batches_with_nan} batches contain NaN values")
        
        return batches_with_data > 0
    
    return False

def test_main_data_file():
    """Check the main data file to see if it contains any non-zero values"""
    print("\nChecking main data file...")
    
    main_data_path = os.path.join(backend_dir, "data", "guitarset_training_data", "guitarset_training_data.npz")
    
    if not os.path.exists(main_data_path):
        print(f"Error: Main data file {main_data_path} does not exist.")
        return False
    
    try:
        # Load the NPZ file
        data = np.load(main_data_path)
        
        # Print the keys in the NPZ file
        print(f"Keys in the NPZ file: {data.files}")
        
        # Check each array in the file
        for key in data.files:
            arr = data[key]
            
            # Basic stats
            print(f"\nArray '{key}':")
            print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")
            
            # Check for non-zero values
            non_zero = np.count_nonzero(arr)
            total = arr.size
            non_zero_pct = 100 * non_zero / total if total > 0 else 0
            
            print(f"  Non-zero values: {non_zero} out of {total} ({non_zero_pct:.4f}%)")
            
            # Value range
            if arr.size > 0:
                print(f"  Range: [{arr.min()}, {arr.max()}]")
                
                # Sample values if non-zero
                if non_zero > 0:
                    # Find indices of some non-zero values
                    non_zero_indices = np.nonzero(arr)
                    if len(non_zero_indices[0]) > 0:
                        idx = tuple(index[0] for index in non_zero_indices)
                        print(f"  Sample non-zero value at {idx}: {arr[idx]}")
            
            # Check for NaN values
            nan_count = np.isnan(arr).sum()
            if nan_count > 0:
                print(f"  WARNING: Contains {nan_count} NaN values")
        
        return True
    
    except Exception as e:
        print(f"Error loading main data file: {str(e)}")
        return False

def test_specific_batch(batch_name="batch_0"):
    """Test a specific batch directory to verify data integrity"""
    print(f"\nTesting specific batch: {batch_name}")
    
    batch_path = os.path.join(backend_dir, "data", "guitarset_training_data", "batches", batch_name)
    
    if not os.path.exists(batch_path):
        print(f"Error: Batch directory {batch_path} does not exist.")
        return False
    
    x_spec_path = os.path.join(batch_path, 'X_spec.npy')
    x_lstm_path = os.path.join(batch_path, 'X_lstm.npy')
    y_tab_path = os.path.join(batch_path, 'y_tab.npy')
    
    if not os.path.exists(x_spec_path) or not os.path.exists(y_tab_path):
        print(f"Error: Required data files not found in {batch_path}")
        return False
    
    try:
        X_spec = np.load(x_spec_path)
        y_tab = np.load(y_tab_path)
        
        # Load X_lstm if available
        if os.path.exists(x_lstm_path):
            X_lstm = np.load(x_lstm_path)
            print(f"X_lstm.shape = {X_lstm.shape}, dtype = {X_lstm.dtype}, range = [{X_lstm.min():.4f}, {X_lstm.max():.4f}]")
        
        # Count non-zero elements in y_tab (active fret positions)
        active_positions = np.count_nonzero(y_tab)
        total_positions = y_tab.size
        active_percentage = 100 * active_positions / total_positions
        
        print(f"X_spec.shape = {X_spec.shape}, dtype = {X_spec.dtype}, range = [{X_spec.min():.4f}, {X_spec.max():.4f}]")
        print(f"y_tab.shape = {y_tab.shape}, dtype = {y_tab.dtype}, range = [{y_tab.min():.4f}, {y_tab.max():.4f}]")
        print(f"Active fret positions: {active_positions} out of {total_positions} ({active_percentage:.4f}%)")
        
        # Verify if this is binary data (just 0s and 1s)
        unique_values = np.unique(y_tab)
        if len(unique_values) <= 5:  # Only show if there are a small number of unique values
            print(f"Unique values in y_tab: {unique_values}")
        else:
            print(f"y_tab has {len(unique_values)} unique values")
        
        # Show a sample of non-zero positions
        if active_positions > 0:
            non_zero_indices = np.nonzero(y_tab)
            sample_size = min(5, len(non_zero_indices[0]))
            print("\nSample of non-zero positions:")
            for i in range(sample_size):
                idx = (non_zero_indices[0][i], non_zero_indices[1][i], non_zero_indices[2][i])
                print(f"  Position {idx}: {y_tab[idx]}")
        
        return active_positions > 0
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False

if __name__ == "__main__":
    print("TabCNN Data Loading Test")
    print("=======================")
    
    batch_success = test_load_batches(5)
    main_file_success = test_main_data_file()
    new_batch_success = test_specific_batch("batch_0")  # Check one of our newly generated batches
    
    if batch_success:
        print("\nTest completed successfully! Found valid training data in old batches.")
    else:
        print("\nWarning: No active fret positions found in old batches.")
    
    if main_file_success:
        print("Successfully examined main data file.")
    else:
        print("Failed to examine main data file.")
    
    if new_batch_success:
        print("Successfully verified new synthetic batch data with active fret positions!")
    else:
        print("Warning: New synthetic batch data does not have active fret positions.")
        
    if not batch_success and new_batch_success:
        print("\nRecommendation: Use the newly generated synthetic data for training.")
        print("The old batch data contains all zeros, but the new synthetic data has proper active fret positions.")
