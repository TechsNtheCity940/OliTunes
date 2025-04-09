"""
Safe NPZ loader module to handle files with overlapped entries
"""
import os
import numpy as np
import zipfile
import tempfile

def safe_load_npz(file_path, required_keys=None, cache=None):
    """
    Safely loads NPZ files even when they have overlapped entries.
    
    Args:
        file_path: Path to the NPZ file
        required_keys: List of required keys to extract (if None, loads all keys)
        cache: Optional dictionary to use as a cache
        
    Returns:
        Dictionary containing the loaded arrays or None if loading failed
    """
    if cache is not None and file_path in cache:
        return cache[file_path]
        
    filename = os.path.basename(file_path)
    loaded = {}
    
    # Method 1: Try the zipfile extraction approach
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Extract the NPZ (which is a ZIP file) contents
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
            
            # Load each NPY file individually
            keys_to_load = required_keys or [os.path.splitext(f)[0] for f in os.listdir(tmp_dir) if f.endswith('.npy')]
            
            for key in keys_to_load:
                npy_path = os.path.join(tmp_dir, f"{key}.npy")
                if os.path.exists(npy_path):
                    loaded[key] = np.load(npy_path)
                elif required_keys is not None:
                    print(f"Missing required key {key} in {filename}")
                    
        # Check if we have all required keys
        if required_keys is not None and not all(key in loaded for key in required_keys):
            print(f"Missing some required keys in {filename}")
            loaded = None
    
    # Method 2: If method 1 fails, try loading with memory mapping
    except Exception as e:
        print(f"Zipfile method failed for {filename}: {str(e)}")
        try:
            loaded = {}
            with np.load(file_path, mmap_mode='r') as data:
                keys_to_load = required_keys or data.files
                for key in keys_to_load:
                    if key in data:
                        # Copy the array to avoid zipfile conflicts
                        loaded[key] = np.array(data[key])
                    elif required_keys is not None:
                        print(f"Missing required key {key} in {filename}")
            
            # Check if we have all required keys
            if required_keys is not None and not all(key in loaded for key in required_keys):
                print(f"Missing some required keys in {filename}")
                loaded = None
        
        # Method 3: Last resort - try basic loading
        except Exception as e2:
            print(f"Memory-mapped loading failed for {filename}: {str(e2)}")
            try:
                # Try basic loading without memory mapping
                data = np.load(file_path, allow_pickle=True)
                loaded = {}
                keys_to_load = required_keys or data.files
                for key in keys_to_load:
                    if key in data:
                        loaded[key] = data[key]
                    elif required_keys is not None:
                        print(f"Missing required key {key} in {filename}")
                
                # Check if we have all required keys
                if required_keys is not None and not all(key in loaded for key in required_keys):
                    print(f"Missing some required keys in {filename}")
                    loaded = None
            except Exception as e3:
                print(f"All loading methods failed for {filename}")
                loaded = None
    
    # Update cache if provided
    if cache is not None and loaded is not None:
        cache[file_path] = loaded
        
    return loaded

# Demonstration usage
if __name__ == "__main__":
    # Example usage
    print("Testing safe_npz_loader module")
    try:
        test_file = "backend/models/tab-cnn/model/saved/c 2025-04-06 082819/4/predictions.npz"
        if os.path.exists(test_file):
            data = safe_load_npz(test_file, required_keys=['repr', 'tab'])
            if data:
                print(f"Successfully loaded with keys: {list(data.keys())}")
            else:
                print("Failed to load file")
    except Exception as e:
        print(f"Test failed: {str(e)}")
