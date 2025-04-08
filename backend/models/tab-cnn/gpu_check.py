"""
GPU availability checker for TensorFlow and CuPy
"""

import os
import sys
import platform

# Print environment information
print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")

# Try to import tensorflow and check GPU availability
print("\n=== TensorFlow GPU Check ===")
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check if TensorFlow was built with CUDA
    print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # List physical devices
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Number of GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"  {gpu}")
    
    # Check if GPU is available
    print(f"TensorFlow GPU available: {tf.test.is_gpu_available()}")
    
    # Show compute capability if GPU is available
    if len(gpus) > 0:
        try:
            for gpu in gpus:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"GPU details: {details}")
        except Exception as e:
            print(f"Error getting device details: {e}")
    
    # Try a simple GPU operation
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"Simple GPU operation result: {c}")
            print("GPU operation successful!")
    except Exception as e:
        print(f"Error running GPU operation: {e}")
        
except ImportError:
    print("TensorFlow not installed")
except Exception as e:
    print(f"Error in TensorFlow check: {e}")

# Try to import CuPy and check GPU availability
print("\n=== CuPy GPU Check ===")
try:
    import cupy as cp
    print(f"CuPy version: {cp.__version__}")
    
    # Get CUDA version
    print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    
    # Get device count and properties
    n_devices = cp.cuda.runtime.getDeviceCount()
    print(f"Number of CUDA devices: {n_devices}")
    
    for i in range(n_devices):
        props = cp.cuda.runtime.getDeviceProperties(i)
        print(f"Device {i}: {props['name'].decode()}")
        print(f"  Compute Capability: {props['major']}.{props['minor']}")
        print(f"  Total Memory: {props['totalGlobalMem'] / (1024**3):.2f} GB")
    
    # Try a simple CuPy operation
    try:
        a = cp.array([[1, 2], [3, 4]])
        b = cp.array([[5, 6], [7, 8]])
        c = cp.matmul(a, b)
        print(f"Simple CuPy operation result: {cp.asnumpy(c)}")
        print("CuPy GPU operation successful!")
    except Exception as e:
        print(f"Error running CuPy operation: {e}")
        
except ImportError:
    print("CuPy not installed")
except Exception as e:
    print(f"Error in CuPy check: {e}")

# Check CUDA environment variables
print("\n=== CUDA Environment Variables ===")
cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH', 'PATH']
for var in cuda_vars:
    print(f"{var}: {os.environ.get(var, 'Not set')}")

print("\nNOTE: If GPU is not detected, try the following:")
print("1. Make sure you have CUDA and cuDNN installed with versions compatible with TensorFlow")
print("2. Check that your GPU drivers are up to date")
print("3. Make sure the CUDA environment variables are set correctly")
print("4. Check if another process is using the GPU (and release it if needed)")
print("5. Consider using tensorflow-gpu or installing TensorFlow with GPU support")
