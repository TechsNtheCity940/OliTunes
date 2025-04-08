''' A CNN to classify 6 fret-string positions
    at the frame level during guitar performance
    with explicit CuPy GPU acceleration
'''

from __future__ import print_function
import tensorflow.keras as keras
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, Lambda
from tensorflow.keras import backend as K
from DataGenerator import DataGenerator
import pandas as pd
import numpy as np
import datetime
from Metrics import *
import tensorflow as tf
from sklearn.model_selection import KFold
import time
import psutil
import gc
import queue
import threading
import zipfile
import tempfile
from safe_npz_loader import safe_load_npz  # Import our custom safe NPZ loader

# GPU acceleration with CuPy for data preprocessing
class GPUAccelerator:
    """
    Class to handle GPU acceleration tasks
    """
    def __init__(self):
        self.gpu_available = False
        self.gpu = None
        self.mem_pool = None
        self.peak_memory_usage = 0
        self.cache = {}  # For storing computed values
        self.pre_allocated_buffers = {}  # For reusing memory
        
        # Try to initialize GPU
        try:
            import cupy as cp
            self.cp = cp
            
            # Try to get GPU device
            self.gpu = cp.cuda.Device(0)  # Use first GPU
            self.gpu.use()
            self.mem_pool = cp.get_default_memory_pool()
            self.pinned_pool = cp.cuda.PinnedMemoryPool()
            cp.cuda.set_pinned_memory_pool(self.pinned_pool)
            self.gpu_available = True
            
            # Get GPU info
            gpu_attrs = cp.cuda.runtime.getDeviceProperties(0)
            self.gpu_name = gpu_attrs['name'].decode('utf-8') if 'name' in gpu_attrs else "Unknown GPU"
            self.total_memory = gpu_attrs['totalGlobalMem'] / (1024**2) if 'totalGlobalMem' in gpu_attrs else 0  # MB
            
            print(f"GPU acceleration enabled: {self.gpu_name} with {self.total_memory:.2f} MB memory")
            
        except Exception as e:
            print(f"GPU acceleration not available: {str(e)}")
            self.gpu_available = False
    
    def log_memory_usage(self, label=""):
        """
        Log GPU memory usage
        """
        if not self.gpu_available:
            return
        
        try:
            # Use CuPy's memory pool stats
            used_bytes = self.mem_pool.used_bytes()
            total_bytes = self.mem_pool.total_bytes()
            
            # Convert to MB
            used_mb = used_bytes / (1024 * 1024)
            total_mb = total_bytes / (1024 * 1024)
            
            # Update peak memory usage
            if used_mb > self.peak_memory_usage:
                self.peak_memory_usage = used_mb
                
            print(f"GPU Memory ({label}): {used_mb:.2f}MB used / {total_mb:.2f}MB allocated / {self.total_memory:.2f}MB total")
            return used_mb, total_mb
        except Exception as e:
            print(f"Error logging GPU memory: {str(e)}")
            return 0, 0

    def to_gpu(self, array, stream=None):
        """Transfer NumPy array to GPU if possible using asynchronous transfer if stream provided."""
        if self.gpu_available:
            try:
                # Track memory before operation
                if hasattr(self.cp.cuda, 'Device'):
                    mem_before = self.mem_pool.used_bytes()
                
                # Transfer to GPU asynchronously if stream provided
                if stream is not None and hasattr(self.cp, 'asarray'):
                    with stream:
                        result = self.cp.asarray(array, order='C')  # 'C' order for better performance
                else:
                    result = self.cp.asarray(array, order='C')
                
                # Force synchronization to accurately measure memory
                if hasattr(self.cp.cuda, 'Stream'):
                    self.cp.cuda.Stream.null.synchronize()
                
                # Check memory after operation
                if hasattr(self.cp.cuda, 'Device'):
                    mem_after = self.mem_pool.used_bytes()
                    mem_used = (mem_after - mem_before) / (1024 * 1024)  # MB
                    
                    if mem_used > self.peak_memory_usage:
                        self.peak_memory_usage = mem_used
                        
                return result
            except Exception as e:
                print(f"Error transferring array to GPU: {e}")
                return array
        return array
        
    def to_cpu(self, array, stream=None):
        """Transfer array back to CPU if it's on GPU."""
        if self.gpu_available and hasattr(array, 'get'):
            try:
                if stream is not None and hasattr(array, 'get'):
                    with stream:
                        return array.get()
                return array.get()
            except Exception as e:
                print(f"Error transferring array to CPU: {e}")
                return array
        return array
    
    def create_stream(self):
        """Create a CUDA stream for asynchronous operations."""
        if self.gpu_available and hasattr(self.cp.cuda, 'Stream'):
            return self.cp.cuda.Stream()
        return None
    
    def gpu_batch_process(self, data_list, process_func, batch_size=10):
        """Process a list of data in batches to avoid GPU memory overflow."""
        results = []
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            
            # Process each item in the batch
            batch_results = [process_func(item) for item in batch]
            results.extend(batch_results)
            
            # Clean up after each batch
            self.clear_memory(partial=True)
        
        return results
    
    def gpu_stft(self, audio_data, window_size=2048, hop_length=512, num_bins=None):
        """GPU-accelerated Short-Time Fourier Transform with optimizations."""
        if not self.gpu_available:
            # Fallback to CPU implementation
            from librosa.core import stft
            return stft(audio_data, n_fft=window_size, hop_length=hop_length)
        
        try:
            # First move data to GPU
            audio_gpu = self.to_gpu(audio_data.astype(np.float32))
            
            # Use cached window or create new one
            if window_size in self.cache:
                window = self.cache[window_size]
            else:
                window = self.cp.hanning(window_size).astype(self.cp.float32)
                self.cache[window_size] = window
            
            # Calculate number of frames
            n_frames = 1 + (len(audio_data) - window_size) // hop_length
            
            # Pre-allocate output array on GPU
            if num_bins is None:
                num_bins = window_size // 2 + 1
            
            # Create stream for asynchronous processing
            stream = self.create_stream()
            
            # Use batched processing for large audio files
            if n_frames > 100:
                # Process in smaller batches to avoid memory overflow
                stft_result = self.cp.zeros((num_bins, n_frames), dtype=self.cp.complex64)
                
                batch_size = 50  # Process 50 frames at a time
                for batch_idx in range(0, n_frames, batch_size):
                    end_idx = min(batch_idx + batch_size, n_frames)
                    batch_frames = end_idx - batch_idx
                    
                    # Process this batch of frames
                    for i in range(batch_frames):
                        frame_idx = batch_idx + i
                        start = frame_idx * hop_length
                        end = start + window_size
                        
                        # Apply window
                        windowed_frame = audio_gpu[start:end] * window
                        
                        # Compute FFT
                        fft_result = self.cp.fft.rfft(windowed_frame)
                        
                        # Store result
                        stft_result[:, frame_idx] = fft_result
                    
                    # Clear temp data after each batch
                    if hasattr(self.cp.cuda, 'Stream'):
                        stream.synchronize()
                
            else:
                # For smaller files, process all at once
                stft_result = self.cp.zeros((num_bins, n_frames), dtype=self.cp.complex64)
                
                # Vectorized approach for small files
                for i in range(n_frames):
                    start = i * hop_length
                    end = start + window_size
                    
                    # Apply window
                    windowed_frame = audio_gpu[start:end] * window
                    
                    # Compute FFT (potentially cached)
                    fft_result = self.cp.fft.rfft(windowed_frame)
                    
                    # Store result
                    stft_result[:, i] = fft_result
            
            # Synchronize and clear temporary GPU arrays
            if hasattr(self.cp.cuda, 'Stream'):
                self.cp.cuda.Stream.null.synchronize()
            
            return stft_result
        except Exception as e:
            print(f"GPU STFT failed: {e}")
            print("Falling back to CPU STFT")
            from librosa.core import stft
            return stft(audio_data, n_fft=window_size, hop_length=hop_length)

    def clear_memory(self, partial=False):
        """Clear GPU memory."""
        if self.gpu_available:
            try:
                if hasattr(self.cp, 'get_default_memory_pool'):
                    mempool = self.cp.get_default_memory_pool()
                    pinned_mempool = self.cp.get_default_pinned_memory_pool()
                    
                    if partial:
                        # Just free unused blocks
                        mempool.free_unused_blocks()
                        pinned_mempool.free_unused_blocks()
                    else:
                        # Free all blocks
                        mempool.free_all_blocks()
                        pinned_mempool.free_all_blocks()
                    
                    # Run garbage collection
                    gc.collect()
                    
                    if not partial:
                        print(f"GPU memory cleared. Peak usage was {self.peak_memory_usage:.2f} MB")
                        self.peak_memory_usage = 0
                        
                        # Reinitialize buffers after full clear
                        self._initialize_buffers()
                
                self.log_memory_usage()
                
            except Exception as e:
                print(f"Error clearing GPU memory: {e}")
                
    def gpu_matmul(self, a, b):
        """Perform matrix multiplication on GPU."""
        if self.gpu_available:
            try:
                a_gpu = self.to_gpu(a)
                b_gpu = self.to_gpu(b)
                result_gpu = self.cp.matmul(a_gpu, b_gpu)
                result = self.to_cpu(result_gpu)
                return result
            except Exception as e:
                print(f"GPU matrix multiplication failed: {e}")
                return np.matmul(a, b)
        return np.matmul(a, b)
    
    def gpu_conv(self, input_data, filters, strides=(1, 1), padding='same'):
        """Perform 2D convolution on GPU."""
        if not self.gpu_available:
            return None  # Let TensorFlow handle it instead
            
        try:
            # Create a 2D convolution layer on the CPU first
            from tensorflow.keras.layers import Conv2D
            conv_layer = Conv2D(filters=filters.shape[0], 
                                kernel_size=filters.shape[1:3],
                                strides=strides, 
                                padding=padding, 
                                use_bias=False)
            
            # Set weights manually
            conv_layer.build((None, input_data.shape[1], input_data.shape[2], input_data.shape[3]))
            conv_layer.set_weights([filters])
            
            # Run on CPU using TensorFlow
            return conv_layer(input_data).numpy()
        except Exception as e:
            print(f"GPU convolution failed: {e}")
            return None  # Let TensorFlow handle it instead

class TabCNN:
    
    def __init__(self, 
                 batch_size=64,  # Reduced from 128 for better GPU memory management
                 epochs=8,
                 con_win_size = 9,
                 spec_repr="c",
                 data_path="f:/newrepos/olitunes/backend/models/tab-cnn/data/spec_repr/",
                 id_file="f:/newrepos/olitunes/backend/models/tab-cnn/data/id.csv",
                 save_path="f:/newrepos/olitunes/backend/models/tab-cnn/model/saved/"):   
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
        self.spec_repr = spec_repr
        self.data_path = data_path
        self.id_file = id_file
        self.save_path = save_path
        
        # Initialize GPU accelerator
        self.gpu = GPUAccelerator()
        
        # Performance monitoring
        self.training_times = []
        self.preprocessing_times = []
        
        self.load_IDs()
        self.validate_data_files()
        
        self.save_folder = self.save_path + self.spec_repr + " " + datetime.datetime.now().strftime("%Y-%m-%d %H%M%S") + "/"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.log_file = self.save_folder + "log.txt"
        
        self.metrics = {}
        self.metrics["pp"] = []
        self.metrics["pr"] = []
        self.metrics["pf"] = []
        self.metrics["tp"] = []
        self.metrics["tr"] = []
        self.metrics["tf"] = []
        self.metrics["tdr"] = []
        self.metrics["data"] = ["g0","g1","g2","g3","g4","g5","mean","std dev"]
        
        if self.spec_repr == "c":
            self.input_shape = (192, self.con_win_size, 1)
        elif self.spec_repr == "m":
            self.input_shape = (128, self.con_win_size, 1)
        elif self.spec_repr == "cm":
            self.input_shape = (320, self.con_win_size, 1)
        elif self.spec_repr == "s":
            self.input_shape = (1025, self.con_win_size, 1)
            
        # these probably won't ever change
        self.num_classes = 21
        self.num_strings = 6

    def load_IDs(self):
        csv_file = self.id_file
        self.list_IDs = list(pd.read_csv(csv_file, header=None)[0])
        
    def validate_data_files(self):
        """Pre-scan all data files to identify which ones are valid and can be used for training."""
        print("Validating data files...")
        valid_IDs = []
        invalid_count = 0
        
        import numpy as np
        
        for ID in self.list_IDs:
            try:
                # determine filename and path
                data_dir = self.data_path + self.spec_repr + "/"
                filename = ID + ".npz"
                filepath = data_dir + filename
                
                # Check if file exists
                import os
                if not os.path.exists(filepath):
                    print(f"File not found: {filepath}")
                    invalid_count += 1
                    continue
                
                # Try to load the file to check if it's valid
                try:
                    # Just try to open the file to check if it's not corrupted
                    # Don't validate the content structure as that will be handled in DataGenerator
                    with np.load(filepath, allow_pickle=True) as data:
                        pass
                    
                    # If we reach here, at least the file can be opened
                    valid_IDs.append(ID)
                    
                except Exception as e:
                    print(f"Error loading {filepath}: {str(e)}")
                    invalid_count += 1
                    continue
                    
            except Exception as e:
                print(f"Error validating {ID}: {str(e)}")
                invalid_count += 1
                continue
        
        self.validated_IDs = valid_IDs
        print(f"Data validation complete: {len(valid_IDs)} valid files, {invalid_count} invalid/missing files.")
        
        # Update the original list_IDs to only contain valid IDs
        if len(valid_IDs) > 0:
            self.list_IDs = valid_IDs
            print("Using validated IDs for training.")
        else:
            print("WARNING: No valid files found! Using original ID list instead.")
            
    def split_data(self):
        """
        Splits list_IDs into five parts for five-fold cross-validation
        """
        print(f"Splitting data with {len(self.list_IDs)} IDs")
        
        # If we have no IDs, log an error
        if len(self.list_IDs) == 0:
            print("ERROR: No IDs available for splitting data. Training cannot proceed.")
            raise ValueError("No IDs found for training. Please check your data files.")
            
        n_folds = 5
        self.n_ids = len(self.list_IDs)
        self.folds = []
        skf = KFold(n_splits=n_folds, shuffle=True)
        
        self.indices = np.arange(self.n_ids)
        for train, test in skf.split(self.indices):
            self.folds.append({
                "training": train,
                "validation": test
            })
            
    def partition_data(self, fold):
        self.partition = {}
        self.partition["training"] = [self.list_IDs[i] for i in self.folds[fold]["training"]]
        self.partition["validation"] = [self.list_IDs[i] for i in self.folds[fold]["validation"]]
        
        self.training_generator = DataGenerator(self.partition['training'], 
                                                data_path=self.data_path, 
                                                batch_size=self.batch_size,  
                                                shuffle=True,
                                                spec_repr=self.spec_repr, 
                                                con_win_size=self.con_win_size)
        
        self.validation_generator = DataGenerator(self.partition['validation'], 
                                                 data_path=self.data_path, 
                                                 batch_size=self.batch_size,  
                                                 shuffle=False,
                                                 spec_repr=self.spec_repr, 
                                                 con_win_size=self.con_win_size)
        
        self.split_folder = self.save_folder + str(fold) + "/"
        if not os.path.exists(self.split_folder):
            os.makedirs(self.split_folder)
                
    def log_model(self):
        with open(self.log_file, 'w', encoding='utf-8') as fh:
            fh.write("\nbatch_size: " + str(self.batch_size))
            fh.write("\nepochs: " + str(self.epochs))
            fh.write("\nspec_repr: " + str(self.spec_repr))
            fh.write("\ndata_path: " + str(self.data_path))
            fh.write("\ncon_win_size: " + str(self.con_win_size))
            fh.write("\nid_file: " + str(self.id_file) + "\n")
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
       
    def softmax_by_string(self, t):
        sh = K.shape(t)
        string_sm = []
        for i in range(self.num_strings):
            string_sm.append(K.expand_dims(K.softmax(t[:,i,:]), axis=1))
        return K.concatenate(string_sm, axis=1)
    
    def catcross_by_string(self, target, output):
        loss = 0
        for i in range(self.num_strings):
            loss += K.categorical_crossentropy(target[:,i,:], output[:,i,:])
        return loss
    
    def avg_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
           
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))   
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes * self.num_strings)) # no activation
        model.add(Reshape((self.num_strings, self.num_classes)))
        model.add(Activation(self.softmax_by_string))

        # Use Adam optimizer for better performance
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(loss=self.catcross_by_string,
                      optimizer=optimizer,
                      metrics=[self.avg_acc])
        
        self.model = model

    def adjust_batch_size(self):
        """
        Adaptively adjust batch size based on available GPU memory
        """
        if not hasattr(self, 'gpu') or not self.gpu.gpu_available:
            print("No GPU available, using default batch size")
            return 32
        
        try:
            # Get available GPU memory
            if self.gpu.mem_pool:
                # Get total and used memory from memory pool
                total_memory = self.gpu.mem_pool.total_bytes()
                used_memory = self.gpu.mem_pool.used_bytes()
                free_memory_mb = (total_memory - used_memory) / (1024 * 1024)
                
                # Heuristic: Each sample needs approximately X MB in model
                # This is a rough estimate and may need adjustment
                memory_per_sample_mb = 150  # Conservative estimate
                
                # Calculate batch size based on available memory
                # Use only 70% of free memory to leave room for intermediate tensors
                safe_memory = free_memory_mb * 0.7
                estimated_batch_size = int(safe_memory / memory_per_sample_mb)
                
                # Constrain batch size within reasonable limits
                batch_size = max(2, min(128, estimated_batch_size))
                
                # Make sure batch size is a power of 2 for optimal performance
                batch_size = 2 ** int(np.log2(batch_size))
                
                print(f"Adaptive batch size: {batch_size} (based on {free_memory_mb:.0f}MB free memory)")
                self.batch_size = batch_size
                return batch_size
            else:
                print("Memory pool not available, using default batch size")
                return 32
        except Exception as e:
            print(f"Error adjusting batch size: {str(e)}")
            return 32

    def preprocessX(self, X):
        """Apply GPU preprocessing to input data if GPU is available."""
        start_time = time.time()
        
        if self.gpu.gpu_available:
            try:
                # Check if GPU acceleration is enabled
                if not hasattr(self, 'gpu') or not self.gpu.gpu_available:
                    # Fall back to CPU processing
                    result = X
                    return result
                    
                # Process on GPU
                X_gpu = self.gpu.to_gpu(X)
                
                # Normalize the data on GPU
                X_mean = self.gpu.cp.mean(X_gpu)
                X_std = self.gpu.cp.std(X_gpu) + 1e-8  # Avoid division by zero
                X_gpu = (X_gpu - X_mean) / X_std
                
                # Apply optional data augmentation on GPU
                if hasattr(self, 'use_augmentation') and self.use_augmentation:
                    # Add small random noise for regularization
                    noise = self.gpu.cp.random.normal(0, 0.01, X_gpu.shape).astype(self.gpu.cp.float32)
                    X_gpu = X_gpu + noise
                
                # Apply bandpass filtering if needed
                if hasattr(self, 'use_bandpass') and self.use_bandpass:
                    # Simplified bandpass filter implementation on GPU
                    # This helps focus on guitar frequency ranges
                    freq_mask = self.gpu.cp.ones_like(X_gpu)
                    # Apply mask to focus on frequencies in guitar range (80-1200Hz approx)
                    # This assumes the frequency bins are ordered
                    if X_gpu.shape[0] > 100:  # Only if we have enough frequency bins
                        # Zero out lower frequencies
                        low_bin = max(0, int(X_gpu.shape[0] * 0.05))
                        freq_mask[:low_bin] = 0.2  # Attenuate but don't completely remove
                        
                        # Zero out higher frequencies
                        high_bin = min(X_gpu.shape[0], int(X_gpu.shape[0] * 0.7))
                        freq_mask[high_bin:] = 0.2
                        
                        # Apply the mask
                        X_gpu = X_gpu * freq_mask
                
                # Transfer back to CPU
                result = self.gpu.to_cpu(X_gpu)
                
                # Clear temporary GPU memory
                del X_gpu
                if hasattr(self.gpu.cp, 'get_default_memory_pool'):
                    self.gpu.cp.get_default_memory_pool().free_all_blocks()
                
                self.preprocessing_times.append(time.time() - start_time)
                return result
            except Exception as e:
                print(f"GPU preprocessing error, falling back to CPU: {str(e)}")
                self.preprocessing_times.append(time.time() - start_time)
                return X
        return X

    def build_model_with_optimizations(self):
        """Build a CNN model optimized for GPU training without relying on TensorFlow GPU."""
        print("Building optimized CNN model...")
        
        model = Sequential()
        
        if self.spec_repr == "c" or self.spec_repr == "m" or self.spec_repr == "cm":
            # First convolutional layer - modified for better gradient flow
            model.add(Conv2D(32, kernel_size=(3, 3),
                            activation='relu',
                            padding='same',
                            input_shape=self.input_shape,
                            kernel_initializer='he_normal'))
            
            # Add batch normalization and spatial dropout for regularization
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.SpatialDropout2D(0.2))
            
            # Second convolutional layer
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.SpatialDropout2D(0.2))
            
            # Third convolutional layer
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(keras.layers.BatchNormalization())
            
            # Flatten layer
            model.add(Flatten())
            
            # Dense layers with stronger regularization
            model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
            model.add(Dropout(0.3))
            model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
            model.add(Dropout(0.3))
        elif self.spec_repr == "s":
            # Handle 's' representation with 1D convolutions
            model.add(Reshape((self.input_shape[0], self.input_shape[1]), input_shape=self.input_shape))
            model.add(Conv1D(32, kernel_size=3, activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(0.2))
            model.add(Conv1D(64, kernel_size=3, activation='relu'))
            model.add(keras.layers.MaxPooling1D(pool_size=2))
            model.add(keras.layers.BatchNormalization())
            model.add(Conv1D(128, kernel_size=3, activation='relu'))
            model.add(keras.layers.MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.3))
            
        # Output layers for each string (6 strings x 21 positions)
        for i in range(self.num_strings):
            model.add(Lambda(lambda x, i=i: x, output_shape=lambda s, i=i: s))
            model.add(Dense(self.num_classes, activation='softmax', name='string'+str(i)))
        
        # Compile with mixed precision
        # Note: We can only use mixed precision if TensorFlow has GPU support
        try:
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("Using mixed precision training")
        except Exception as e:
            print(f"Mixed precision not available: {e}")
        
        # Use a lower learning rate with decay for stability
        opt = keras.optimizers.Adam(learning_rate=0.001)
        
        # Compile the model
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        
        self.model = model
        return model
    
    def save_results_csv(self):
        """
        Save results to a CSV file
        """
        try:
            # Ensure all metric arrays are of the same length by padding with NaN
            max_length = 0
            for metric_name, metric_values in self.metrics.items():
                if isinstance(metric_values, (list, np.ndarray)) and len(metric_values) > max_length:
                    max_length = len(metric_values)
            
            if max_length == 0:
                print("No metrics to save")
                return
                
            output = {}
            # Pad all arrays to the same length
            for metric_name, metric_values in self.metrics.items():
                if isinstance(metric_values, (list, np.ndarray)) and len(metric_values) > 0:
                    # Convert to list for consistent handling
                    padded_values = list(metric_values)
                    # Pad with NaN to match max length
                    padded_values.extend([float('nan')] * (max_length - len(padded_values)))
                    output[metric_name] = padded_values
            
            # Create DataFrame only if we have valid data
            if len(output) > 0:
                df = pd.DataFrame.from_dict(output)
                
                # Save to CSV
                output_file = os.path.join(self.save_folder, 'results.csv')
                df.to_csv(output_file, index=False)
                print(f"Results saved to {output_file}")
            else:
                print("No valid metrics to save to CSV")
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def train(self):
        # Consider using adaptive batch size based on available GPU memory
        if hasattr(self, 'batch_size_adaptive') and self.batch_size_adaptive:
            self.adjust_batch_size()
        
        print(f"Training with batch size: {self.batch_size}")
        
        # Monitor system resources
        print(f"CPU usage: {psutil.cpu_percent()}%")
        print(f"RAM usage: {psutil.virtual_memory().percent}%")
        
        # Datasets - Use the proper batch size
        training_generator = DataGenerator(self.partition['training'], 
                                           data_path=self.data_path, 
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           spec_repr=self.spec_repr, 
                                           con_win_size=self.con_win_size)
        validation_generator = DataGenerator(self.partition['validation'], 
                                             data_path=self.data_path, 
                                             batch_size=self.batch_size,
                                             shuffle=False,
                                             spec_repr=self.spec_repr, 
                                             con_win_size=self.con_win_size)
        
        # Track how many valid samples we're using
        print(f"Training with {len(self.partition['training'])} samples, validating with {len(self.partition['validation'])} samples")
        print(f"Note: Some samples may be skipped due to file issues (see warnings above)")
        
        # Configure callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.0001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.split_folder + "best_model.weights.h5",
                save_best_only=True,
                monitor='val_loss'
            ),
            # Add a callback to clear GPU memory after each epoch
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.gpu.clear_memory(partial=True)
            ),
            # Add learning rate scheduler for better convergence
            tf.keras.callbacks.LambdaCallback(
                on_epoch_begin=lambda epoch, logs: print(f"\nEpoch {epoch+1}/{self.epochs} - Current LR: {K.get_value(self.model.optimizer.learning_rate):.6f}")
            )
        ]
        
        # Add TensorBoard callback if TensorBoard is available
        try:
            log_dir = self.split_folder + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                profile_batch=0  # Don't profile as it can cause memory issues
            )
            callbacks.append(tensorboard_callback)
            print(f"TensorBoard logs will be saved to {log_dir}")
        except Exception as e:
            print(f"TensorBoard callback not available: {e}")
        
        # Use cupy-accelerated preprocessing before training
        print("Starting GPU-accelerated training...")
        
        # Train model on dataset
        training_start = time.time()
        model_history = self.model.fit(
            x=training_generator,
            validation_data=validation_generator,
            epochs=self.epochs,
            verbose=1,
            callbacks=callbacks
        )
        training_end = time.time()
        
        # Record training time
        training_time = training_end - training_start
        self.training_times.append(training_time)
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Average preprocessing time per batch: {np.mean(self.preprocessing_times):.4f} seconds")
        
        # Print memory usage
        if self.gpu.gpu_available:
            print(f"Peak GPU memory usage: {self.gpu.peak_memory_usage:.2f} MB")
        print(f"CPU RAM usage: {psutil.virtual_memory().percent}%")
        
        # Print training summary
        if hasattr(model_history, 'history'):
            # Check if we have any metrics to report
            if 'string0_accuracy' in model_history.history:
                print("\nFinal training metrics:")
                for i in range(self.num_strings):
                    key = f'string{i}_accuracy'
                    if key in model_history.history:
                        print(f"  String {i+1} accuracy: {model_history.history[key][-1]:.4f}")
            
            # Overall accuracy if available
            if 'accuracy' in model_history.history:
                print(f"Overall accuracy: {model_history.history['accuracy'][-1]:.4f}")
                
            # Validation metrics
            if 'val_string0_accuracy' in model_history.history:
                print("\nFinal validation metrics:")
                for i in range(self.num_strings):
                    key = f'val_string{i}_accuracy'
                    if key in model_history.history:
                        print(f"  String {i+1} validation accuracy: {model_history.history[key][-1]:.4f}")
                        
            # Overall validation accuracy if available
            if 'val_accuracy' in model_history.history:
                print(f"Overall validation accuracy: {model_history.history['val_accuracy'][-1]:.4f}")
        
        # Clean up GPU memory
        self.gpu.clear_memory()
        
        # Save the model weights after training
        print("Saving model weights...")
        self.save_weights()
        print("Model weights saved successfully.")

    def save_weights(self):
        # In newer Keras versions, the filename must end in .weights.h5
        self.model.save_weights(self.split_folder + "model.weights.h5")
        
    def test(self):
        X_test, self.y_gt = self.validation_generator[0]
        X_test = self.preprocessX(X_test)  # Apply GPU preprocessing
        self.y_pred = self.model.predict(X_test)
        
        # Clean up GPU memory after testing
        self.gpu.clear_memory()

    def save_predictions(self):
        np.savez(self.split_folder + "predictions.npz", y_pred=self.y_pred, y_gt=self.y_gt)
        
    def evaluate(self):
        self.metrics["pp"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pr"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pf"].append(pitch_f_measure(self.y_pred, self.y_gt))
        self.metrics["tp"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tr"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tf"].append(tab_f_measure(self.y_pred, self.y_gt))
        self.metrics["tdr"].append(tab_disamb(self.y_pred, self.y_gt))
        
##################################
########### EXPERIMENT ###########
##################################
def main():
    try:
        print("Starting TabCNN training process with GPU acceleration...")
        
        # Performance metrics
        start_time = time.time()
        
        # Create TabCNN with adaptive batch sizing
        tabcnn = TabCNN(batch_size=32)  # Initial batch size, will be adjusted adaptively
        tabcnn.batch_size_adaptive = True  # Enable adaptive batch sizing
        tabcnn.use_augmentation = True  # Enable data augmentation
        tabcnn.use_bandpass = True  # Enable bandpass filtering for guitar frequencies
        
        tabcnn.split_data()
        print("Logging model...")
        
        # Use our optimized model builder instead of the standard one
        tabcnn.build_model_with_optimizations()
        tabcnn.log_model()
        
        # Track performance metrics
        fold_times = []
        fold_accuracies = []
        gpu_memories = []
        
        for fold in range(5):
            try:
                fold_start = time.time()
                print(f"\n{'='*20} FOLD {fold+1}/5 {'='*20}")
                tabcnn.partition_data(fold)
                
                # Clear GPU memory before building model
                tabcnn.gpu.clear_memory()
                
                print("Building optimized model...")
                tabcnn.build_model_with_optimizations()  
                
                print("Training with GPU acceleration...")
                tabcnn.train()
                
                print("Testing model performance...")
                tabcnn.test()
                tabcnn.save_predictions()
                tabcnn.evaluate()
                
                fold_end = time.time()
                fold_time = fold_end - fold_start
                fold_times.append(fold_time)
                
                # Record metrics for this fold
                if 'tr' in tabcnn.metrics and len(tabcnn.metrics['tr']) > 0:
                    fold_accuracies.append(tabcnn.metrics['tr'][-1])
                
                if tabcnn.gpu.gpu_available:
                    gpu_memories.append(tabcnn.gpu.peak_memory_usage)
                
                print(f"\nFold {fold+1} completed in {fold_time:.2f} seconds:")
                print(f"  Pitch Precision: {tabcnn.metrics['pp'][-1]:.4f}")
                print(f"  Pitch Recall: {tabcnn.metrics['pr'][-1]:.4f}")
                print(f"  Tab Precision: {tabcnn.metrics['tp'][-1]:.4f}")
                print(f"  Tab Recall: {tabcnn.metrics['tr'][-1]:.4f}")
                
                # Clear memory after each fold
                gc.collect()
                if tabcnn.gpu.gpu_available:
                    tabcnn.gpu.clear_memory()
                
            except Exception as e:
                print(f"Error in fold {fold+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                print("Continuing with next fold...")
                continue
        
        # Save overall results
        tabcnn.save_results_csv()
        
        # Print performance summary
        total_time = time.time() - start_time
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY:")
        print("="*50)
        print(f"Total training time: {total_time:.2f} seconds")
        if fold_times:
            print(f"Average time per fold: {np.mean(fold_times):.2f} seconds")
            print(f"Fastest fold: {np.min(fold_times):.2f} seconds")
            print(f"Slowest fold: {np.max(fold_times):.2f} seconds")
        
        if fold_accuracies:
            print(f"\nAverage tab recall across folds: {np.mean(fold_accuracies):.4f}")
            print(f"Best fold accuracy: {np.max(fold_accuracies):.4f}")
        
        if gpu_memories and tabcnn.gpu.gpu_available:
            print(f"\nGPU Memory Usage:")
            print(f"  Peak: {np.max(gpu_memories):.2f} MB")
            print(f"  Average: {np.mean(gpu_memories):.2f} MB")
        
        print("\nCPU Utilization:")
        print(f"  Current: {psutil.cpu_percent()}%")
        print(f"  RAM: {psutil.virtual_memory().percent}%")
        
        print("\nTraining Configuration:")
        print(f"  Batch size: {tabcnn.batch_size}")
        print(f"  Spectral representation: {tabcnn.spec_repr}")
        print(f"  Context window size: {tabcnn.con_win_size}")
        print(f"  Data augmentation: {hasattr(tabcnn, 'use_augmentation') and tabcnn.use_augmentation}")
        
        print("\nTabCNN training process completed. Results saved to CSV.")
        print(f"Model weights saved to: {tabcnn.save_folder}")
    except Exception as e:
        print(f"Error in TabCNN training process: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Training interrupted.")
        
if __name__ == '__main__':
    main()
