import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence
import random

class DataGenerator(Sequence):
    
    def __init__(self, list_IDs, data_path="f:/newrepos/olitunes/backend/models/tab-cnn/data/spec_repr/", batch_size=128, shuffle=True, label_dim = (6,21), spec_repr="c", con_win_size=9):
        
        self.list_IDs = list_IDs
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.label_dim = label_dim
        self.spec_repr = spec_repr
        self.con_win_size = con_win_size
        self.halfwin = con_win_size // 2
        
        if self.spec_repr == "c":
            self.X_dim = (self.batch_size, 192, self.con_win_size, 1)
        elif self.spec_repr == "m":
            self.X_dim = (self.batch_size, 128, self.con_win_size, 1)
        elif self.spec_repr == "cm":
            self.X_dim = (self.batch_size, 320, self.con_win_size, 1)
        elif self.spec_repr == "s":
            self.X_dim = (self.batch_size, 1025, self.con_win_size, 1)
            
        self.y_dim = (self.batch_size, self.label_dim[0], self.label_dim[1])
        
        # Store frames for each ID to allow full sample utilization
        self.id_frames = self._preprocess_ids()
        
        self.on_epoch_end()
        
    def _preprocess_ids(self):
        """Process all IDs to detect number of frames for each sample."""
        id_frames = {}
        
        for ID in self.list_IDs:
            try:
                # determine filename and path
                data_dir = self.data_path + self.spec_repr + "/"
                filename = ID + ".npz"
                filepath = data_dir + filename
                
                # Check if file exists
                import os
                if not os.path.exists(filepath):
                    print(f"Warning: File not found during preprocessing: {filepath}")
                    id_frames[ID] = []
                    continue
                
                # Try to load the file
                try:
                    loaded = np.load(filepath, allow_pickle=True)
                    
                    if "repr" not in loaded:
                        print(f"Warning: 'repr' key missing in {filename}")
                        id_frames[ID] = []
                        continue
                    
                    repr_data = loaded["repr"]
                    
                    if repr_data.ndim == 3:
                        # This is a 3D array with multiple frames
                        n_frames = repr_data.shape[0]
                        id_frames[ID] = list(range(n_frames))
                        print(f"Found {n_frames} frames in {filename}")
                    else:
                        # Single frame case
                        id_frames[ID] = [0]
                        
                except Exception as e:
                    print(f"Error preprocessing {filepath}: {str(e)}")
                    id_frames[ID] = []
                    
            except Exception as e:
                print(f"Error processing ID {ID} during preprocessing: {str(e)}")
                id_frames[ID] = []
                
        # Create a flattened list of (ID, frame_idx) pairs for training
        self.all_samples = []
        for ID, frames in id_frames.items():
            for frame_idx in frames:
                self.all_samples.append((ID, frame_idx))
                
        print(f"Preprocessed {len(self.list_IDs)} files into {len(self.all_samples)} total frames")
        return id_frames
        
    def __len__(self):
        # number of batches per epoch (based on total frames now, not just IDs)
        return int(np.floor(float(len(self.all_samples)) / self.batch_size))
    
    def __getitem__(self, index):
        # generate indices of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # find list of samples (ID, frame_idx pairs)
        batch_samples = [self.all_samples[k] for k in indexes]
        
        # generate data
        X, y = self.__data_generation(batch_samples)
        
        return X, y
    
    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.all_samples))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, batch_samples):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.zeros(self.X_dim)
        y = np.zeros(self.y_dim)

        # Generate data
        for i, (ID, frame_idx) in enumerate(batch_samples):
            try:
                # determine filename and path
                data_dir = self.data_path + self.spec_repr + "/"
                filename = ID + ".npz"
                filepath = data_dir + filename
                
                # Check if file exists
                import os
                if not os.path.exists(filepath):
                    print(f"Warning: File not found: {filepath}, using zeros")
                    X[i,] = np.zeros(X[i,].shape)
                    y[i,] = np.zeros(y[i,].shape)
                    continue
                
                # load data with more robust handling
                try:
                    loaded = np.load(filepath, allow_pickle=True)
                    
                    # Confirm necessary keys exist
                    if "repr" not in loaded:
                        print(f"Warning: 'repr' key missing in {filename}, using zeros")
                        X[i,] = np.zeros(X[i,].shape)
                        y[i,] = np.zeros(y[i,].shape)
                        continue
                    
                    # Get the representation data
                    full_x = loaded["repr"]
                    
                    # Handle arrays of different dimensions
                    if full_x.ndim == 3:
                        # For 3D arrays, use the specified frame_idx
                        # No longer just using index 0
                        if frame_idx < full_x.shape[0]:
                            full_x = full_x[frame_idx]
                        else:
                            print(f"Warning: Frame index {frame_idx} out of bounds in {filename}, using zeros")
                            X[i,] = np.zeros(X[i,].shape)
                            y[i,] = np.zeros(y[i,].shape)
                            continue
                    elif full_x.ndim > 3:
                        print(f"Warning: Array dimension too high in {filename}: {full_x.ndim}D, using zeros")
                        X[i,] = np.zeros(X[i,].shape)
                        y[i,] = np.zeros(y[i,].shape)
                        continue
                    
                    # Ensure we have enough dimensions for context window
                    try:
                        full_x = np.pad(full_x, [(self.halfwin, self.halfwin), (0, 0)], mode='constant')
                    except Exception as pad_error:
                        print(f"Error padding array in {filename}: {str(pad_error)}, shape={full_x.shape}, using zeros")
                        X[i,] = np.zeros(X[i,].shape)
                        y[i,] = np.zeros(y[i,].shape)
                        continue
                    
                    # Extract the sample window
                    try:
                        sample_x = full_x[0 : self.con_win_size]
                        
                        # Check if sample has expected shape for swapaxes operation
                        if len(sample_x.shape) != 2:
                            print(f"Warning: Unexpected sample shape in {filename}: {sample_x.shape}, using zeros")
                            X[i,] = np.zeros(X[i,].shape)
                            y[i,] = np.zeros(y[i,].shape)
                            continue
                            
                        # Reshape the data for the model
                        X[i,] = np.expand_dims(np.swapaxes(sample_x, 0, 1), -1)
                    except Exception as e:
                        print(f"Error reshaping data from {filepath}: {str(e)}, shape={full_x.shape}, using zeros")
                        X[i,] = np.zeros(X[i,].shape)
                        y[i,] = np.zeros(y[i,].shape)
                        continue
        
                    # Store label data if available
                    try:
                        if "labels" in loaded and frame_idx < loaded["labels"].shape[0]:
                            y[i,] = loaded["labels"][frame_idx]
                        else:
                            print(f"Warning: No valid labels found in {filename} for frame {frame_idx}, using zeros")
                            y[i,] = np.zeros(y[i,].shape)
                    except Exception as e:
                        print(f"Error loading labels from {filepath}: {str(e)}, using zeros")
                        y[i,] = np.zeros(y[i,].shape)
                    
                except Exception as e:
                    print(f"Error processing file {filepath}: {str(e)}, using zeros")
                    X[i,] = np.zeros(X[i,].shape)
                    y[i,] = np.zeros(y[i,].shape)
            except Exception as e:
                print(f"Error processing ID {ID}: {str(e)}, using zeros")
                X[i,] = np.zeros(X[i,].shape)
                y[i,] = np.zeros(y[i,].shape)

        return X, y