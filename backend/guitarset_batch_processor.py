import os
import json
import shutil
import logging
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
import h5py
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("guitarset_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GuitarsetBatchProcessor:
    """
    Process GuitarSet training data in batches, extract features, and save in compressed format.
    This allows for efficient processing of large datasets and subsequent removal of original files.
    """
    
    def __init__(self, source_dir, output_dir, batch_size=10, max_workers=None):
        """
        Initialize the batch processor.
        
        Args:
            source_dir (str): Directory containing the original guitarset training data
            output_dir (str): Directory to save the processed data
            batch_size (int): Number of batches to process at once
            max_workers (int): Maximum number of worker processes (defaults to CPU count)
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.max_workers = max_workers or multiprocessing.cpu_count()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize metadata
        self.metadata = {
            'total_batches': 0,
            'processed_batches': 0,
            'total_examples': 0,
            'feature_shape': None,
            'label_shape': None,
            'compression_ratio': 0,
            'original_size_bytes': 0,
            'compressed_size_bytes': 0,
            'batch_directories': []
        }
        
        # Load existing metadata if available
        metadata_path = self.output_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def discover_batches(self):
        """
        Discover all batch directories in the source directory.
        
        Returns:
            list: List of batch directory paths
        """
        batch_dirs = []
        
        # Check if there's a 'batches' subdirectory
        batches_dir = self.source_dir / 'batches'
        if batches_dir.exists() and batches_dir.is_dir():
            # Look for batch directories inside the 'batches' subdirectory
            for item in batches_dir.iterdir():
                if item.is_dir() and item.name.startswith('batch_'):
                    batch_dirs.append(item)
        else:
            # Fallback to the original behavior - look directly in source_dir
            for item in self.source_dir.iterdir():
                if item.is_dir() and item.name.startswith('batch_'):
                    batch_dirs.append(item)
                
        # Sort by batch number
        batch_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
        
        # Update metadata
        self.metadata['total_batches'] = len(batch_dirs)
        self.metadata['batch_directories'] = [str(d.relative_to(self.source_dir.parent if 'batches' in str(d) else self.source_dir)) for d in batch_dirs]
        
        return batch_dirs
    
    def process_batch(self, batch_dir):
        """
        Process a single batch directory.
        
        Args:
            batch_dir (Path): Path to the batch directory
            
        Returns:
            dict: Processing statistics
        """
        try:
            batch_name = batch_dir.name
            output_file = self.output_dir / f"{batch_name}.h5"
            
            # Skip if already processed
            if output_file.exists():
                logger.info(f"Batch {batch_name} already processed, skipping")
                return {
                    'batch': batch_name,
                    'status': 'skipped',
                    'examples': 0,
                    'original_size': 0,
                    'compressed_size': 0
                }
            
            # Calculate original size
            original_size = sum(f.stat().st_size for f in batch_dir.glob('**/*') if f.is_file())
            
            # Load the data
            X, y = self._load_batch_data(batch_dir)
            
            if X is None or y is None:
                logger.warning(f"Failed to load data from {batch_name}, skipping")
                return {
                    'batch': batch_name,
                    'status': 'failed',
                    'examples': 0,
                    'original_size': original_size,
                    'compressed_size': 0
                }
            
            # Update metadata with shapes if not set
            if self.metadata['feature_shape'] is None:
                self.metadata['feature_shape'] = list(X.shape[1:])
            if self.metadata['label_shape'] is None:
                self.metadata['label_shape'] = list(y.shape[1:])
            
            # Save to compressed HDF5
            with h5py.File(output_file, 'w') as f:
                # Use gzip compression
                f.create_dataset('features', data=X, compression='gzip', compression_opts=9)
                f.create_dataset('labels', data=y, compression='gzip', compression_opts=9)
                
                # Add metadata
                f.attrs['examples'] = X.shape[0]
                f.attrs['feature_shape'] = X.shape[1:]
                f.attrs['label_shape'] = y.shape[1:]
                f.attrs['batch'] = batch_name
            
            # Get compressed size
            compressed_size = output_file.stat().st_size
            
            # Return statistics
            return {
                'batch': batch_name,
                'status': 'success',
                'examples': X.shape[0],
                'original_size': original_size,
                'compressed_size': compressed_size
            }
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_dir.name}: {str(e)}")
            return {
                'batch': batch_dir.name,
                'status': 'error',
                'error': str(e),
                'examples': 0,
                'original_size': 0,
                'compressed_size': 0
            }
    
    def _load_batch_data(self, batch_dir):
        """
        Load data from a batch directory.
        
        Args:
            batch_dir (Path): Path to the batch directory
            
        Returns:
            tuple: (features, labels) as numpy arrays
        """
        try:
            # Look for the specific numpy files we found in the batch directories
            x_lstm_file = batch_dir / 'X_lstm.npy'
            x_spec_file = batch_dir / 'X_spec.npy'
            y_tab_file = batch_dir / 'y_tab.npy'
            
            # Check if all required files exist
            if x_lstm_file.exists() and y_tab_file.exists():
                # Load numpy files
                X_lstm = np.load(x_lstm_file)
                y_tab = np.load(y_tab_file)
                
                # If the spectrogram features exist, load them too
                if x_spec_file.exists():
                    X_spec = np.load(x_spec_file)
                    # Combine features if needed
                    # For now, we'll just use the LSTM features as our primary features
                    X = X_lstm
                else:
                    X = X_lstm
                
                logger.info(f"Successfully loaded data from {batch_dir.name}: X shape {X.shape}, y shape {y_tab.shape}")
                return X, y_tab
            
            # If the specific files weren't found, try the original approach
            # Look for numpy files with generic patterns
            feature_files = list(batch_dir.glob('*_features.npy')) + list(batch_dir.glob('X_*.npy'))
            label_files = list(batch_dir.glob('*_labels.npy')) + list(batch_dir.glob('y_*.npy'))
            
            if feature_files and label_files:
                # Load from numpy files
                X = np.load(feature_files[0])
                y = np.load(label_files[0])
                logger.info(f"Loaded generic numpy files from {batch_dir.name}")
                return X, y
            
            # Look for TensorFlow data files
            tf_files = list(batch_dir.glob('*.tfrecord'))
            if tf_files:
                # Load from TensorFlow records
                X, y = self._load_from_tfrecords(tf_files)
                return X, y
            
            # Look for JSON files
            json_files = list(batch_dir.glob('*.json'))
            if json_files:
                # Load from JSON files
                X, y = self._load_from_json(json_files)
                return X, y
            
            # If no recognized format is found, try to infer from directory structure
            return self._infer_data_from_directory(batch_dir)
            
        except Exception as e:
            logger.warning(f"Failed to load data from {batch_dir.name}, skipping")
            logger.debug(f"Error details: {str(e)}")
            return None, None
    
    def _load_from_tfrecords(self, tf_files):
        """
        Load data from TensorFlow record files.
        
        Args:
            tf_files (list): List of TensorFlow record file paths
            
        Returns:
            tuple: (features, labels) as numpy arrays
        """
        # This is a placeholder - implement based on your TFRecord schema
        features_list = []
        labels_list = []
        
        # Example parsing function - adjust to your TFRecord schema
        def _parse_function(example_proto):
            feature_description = {
                'features': tf.io.FixedLenFeature([], tf.string),
                'labels': tf.io.FixedLenFeature([], tf.string)
            }
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            features = tf.io.parse_tensor(parsed['features'], tf.float32)
            labels = tf.io.parse_tensor(parsed['labels'], tf.float32)
            return features, labels
        
        for file_path in tf_files:
            dataset = tf.data.TFRecordDataset(str(file_path))
            dataset = dataset.map(_parse_function)
            
            for features, labels in dataset:
                features_list.append(features.numpy())
                labels_list.append(labels.numpy())
        
        if not features_list:
            return None, None
            
        return np.array(features_list), np.array(labels_list)
    
    def _load_from_json(self, json_files):
        """
        Load data from JSON files.
        
        Args:
            json_files (list): List of JSON file paths
            
        Returns:
            tuple: (features, labels) as numpy arrays
        """
        features_list = []
        labels_list = []
        
        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Adapt this to your JSON structure
                if 'features' in data and 'labels' in data:
                    features_list.append(np.array(data['features']))
                    labels_list.append(np.array(data['labels']))
        
        if not features_list:
            return None, None
            
        return np.array(features_list), np.array(labels_list)
    
    def _infer_data_from_directory(self, batch_dir):
        """
        Attempt to infer data structure from directory contents.
        
        Args:
            batch_dir (Path): Path to the batch directory
            
        Returns:
            tuple: (features, labels) as numpy arrays
        """
        # This is a placeholder - implement based on your directory structure
        # For example, you might have subdirectories for each example
        
        features_list = []
        labels_list = []
        
        # Example: Look for subdirectories that might contain example data
        for subdir in batch_dir.iterdir():
            if subdir.is_dir():
                # Look for feature and label files in each example directory
                feature_file = subdir / 'features.npy'
                label_file = subdir / 'labels.npy'
                
                if feature_file.exists() and label_file.exists():
                    features_list.append(np.load(feature_file))
                    labels_list.append(np.load(label_file))
        
        if not features_list:
            return None, None
            
        return np.array(features_list), np.array(labels_list)
    
    def process_batches(self, batch_dirs=None, remove_processed=False):
        """
        Process multiple batch directories in parallel.
        
        Args:
            batch_dirs (list, optional): List of batch directories to process.
                                        If None, processes all discovered batches.
            remove_processed (bool): Whether to remove original data after processing
            
        Returns:
            dict: Processing statistics
        """
        if batch_dirs is None:
            batch_dirs = self.discover_batches()
        
        # Process in chunks of batch_size
        results = []
        
        for i in range(0, len(batch_dirs), self.batch_size):
            chunk = batch_dirs[i:i+self.batch_size]
            logger.info(f"Processing batch chunk {i//self.batch_size + 1}/{(len(batch_dirs)-1)//self.batch_size + 1} ({len(chunk)} batches)")
            
            # Process batches in parallel
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.process_batch, batch_dir) for batch_dir in chunk]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                    result = future.result()
                    results.append(result)
                    
                    # Update metadata
                    if result['status'] == 'success':
                        self.metadata['processed_batches'] += 1
                        self.metadata['total_examples'] += result['examples']
                        self.metadata['original_size_bytes'] += result['original_size']
                        self.metadata['compressed_size_bytes'] += result['compressed_size']
                    
                    # Save metadata after each batch
                    self._save_metadata()
            
            # Remove processed batches if requested
            if remove_processed:
                for result in results[-len(chunk):]:
                    if result['status'] == 'success':
                        batch_path = self.source_dir / result['batch']
                        if batch_path.exists():
                            logger.info(f"Removing processed batch: {result['batch']}")
                            shutil.rmtree(batch_path)
        
        # Calculate compression ratio
        if self.metadata['original_size_bytes'] > 0:
            self.metadata['compression_ratio'] = self.metadata['original_size_bytes'] / max(1, self.metadata['compressed_size_bytes'])
        
        # Save final metadata
        self._save_metadata()
        
        return {
            'total_batches': len(batch_dirs),
            'processed_batches': self.metadata['processed_batches'],
            'total_examples': self.metadata['total_examples'],
            'original_size_gb': self.metadata['original_size_bytes'] / (1024**3),
            'compressed_size_gb': self.metadata['compressed_size_bytes'] / (1024**3),
            'compression_ratio': self.metadata['compression_ratio'],
            'space_saved_gb': (self.metadata['original_size_bytes'] - self.metadata['compressed_size_bytes']) / (1024**3)
        }
    
    def _save_metadata(self):
        """Save metadata to JSON file."""
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load_processed_data(self, batch_indices=None):
        """
        Load processed data from HDF5 files.
        
        Args:
            batch_indices (list, optional): Indices of batches to load.
                                          If None, loads all processed batches.
            
        Returns:
            tuple: (features, labels) as numpy arrays
        """
        h5_files = list(self.output_dir.glob('batch_*.h5'))
        h5_files.sort(key=lambda x: int(x.stem.split('_')[1]))
        
        if batch_indices is not None:
            h5_files = [h5_files[i] for i in batch_indices if i < len(h5_files)]
        
        features_list = []
        labels_list = []
        
        for h5_file in tqdm(h5_files, desc="Loading processed data"):
            with h5py.File(h5_file, 'r') as f:
                features = f['features'][:]
                labels = f['labels'][:]
                
                features_list.append(features)
                labels_list.append(labels)
        
        if not features_list:
            return None, None
            
        return np.concatenate(features_list), np.concatenate(labels_list)


def main():
    parser = argparse.ArgumentParser(description='Process GuitarSet training data in batches')
    parser.add_argument('--source', default='F:/newrepos/olitunes/backend/data/guitarset_training_data',
                        help='Source directory containing original data (default: %(default)s)')
    parser.add_argument('--output', default='F:/newrepos/olitunes/processed_guitarset_data',
                        help='Output directory for processed data (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of batches to process at once')
    parser.add_argument('--workers', type=int, default=None, help='Maximum number of worker processes')
    parser.add_argument('--remove', action='store_true', help='Remove original data after processing')
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not os.path.exists(args.source):
        logger.error(f"Source directory not found: {args.source}")
        logger.info("Please specify a valid source directory using the --source argument")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    processor = GuitarsetBatchProcessor(
        source_dir=args.source,
        output_dir=args.output,
        batch_size=args.batch_size,
        max_workers=args.workers
    )
    
    logger.info(f"Starting batch processing from {args.source} to {args.output}")
    
    try:
        stats = processor.process_batches(remove_processed=args.remove)
        
        logger.info("Processing complete!")
        logger.info(f"Processed {stats['processed_batches']}/{stats['total_batches']} batches")
        logger.info(f"Total examples: {stats['total_examples']}")
        logger.info(f"Original size: {stats['original_size_gb']:.2f} GB")
        logger.info(f"Compressed size: {stats['compressed_size_gb']:.2f} GB")
        logger.info(f"Compression ratio: {stats['compression_ratio']:.2f}x")
        logger.info(f"Space saved: {stats['space_saved_gb']:.2f} GB")
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        logger.info("Please check that the source directory exists and contains batch directories")
    except Exception as e:
        logger.error(f"Error during processing: {e}")


if __name__ == "__main__":
    main()
