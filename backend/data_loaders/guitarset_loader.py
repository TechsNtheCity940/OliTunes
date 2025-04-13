"""
GuitarSet Data Loader

Handles loading of GuitarSet batch files with error recovery and validation.
"""
import os
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class GuitarSetLoader:
    """Loader for GuitarSet batch files"""
    
    def __init__(self, data_dir: str):
        """
        Initialize the loader with GuitarSet data directory
        
        Args:
            data_dir: Path to GuitarSet training data directory
        """
        self.data_dir = data_dir
        
    def load_batch(self, batch_id: int) -> Optional[Dict]:
        """
        Load a single batch of GuitarSet data
        
        Args:
            batch_id: ID of the batch to load
            
        Returns:
            Dictionary containing X and y arrays, or None if failed
        """
        batch_dir = os.path.join(self.data_dir, f"batch_{batch_id}")
        
        if not os.path.exists(batch_dir):
            logger.warning(f"Batch directory not found: {batch_dir}")
            return None
            
        try:
            # Load X (features)
            X_files = [f for f in os.listdir(batch_dir) if f.startswith('x_')]
            X = np.stack([
                np.load(os.path.join(batch_dir, f))
                for f in sorted(X_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            ])
            
            # Load y (labels)
            y_files = [f for f in os.listdir(batch_dir) if f.startswith('y_')]
            y = np.stack([
                np.load(os.path.join(batch_dir, f))
                for f in sorted(y_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            ])
            
            # Validate shapes
            if X.shape[0] != y.shape[0]:
                logger.error(f"Mismatched X/y sizes in batch {batch_id}: {X.shape[0]} vs {y.shape[0]}")
                return None
                
            return {'X': X, 'y': y}
            
        except Exception as e:
            logger.error(f"Error loading batch {batch_id}: {e}")
            return None
    
    def load_all_batches(self, max_batches: int = None) -> Dict:
        """
        Load all available batches with error recovery
        
        Args:
            max_batches: Maximum number of batches to load (None for all)
            
        Returns:
            Dictionary with combined X and y arrays from all batches
        """
        all_X = []
        all_y = []
        failed_batches = []
        
        # Find how many batches exist
        batch_dirs = [d for d in os.listdir(self.data_dir) if d.startswith('batch_')]
        total_batches = len(batch_dirs)
        
        if max_batches is not None:
            total_batches = min(total_batches, max_batches)
        
        for batch_id in range(total_batches):
            batch_data = self.load_batch(batch_id)
            if batch_data is not None:
                all_X.append(batch_data['X'])
                all_y.append(batch_data['y'])
            else:
                failed_batches.append(batch_id)
        
        if failed_batches:
            logger.warning(f"Failed to load {len(failed_batches)} batches: {failed_batches}")
        
        if not all_X:
            logger.error("No batches loaded successfully")
            return None
            
        return {
            'X': np.concatenate(all_X, axis=0),
            'y': np.concatenate(all_y, axis=0),
            'failed_batches': failed_batches
        }
