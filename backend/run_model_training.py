#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OliTunes Run Model Training
===========================

This script orchestrates the full model training process, including:
1. Data preparation from real tabs and synthetic data
2. Training individual models (LSTM, Transformer, Hybrid)
3. Creating and training an ensemble model
4. Testing the models for tablature generation

This implements real ML models trained on actual tab data - no mock training.
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_training_runner")

# Try importing our modules, with helpful error messages if dependencies are missing
try:
    import numpy as np
    import tensorflow as tf
    logger.info(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    logger.error(f"Failed to import TensorFlow: {e}")
    logger.error("Please install TensorFlow: pip install tensorflow")
    sys.exit(1)

# Local imports
from tab_data_pipeline import TabDataPipeline
from train_real_models import RealModelTrainer
from ensemble_models import create_ensemble_model, train_ensemble_model
from model_integration import ModelIntegrator

# Constants and paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TAB_DATA_DIR = os.path.join(DATA_DIR, "tab_data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "backend", "models")
TAB_FILES_DIR = os.path.join(TAB_DATA_DIR, "tab_files")
TRAINING_DATA_DIR = os.path.join(TAB_DATA_DIR, "training")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "ensemble_model"), exist_ok=True)


def prepare_tab_data(context_window: int = 3, 
                    synthetic_ratio: float = 0.5,
                    total_samples: int = 10000) -> tuple:
    """
    Prepare training data from real tabs and synthetic data.
    
    Args:
        context_window: Number of previous notes to use as context
        synthetic_ratio: Ratio of synthetic to real data
        total_samples: Target number of total samples
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    logger.info("=== Preparing Training Data ===")
    
    # Initialize data pipeline
    pipeline = TabDataPipeline(
        context_window=context_window,
        validation_split=0.2
    )
    
    # Use the correct tab files directory path
    backend_tab_files_dir = os.path.join(PROJECT_ROOT, "backend", "data", "tab_data", "tab_files")
    
    # Check for both possible tab file locations
    tab_files_dir = None
    if os.path.exists(backend_tab_files_dir) and os.listdir(backend_tab_files_dir):
        tab_files_dir = backend_tab_files_dir
        logger.info(f"Using tab files from backend directory: {tab_files_dir}")
    elif os.path.exists(TAB_FILES_DIR) and os.listdir(TAB_FILES_DIR):
        tab_files_dir = TAB_FILES_DIR
        logger.info(f"Using tab files from data directory: {tab_files_dir}")
    else:
        logger.error(f"No tab files found in either {backend_tab_files_dir} or {TAB_FILES_DIR}")
        return None
        
    # Count tab files
    tab_files = [f for f in os.listdir(tab_files_dir) if f.endswith('.txt')]
    logger.info(f"Found {len(tab_files)} tab files in {tab_files_dir}")
    
    # Create training data
    try:
        X_train, X_val, y_train, y_val = pipeline.load_training_data()
        
        logger.info(f"Successfully prepared training data:")
        logger.info(f"  X_train shape: {X_train.shape}")
        logger.info(f"  X_val shape: {X_val.shape}")
        
        if len(X_train) < 100:
            logger.warning("Very small training dataset. Results may be poor.")
        
        return X_train, X_val, y_train, y_val
    
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        return None


def train_individual_models(X_train, X_val, y_train, y_val, 
                           context_window: int = 3,
                           epochs: int = 100,
                           batch_size: int = 64) -> Dict[str, str]:
    """
    Train individual models (LSTM, Transformer, Hybrid).
    
    Args:
        X_train: Training input features
        X_val: Validation input features
        y_train: Training targets
        y_val: Validation targets
        context_window: Context window size
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary with model types and paths
    """
    logger.info("=== Training Individual Models ===")
    
    model_types = ["lstm", "transformer", "hybrid"]
    model_paths = {}
    
    # Train each model type
    for model_type in model_types:
        logger.info(f"Training {model_type.upper()} model...")
        
        try:
            # Create model trainer
            trainer = RealModelTrainer(
                model_type=model_type,
                model_name=f"{model_type}_model",
                context_window=context_window
            )
            
            # Train the model
            history = trainer.train(
                epochs=epochs,
                batch_size=batch_size,
                patience=20  # Early stopping patience
            )
            
            # Save model path
            model_paths[model_type] = trainer.model_path
            
            # Evaluate on validation data
            metrics = trainer.evaluate(X_val, y_val)
            logger.info(f"{model_type.upper()} model validation results:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
    
    return model_paths


def train_ensemble(X_train, X_val, y_train, y_val,
                  model_paths: Dict[str, str],
                  context_window: int = 3,
                  features_per_note: int = 8,
                  epochs: int = 50,
                  batch_size: int = 64) -> str:
    """
    Create and train an ensemble model.
    
    Args:
        X_train: Training input features
        X_val: Validation input features
        y_train: Training targets
        y_val: Validation targets
        model_paths: Dictionary with model types and paths
        context_window: Context window size
        features_per_note: Features per note
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Path to the trained ensemble model
    """
    logger.info("=== Creating and Training Ensemble Model ===")
    
    # Calculate input dimension
    input_dim = context_window * features_per_note
    
    try:
        # Create ensemble model
        ensemble_model = create_ensemble_model(
            input_dim=input_dim,
            context_window=context_window,
            features_per_note=features_per_note,
            ensemble_type="weighted_average"
        )
        
        # Train ensemble model
        ensemble_path = os.path.join(MODELS_DIR, "ensemble_model", "ensemble_model.h5")
        
        history = train_ensemble_model(
            ensemble_model=ensemble_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            model_path=ensemble_path
        )
        
        logger.info(f"Ensemble model trained and saved to {ensemble_path}")
        return ensemble_path
    
    except Exception as e:
        logger.error(f"Error training ensemble model: {e}")
        return None


def test_model_integration():
    """Test the model integration with the tablature generator."""
    logger.info("=== Testing Model Integration ===")
    
    try:
        # Initialize model integrator
        integrator = ModelIntegrator()
        
        # Load the model
        success = integrator.load_model()
        
        if success:
            logger.info("Successfully loaded model for integration")
            
            # Test prediction with some example context
            context = [
                {"string": 0, "fret": 5, "midi_note": 64 + 5},
                {"string": 1, "fret": 3, "midi_note": 59 + 3},
                {"string": 2, "fret": 2, "midi_note": 55 + 2}
            ]
            
            # Predict position for a new note
            prediction = integrator.predict_position(context, midi_note=60)
            
            logger.info(f"Test prediction:")
            logger.info(f"  Note: MIDI 60 (C4)")
            logger.info(f"  Predicted position: String {prediction['string'] + 1}, Fret {prediction['fret']}")
            logger.info(f"  Confidence: {prediction.get('confidence', 0.0):.2f}")
            
            return True
        else:
            logger.error("Failed to load model for integration")
            return False
    
    except Exception as e:
        logger.error(f"Error testing model integration: {e}")
        return False


def main():
    """Main function to run the full training pipeline."""
    parser = argparse.ArgumentParser(description="OliTunes ML Model Training Pipeline")
    
    parser.add_argument("--context_window", type=int, default=3,
                       help="Context window size (number of previous notes)")
    
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    
    parser.add_argument("--synthetic_ratio", type=float, default=0.5,
                       help="Ratio of synthetic to real data")
    
    parser.add_argument("--total_samples", type=int, default=10000,
                       help="Target number of total samples")
    
    parser.add_argument("--skip_individual", action="store_true",
                       help="Skip training individual models")
    
    parser.add_argument("--skip_ensemble", action="store_true",
                       help="Skip training ensemble model")
    
    parser.add_argument("--test_only", action="store_true",
                       help="Only test model integration without training")
    
    args = parser.parse_args()
    
    # Print ASCII art header
    print(r"""
 _____  _  _ _____                          ___  _____        _       _             
|     || ||_|_   _|_ _ ___ ___ ___     ___ |  _||_   _|_ _ _|_|___ _|_|___ ___ ___ 
|  |  || || | | | | | |   | .'|_ -|___|___|| |_   | | | '_| | |   | | |   | . |  _|
|_____||_||_| |_| |___|_|_|__,|___|        |___|  |_| |_,_|_|_|_|_|_|_|_|_|_  |_|  
                                                                           |___|                                                    
""")
    
    print("\nOliTunes Real ML Model Training Pipeline")
    print("========================================\n")
    
    start_time = time.time()
    
    # Just test the model integration if requested
    if args.test_only:
        test_model_integration()
        sys.exit(0)
    
    # Step 1: Prepare training data
    data = prepare_tab_data(
        context_window=args.context_window,
        synthetic_ratio=args.synthetic_ratio,
        total_samples=args.total_samples
    )
    
    if data is None:
        logger.error("Failed to prepare training data. Exiting.")
        sys.exit(1)
    
    X_train, X_val, y_train, y_val = data
    
    # Step 2: Train individual models
    model_paths = {}
    if not args.skip_individual:
        model_paths = train_individual_models(
            X_train, X_val, y_train, y_val,
            context_window=args.context_window,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    # Step 3: Train ensemble model
    if not args.skip_ensemble and model_paths:
        ensemble_path = train_ensemble(
            X_train, X_val, y_train, y_val,
            model_paths=model_paths,
            context_window=args.context_window,
            features_per_note=8,  # STRINGS + 2
            epochs=50,  # Fewer epochs for ensemble
            batch_size=args.batch_size
        )
    
    # Step 4: Test model integration
    test_model_integration()
    
    # Print summary
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n=== Training Complete ===")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Models saved in {MODELS_DIR}")
    print("\nYou can now use these models for tablature generation!")


if __name__ == "__main__":
    main()
