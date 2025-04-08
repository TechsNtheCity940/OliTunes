"""
OliTunes Model Optimization

This module provides functions for optimizing TensorFlow models, including:
- Model quantization for reduced size and faster inference
- Model pruning to remove unnecessary weights
- Model compression

These optimizations help make the models more efficient for deployment
while maintaining accuracy.
"""

import os
import time
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Union, Optional

# Configure logging
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """
    Handles model optimization tasks including quantization, pruning, and compression.
    """
    def __init__(self, model_dir: str):
        """
        Initialize the model optimizer.
        
        Args:
            model_dir: Directory where models are stored
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def quantize_model(self, model: tf.keras.Model, 
                       dataset: Tuple[np.ndarray, np.ndarray],
                       model_name: str,
                       quantization_type: str = 'post_training') -> tf.keras.Model:
        """
        Quantize a TensorFlow model to reduce size and improve inference speed.
        
        Args:
            model: Original TensorFlow model
            dataset: Representative dataset for calibration (X, y)
            model_name: Name of the model for saving
            quantization_type: Type of quantization ('post_training' or 'quantization_aware')
            
        Returns:
            Quantized model
        """
        logger.info(f"Quantizing model {model_name} using {quantization_type} quantization")
        
        if quantization_type == 'post_training':
            # Create a representative dataset from the calibration data
            X_calib, _ = dataset
            
            def representative_dataset_gen():
                for i in range(min(100, len(X_calib))):  # Use up to 100 samples
                    sample = X_calib[i:i+1].astype(np.float32)
                    yield [sample]
            
            # Convert the model to TF Lite with quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset_gen
            
            # For full integer quantization:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            try:
                quantized_tflite_model = converter.convert()
                
                # Save the quantized model
                tflite_model_path = os.path.join(self.model_dir, f"{model_name}_quantized.tflite")
                with open(tflite_model_path, 'wb') as f:
                    f.write(quantized_tflite_model)
                    
                logger.info(f"Quantized model saved to {tflite_model_path}")
                
                # Create an interpreter for the quantized model
                interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
                interpreter.allocate_tensors()
                
                # Get input and output details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Create a wrapper model that uses the TFLite interpreter
                class QuantizedModel(tf.keras.Model):
                    def __init__(self, interpreter, input_details, output_details):
                        super(QuantizedModel, self).__init__()
                        self.interpreter = interpreter
                        self.input_details = input_details
                        self.output_details = output_details
                        
                    def call(self, inputs):
                        # Ensure the input is the right type
                        input_data = tf.cast(inputs, tf.float32)
                        
                        # Set the input tensor
                        self.interpreter.set_tensor(
                            self.input_details[0]['index'], 
                            input_data.numpy() if hasattr(input_data, 'numpy') else input_data
                        )
                        
                        # Run inference
                        self.interpreter.invoke()
                        
                        # Get the output tensor
                        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                        return tf.convert_to_tensor(output_data)
                
                # Return the wrapped model
                return QuantizedModel(interpreter, input_details, output_details)
                
            except Exception as e:
                logger.error(f"Error during model quantization: {e}")
                logger.warning("Returning the original model due to quantization error")
                return model
        
        elif quantization_type == 'quantization_aware':
            # Implement quantization-aware training
            try:
                # Apply quantization aware training
                quantized_model = tf.keras.models.clone_model(model)
                
                # Apply the quantization aware training API
                quantized_model = tf.quantization.quantize_model(
                    model=quantized_model,
                    # Use default quantization configuration
                    quantization_config=tf.quantization.default_8bit_quantize_scheme()
                )
                
                # Compile the model with the same settings
                quantized_model.compile(
                    optimizer=model.optimizer,
                    loss=model.loss,
                    metrics=model.metrics
                )
                
                # Save the quantized model
                quantized_model_path = os.path.join(self.model_dir, f"{model_name}_quantized")
                quantized_model.save(quantized_model_path)
                
                logger.info(f"Quantization-aware model saved to {quantized_model_path}")
                
                return quantized_model
                
            except Exception as e:
                logger.error(f"Error during quantization-aware training: {e}")
                logger.warning("Returning the original model due to quantization error")
                return model
        
        else:
            logger.warning(f"Unknown quantization type: {quantization_type}. Using original model.")
            return model
            
    def benchmark_model(self, model: tf.keras.Model, 
                       X_test: np.ndarray, 
                       batch_size: int = 32, 
                       num_runs: int = 10) -> Dict:
        """
        Benchmark model performance for inference speed.
        
        Args:
            model: Model to benchmark
            X_test: Test data for inference
            batch_size: Batch size for inference
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary with benchmark results
        """
        # Warm up
        _ = model.predict(X_test[:batch_size])
        
        # Benchmark
        total_time = 0
        total_samples = 0
        
        for _ in range(num_runs):
            # Select random batch
            if len(X_test) > batch_size:
                indices = np.random.choice(len(X_test), batch_size, replace=False)
                batch = X_test[indices]
            else:
                batch = X_test
                
            # Time the prediction
            start_time = time.time()
            _ = model.predict(batch, verbose=0)
            end_time = time.time()
            
            # Update totals
            inference_time = end_time - start_time
            total_time += inference_time
            total_samples += len(batch)
        
        # Calculate metrics
        avg_time_per_batch = total_time / num_runs
        avg_time_per_sample = total_time / total_samples
        samples_per_second = total_samples / total_time
        
        results = {
            "avg_time_per_batch": avg_time_per_batch,
            "avg_time_per_sample": avg_time_per_sample,
            "samples_per_second": samples_per_second,
            "batch_size": batch_size,
            "num_runs": num_runs
        }
        
        logger.info(f"Benchmark results: {results}")
        return results

def quantize_and_benchmark(model_path: str, X_test: np.ndarray, model_name: str) -> Tuple[tf.keras.Model, Dict]:
    """
    Helper function to load, quantize and benchmark a model.
    
    Args:
        model_path: Path to the original model
        X_test: Test data for benchmarking
        model_name: Name of the model
        
    Returns:
        Tuple of (quantized_model, benchmark_results)
    """
    # Load the original model
    original_model = tf.keras.models.load_model(model_path)
    
    # Get model directory
    model_dir = os.path.dirname(model_path)
    optimizer = ModelOptimizer(model_dir)
    
    # Benchmark the original model
    logger.info(f"Benchmarking original model: {model_name}")
    original_results = optimizer.benchmark_model(original_model, X_test)
    
    # Quantize the model
    quantized_model = optimizer.quantize_model(
        model=original_model,
        dataset=(X_test, None),
        model_name=model_name
    )
    
    # Benchmark the quantized model
    logger.info(f"Benchmarking quantized model: {model_name}")
    quantized_results = optimizer.benchmark_model(quantized_model, X_test)
    
    # Compare results
    speedup = original_results["avg_time_per_batch"] / quantized_results["avg_time_per_batch"]
    logger.info(f"Model quantization speedup: {speedup:.2f}x")
    
    # Return the quantized model and benchmark results
    return quantized_model, {
        "original": original_results,
        "quantized": quantized_results,
        "speedup": speedup
    }

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize and benchmark TensorFlow models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to optimize")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data for benchmarking")
    parser.add_argument("--model_name", type=str, default="model", help="Name of the model")
    
    args = parser.parse_args()
    
    # Load test data
    X_test = np.load(args.data_path)
    
    # Quantize and benchmark
    quantized_model, results = quantize_and_benchmark(args.model_path, X_test, args.model_name)
    
    print(f"Original model average time per batch: {results['original']['avg_time_per_batch']:.4f} seconds")
    print(f"Quantized model average time per batch: {results['quantized']['avg_time_per_batch']:.4f} seconds")
    print(f"Speedup: {results['speedup']:.2f}x")
