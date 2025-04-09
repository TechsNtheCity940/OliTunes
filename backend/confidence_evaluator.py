"""
OliTunes Confidence Threshold Module

This module implements confidence threshold mechanisms for model predictions.
It helps determine when to trust ML model outputs vs. when to fall back to
rule-based algorithms.
"""

import os
import numpy as np
import tensorflow as tf
import logging
from typing import Dict, List, Tuple, Union, Optional
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

class ConfidenceEvaluator:
    """
    Evaluates the confidence of model predictions and determines when to use them.
    """
    def __init__(self, threshold: float = 0.6, 
                 min_threshold: float = 0.4,
                 fallback_threshold: float = 0.2):
        """
        Initialize confidence evaluator.
        
        Args:
            threshold: Minimum confidence to accept a prediction (0.0-1.0)
            min_threshold: Minimum threshold to consider a prediction useful
            fallback_threshold: Threshold below which to ignore predictions entirely
        """
        self.threshold = threshold
        self.min_threshold = min_threshold
        self.fallback_threshold = fallback_threshold
        self.confidence_history = defaultdict(list)
        self.max_history_size = 100
        self.calibration_data = None

    def get_prediction_confidence(self, prediction: np.ndarray) -> float:
        """
        Calculate confidence score for a prediction.
        
        Args:
            prediction: Model output prediction array
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # For one-hot encoded outputs, use the max probability
        if len(prediction.shape) > 1 and prediction.shape[1] > 1:
            # Get max probability
            confidence = float(np.max(prediction))
        else:
            # For regression outputs, use a different metric
            # Scale based on expected distribution
            confidence = float(1.0 - min(1.0, np.std(prediction)))
            
        return confidence
    
    def update_thresholds(self, prediction_type: str, 
                          confidence: float, 
                          was_correct: bool) -> None:
        """
        Update confidence thresholds based on feedback.
        
        Args:
            prediction_type: Type of prediction (e.g., 'string', 'fret', 'position')
            confidence: Confidence value of the prediction
            was_correct: Whether the prediction was correct
        """
        # Add to history
        self.confidence_history[prediction_type].append((confidence, was_correct))
        
        # Trim history if needed
        if len(self.confidence_history[prediction_type]) > self.max_history_size:
            self.confidence_history[prediction_type] = self.confidence_history[prediction_type][-self.max_history_size:]
        
        # Recalibrate thresholds if we have enough data
        if len(self.confidence_history[prediction_type]) >= 20:
            confidences = np.array([c for c, _ in self.confidence_history[prediction_type]])
            correctness = np.array([1.0 if c else 0.0 for _, c in self.confidence_history[prediction_type]])
            
            # Find confidence level that gives ~80% correct predictions
            sorted_indices = np.argsort(confidences)
            cumulative_correct = np.cumsum(correctness[sorted_indices])
            cumulative_total = np.arange(1, len(correctness) + 1)
            accuracy_ratio = cumulative_correct / cumulative_total
            
            # Find where accuracy crosses our target (80%)
            target_accuracy = 0.8
            threshold_idx = np.searchsorted(accuracy_ratio, target_accuracy)
            if threshold_idx < len(confidences):
                # Update the threshold based on this calibration
                new_threshold = confidences[sorted_indices[threshold_idx]]
                self.threshold = min(0.9, max(0.5, new_threshold))  # Keep in reasonable range
                logger.info(f"Updated confidence threshold for {prediction_type} to {self.threshold:.2f}")
    
    def should_use_prediction(self, prediction: np.ndarray, 
                              prediction_type: str = 'default') -> Tuple[bool, float, str]:
        """
        Determine if a prediction should be used or fallback to rule-based.
        
        Args:
            prediction: Model prediction output
            prediction_type: Type of prediction for tracking
            
        Returns:
            Tuple of (use_prediction, confidence, recommendation)
        """
        confidence = self.get_prediction_confidence(prediction)
        
        # Determine whether to use the prediction
        if confidence >= self.threshold:
            use_prediction = True
            recommendation = "use_ml"
        elif confidence >= self.min_threshold:
            use_prediction = True
            recommendation = "hybrid"
        elif confidence >= self.fallback_threshold:
            use_prediction = False
            recommendation = "weighted"
        else:
            use_prediction = False
            recommendation = "use_rules"
            
        # Track this for auto-adjusting thresholds
        self.confidence_history[prediction_type].append((confidence, None))
        
        return use_prediction, confidence, recommendation
    
    def calibrate_with_dataset(self, model: tf.keras.Model, 
                               X_val: np.ndarray, 
                               y_val: np.ndarray) -> Dict:
        """
        Calibrate confidence thresholds using validation data.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary of calibration results
        """
        logger.info("Calibrating confidence thresholds with validation data")
        
        # Get predictions
        y_pred = model.predict(X_val)
        
        # Calculate confidences
        confidences = np.array([self.get_prediction_confidence(pred) for pred in y_pred])
        
        # Determine correctness (using argmax for classification tasks)
        if len(y_val.shape) > 1 and y_val.shape[1] > 1:
            # One-hot encoded targets
            correct_predictions = np.argmax(y_pred, axis=1) == np.argmax(y_val, axis=1)
        else:
            # Regression task - use threshold for correctness
            error = np.abs(y_pred - y_val)
            correct_predictions = error < 0.1  # Within 10% error
            
        # Calculate accuracy at different confidence thresholds
        thresholds = np.linspace(0.1, 0.9, 9)
        results = {}
        
        for threshold in thresholds:
            # Predictions above this threshold
            mask = confidences >= threshold
            if np.sum(mask) > 0:
                accuracy = np.mean(correct_predictions[mask])
                coverage = np.mean(mask)
                results[f"{threshold:.1f}"] = {
                    "accuracy": float(accuracy),
                    "coverage": float(coverage),
                    "count": int(np.sum(mask))
                }
                
        # Determine optimal threshold (>= 80% accuracy)
        optimal_threshold = 0.5
        for threshold in sorted(thresholds):
            if threshold in results and results[f"{threshold:.1f}"]["accuracy"] >= 0.8:
                optimal_threshold = threshold
                break
                
        # Update the threshold
        self.threshold = optimal_threshold
        self.min_threshold = max(0.3, optimal_threshold - 0.2)
        self.fallback_threshold = max(0.2, optimal_threshold - 0.3)
        
        logger.info(f"Calibrated confidence thresholds: {self.threshold:.2f}, {self.min_threshold:.2f}, {self.fallback_threshold:.2f}")
        
        # Save calibration data
        self.calibration_data = {
            "threshold": self.threshold,
            "min_threshold": self.min_threshold,
            "fallback_threshold": self.fallback_threshold,
            "results": results
        }
        
        return self.calibration_data
    
    def blend_predictions(self, 
                         ml_prediction: np.ndarray, 
                         rule_prediction: np.ndarray,
                         confidence: float) -> np.ndarray:
        """
        Blend ML and rule-based predictions based on confidence.
        
        Args:
            ml_prediction: Model prediction
            rule_prediction: Rule-based prediction
            confidence: Confidence in the model prediction
            
        Returns:
            Blended prediction
        """
        # Normalize confidence to weight
        if confidence >= self.threshold:
            ml_weight = 0.9
        elif confidence >= self.min_threshold:
            # Scale between 0.5-0.9 based on confidence
            ml_weight = 0.5 + (confidence - self.min_threshold) / (self.threshold - self.min_threshold) * 0.4
        elif confidence >= self.fallback_threshold:
            # Scale between 0.1-0.5 based on confidence
            ml_weight = 0.1 + (confidence - self.fallback_threshold) / (self.min_threshold - self.fallback_threshold) * 0.4
        else:
            ml_weight = 0.0
            
        # Blend predictions
        rule_weight = 1.0 - ml_weight
        blended = (ml_prediction * ml_weight) + (rule_prediction * rule_weight)
        
        return blended

def integrate_with_music_theory(ml_prediction: np.ndarray, 
                               notes: List[int], 
                               key: Optional[str] = None,
                               confidence: float = 0.5) -> np.ndarray:
    """
    Integrate ML predictions with music theory constraints.
    
    Args:
        ml_prediction: Raw ML prediction
        notes: MIDI notes being played
        key: Optional key of the music
        confidence: Model confidence
        
    Returns:
        Adjusted prediction incorporating music theory
    """
    # This is a placeholder - the actual implementation would:
    # 1. Use music theory to constrain the possible positions
    # 2. Blend the ML prediction with these constraints
    # 3. Return a modified prediction
    
    # For now, we'll just return the original prediction
    return ml_prediction
