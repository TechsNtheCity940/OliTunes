#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OliTunes ML Model Integration

This module provides integration between trained ML models and the OliTunes
tablature generation system. It handles model loading, prediction,
and blending ML predictions with music theory constraints.
"""

import os
import sys
import json
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Union, Optional
from functools import lru_cache
import time
import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Import OliTunes components - handle different directory structures
try:
    # Import our enhanced components
    from model_optimization import ModelOptimizer
    from confidence_threshold import ConfidenceEvaluator, integrate_with_music_theory
    from user_feedback import FeedbackCollector
    
    # Import existing OliTunes modules
    import music_theory
    from music_theory import MusicTheoryAnalyzer
except ImportError:
    # Handle import from different directory structures
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.append(base_dir)
    
    # Try again with updated path
    from model_optimization import ModelOptimizer
    from confidence_threshold import ConfidenceEvaluator, integrate_with_music_theory
    from user_feedback import FeedbackCollector
    from music_theory import MusicTheoryAnalyzer

# Define constants
NUM_STRINGS = 6
MAX_FRET = 24
MODEL_CACHE_SIZE = 10  # Number of models to keep in LRU cache

class ModelIntegration:
    """
    Integrates ML models with the tablature generation system.
    """
    def __init__(self, model_dir: str = None, use_quantized: bool = True, enable_caching: bool = True):
        """
        Initialize model integration.
        
        Args:
            model_dir: Directory containing trained models
            use_quantized: Whether to use quantized models (if available)
            enable_caching: Whether to enable model prediction caching
        """
        # Set model directory
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_dir = model_dir
        logger.info(f"Model integration initialized with model directory: {self.model_dir}")
        
        # Initialize components
        self.context_window = 3  # Default context window size
        self.features_per_note = 8  # Default features per note
        logger.info(f"Context window: {self.context_window}, Features per note: {self.features_per_note}")
        
        # Performance tracking
        self.performance_metrics = {
            "load_time": {},
            "inference_time": {},
            "batch_size": 16,  # Default batch size for batch processing
            "cache_hits": 0,
            "cache_misses": 0,
            "total_predictions": 0
        }
        
        # Enable caching
        self.enable_caching = enable_caching
        if enable_caching:
            # Setup prediction cache with LRU mechanism
            self._prediction_cache = {}
            self._cache_max_size = 1000  # Store up to 1000 prediction results
            logger.info(f"Model caching enabled with max size: {self._cache_max_size}")
        
        # Load models
        self.models = self._load_models(use_quantized)
        
        # Initialize confidence evaluator
        self.confidence_evaluator = ConfidenceEvaluator()
        
        # Initialize feedback collector
        self.feedback_collector = FeedbackCollector()
        
        # Initialize model optimizer
        self.model_optimizer = ModelOptimizer(model_dir)
        
        # Model versioning
        self.model_versions = self._get_model_versions()
    
    def _get_model_versions(self) -> Dict[str, str]:
        """
        Get model versions from metadata files.
        
        Returns:
            Dictionary mapping model names to version strings
        """
        versions = {}
        
        # Check metadata file for each model
        for model_name in self.models.keys():
            metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        versions[model_name] = metadata.get('version', 'unknown')
                except Exception as e:
                    logger.error(f"Error loading metadata for {model_name}: {e}")
                    versions[model_name] = 'unknown'
            else:
                versions[model_name] = 'unknown'
                
        return versions
    
    def _load_models(self, use_quantized: bool = True) -> Dict[str, tf.keras.Model]:
        """
        Load all available models.
        
        Args:
            use_quantized: Whether to use quantized models if available
            
        Returns:
            Dictionary of model name to model object
        """
        models = {}
        
        # Check if the model directory exists
        if not os.path.exists(self.model_dir):
            logger.warning(f"Model directory {self.model_dir} does not exist")
            return models
            
        # Standard model names
        model_names = ["lstm", "transformer", "hybrid", "ensemble"]
        
        for name in model_names:
            # Track loading time for performance metrics
            start_time = time.time()
            
            # Try quantized model first if requested
            model_loaded = False
            
            if use_quantized:
                quantized_path = os.path.join(self.model_dir, f"{name}_quantized.tflite")
                if os.path.exists(quantized_path):
                    try:
                        # Load TFLite model
                        interpreter = tf.lite.Interpreter(model_path=quantized_path)
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
                                
                            def predict(self, x, verbose=0, **kwargs):
                                """
                                Handle batch prediction for TFLite models
                                """
                                # For single samples
                                if x.ndim == 2:
                                    return self(x)
                                
                                # For batches, process each sample and combine results
                                results = []
                                for sample in x:
                                    # Add batch dimension for interpreter
                                    sample = np.expand_dims(sample, axis=0)
                                    results.append(self(sample))
                                
                                # Stack results
                                return tf.stack(results)
                        
                        # Add the model to our collection
                        models[name] = QuantizedModel(interpreter, input_details, output_details)
                        logger.info(f"Loaded quantized {name} model")
                        model_loaded = True
                        
                    except Exception as e:
                        logger.error(f"Error loading quantized {name} model: {e}")
                        model_loaded = False
            
            # Try standard model if quantized model not loaded
            if not model_loaded:
                standard_path = os.path.join(self.model_dir, name)
                if os.path.exists(standard_path):
                    try:
                        model = tf.keras.models.load_model(standard_path)
                        models[name] = model
                        logger.info(f"Loaded standard {name} model")
                    except Exception as e:
                        logger.error(f"Error loading {name} model: {e}")
            
            # Record model loading time
            if name in models:
                self.performance_metrics["load_time"][name] = time.time() - start_time
        
        # Load calibration data if available
        calibration_path = os.path.join(self.model_dir, "confidence_calibration.json")
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, "r") as f:
                    calibration = json.load(f)
                    self.confidence_evaluator.threshold = calibration.get("threshold", 0.6)
                    self.confidence_evaluator.min_threshold = calibration.get("min_threshold", 0.4)
                    self.confidence_evaluator.fallback_threshold = calibration.get("fallback_threshold", 0.2)
                    logger.info(f"Loaded confidence calibration: {self.confidence_evaluator.threshold}")
            except Exception as e:
                logger.error(f"Error loading confidence calibration: {e}")
        
        return models
    
    def _get_prediction_cache_key(self, context_positions, target_midi_note, model_name, key, style):
        """
        Generate a cache key for model predictions to enable lookup.
        
        Args:
            context_positions: List of position dictionaries
            target_midi_note: Target MIDI note
            model_name: Model name
            key: Musical key
            style: Playing style
            
        Returns:
            String cache key
        """
        # Create a string representation of the context positions
        context_str = "|".join([f"{p.get('string', 0)}-{p.get('fret', 0)}-{p.get('midi_note', 0)}" 
                              for p in context_positions[-self.context_window:]])
        
        # Create the complete cache key
        cache_key = f"{context_str}_{target_midi_note}_{model_name}_{key or 'None'}_{style or 'None'}"
        return cache_key
    
    def prepare_input_features(self, 
                              contexts: List[List[Dict]], 
                              target_midi_notes: Optional[List[int]] = None) -> np.ndarray:
        """
        Prepare input features for batch model prediction.
        
        Args:
            contexts: List of context position lists
            target_midi_notes: Optional list of target MIDI notes
            
        Returns:
            Numpy array of input features for batch prediction
        """
        # Handle single context case
        if not isinstance(contexts[0], list):
            contexts = [contexts]
            if target_midi_notes is not None and not isinstance(target_midi_notes, list):
                target_midi_notes = [target_midi_notes]
                
        # Create batched features
        batch_features = []
        
        for i, context_positions in enumerate(contexts):
            # Pad if needed
            if len(context_positions) < self.context_window:
                padding = [{"string": 0, "fret": 0, "midi_note": 0}] * (self.context_window - len(context_positions))
                context_positions = padding + context_positions
            
            # Take the most recent positions up to context_window
            context_positions = context_positions[-self.context_window:]
            
            # Create features array for this context
            features = []
            
            # Process each position in the context window
            for pos in context_positions:
                # Get position data (with defaults)
                string = int(pos.get("string", 0))
                fret = int(pos.get("fret", 0))
                midi_note = int(pos.get("midi_note", 0))
                
                # One-hot encode string
                string_features = [0] * NUM_STRINGS
                if 0 <= string < NUM_STRINGS:
                    string_features[string] = 1
                else:
                    # Default to first string if invalid
                    string_features[0] = 1
                    
                # Add string features
                features.extend(string_features)
                
                # Add fret (normalized)
                features.append(fret / MAX_FRET)
                
                # Add MIDI note (normalized)
                features.append(midi_note / 127.0 if midi_note > 0 else 0)
                
            batch_features.append(features)
        
        # Convert to numpy array for model input
        X = np.array(batch_features, dtype=np.float32)
        
        return X
    
    def predict_batch_positions(self,
                               contexts: List[List[Dict]],
                               target_midi_notes: List[int],
                               model_name: str = "ensemble",
                               key: Optional[str] = None,
                               style: Optional[str] = None) -> List[Dict]:
        """
        Predict positions for a batch of notes.
        
        Args:
            contexts: List of context position lists
            target_midi_notes: List of target MIDI notes
            model_name: Name of model to use
            key: Optional musical key for context
            style: Optional playing style
            
        Returns:
            List of dictionaries with predicted positions and metadata
        """
        if not self.models:
            logger.warning("No models available for prediction")
            return [{"string": 0, "fret": 0, "confidence": 0, "method": "default"} for _ in target_midi_notes]
        
        # Use ensemble model by default if available
        if model_name not in self.models and "ensemble" in self.models:
            model_name = "ensemble"
        elif model_name not in self.models and self.models:
            # Use first available model
            model_name = list(self.models.keys())[0]
            
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not available, using rule-based position")
            return [{"string": 0, "fret": 0, "confidence": 0, "method": "rule_based"} for _ in target_midi_notes]
        
        # Check cache for each prediction if caching is enabled
        if self.enable_caching:
            results = []
            uncached_indices = []
            uncached_contexts = []
            uncached_notes = []
            
            # Check which predictions are in the cache
            for i, (context, note) in enumerate(zip(contexts, target_midi_notes)):
                cache_key = self._get_prediction_cache_key(context, note, model_name, key, style)
                if cache_key in self._prediction_cache:
                    # Cache hit
                    results.append(self._prediction_cache[cache_key])
                    self.performance_metrics["cache_hits"] += 1
                else:
                    # Cache miss - need to make prediction
                    results.append(None)  # Placeholder
                    uncached_indices.append(i)
                    uncached_contexts.append(context)
                    uncached_notes.append(note)
                    self.performance_metrics["cache_misses"] += 1
            
            # If all results were cached, return them
            if len(uncached_indices) == 0:
                return results
            
            # Only prepare uncached predictions
            contexts = uncached_contexts
            target_midi_notes = uncached_notes
        else:
            # If caching is disabled, just create empty results list
            results = [None] * len(target_midi_notes)
            uncached_indices = list(range(len(target_midi_notes)))
        
        # Track inference time
        start_time = time.time()
        
        try:
            # Prepare input features
            X = self.prepare_input_features(contexts)
            
            # Get model
            model = self.models[model_name]
            
            # Make batch prediction
            predictions = model.predict(X, verbose=0)
            
            # Update performance metrics
            self.performance_metrics["total_predictions"] += len(target_midi_notes)
            inference_time = time.time() - start_time
            self.performance_metrics["inference_time"][model_name] = inference_time
            
            # Process each prediction
            for i, (prediction, context, target_note) in enumerate(zip(predictions, contexts, target_midi_notes)):
                # Evaluate prediction confidence
                confidence = self.confidence_evaluator.get_prediction_confidence(prediction)
                use_prediction, confidence, recommendation = self.confidence_evaluator.should_use_prediction(
                    prediction, prediction_type="position"
                )
                
                # Calculate rule-based position using music theory
                rule_based_position = self._calculate_rule_based_position(
                    context, target_note, key, style
                )
                
                # Determine prediction method
                if use_prediction and recommendation == "use_ml":
                    # Use pure ML prediction
                    method = "ml"
                    position = self._decode_position_from_prediction(prediction, target_note)
                elif recommendation == "hybrid" or recommendation == "weighted":
                    # Blend ML and rule-based predictions
                    method = "hybrid"
                    
                    # Create rule-based prediction in same format as ML prediction
                    rule_prediction = np.zeros_like(prediction)
                    
                    # Set string one-hot (first NUM_STRINGS elements)
                    if 0 <= rule_based_position["string"] < NUM_STRINGS:
                        rule_prediction[rule_based_position["string"]] = 1
                    
                    # Set fret one-hot (next MAX_FRET elements)
                    if 0 <= rule_based_position["fret"] < MAX_FRET:
                        rule_prediction[NUM_STRINGS + rule_based_position["fret"]] = 1
                    
                    # Blend predictions
                    blended_prediction = self.confidence_evaluator.blend_predictions(
                        prediction, rule_prediction, confidence
                    )
                    
                    position = self._decode_position_from_prediction(blended_prediction, target_note)
                else:
                    # Use rule-based prediction
                    method = "rule_based"
                    position = rule_based_position
                    
                # Ensure the position produces the target MIDI note
                position = self._validate_position(position, target_note)
                
                # Add metadata
                position["confidence"] = float(confidence)
                position["method"] = method
                position["model"] = model_name
                
                # Add to results and cache if enabled
                if self.enable_caching:
                    # Only update the uncached positions
                    original_index = uncached_indices[i]
                    results[original_index] = position
                    
                    # Add to cache
                    cache_key = self._get_prediction_cache_key(context, target_note, model_name, key, style)
                    self._prediction_cache[cache_key] = position
                    
                    # Manage cache size
                    if len(self._prediction_cache) > self._cache_max_size:
                        # Remove oldest entry (first key)
                        oldest_key = next(iter(self._prediction_cache))
                        del self._prediction_cache[oldest_key]
                else:
                    # Without caching, just add to results directly
                    results[i] = position
                
            return results
                
        except Exception as e:
            logger.error(f"Error predicting batch positions: {e}")
            
            # Fallback to rule-based position for each note
            fallback_results = []
            for i, (context, target_note) in enumerate(zip(contexts, target_midi_notes)):
                rule_based_position = self._calculate_rule_based_position(
                    context, target_note, key, style
                )
                rule_based_position["confidence"] = 0.0
                rule_based_position["method"] = "rule_based_fallback"
                rule_based_position["model"] = "none"
                
                if self.enable_caching:
                    # Only update the uncached positions
                    original_index = uncached_indices[i]
                    results[original_index] = rule_based_position
                else:
                    # Without caching, just add to results directly
                    fallback_results.append(rule_based_position)
            
            return results if self.enable_caching else fallback_results
    
    def predict_next_position(self, 
                             context_positions: List[Dict],
                             target_midi_note: int,
                             model_name: str = "ensemble",
                             key: Optional[str] = None,
                             style: Optional[str] = None) -> Dict:
        """
        Predict the next fretboard position.
        
        Args:
            context_positions: List of previous position dictionaries
            target_midi_note: MIDI note to predict position for
            model_name: Name of model to use
            key: Optional musical key for context
            style: Optional playing style
            
        Returns:
            Dictionary with predicted position and metadata
        """
        # Use batch prediction for a single note
        results = self.predict_batch_positions(
            [context_positions], [target_midi_note], model_name, key, style
        )
        
        # Return the single result
        return results[0]
    
    def _decode_position_from_prediction(self, prediction: np.ndarray, target_midi_note: int) -> Dict:
        """
        Decode model prediction into a fretboard position.
        
        Args:
            prediction: Model prediction array
            target_midi_note: MIDI note to position
            
        Returns:
            Position dictionary
        """
        # Extract string prediction (first NUM_STRINGS elements)
        string_probs = prediction[:NUM_STRINGS]
        
        # Extract fret prediction (next MAX_FRET elements)
        fret_probs = prediction[NUM_STRINGS:NUM_STRINGS+MAX_FRET]
        
        # Get most likely string and fret
        string = int(np.argmax(string_probs))
        fret = int(np.argmax(fret_probs))
        
        return {"string": string, "fret": fret, "midi_note": target_midi_note}
    
    def _calculate_rule_based_position(self, 
                                      context_positions: List[Dict],
                                      target_midi_note: int,
                                      key: Optional[str] = None,
                                      style: Optional[str] = None) -> Dict:
        """
        Calculate a rule-based fretboard position using music theory.
        
        Args:
            context_positions: Previous position context
            target_midi_note: MIDI note to position
            key: Optional musical key
            style: Optional playing style
            
        Returns:
            Position dictionary
        """
        # Standard tuning MIDI notes for open strings (E2, A2, D3, G3, B3, E4)
        open_strings = [40, 45, 50, 55, 59, 64]
        
        # Calculate all possible positions for this note
        positions = []
        for string in range(NUM_STRINGS):
            open_note = open_strings[string]
            fret = target_midi_note - open_note
            
            if 0 <= fret < MAX_FRET:
                positions.append({"string": string, "fret": fret, "midi_note": target_midi_note})
        
        if not positions:
            # Note can't be played on guitar, find closest match
            closest_note = min(range(128), key=lambda n: abs(n - target_midi_note))
            
            # Try again with closest note
            return self._calculate_rule_based_position(context_positions, closest_note, key, style)
        
        # Apply style bias if specified
        if style:
            positions = self._apply_style_bias(positions, style)
        
        # Weight positions by proximity to previous positions
        if context_positions:
            last_position = context_positions[-1]
            
            # Calculate distance score for each position
            for pos in positions:
                # Calculate fretboard distance
                string_distance = abs(pos["string"] - last_position.get("string", 0))
                fret_distance = abs(pos["fret"] - last_position.get("fret", 0))
                
                # Manhattan distance on fretboard
                distance = string_distance + fret_distance
                
                # Add distance score (lower is better)
                pos["distance_score"] = -distance
            
            # Sort by distance score (highest score first)
            positions.sort(key=lambda p: p.get("distance_score", 0), reverse=True)
        
        # Return best position
        return positions[0]
    
    def _apply_style_bias(self, positions: List[Dict], style: str) -> List[Dict]:
        """
        Apply playing style bias to positions.
        
        Args:
            positions: List of possible positions
            style: Playing style
            
        Returns:
            List of positions with style score added
        """
        # Define style biases
        style_biases = {
            "blues": {"open": 0, "low": 3, "mid": 5, "high": 2},
            "rock": {"open": 1, "low": 4, "mid": 5, "high": 3},
            "jazz": {"open": 0, "low": 2, "mid": 4, "high": 5},
            "folk": {"open": 5, "low": 4, "mid": 2, "high": 0},
            "country": {"open": 4, "low": 5, "mid": 3, "high": 1},
            "metal": {"open": 2, "low": 3, "mid": 5, "high": 4},
            "classical": {"open": 5, "low": 3, "mid": 2, "high": 1},
            "fingerstyle": {"open": 5, "low": 4, "mid": 3, "high": 1}
        }
        
        # Default to rock if style not found
        if style.lower() not in style_biases:
            style = "rock"
            
        bias = style_biases[style.lower()]
        
        # Apply bias to each position
        for pos in positions:
            fret = pos["fret"]
            
            # Categorize position
            if fret == 0:
                category = "open"
            elif fret < 5:
                category = "low"
            elif fret < 12:
                category = "mid"
            else:
                category = "high"
                
            # Add style score
            pos["style_score"] = bias[category]
        
        # Sort by style score (highest first)
        positions.sort(key=lambda p: p.get("style_score", 0), reverse=True)
        
        return positions
    
    def _validate_position(self, position: Dict, target_midi_note: int) -> Dict:
        """
        Validate that a position produces the target MIDI note.
        
        Args:
            position: Position dictionary
            target_midi_note: Target MIDI note
            
        Returns:
            Validated position
        """
        # Standard tuning MIDI notes for open strings (E2, A2, D3, G3, B3, E4)
        open_strings = [40, 45, 50, 55, 59, 64]
        
        string = position["string"]
        fret = position["fret"]
        
        # Check if string is valid
        if not (0 <= string < NUM_STRINGS):
            # Default to string 0
            string = 0
            
        # Check if fret is valid
        if not (0 <= fret < MAX_FRET):
            # Cap at valid range
            fret = max(0, min(MAX_FRET - 1, fret))
            
        # Calculate actual MIDI note
        if 0 <= string < len(open_strings):
            actual_midi = open_strings[string] + fret
        else:
            actual_midi = target_midi_note
            
        # Update position
        position["string"] = string
        position["fret"] = fret
        position["midi_note"] = actual_midi
        
        return position
    
    def predict_chord_positions(self, 
                               chord_notes: List[int],
                               context_positions: Optional[List[Dict]] = None,
                               key: Optional[str] = None,
                               style: Optional[str] = None) -> List[Dict]:
        """
        Predict guitar positions for a chord.
        
        Args:
            chord_notes: List of MIDI notes in the chord
            context_positions: Optional previous position context
            key: Optional musical key
            style: Optional playing style
            
        Returns:
            List of positions for the chord
        """
        if not chord_notes:
            return []
            
        # Initialize music theory analyzer
        analyzer = MusicTheoryAnalyzer()
        
        # Identify the chord if possible
        chord_info = analyzer.determine_chord_from_notes(chord_notes)
        chord_name = chord_info.get('root', '') + ' ' + chord_info.get('type', '')
        
        # Initialize empty positions list
        positions = []
        
        # Predict position for each note
        for note in chord_notes:
            position = self.predict_next_position(
                context_positions=context_positions if context_positions else [],
                target_midi_note=note,
                key=key,
                style=style
            )
            positions.append(position)
            
            # Add this position to context for next prediction
            if context_positions is not None:
                context_positions.append(position)
                
        # Ensure chord is playable (no overlapping positions)
        positions = self._ensure_playable_chord(positions, chord_notes)
        
        # Add chord metadata
        for pos in positions:
            pos["chord"] = chord_name
            
        return positions
    
    def _ensure_playable_chord(self, positions: List[Dict], notes: List[int]) -> List[Dict]:
        """
        Ensure a chord is playable on guitar (one note per string max).
        
        Args:
            positions: Predicted positions
            notes: MIDI notes in the chord
            
        Returns:
            Adjusted positions for a playable chord
        """
        # Group positions by string
        string_positions = {}
        for pos in positions:
            string = pos["string"]
            if string not in string_positions:
                string_positions[string] = []
            string_positions[string].append(pos)
            
        # Check for strings with multiple notes
        playable_positions = []
        used_strings = set()
        
        # First, add positions on unique strings
        for pos in positions:
            string = pos["string"]
            if string not in used_strings and len(string_positions[string]) == 1:
                playable_positions.append(pos)
                used_strings.add(string)
                
        # Then, handle strings with multiple notes
        # (take the one with highest confidence)
        for string, pos_list in string_positions.items():
            if string not in used_strings and pos_list:
                # Sort by confidence
                pos_list.sort(key=lambda p: p.get("confidence", 0), reverse=True)
                playable_positions.append(pos_list[0])
                used_strings.add(string)
                
        # Ensure all notes are represented if possible
        included_notes = {pos["midi_note"] for pos in playable_positions}
        missing_notes = [note for note in notes if note not in included_notes]
        
        if missing_notes:
            # Try to add positions for missing notes
            for note in missing_notes:
                candidate_positions = [
                    pos for pos in positions 
                    if pos["midi_note"] == note and pos["string"] not in used_strings
                ]
                
                if candidate_positions:
                    # Sort by confidence
                    candidate_positions.sort(key=lambda p: p.get("confidence", 0), reverse=True)
                    best_pos = candidate_positions[0]
                    playable_positions.append(best_pos)
                    used_strings.add(best_pos["string"])
                    
        # Sort by string number
        playable_positions.sort(key=lambda p: p["string"])
        
        return playable_positions
    
    def collect_user_feedback(self, 
                             tab_id: str, 
                             feedback: Dict, 
                             positions: List[Dict]) -> bool:
        """
        Collect user feedback on generated tablature.
        
        Args:
            tab_id: Identifier for the tablature
            feedback: User feedback data
            positions: Positions used in the tablature
            
        Returns:
            True if feedback was successfully collected
        """
        try:
            # Add additional context to the feedback
            context = {
                "positions": positions,
                "timestamp": datetime.datetime.now().isoformat(),
                "models_used": list(set([pos.get("model", "unknown") for pos in positions if "model" in pos])),
                "methods_used": list(set([pos.get("method", "unknown") for pos in positions if "method" in pos])),
                "model_versions": self.model_versions,
                "average_confidence": sum([pos.get("confidence", 0) for pos in positions]) / len(positions) if positions else 0
            }
            
            # Collect using feedback collector
            result = self.feedback_collector.add_feedback(tab_id, feedback, context)
            
            # If successful and rating is negative, adjust confidence thresholds
            if result and feedback.get("rating", 0) < 3.0:
                # Lower confidence threshold slightly for poor results
                self.confidence_evaluator.threshold *= 0.95
                self.confidence_evaluator.threshold = max(self.confidence_evaluator.threshold, 0.4)
                
                # Log the change
                logger.info(f"Adjusted confidence threshold to {self.confidence_evaluator.threshold} based on feedback")
                
                # Save updated calibration
                self.save_calibration_data()
            
            return result
            
        except Exception as e:
            logger.error(f"Error collecting user feedback: {e}")
            return False
            
    def get_calibration_data(self) -> Dict:
        """
        Get current calibration data for confidence thresholds.
        
        Returns:
            Dictionary of calibration data
        """
        return {
            "threshold": self.confidence_evaluator.threshold,
            "min_threshold": self.confidence_evaluator.min_threshold,
            "fallback_threshold": self.confidence_evaluator.fallback_threshold
        }
    
    def save_calibration_data(self) -> bool:
        """
        Save current calibration data to disk.
        
        Returns:
            True if successful, False otherwise
        """
        calibration_path = os.path.join(self.model_dir, "confidence_calibration.json")
        
        try:
            calibration = self.get_calibration_data()
            
            with open(calibration_path, "w") as f:
                json.dump(calibration, f, indent=2)
                
            logger.info(f"Saved confidence calibration to {calibration_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")
            return False
    
    def quantize_models(self, X_calib: np.ndarray) -> Dict:
        """
        Quantize all loaded models for faster inference.
        
        Args:
            X_calib: Calibration data for quantization
            
        Returns:
            Dictionary with quantization results
        """
        results = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"Quantizing {name} model")
                
                # Skip if already quantized
                if isinstance(model, tf.lite.Interpreter):
                    logger.info(f"Model {name} is already quantized")
                    results[name] = {"status": "already_quantized"}
                    continue
                    
                # Quantize the model
                quantized_model = self.model_optimizer.quantize_model(
                    model=model,
                    dataset=(X_calib, None),
                    model_name=name
                )
                
                # Update model reference
                self.models[name] = quantized_model
                
                # Benchmark speedup
                original_results = self.model_optimizer.benchmark_model(model, X_calib)
                quantized_results = self.model_optimizer.benchmark_model(quantized_model, X_calib)
                
                # Calculate speedup
                speedup = original_results["avg_time_per_batch"] / quantized_results["avg_time_per_batch"]
                
                results[name] = {
                    "status": "success",
                    "speedup": speedup,
                    "original_time": original_results["avg_time_per_batch"],
                    "quantized_time": quantized_results["avg_time_per_batch"]
                }
                
                logger.info(f"Quantized {name} model with {speedup:.2f}x speedup")
                
            except Exception as e:
                logger.error(f"Error quantizing {name} model: {e}")
                results[name] = {"status": "error", "error": str(e)}
                
        return results
    
    def get_model_performance_stats(self) -> Dict:
        """
        Get model performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            "models": list(self.models.keys()),
            "versions": self.model_versions,
            "load_times": self.performance_metrics["load_time"],
            "inference_times": self.performance_metrics["inference_time"],
            "predictions": {
                "total": self.performance_metrics["total_predictions"],
                "cache_hits": self.performance_metrics["cache_hits"],
                "cache_misses": self.performance_metrics["cache_misses"],
                "hit_rate": self.performance_metrics["cache_hits"] / max(1, (self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]))
            },
            "batch_size": self.performance_metrics["batch_size"]
        }
        
        return stats
    
    def collect_user_feedback(self, tab_id: str, feedback: Dict, positions: List[Dict] = None) -> bool:
        """
        Collect user feedback on generated tablature and use it to calibrate confidence thresholds.
        
        Args:
            tab_id: Unique identifier for the tablature
            feedback: Dictionary containing feedback data (rating, comments, etc.)
            positions: Optional list of position dictionaries used in the tablature
            
        Returns:
            bool: True if feedback was successfully collected, False otherwise
        """
        try:
            # Add prediction metadata to feedback
            feedback["model_version"] = self.model_metadata.get("version", "unknown")
            feedback["model_timestamp"] = self.model_metadata.get("timestamp", datetime.now().isoformat())
            
            # Add performance metrics if available
            if hasattr(self, 'performance_metrics'):
                feedback["avg_confidence"] = self.performance_metrics.get("avg_confidence", 0.0)
                feedback["prediction_time"] = self.performance_metrics.get("last_inference_time", 0.0)
            
            # Create feedback collector and save feedback
            collector = FeedbackCollector()
            success = collector.add_feedback(tab_id, feedback, positions)
            
            # Adjust confidence thresholds based on feedback if rating is provided
            if success and "rating" in feedback:
                rating = float(feedback["rating"])
                # Use feedback to adjust confidence thresholds
                if rating >= 4.0:  # Good feedback - our thresholds might be too high
                    self.confidence_threshold = max(0.5, self.confidence_threshold * 0.95)
                elif rating <= 2.0:  # Bad feedback - our thresholds might be too low
                    self.confidence_threshold = min(0.95, self.confidence_threshold * 1.05)
                    
                logger.info(f"Adjusted confidence threshold to {self.confidence_threshold} based on user feedback")
            
            return success
        except Exception as e:
            logger.error(f"Error collecting user feedback: {str(e)}", exc_info=True)
            return False
    
    def get_model_performance_stats(self) -> Dict:
        """
        Get performance statistics for the model.
        
        Returns:
            Dict: Dictionary containing performance statistics
        """
        try:
            # Calculate cache hit ratio
            total_predictions = self._cache_hits + self._cache_misses
            cache_hit_ratio = self._cache_hits / total_predictions if total_predictions > 0 else 0
            
            # Get time series data (last 10 predictions)
            time_series = self._performance_history[-10:] if hasattr(self, '_performance_history') else []
            
            stats = {
                "model_version": self.model_metadata.get("version", "unknown"),
                "is_quantized": self._use_quantized,
                "model_size_mb": self._get_model_size_mb(),
                "last_updated": self.model_metadata.get("timestamp", datetime.now().isoformat()),
                
                # Cache statistics
                "cache_size": len(self._prediction_cache),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_ratio": cache_hit_ratio,
                
                # Timing information
                "model_load_time": self._model_load_time,
                "avg_inference_time": self._calculate_avg_inference_time(),
                "avg_batch_time": self._calculate_avg_batch_time(),
                "avg_batch_size": self._calculate_avg_batch_size(),
                
                # Time series data
                "time_series": time_series
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting model performance stats: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def get_calibration_data(self) -> Dict:
        """
        Get confidence calibration data for the model.
        
        Returns:
            Dict: Dictionary containing calibration data
        """
        try:
            # If we don't have ground truth data, return placeholder data
            if not hasattr(self, '_calibration_data') or not self._calibration_data:
                # Create placeholder bins
                bins = []
                bin_step = 0.1
                for i in range(10):
                    min_conf = i * bin_step
                    max_conf = (i + 1) * bin_step
                    # Ideally, model confidence should match accuracy
                    expected_accuracy = (min_conf + max_conf) / 2
                    # Add some noise to make it realistic
                    accuracy = expected_accuracy + (np.random.random() * 0.1 - 0.05)
                    accuracy = max(0, min(1, accuracy))
                    
                    bins.append({
                        "min_confidence": min_conf,
                        "max_confidence": max_conf,
                        "sample_count": np.random.randint(10, 100),
                        "accuracy": accuracy,
                        "calibration_error": expected_accuracy - accuracy
                    })
                
                return {
                    "bins": bins,
                    "overall_error": np.random.random() * 0.1,
                    "is_placeholder": True
                }
            
            # Calculate real calibration data from stored ground truth comparisons
            bins = []
            bin_step = 0.1
            overall_samples = 0
            overall_error = 0
            
            for i in range(10):
                min_conf = i * bin_step
                max_conf = (i + 1) * bin_step
                
                # Filter calibration data for this confidence bin
                bin_data = [item for item in self._calibration_data 
                           if min_conf <= item["confidence"] < max_conf]
                
                if bin_data:
                    sample_count = len(bin_data)
                    # Calculate accuracy as fraction of correct predictions
                    accuracy = sum(1 for item in bin_data if item["correct"]) / sample_count
                    # Calculate calibration error (difference between confidence and accuracy)
                    expected_confidence = (min_conf + max_conf) / 2
                    calibration_error = expected_confidence - accuracy
                    
                    bins.append({
                        "min_confidence": min_conf,
                        "max_confidence": max_conf,
                        "sample_count": sample_count,
                        "accuracy": accuracy,
                        "calibration_error": calibration_error
                    })
                    
                    overall_samples += sample_count
                    overall_error += abs(calibration_error) * sample_count
                else:
                    # No samples in this bin
                    bins.append({
                        "min_confidence": min_conf,
                        "max_confidence": max_conf,
                        "sample_count": 0,
                        "accuracy": 0,
                        "calibration_error": 0
                    })
            
            # Calculate overall calibration error
            overall_error = overall_error / overall_samples if overall_samples > 0 else 0
            
            return {
                "bins": bins,
                "overall_error": overall_error,
                "is_placeholder": False
            }
        
        except Exception as e:
            logger.error(f"Error getting calibration data: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _get_model_size_mb(self) -> float:
        """Get the size of the model file in megabytes."""
        try:
            if not self.model_path or not os.path.exists(self.model_path):
                return 0.0
            
            size_bytes = os.path.getsize(self.model_path)
            return size_bytes / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.error(f"Error getting model size: {str(e)}")
            return 0.0
    
    def _calculate_avg_inference_time(self) -> float:
        """Calculate the average inference time from performance history."""
        if not hasattr(self, '_inference_times') or not self._inference_times:
            return 0.0
        
        return sum(self._inference_times) / len(self._inference_times)
    
    def _calculate_avg_batch_time(self) -> float:
        """Calculate the average batch processing time from performance history."""
        if not hasattr(self, '_batch_times') or not self._batch_times:
            return 0.0
        
        return sum(self._batch_times) / len(self._batch_times)
    
    def _calculate_avg_batch_size(self) -> float:
        """Calculate the average batch size from performance history."""
        if not hasattr(self, '_batch_sizes') or not self._batch_sizes:
            return 0.0
        
        return sum(self._batch_sizes) / len(self._batch_sizes)


def create_model_integration(model_dir: str = None, use_quantized: bool = True) -> ModelIntegration:
    """
    Factory function to create a ModelIntegration instance.
    
    Args:
        model_dir: Directory containing the model files
        use_quantized: Whether to use quantized models
        
    Returns:
        ModelIntegration: A new ModelIntegration instance
    """
    return ModelIntegration(model_dir, use_quantized)


def get_note_positions(integration: ModelIntegration, 
                     midi_notes: List[int],
                     context: Optional[List[Dict]] = None,
                     key: Optional[str] = None,
                     style: Optional[str] = None) -> List[Dict]:
    """
    Get positions for a sequence of notes.
    
    Args:
        integration: ModelIntegration instance
        midi_notes: List of MIDI notes to position
        context: Optional position context
        key: Optional musical key
        style: Optional playing style
        
    Returns:
        List of position dictionaries
    """
    positions = []
    current_context = [] if context is None else context.copy()
    
    for note in midi_notes:
        position = integration.predict_next_position(
            context_positions=current_context,
            target_midi_note=note,
            key=key,
            style=style
        )
        positions.append(position)
        current_context.append(position)
        
    return positions

def get_chord_positions(integration: ModelIntegration,
                      chord_notes: List[int],
                      context: Optional[List[Dict]] = None,
                      key: Optional[str] = None,
                      style: Optional[str] = None) -> List[Dict]:
    """
    Get positions for a chord.
    
    Args:
        integration: ModelIntegration instance
        chord_notes: List of MIDI notes in the chord
        context: Optional position context
        key: Optional musical key
        style: Optional playing style
        
    Returns:
        List of position dictionaries for the chord
    """
    return integration.predict_chord_positions(
        chord_notes=chord_notes,
        context_positions=context,
        key=key,
        style=style
    )
