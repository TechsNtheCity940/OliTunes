"""
OliTunes User Feedback Collection System

This module implements a system for collecting and processing user feedback
on generated tablature. This feedback is used to improve model training
and adjust confidence thresholds.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# Define paths
FEEDBACK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "feedback")
os.makedirs(FEEDBACK_DIR, exist_ok=True)

class FeedbackCollector:
    """
    Collects and processes user feedback on generated tablature
    """
    def __init__(self, feedback_dir: str = FEEDBACK_DIR):
        """
        Initialize the feedback collector.
        
        Args:
            feedback_dir: Directory to store feedback data
        """
        self.feedback_dir = feedback_dir
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Load existing feedback data
        self.feedback_file = os.path.join(feedback_dir, "feedback_data.json")
        self.feedback_data = self._load_feedback_data()
        
        # Setup aggregated feedback statistics
        self.feedback_stats = self._compute_feedback_stats()
        
    def _load_feedback_data(self) -> Dict:
        """
        Load existing feedback data from disk.
        
        Returns:
            Dictionary of feedback data
        """
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading feedback data: {e}")
                return {"feedback": [], "metadata": {"last_updated": str(datetime.now())}}
        else:
            return {"feedback": [], "metadata": {"last_updated": str(datetime.now())}}
            
    def _save_feedback_data(self) -> bool:
        """
        Save feedback data to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update metadata
            self.feedback_data["metadata"]["last_updated"] = str(datetime.now())
            self.feedback_data["metadata"]["count"] = len(self.feedback_data["feedback"])
            
            # Save to disk
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
                
            logger.info(f"Saved {len(self.feedback_data['feedback'])} feedback entries to {self.feedback_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving feedback data: {e}")
            return False
            
    def _compute_feedback_stats(self) -> Dict:
        """
        Compute statistics from feedback data.
        
        Returns:
            Dictionary of feedback statistics
        """
        stats = {
            "total_count": len(self.feedback_data["feedback"]),
            "average_rating": 0.0,
            "model_performance": {},
            "position_accuracy": 0.0,
            "playability": 0.0,
            "common_issues": {}
        }
        
        if not self.feedback_data["feedback"]:
            return stats
            
        # Calculate average rating
        ratings = [entry.get("rating", 0) for entry in self.feedback_data["feedback"]]
        stats["average_rating"] = sum(ratings) / len(ratings)
        
        # Calculate position accuracy and playability
        position_scores = [entry.get("position_score", 0) for entry in self.feedback_data["feedback"] 
                          if "position_score" in entry]
        playability_scores = [entry.get("playability", 0) for entry in self.feedback_data["feedback"]
                             if "playability" in entry]
        
        if position_scores:
            stats["position_accuracy"] = sum(position_scores) / len(position_scores)
        
        if playability_scores:
            stats["playability"] = sum(playability_scores) / len(playability_scores)
        
        # Count common issues
        issues = {}
        for entry in self.feedback_data["feedback"]:
            if "issues" in entry:
                for issue in entry["issues"]:
                    issues[issue] = issues.get(issue, 0) + 1
                    
        # Get top 5 issues
        stats["common_issues"] = dict(sorted(issues.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Calculate model performance
        model_ratings = {}
        for entry in self.feedback_data["feedback"]:
            if "model" in entry:
                model = entry["model"]
                rating = entry.get("rating", 0)
                if model not in model_ratings:
                    model_ratings[model] = []
                model_ratings[model].append(rating)
                
        # Average ratings by model
        for model, ratings in model_ratings.items():
            stats["model_performance"][model] = sum(ratings) / len(ratings)
            
        return stats
        
    def add_feedback(self, tab_id: str, user_feedback: Dict, positions: List[Dict] = None) -> bool:
        """
        Add user feedback for a generated tablature.
        
        Args:
            tab_id: Identifier for the tablature
            user_feedback: Dictionary with user feedback data (rating, comments, etc.)
            positions: Optional list of position dictionaries used in the tablature
            
        Returns:
            bool: True if feedback was successfully added, False otherwise
        """
        try:
            self._ensure_feedback_dir_exists()
            
            # Generate a unique ID for this feedback
            feedback_id = str(uuid.uuid4())
            
            # Add metadata
            feedback_data = {
                "id": feedback_id,
                "tab_id": tab_id,
                "timestamp": datetime.now().isoformat() if "timestamp" not in user_feedback else user_feedback["timestamp"],
                "feedback": user_feedback
            }
            
            # Add positions if provided
            if positions:
                feedback_data["positions"] = positions
            
            # Save to file
            feedback_file = os.path.join(self.feedback_dir, f"{feedback_id}.json")
            with open(feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=2)
            
            logger.info(f"Saved user feedback {feedback_id} for tab {tab_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding feedback: {str(e)}", exc_info=True)
            return False
            
    def get_feedback_for_tab(self, tab_id: str) -> List[Dict]:
        """
        Get all feedback for a specific tablature.
        
        Args:
            tab_id: Identifier for the tablature
            
        Returns:
            List of feedback entries
        """
        return [entry for entry in self.feedback_data["feedback"] 
                if entry["id"] == tab_id]
                
    def get_feedback_stats(self) -> Dict:
        """
        Get aggregated statistics for user feedback.
        
        Returns:
            Dict: Dictionary containing feedback statistics
        """
        try:
            self._ensure_feedback_dir_exists()
            
            # Load all feedback files
            feedback_files = [os.path.join(self.feedback_dir, f) for f in os.listdir(self.feedback_dir) 
                             if f.endswith('.json')]
            
            if not feedback_files:
                return {
                    "total_feedbacks": 0,
                    "average_rating": 0,
                    "rating_distribution": {},
                    "difficulty_distribution": {},
                    "rating_breakdown": {
                        "playability": 0,
                        "accuracy": 0,
                        "musicality": 0
                    },
                    "recent_feedback": [],
                    "top_rated_model": None,
                    "rating_trend": 0
                }
            
            # Load feedback data
            all_feedback = []
            for file_path in feedback_files:
                try:
                    with open(file_path, 'r') as f:
                        feedback_data = json.load(f)
                        all_feedback.append(feedback_data)
                except Exception as e:
                    logger.error(f"Error loading feedback file {file_path}: {str(e)}")
            
            # Sort by timestamp (newest first)
            all_feedback.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Calculate statistics
            total_feedbacks = len(all_feedback)
            
            # Rating stats
            ratings = [float(fb["feedback"].get("rating", 0)) for fb in all_feedback 
                      if "rating" in fb["feedback"]]
            
            average_rating = sum(ratings) / len(ratings) if ratings else 0
            
            # Rating distribution
            rating_distribution = {}
            for rating in ratings:
                # Round to nearest 0.5
                rounded_rating = round(rating * 2) / 2
                rating_distribution[str(rounded_rating)] = rating_distribution.get(str(rounded_rating), 0) + 1
            
            # Difficulty distribution
            difficulty_distribution = {}
            for fb in all_feedback:
                difficulty = fb["feedback"].get("difficulty", "medium")
                difficulty_distribution[difficulty] = difficulty_distribution.get(difficulty, 0) + 1
            
            # Rating breakdown
            playability_ratings = [float(fb["feedback"].get("playability", 0)) for fb in all_feedback 
                                 if "playability" in fb["feedback"]]
            accuracy_ratings = [float(fb["feedback"].get("accuracy", 0)) for fb in all_feedback 
                              if "accuracy" in fb["feedback"]]
            musicality_ratings = [float(fb["feedback"].get("musicality", 0)) for fb in all_feedback 
                                if "musicality" in fb["feedback"]]
            
            rating_breakdown = {
                "playability": sum(playability_ratings) / len(playability_ratings) if playability_ratings else 0,
                "accuracy": sum(accuracy_ratings) / len(accuracy_ratings) if accuracy_ratings else 0,
                "musicality": sum(musicality_ratings) / len(musicality_ratings) if musicality_ratings else 0
            }
            
            # Recent feedback (last 10)
            recent_feedback = []
            for fb in all_feedback[:10]:
                recent_item = {
                    "timestamp": fb.get("timestamp", ""),
                    "rating": fb["feedback"].get("rating", 0),
                    "difficulty": fb["feedback"].get("difficulty", "medium"),
                    "comments": fb["feedback"].get("comments", ""),
                    "model_version": fb["feedback"].get("model_version", "unknown")
                }
                recent_feedback.append(recent_item)
            
            # Top rated model
            model_ratings = {}
            for fb in all_feedback:
                model_version = fb["feedback"].get("model_version", "unknown")
                if model_version not in model_ratings:
                    model_ratings[model_version] = {"total": 0, "count": 0}
                
                if "rating" in fb["feedback"]:
                    model_ratings[model_version]["total"] += float(fb["feedback"]["rating"])
                    model_ratings[model_version]["count"] += 1
            
            top_rated_model = None
            top_rating = 0
            
            for model, data in model_ratings.items():
                if data["count"] > 0:
                    avg_rating = data["total"] / data["count"]
                    if avg_rating > top_rating:
                        top_rating = avg_rating
                        top_rated_model = model
            
            # Calculate rating trend (comparing last 10 vs previous 10)
            recent_ratings = [float(fb["feedback"].get("rating", 0)) for fb in all_feedback[:10] 
                           if "rating" in fb["feedback"]]
            previous_ratings = [float(fb["feedback"].get("rating", 0)) for fb in all_feedback[10:20] 
                             if "rating" in fb["feedback"]]
            
            if recent_ratings and previous_ratings:
                recent_avg = sum(recent_ratings) / len(recent_ratings)
                previous_avg = sum(previous_ratings) / len(previous_ratings)
                rating_trend = recent_avg - previous_avg
            else:
                rating_trend = 0
            
            return {
                "total_feedbacks": total_feedbacks,
                "average_rating": average_rating,
                "rating_distribution": rating_distribution,
                "difficulty_distribution": difficulty_distribution,
                "rating_breakdown": rating_breakdown,
                "recent_feedback": recent_feedback,
                "top_rated_model": top_rated_model,
                "rating_trend": rating_trend
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback stats: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "total_feedbacks": 0,
                "average_rating": 0
            }
            
    def _ensure_feedback_dir_exists(self):
        if not os.path.exists(self.feedback_dir):
            os.makedirs(self.feedback_dir)
            
    def export_feedback_to_csv(self, output_path: Optional[str] = None) -> str:
        """
        Export feedback data to CSV for analysis.
        
        Args:
            output_path: Path to save the CSV file
            
        Returns:
            Path to the saved CSV file
        """
        if output_path is None:
            output_path = os.path.join(self.feedback_dir, f"feedback_export_{int(time.time())}.csv")
            
        try:
            # Convert feedback to a flat structure for DataFrame
            flat_feedback = []
            
            for entry in self.feedback_data["feedback"]:
                flat_entry = {
                    "id": entry.get("id", ""),
                    "timestamp": entry.get("timestamp", ""),
                }
                
                # Add feedback fields
                feedback = entry.get("feedback", {})
                for key, value in feedback.items():
                    flat_entry[f"feedback_{key}"] = value
                    
                # Add context fields
                context = entry.get("context", {})
                for key, value in context.items():
                    flat_entry[f"context_{key}"] = value
                    
                flat_feedback.append(flat_entry)
                
            # Create DataFrame and save to CSV
            df = pd.DataFrame(flat_feedback)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Exported {len(df)} feedback entries to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting feedback to CSV: {e}")
            return ""
            
    def create_training_data_from_feedback(self, 
                                          min_rating: float = 4.0, 
                                          output_dir: Optional[str] = None) -> str:
        """
        Create training data from highly-rated feedback.
        
        Args:
            min_rating: Minimum rating to include in training data
            output_dir: Directory to save training data
            
        Returns:
            Path to the saved training data file
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(self.feedback_dir), "tab_data")
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "feedback_training_data.csv")
        
        try:
            # Filter feedback by rating
            high_rated_feedback = [
                entry for entry in self.feedback_data["feedback"]
                if entry.get("feedback", {}).get("rating", 0) >= min_rating
            ]
            
            if not high_rated_feedback:
                logger.warning(f"No feedback with rating >= {min_rating} found")
                return ""
            
            # Extract position data
            position_data = []
            
            for entry in high_rated_feedback:
                if "context" in entry and "positions" in entry["context"]:
                    positions = entry["context"]["positions"]
                    
                    # Ensure these are in the right format for training
                    for pos in positions:
                        if isinstance(pos, dict) and "string" in pos and "fret" in pos:
                            position_data.append(pos)
            
            if not position_data:
                logger.warning("No position data found in filtered feedback")
                return ""
                
            # Convert to DataFrame and save
            df = pd.DataFrame(position_data)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Created training data with {len(df)} positions from feedback")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating training data from feedback: {e}")
            return ""
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """
        Get user preferences based on feedback history.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of user preferences
        """
        # Filter feedback by user ID
        user_feedback = [
            entry for entry in self.feedback_data["feedback"]
            if entry.get("context", {}).get("user_id") == user_id
        ]
        
        preferences = {
            "preferred_style": None,
            "preferred_positions": None,
            "preferred_difficulty": None
        }
        
        if not user_feedback:
            return preferences
            
        # Extract preferred style
        style_ratings = {}
        for entry in user_feedback:
            style = entry.get("context", {}).get("style")
            rating = entry.get("feedback", {}).get("rating", 0)
            
            if style:
                if style not in style_ratings:
                    style_ratings[style] = []
                style_ratings[style].append(rating)
                
        # Get highest rated style
        if style_ratings:
            avg_ratings = {
                style: sum(ratings) / len(ratings)
                for style, ratings in style_ratings.items()
            }
            preferences["preferred_style"] = max(avg_ratings.items(), key=lambda x: x[1])[0]
            
        # Extract preferred positions (low, mid, high)
        position_ratings = {"low": [], "mid": [], "high": []}
        for entry in user_feedback:
            positions = entry.get("context", {}).get("positions", [])
            rating = entry.get("feedback", {}).get("rating", 0)
            
            if positions:
                # Calculate average fret position
                frets = [pos.get("fret", 0) for pos in positions if isinstance(pos, dict)]
                if frets:
                    avg_fret = sum(frets) / len(frets)
                    
                    # Categorize as low, mid, or high
                    if avg_fret < 5:
                        position_type = "low"
                    elif avg_fret < 12:
                        position_type = "mid"
                    else:
                        position_type = "high"
                        
                    position_ratings[position_type].append(rating)
        
        # Get highest rated position type
        if all(position_ratings.values()):
            avg_ratings = {
                pos_type: sum(ratings) / len(ratings)
                for pos_type, ratings in position_ratings.items()
                if ratings
            }
            if avg_ratings:
                preferences["preferred_positions"] = max(avg_ratings.items(), key=lambda x: x[1])[0]
                
        # Extract preferred difficulty
        difficulty_ratings = {"easy": [], "medium": [], "hard": []}
        for entry in user_feedback:
            difficulty = entry.get("context", {}).get("difficulty")
            rating = entry.get("feedback", {}).get("rating", 0)
            
            if difficulty in difficulty_ratings:
                difficulty_ratings[difficulty].append(rating)
                
        # Get highest rated difficulty
        if any(difficulty_ratings.values()):
            avg_ratings = {
                diff: sum(ratings) / len(ratings)
                for diff, ratings in difficulty_ratings.items()
                if ratings
            }
            if avg_ratings:
                preferences["preferred_difficulty"] = max(avg_ratings.items(), key=lambda x: x[1])[0]
                
        return preferences
