"""
OliTunes Web Tab Integration

This module integrates the Ultimate Guitar tab extractor with the OliTunes web UI,
allowing users to download tabs for model training directly from the interface.
"""

import os
import json
import logging
import threading
import time
from typing import Dict, List, Union, Optional
from flask import Blueprint, request, jsonify

# Import OliTunes modules
try:
    from browser_ug_extractor import BrowserUGExtractor
    from tab_data_processor import TabDataProcessor
    from tab_data_pipeline import TabDataPipeline
except ImportError:
    # Handle import from different directory structures
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from browser_ug_extractor import BrowserUGExtractor
    from tab_data_processor import TabDataProcessor
    from tab_data_pipeline import TabDataPipeline

# Configure logging
logger = logging.getLogger(__name__)

# Define paths
TAB_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tab_data")
DOWNLOAD_DIR = os.path.join(TAB_DATA_DIR, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Blueprint for tab extraction API - Updated URL prefix to match frontend requests
tab_extractor_bp = Blueprint('tab_extractor', __name__, url_prefix='/tab_extractor')

# Active extraction jobs
active_jobs = {}
job_results = {}

class WebTabExtractor:
    """
    Manages tab extraction through the web interface
    """
    def __init__(self, download_dir: str = DOWNLOAD_DIR):
        """
        Initialize the web tab extractor.
        
        Args:
            download_dir: Directory to save downloaded tabs
        """
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
        
        # Initialize with correct parameters
        self.extractor = BrowserUGExtractor(headless=True, rate_limit=2, browser_type='firefox')
        self.data_processor = TabDataProcessor()
        self.data_pipeline = TabDataPipeline()
        
    def start_extraction_job(self, 
                            artist: str, 
                            song_limit: int = 20, 
                            job_id: Optional[str] = None) -> str:
        """
        Start a tab extraction job in a separate thread.
        
        Args:
            artist: Artist name to search for
            song_limit: Maximum number of songs to download
            job_id: Optional job identifier
            
        Returns:
            Job ID for tracking progress
        """
        if not job_id:
            job_id = f"job_{int(time.time())}_{artist.replace(' ', '_')}"
            
        # Setup job tracking
        active_jobs[job_id] = {
            "status": "starting",
            "artist": artist,
            "song_limit": song_limit,
            "start_time": time.time(),
            "progress": 0,
            "downloaded": 0,
            "errors": []
        }
        
        # Start extraction in a separate thread
        thread = threading.Thread(
            target=self._run_extraction_job,
            args=(artist, song_limit, job_id)
        )
        thread.daemon = True
        thread.start()
        
        return job_id
        
    def _run_extraction_job(self, artist: str, song_limit: int, job_id: str):
        """
        Run the extraction job in the background thread.
        
        Args:
            artist: Artist name to search for
            song_limit: Maximum number of songs to download
            job_id: Unique identifier for this job
        """
        try:
            logger.info(f"Starting extraction job for artist: {artist}, limit: {song_limit}, job_id: {job_id}")
            active_jobs[job_id]["status"] = "running"
            active_jobs[job_id]["progress"] = 0.1

            # Phase 1: Get tabs using the browser extractor
            try:
                logger.info(f"Downloading tabs for artist: {artist}")
                
                # Use batch download method of BrowserUGExtractor
                downloaded_files = self.extractor.batch_download_for_artist(
                    artist_name=artist,
                    max_tab_count=song_limit,
                    max_chord_count=0  # Only download tabs for now
                )
                
                # Update job progress
                active_jobs[job_id]["progress"] = 0.5
                
                # Copy files to our download directory if they're not already there
                if downloaded_files:
                    for file_path in downloaded_files:
                        if not file_path.startswith(self.download_dir):
                            filename = os.path.basename(file_path)
                            target_path = os.path.join(self.download_dir, filename)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as src_file:
                                    content = src_file.read()
                                with open(target_path, 'w', encoding='utf-8') as dest_file:
                                    dest_file.write(content)
                                logger.info(f"Copied tab file to: {target_path}")
                            except Exception as e:
                                logger.error(f"Error copying tab file: {str(e)}")
                
                # Store the results
                job_results[job_id] = {
                    "files": downloaded_files,
                    "count": len(downloaded_files)
                }
                
                logger.info(f"Downloaded {len(downloaded_files)} tab files for {artist}")
            except Exception as e:
                logger.error(f"Error downloading tabs: {str(e)}")
                raise

            # Update job status
            active_jobs[job_id]["status"] = "completed"
            active_jobs[job_id]["end_time"] = time.time()
            active_jobs[job_id]["processed_tabs"] = len(downloaded_files)
            
        except Exception as e:
            # Handle errors
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["end_time"] = time.time()
            active_jobs[job_id]["errors"].append(str(e))
            
            # Store results
            job_results[job_id] = {
                "artist": artist,
                "error": str(e),
                "duration": active_jobs[job_id]["end_time"] - active_jobs[job_id]["start_time"]
            }
    
    def _update_progress(self, job_id: str, progress: float, downloaded: int, total: int) -> None:
        """
        Update job progress.
        
        Args:
            job_id: Job identifier
            progress: Progress percentage (0-100)
            downloaded: Number of tabs downloaded
            total: Total tabs to download
        """
        if job_id in active_jobs:
            active_jobs[job_id]["progress"] = progress
            active_jobs[job_id]["downloaded"] = downloaded
            active_jobs[job_id]["total"] = total
            
    def get_job_status(self, job_id: str) -> Dict:
        """
        Get current status of a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary with job status
        """
        if job_id in active_jobs:
            return active_jobs[job_id]
        elif job_id in job_results:
            return {"status": "completed", "results": job_results[job_id]}
        else:
            return {"status": "not_found"}
            
    def get_all_jobs(self) -> Dict:
        """
        Get status of all jobs.
        
        Returns:
            Dictionary with all job statuses
        """
        return {
            "active": active_jobs,
            "completed": job_results
        }
        
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was canceled, False otherwise
        """
        if job_id in active_jobs and active_jobs[job_id]["status"] == "running":
            active_jobs[job_id]["status"] = "canceled"
            
            # Note: We can't actually stop the thread, but we can signal cancellation
            # The extractor class should periodically check for this status
            
            return True
        return False
        
    def prepare_downloaded_tabs_for_training(self) -> Dict:
        """
        Process all downloaded tabs and prepare for training.
        
        Returns:
            Dictionary with processing results
        """
        results = {
            "processed_files": 0,
            "extracted_positions": 0,
            "training_ready": False,
            "errors": []
        }
        
        try:
            # Process real tabs
            tab_count = self.data_pipeline.process_real_tabs(tab_dir=self.download_dir)
            results["processed_files"] = tab_count
            
            # Check if we have position data
            positions_path = os.path.join(TAB_DATA_DIR, "processed", "real_positions.csv")
            if os.path.exists(positions_path):
                try:
                    import pandas as pd
                    positions_df = pd.read_csv(positions_path)
                    results["extracted_positions"] = len(positions_df)
                    results["training_ready"] = results["extracted_positions"] > 0
                except Exception as e:
                    results["errors"].append(f"Error reading positions: {str(e)}")
            
        except Exception as e:
            results["errors"].append(f"Error preparing tabs for training: {str(e)}")
            
        return results

# Initialize extractor
web_extractor = WebTabExtractor()

# API routes - Updated to match frontend expected paths
@tab_extractor_bp.route('/search', methods=['GET'])
def search_tabs():
    """API endpoint to search for tabs"""
    query = request.args.get('q')
    if not query:
        return jsonify({"error": "Search query is required"}), 400
        
    try:
        extractor = web_extractor.extractor
        results = extractor.search_tabs(query)
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Error searching tabs: {str(e)}")
        return jsonify({"error": str(e)}), 500

@tab_extractor_bp.route('/extract', methods=['POST'])
def extract_tabs():
    """API endpoint to start tab extraction"""
    data = request.json
    artist = data.get('artist')
    song_limit = int(data.get('limit', 20))
    
    if not artist:
        return jsonify({"error": "Artist name is required"}), 400
        
    job_id = web_extractor.start_extraction_job(artist, song_limit)
    
    return jsonify({
        "job_id": job_id,
        "status": "started",
        "message": f"Started extraction for artist: {artist}"
    })

@tab_extractor_bp.route('/status/<job_id>', methods=['GET'])
def job_status(job_id):
    """API endpoint to check job status"""
    status = web_extractor.get_job_status(job_id)
    return jsonify(status)

@tab_extractor_bp.route('/jobs', methods=['GET'])
def list_jobs():
    """API endpoint to list all jobs"""
    jobs = web_extractor.get_all_jobs()
    return jsonify(jobs)

@tab_extractor_bp.route('/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """API endpoint to cancel a job"""
    success = web_extractor.cancel_job(job_id)
    
    if success:
        return jsonify({"status": "canceled", "job_id": job_id})
    else:
        return jsonify({"error": "Job not found or not running"}), 404

@tab_extractor_bp.route('/prepare', methods=['POST'])
def prepare_for_training():
    """API endpoint to prepare downloaded tabs for training"""
    results = web_extractor.prepare_downloaded_tabs_for_training()
    return jsonify(results)

@tab_extractor_bp.route('/process_existing_tabs', methods=['POST'])
def process_existing_tabs():
    """Process existing tab files without downloading new ones"""
    try:
        # Initialize the processor
        processor = TabDataProcessor()
        
        # Process all tabs in the tab_files directory
        num_positions = processor.process_downloaded_tabs()
        
        # Convert to numpy format
        processor.convert_to_numpy_dataset()
        
        return jsonify({
            "success": True,
            "message": f"Processed {num_positions} positions from existing tab files",
            "positions_extracted": num_positions
        })
    except Exception as e:
        logger.error(f"Error processing existing tabs: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def register_blueprint(app):
    """Register blueprint with Flask app"""
    app.register_blueprint(tab_extractor_bp)
    logger.info("Registered tab extractor API endpoints")
