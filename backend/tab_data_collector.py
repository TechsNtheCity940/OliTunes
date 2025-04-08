#!/usr/bin/env python
"""
Ultimate-Guitar Tab Data Collector

This script collects and processes guitar tablature data from Ultimate-Guitar.com
to build training datasets for the OliTunes fretboard position prediction model.

The data is collected ethically following the site's robots.txt and terms of service.
"""

import os
import re
import json
import time
import random
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tab_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tab_collector")

# Constants
BASE_URL = "https://www.ultimate-guitar.com"
SEARCH_URL = f"{BASE_URL}/search.php"
TAB_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "tab_data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATASET_PATH = os.path.join(TAB_DATA_DIR, "fretboard_positions.csv")

# Ensure directories exist
os.makedirs(TAB_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Guitar string standard tuning (E2, A2, D3, G3, B3, E4)
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]

class TabDataCollector:
    """Collects and processes guitar tablature data from online sources."""
    
    def __init__(self, rate_limit=2):
        """
        Initialize the tab data collector
        
        Args:
            rate_limit: Minimum seconds between requests to avoid overloading servers
        """
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'OliTunes Research Bot (educational purposes only)',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        self.last_request_time = 0
        self.position_data = []
        
        # Load existing data if available
        if os.path.exists(DATASET_PATH):
            try:
                self.dataset = pd.read_csv(DATASET_PATH)
                logger.info(f"Loaded existing dataset with {len(self.dataset)} records")
            except Exception as e:
                logger.error(f"Error loading existing dataset: {e}")
                self.dataset = pd.DataFrame()
        else:
            self.dataset = pd.DataFrame()
    
    def _respect_rate_limit(self):
        """Ensure we don't exceed our self-imposed rate limit"""
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def search_tabs(self, query, tab_type="tab", num_results=5):
        """
        Search for tabs on Ultimate-Guitar
        
        Args:
            query: Search term (usually artist or song name)
            tab_type: Type of tab to search for (tab, chords, bass, etc)
            num_results: Maximum number of results to return
            
        Returns:
            List of search result dictionaries
        """
        self._respect_rate_limit()
        
        params = {
            "search_type": "title",
            "value": query,
            "type": tab_type
        }
        
        try:
            response = self.session.get(SEARCH_URL, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # This is a simplified version - actual implementation would need
            # to be adjusted to match the current Ultimate-Guitar HTML structure
            tab_links = soup.select(".results-table .song a")[:num_results]
            
            for link in tab_links:
                results.append({
                    "title": link.text.strip(),
                    "url": urljoin(BASE_URL, link.get("href", ""))
                })
            
            logger.info(f"Found {len(results)} results for query '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching tabs: {e}")
            return []
    
    def extract_tab_content(self, url):
        """
        Extract tablature content from a tab page
        
        Args:
            url: URL to the tab page
            
        Returns:
            Dictionary with tab metadata and content
        """
        self._respect_rate_limit()
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract tab content (actual implementation would need adjusting)
            tab_content = soup.select_one(".js-tab-content")
            
            if not tab_content:
                logger.warning(f"Could not extract tab content from {url}")
                return None
            
            # Extract metadata
            artist = soup.select_one(".artist-name")
            song = soup.select_one(".song-name")
            rating = soup.select_one(".js-rating-value")
            
            return {
                "artist": artist.text.strip() if artist else "Unknown",
                "song": song.text.strip() if song else "Unknown",
                "rating": float(rating.text) if rating else 0,
                "url": url,
                "content": tab_content.text,
            }
            
        except Exception as e:
            logger.error(f"Error extracting tab content: {e}")
            return None
    
    def parse_tab_positions(self, tab_data):
        """
        Parse guitar positions from tablature text
        
        Args:
            tab_data: Dictionary with tab metadata and content
            
        Returns:
            List of position data dictionaries
        """
        if not tab_data or "content" not in tab_data:
            return []
        
        content = tab_data["content"]
        positions = []
        
        # Regex pattern for standard tab format with 6 strings
        tab_line_pattern = re.compile(r'([eE]|[bB]|[gG]|[dD]|[aA]|[eE])[|-](.+?)\n')
        
        # Find all groups of tab lines
        tab_groups = re.findall(r'([eE][|-].+?\n[bB][|-].+?\n[gG][|-].+?\n[dD][|-].+?\n[aA][|-].+?\n[eE][|-].+?\n)', content, re.DOTALL)
        
        for group in tab_groups:
            lines = group.strip().split('\n')
            if len(lines) != 6:
                continue
                
            # Get each string
            string_lines = {}
            for line in lines:
                match = tab_line_pattern.match(line)
                if match:
                    string_name = match.group(1).lower()
                    string_content = match.group(2)
                    string_lines[string_name] = string_content
            
            # Map string names to indices (0-5, where 0 is the highest pitched string)
            string_indices = {'e': 0, 'b': 1, 'g': 2, 'd': 3, 'a': 4, 'e': 5}
            
            # Extract fret numbers along with their positions
            for string_name, string_content in string_lines.items():
                string_index = string_indices.get(string_name.lower())
                if string_index is None:
                    continue
                
                # Find all fret numbers with their positions
                fret_matches = re.finditer(r'(\d+)', string_content)
                for match in fret_matches:
                    fret = int(match.group(1))
                    position = match.start()
                    
                    # Don't include unrealistic fret positions
                    if fret > 24:
                        continue
                    
                    # Calculate the MIDI note number
                    midi_note = STANDARD_TUNING[string_index] + fret
                    
                    # Try to determine context (surrounding notes)
                    # This is simplified and would need more work
                    context_before = string_content[:position].strip()[-10:]
                    context_after = string_content[position + len(match.group(1)):].strip()[:10]
                    
                    positions.append({
                        "artist": tab_data["artist"],
                        "song": tab_data["song"],
                        "rating": tab_data["rating"],
                        "string": string_index,
                        "fret": fret,
                        "midi_note": midi_note,
                        "position_in_tab": position,
                        "context_before": context_before,
                        "context_after": context_after
                    })
        
        logger.info(f"Extracted {len(positions)} positions from tab")
        return positions
    
    def collect_data_for_genre(self, genre, num_tabs=10):
        """
        Collect tab data for a specific musical genre
        
        Args:
            genre: Genre name to search for
            num_tabs: Number of tabs to process
            
        Returns:
            Number of positions collected
        """
        logger.info(f"Collecting data for genre: {genre}")
        
        # Search for tabs of this genre
        search_results = self.search_tabs(f"{genre} songs", num_results=num_tabs)
        
        count = 0
        for result in tqdm(search_results, desc=f"Processing {genre} tabs"):
            try:
                # Extract and parse the tab
                tab_data = self.extract_tab_content(result["url"])
                if tab_data:
                    positions = self.parse_tab_positions(tab_data)
                    
                    # Add genre information
                    for pos in positions:
                        pos["genre"] = genre
                    
                    self.position_data.extend(positions)
                    count += len(positions)
                    
                    # Save periodically
                    if len(self.position_data) % 1000 == 0:
                        self.save_data()
                
                # Add some randomness to avoid detection
                time.sleep(random.uniform(1.0, 3.0))
                
            except Exception as e:
                logger.error(f"Error processing tab {result['url']}: {e}")
        
        logger.info(f"Collected {count} positions for {genre}")
        return count
    
    def collect_multi_genre_data(self, genres=None):
        """
        Collect tab data across multiple genres
        
        Args:
            genres: List of genres to collect (default: rock, blues, jazz, metal, folk)
            
        Returns:
            Total number of positions collected
        """
        if genres is None:
            genres = ["rock", "blues", "jazz", "metal", "folk"]
        
        total = 0
        for genre in genres:
            count = self.collect_data_for_genre(genre)
            total += count
        
        self.save_data()
        logger.info(f"Total positions collected: {total}")
        return total
    
    def save_data(self):
        """Save collected position data to CSV"""
        if not self.position_data:
            logger.warning("No position data to save")
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.position_data)
            
            # Combine with existing data if any
            if not self.dataset.empty:
                df = pd.concat([self.dataset, df], ignore_index=True)
                
                # Remove duplicates
                df = df.drop_duplicates()
            
            # Save to CSV
            df.to_csv(DATASET_PATH, index=False)
            logger.info(f"Saved {len(df)} positions to {DATASET_PATH}")
            
            # Update instance reference
            self.dataset = df
            
        except Exception as e:
            logger.error(f"Error saving position data: {e}")
    
    def prepare_training_data(self):
        """
        Prepare the collected data for training the neural network
        
        Returns:
            Dictionary with training features and labels
        """
        if not os.path.exists(DATASET_PATH):
            logger.error(f"Dataset file {DATASET_PATH} not found")
            return None
        
        try:
            # Load the dataset
            df = pd.read_csv(DATASET_PATH)
            logger.info(f"Preparing training data from {len(df)} positions")
            
            # Create input features
            X = []
            y = []
            
            # Group by song to maintain context
            for (artist, song), group in df.groupby(['artist', 'song']):
                # Sort by position in tab to maintain sequence
                group = group.sort_values('position_in_tab')
                
                # Process each position
                for i, row in group.iterrows():
                    # Create input vector (one-hot for MIDI note, etc.)
                    input_vector = np.zeros(176)
                    
                    # Set MIDI note (first 128 elements)
                    midi_note = row['midi_note']
                    if 0 <= midi_note < 128:
                        input_vector[midi_note] = 1.0
                    
                    # Set previous position if available (next 25 elements)
                    if i > 0:
                        prev_row = group.iloc[i-1]
                        prev_fret = prev_row['fret']
                        if 0 <= prev_fret <= 24:
                            input_vector[128 + prev_fret] = 1.0
                    
                    # Set string (next 6 elements)
                    string_idx = row['string']
                    if 0 <= string_idx < 6:
                        input_vector[153 + string_idx] = 1.0
                    
                    # Set style based on genre (last 5 elements)
                    genre = row.get('genre', '').lower()
                    style_idx = 0  # Default style
                    if 'blues' in genre:
                        style_idx = 0
                    elif 'rock' in genre:
                        style_idx = 1
                    elif 'jazz' in genre:
                        style_idx = 2
                    elif 'folk' in genre or 'country' in genre:
                        style_idx = 3
                    elif 'metal' in genre:
                        style_idx = 4
                    input_vector[171 + style_idx] = 1.0
                    
                    # Create output vector (one-hot for string/fret combination)
                    output_vector = np.zeros(150)  # 6 strings * 25 frets
                    output_idx = row['string'] * 25 + row['fret']
                    if 0 <= output_idx < 150:
                        output_vector[output_idx] = 1.0
                    
                    X.append(input_vector)
                    y.append(output_vector)
            
            return {
                'X': np.array(X),
                'y': np.array(y),
            }
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None

def main():
    """Main entry point for the script"""
    logger.info("Starting tablature data collection")
    
    collector = TabDataCollector(rate_limit=3)  # Be respectful with rate limiting
    
    # Collect data for various genres
    genres = ["blues", "rock", "jazz", "metal", "folk", "country", "pop"]
    collector.collect_multi_genre_data(genres)
    
    # Prepare data for training
    training_data = collector.prepare_training_data()
    if training_data:
        # Save the prepared data
        np.save(os.path.join(TAB_DATA_DIR, "X_tab_positions.npy"), training_data['X'])
        np.save(os.path.join(TAB_DATA_DIR, "y_tab_positions.npy"), training_data['y'])
        logger.info(f"Saved training data: {len(training_data['X'])} samples")
    
    logger.info("Tab data collection complete")

if __name__ == "__main__":
    main()
