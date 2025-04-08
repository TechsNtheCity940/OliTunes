#!/usr/bin/env python
"""
Ultimate Guitar Tab Extractor

This script allows users to extract tabs from Ultimate-Guitar.com
through their search functionality. It can search by artist name,
download specific tabs, or batch download multiple tabs.
"""

import os
import re
import time
import json
import random
import logging
import requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ug_extractor")

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "tab_data", "tab_files")
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
]

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

class UltimateGuitarExtractor:
    """Tool for extracting tabs from Ultimate-Guitar.com"""
    
    def __init__(self, rate_limit=2):
        """
        Initialize the extractor
        
        Args:
            rate_limit: Seconds to wait between requests
        """
        self.rate_limit = rate_limit
        self.session = requests.Session()
    
    def _get_random_user_agent(self):
        """Get a random user agent to avoid detection"""
        return random.choice(USER_AGENTS)
    
    def _make_request(self, url, headers=None, params=None):
        """
        Make an HTTP request with proper headers and rate limiting
        
        Args:
            url: URL to request
            headers: Optional headers dictionary
            params: Optional query parameters
            
        Returns:
            Response object or None if failed
        """
        if headers is None:
            headers = {
                'User-Agent': self._get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        
        try:
            time.sleep(self.rate_limit)  # Rate limiting
            response = self.session.get(url, headers=headers, params=params, timeout=30)
            if response.status_code != 200:
                logger.error(f"Request failed with status code: {response.status_code}")
                return None
            return response
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
    
    def search_for_artist(self, artist_name, max_pages=3):
        """
        Search for all tabs by a specific artist
        
        Args:
            artist_name: Name of the artist to search for
            max_pages: Maximum number of pages to search
            
        Returns:
            List of tab information dictionaries
        """
        logger.info(f"Searching for tabs by artist: {artist_name}")
        
        search_url = f"https://www.ultimate-guitar.com/search.php?search_type=band&value={quote_plus(artist_name)}"
        all_tabs = []
        
        for page in range(1, max_pages + 1):
            logger.info(f"Scanning page {page} of results")
            
            page_url = f"{search_url}&page={page}"
            response = self._make_request(page_url)
            
            if not response:
                logger.warning(f"Failed to get page {page}")
                break
            
            # Parse the search results page
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for the JSON data in the js-store div
            js_store_div = soup.find('div', class_='js-store')
            
            if js_store_div and js_store_div.get('data-content'):
                try:
                    data = json.loads(js_store_div['data-content'])
                    
                    # Extract tab data from the JSON
                    if 'store' in data and 'page' in data['store'] and 'data' in data['store']['page'] and 'results' in data['store']['page']['data']:
                        results = data['store']['page']['data']['results']
                        
                        if not results:
                            logger.info(f"No results found on page {page}")
                            break
                        
                        for tab in results:
                            try:
                                # Extract tab information
                                tab_info = {
                                    'name': tab.get('song_name', ''),
                                    'url': tab.get('tab_url', ''),
                                    'type': tab.get('type_name', ''),
                                    'rating': float(tab.get('rating', 0)),
                                    'votes': int(tab.get('votes', 0)),
                                    'version': tab.get('version', 1),
                                    'verified': bool(tab.get('verified', False))
                                }
                                
                                # Fix URL if needed
                                if tab_info['url'] and not tab_info['url'].startswith('http'):
                                    tab_info['url'] = f"https://tabs.ultimate-guitar.com{tab_info['url']}"
                                
                                # Only add if it's a tab or chord
                                if tab_info['type'] in ['Tab', 'Chords', 'Bass Tab']:
                                    all_tabs.append(tab_info)
                            except Exception as e:
                                logger.warning(f"Error parsing tab result: {e}")
                                continue
                    
                    # Check if there are more pages
                    if 'pagination' in data['store']['page']['data']:
                        if page >= data['store']['page']['data']['pagination'].get('total', 1):
                            logger.info("Reached the last page of results")
                            break
                
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON data from search results")
                    continue
                except KeyError as e:
                    logger.warning(f"Missing expected key in search results JSON: {e}")
                    continue
            else:
                logger.warning("Could not find tab data in the search results")
            
            # Stop if we found a good number of tabs
            if len(all_tabs) >= 30:  # Plenty to filter later
                break
        
        logger.info(f"Found {len(all_tabs)} tabs for artist: {artist_name}")
        return all_tabs
    
    def get_top_tabs_for_artist(self, artist_name, max_tab_count=5, max_chord_count=5):
        """
        Get the top-rated tabs and chord sheets for an artist
        
        Args:
            artist_name: Name of the artist to search for
            max_tab_count: Maximum number of tabs to return
            max_chord_count: Maximum number of chord sheets to return
            
        Returns:
            Tuple of (top tabs list, top chords list)
        """
        all_artist_tabs = self.search_for_artist(artist_name)
        
        # Separate tabs and chords
        tabs = [t for t in all_artist_tabs if t['type'] == 'Tab' or t['type'] == 'Bass Tab']
        chords = [t for t in all_artist_tabs if t['type'] == 'Chords']
        
        # Sort by rating and then by votes to break ties
        tabs.sort(key=lambda x: (x['rating'], x['votes']), reverse=True)
        chords.sort(key=lambda x: (x['rating'], x['votes']), reverse=True)
        
        # Get top results
        top_tabs = tabs[:max_tab_count]
        top_chords = chords[:max_chord_count]
        
        logger.info(f"Selected {len(top_tabs)} top tabs and {len(top_chords)} top chords for {artist_name}")
        
        return top_tabs, top_chords
    
    def download_tab(self, url, song_name, artist_name):
        """
        Download a tab from Ultimate-Guitar
        
        Args:
            url: URL of the tab
            song_name: Name of the song
            artist_name: Name of the artist
            
        Returns:
            Path to the saved tab file or None if failed
        """
        logger.info(f"Downloading tab: {song_name} by {artist_name}")
        
        response = self._make_request(url)
        if not response:
            return None
        
        # Parse the page
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for the JSON data in the js-store div
        js_store_div = soup.find('div', class_='js-store')
        
        if js_store_div and js_store_div.get('data-content'):
            try:
                data = json.loads(js_store_div['data-content'])
                
                # Extract tab data based on the new structure
                if 'store' in data and 'page' in data['store'] and 'data' in data['store']['page'] and 'tab_view' in data['store']['page']['data']:
                    tab_data = data['store']['page']['data']['tab_view']
                    
                    # Extract information
                    tab_type = tab_data.get('type_name', 'Tab')
                    content = tab_data.get('wiki_tab', {}).get('content', '')
                    
                    if not content:
                        content = tab_data.get('content', '')
                    
                    # Clean up content - remove chord formatting tags if present
                    content = re.sub(r'\[ch\](.*?)\[/ch\]', r'\1', content)
                    
                    # Create folder structure
                    artist_dir = os.path.join(DATA_DIR, artist_name)
                    os.makedirs(artist_dir, exist_ok=True)
                    
                    # Create safe filename
                    safe_song = re.sub(r'[\\/*?:"<>|]', '', song_name)
                    filename = f"{safe_song}.txt"
                    filepath = os.path.join(artist_dir, filename)
                    
                    # Save to file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        # Add header with metadata
                        f.write(f"Title: {song_name}\n")
                        f.write(f"Artist: {artist_name}\n")
                        f.write(f"Type: {tab_type}\n")
                        f.write(f"Source: {url}\n")
                        f.write("-" * 50 + "\n\n")
                        f.write(content)
                    
                    logger.info(f"Tab saved to {filepath}")
                    return filepath
                else:
                    logger.error("Tab data not found in JSON response")
                    
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON data")
            except KeyError as e:
                logger.error(f"Missing expected key in tab data JSON: {e}")
        else:
            logger.error("Could not find tab data on page")
        
        return None
    
    def batch_download_for_artist(self, artist_name, max_tab_count=5, max_chord_count=5):
        """
        Download the top tabs and chords for an artist
        
        Args:
            artist_name: Name of the artist to search for
            max_tab_count: Number of tab files to download
            max_chord_count: Number of chord files to download
            
        Returns:
            List of downloaded file paths
        """
        logger.info(f"Batch downloading tabs and chords for {artist_name}")
        
        downloaded_files = []
        
        # Get top tabs and chords
        top_tabs, top_chords = self.get_top_tabs_for_artist(
            artist_name, 
            max_tab_count=max_tab_count, 
            max_chord_count=max_chord_count
        )
        
        # Download tabs
        for tab in top_tabs:
            try:
                tab_path = self.download_tab(tab['url'], tab['name'], artist_name)
                if tab_path:
                    downloaded_files.append(tab_path)
            except Exception as e:
                logger.error(f"Error downloading tab {tab['name']}: {e}")
        
        # Download chords
        for chord in top_chords:
            try:
                chord_path = self.download_tab(chord['url'], chord['name'], artist_name)
                if chord_path:
                    downloaded_files.append(chord_path)
            except Exception as e:
                logger.error(f"Error downloading chord {chord['name']}: {e}")
        
        logger.info(f"Downloaded {len(downloaded_files)} files for {artist_name}")
        return downloaded_files

def batch_download_tabs(artists_list, tabs_per_artist=5, chords_per_artist=5):
    """
    Download top tabs and chords for multiple artists
    
    Args:
        artists_list: List of artist names
        tabs_per_artist: Number of tabs to download per artist
        chords_per_artist: Number of chord sheets to download per artist
        
    Returns:
        Dictionary with artists and their downloaded files
    """
    extractor = UltimateGuitarExtractor(rate_limit=3)
    results = {}
    
    for artist in artists_list:
        try:
            logger.info(f"Processing artist: {artist}")
            files = extractor.batch_download_for_artist(
                artist,
                max_tab_count=tabs_per_artist,
                max_chord_count=chords_per_artist
            )
            results[artist] = files
        except Exception as e:
            logger.error(f"Error processing artist {artist}: {e}")
    
    return results

def main():
    """Main entry point for command-line usage"""
    # Ensure output directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    extractor = UltimateGuitarExtractor()
    
    print("\n=== Ultimate Guitar Tab Downloader ===\n")
    print("This tool helps you download tabs from Ultimate-Guitar.com")
    print("You can search for tabs or paste a tab URL directly.\n")
    
    while True:
        print("\nOptions:")
        print("1. Search for tabs")
        print("2. Download from URL")
        print("3. Batch download")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            artist = input("\nEnter artist name: ")
            print(f"Searching for tabs by {artist}...")
            
            top_results = extractor.get_top_tabs_for_artist(artist)
            
            if top_results[0] or top_results[1]:
                print("\nTop Tabs:")
                for i, tab in enumerate(top_results[0]):
                    print(f"{i+1}. {tab['name']} - Rating: {tab['rating']}/5 ({tab['votes']} votes)")
                
                print("\nTop Chords:")
                for i, chord in enumerate(top_results[1]):
                    print(f"{i+1}. {chord['name']} - Rating: {chord['rating']}/5 ({chord['votes']} votes)")
                
                download = input("\nDownload these tabs? (y/n): ")
                if download.lower() == 'y':
                    files = extractor.batch_download_for_artist(artist)
                    print(f"\nDownloaded {len(files)} files to {DATA_DIR}")
            else:
                print("No tabs found for this artist.")
        
        elif choice == "2":
            url = input("\nEnter Ultimate-Guitar tab URL: ")
            song_name = input("\nEnter song name: ")
            artist_name = input("\nEnter artist name: ")
            filepath = extractor.download_tab(url, song_name, artist_name)
            if filepath:
                print(f"\nTab downloaded to {filepath}")
            else:
                print("Failed to download tab.")
        
        elif choice == "3":
            artists = input("\nEnter artist names (separated by commas): ").split(',')
            artists = [a.strip() for a in artists if a.strip()]
            
            if artists:
                tabs_per_artist = int(input("Number of tabs per artist (default 5): ") or "5")
                chords_per_artist = int(input("Number of chord sheets per artist (default 5): ") or "5")
                
                print(f"\nDownloading tabs for {len(artists)} artists...")
                results = batch_download_tabs(artists, tabs_per_artist, chords_per_artist)
                
                print("\nDownload Summary:")
                total_files = 0
                for artist, files in results.items():
                    print(f"- {artist}: {len(files)} files")
                    total_files += len(files)
                
                print(f"\nTotal files downloaded: {total_files}")
            else:
                print("No valid artists entered.")
        
        elif choice == "4":
            print("\nExiting program. Goodbye!")
            break
        
        else:
            print("\nInvalid choice, please try again.")

if __name__ == "__main__":
    main()
