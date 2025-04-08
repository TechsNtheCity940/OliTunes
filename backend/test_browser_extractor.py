#!/usr/bin/env python
"""
Test script for the Browser-based Ultimate Guitar Tab Extractor
"""

import os
import sys
import time
import logging
import argparse
from browser_ug_extractor import BrowserUGExtractor, DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_browser_extractor")

def test_artist_search(extractor, artist_name):
    """Test the artist search functionality"""
    print(f"\nTesting search for artist: {artist_name}")
    
    results = extractor.search_for_artist(artist_name, max_pages=1)
    
    print(f"Found {len(results)} results for {artist_name}")
    
    # Display the first few results
    for i, tab in enumerate(results[:5]):
        print(f"{i+1}. {tab.get('name', 'Unknown')} - {tab.get('type', 'Unknown')} - Rating: {tab.get('rating', 0)}")
    
    return results

def test_top_tabs(extractor, artist_name):
    """Test the top tabs extraction"""
    print(f"\nTesting top tabs for: {artist_name}")
    
    tabs, chords = extractor.get_top_tabs_for_artist(artist_name, max_tab_count=3, max_chord_count=2)
    
    print(f"Found {len(tabs)} top tabs and {len(chords)} top chords")
    
    # Display the tabs
    print("\nTop Tabs:")
    for i, tab in enumerate(tabs):
        print(f"{i+1}. {tab.get('name', 'Unknown')} - Rating: {tab.get('rating', 0)}")
    
    # Display the chords
    print("\nTop Chords:")
    for i, chord in enumerate(chords):
        print(f"{i+1}. {chord.get('name', 'Unknown')} - Rating: {tab.get('rating', 0)}")
    
    return tabs, chords

def test_download(extractor, artist_name, tabs, chords):
    """Test downloading tabs"""
    print(f"\nTesting download for: {artist_name}")
    
    downloaded_files = []
    
    # Download the first tab if available
    if tabs:
        tab = tabs[0]
        print(f"Downloading tab: {tab['name']}")
        
        # Use the new download_tab method signature (url, save_to_file)
        tab_content = extractor.download_tab(tab['url'], save_to_file=True)
        
        if tab_content:
            print(f"Successfully downloaded tab: {tab['name']}")
            downloaded_files.append(tab['name'])
            
            # Print a preview of the tab content
            lines = tab_content.split('\n')
            preview_lines = lines[:min(10, len(lines))]
            print("\nPreview of tab content:")
            for line in preview_lines:
                print(line)
        else:
            print(f"Failed to download tab: {tab['name']}")
    
    # Download the first chord if available
    if chords:
        chord = chords[0]
        print(f"\nDownloading chord: {chord['name']}")
        
        # Use the new download_tab method signature (url, save_to_file)
        chord_content = extractor.download_tab(chord['url'], save_to_file=True)
        
        if chord_content:
            print(f"Successfully downloaded chord: {chord['name']}")
            downloaded_files.append(chord['name'])
            
            # Print a preview of the chord content
            lines = chord_content.split('\n')
            preview_lines = lines[:min(10, len(lines))]
            print("\nPreview of chord content:")
            for line in preview_lines:
                print(line)
        else:
            print(f"Failed to download chord: {chord['name']}")
    
    print(f"\nDownloaded {len(downloaded_files)} files for {artist_name}")
    return downloaded_files

def main():
    """Main test function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the browser-based Ultimate Guitar extractor")
    parser.add_argument("--artist", default="Metallica", help="Artist name to test")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no browser UI)")
    parser.add_argument("--browser", type=str, choices=["chrome", "firefox"], default="firefox", help="Browser to use")
    args = parser.parse_args()
    
    artist_name = args.artist
    
    print("\nBrowser-based Ultimate Guitar Tab Extractor Test")
    print("===============================================")
    print(f"Testing with artist: {artist_name}")
    
    try:
        # Initialize the extractor
        extractor = BrowserUGExtractor(headless=args.headless, browser_type=args.browser, rate_limit=5)
        
        # Test artist search
        results = test_artist_search(extractor, artist_name)
        
        # Pause to allow page loading and avoid rate limiting
        time.sleep(3)
        
        if results:
            # Test top tabs extraction
            tabs, chords = test_top_tabs(extractor, artist_name)
            
            # Pause to allow page loading and avoid rate limiting
            time.sleep(3)
            
            # Test downloading
            if tabs or chords:
                downloaded_files = test_download(extractor, artist_name, tabs, chords)
                print(f"Downloaded {len(downloaded_files)} files for {artist_name}")
            else:
                print(f"No tabs or chords available for {artist_name}")
        else:
            print(f"No results found for {artist_name}")
        
        print("\nTest completed successfully!")
        print(f"Downloaded files can be found in: {DATA_DIR}")
        
    except Exception as e:
        logger.exception("Error during testing")
        print(f"Error: {str(e)}")
    
    finally:
        # Make sure to close the browser
        if 'extractor' in locals():
            extractor.close()

if __name__ == "__main__":
    main()
