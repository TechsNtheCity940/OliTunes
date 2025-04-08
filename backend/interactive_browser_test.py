#!/usr/bin/env python
"""
Interactive test tool for Browser-based Ultimate Guitar Tab Extractor.
This allows manual testing and debugging of the browser tab extractor.
"""

import os
import sys
import json
import logging
import time
from browser_ug_extractor import BrowserUGExtractor, DATA_DIR, TAB_FILES_DIR

# Configure logging with more details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("browser_extractor_debug.log")
    ]
)
logger = logging.getLogger("interactive_browser_test")

def print_separator(title=None):
    """Print a separator line with optional title"""
    width = 80
    if title:
        print(f"\n{'=' * ((width - len(title) - 2) // 2)} {title} {'=' * ((width - len(title) - 2) // 2)}")
    else:
        print("\n" + "=" * width)

def save_debug_data(data, filename):
    """Save debug data to a file"""
    filepath = os.path.join("debug_output", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if isinstance(data, (dict, list)):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(data))
    
    print(f"Debug data saved to {filepath}")

def search_and_analyze_artist(extractor, artist_name):
    """Search for an artist and analyze the results"""
    print_separator(f"Searching for artist: {artist_name}")
    
    # Perform the search
    results = extractor.search_for_artist(artist_name)
    
    # Save the raw results for analysis
    save_debug_data(results, f"{artist_name}_search_results.json")
    
    # Display summary
    print(f"Found {len(results)} results for {artist_name}")
    
    # Show the first few results
    for i, tab in enumerate(results[:5]):
        print(f"{i+1}. {tab.get('name', 'Unknown')} - Type: {tab.get('type', 'Unknown')} - Rating: {tab.get('rating', 0)}")
    
    return results

def test_top_tabs_extraction(extractor, artist_name):
    """Test the extraction of top tabs for an artist"""
    print_separator(f"Getting top tabs for: {artist_name}")
    
    # First, search for the artist to get all tabs
    results = extractor.search_for_artist(artist_name)
    
    if not results:
        print(f"No results found for {artist_name}")
        return [], []
    
    # Filter and sort tabs
    tabs = sorted(
        [t for t in results if t['type'] == 'Tab'], 
        key=lambda x: float(x.get('rating', 0) or 0),
        reverse=True
    )[:3]
    
    # Filter and sort chords
    chords = sorted(
        [c for c in results if c['type'] == 'Chords'],
        key=lambda x: float(x.get('rating', 0) or 0),
        reverse=True
    )[:3]
    
    # Save the results for analysis
    save_debug_data({"tabs": tabs, "chords": chords}, f"{artist_name}_top_tabs.json")
    
    # Display tabs
    print(f"\nTop {len(tabs)} tabs:")
    for i, tab in enumerate(tabs):
        print(f"{i+1}. {tab.get('name', 'Unknown')} - Rating: {tab.get('rating', 0)}")
    
    # Display chords
    print(f"\nTop {len(chords)} chords:")
    for i, chord in enumerate(chords):
        print(f"{i+1}. {chord.get('name', 'Unknown')} - Rating: {chord.get('rating', 0)}")
    
    return tabs, chords

def download_sample_tab(extractor, artist_name, tabs):
    """Download a sample tab if available"""
    print_separator(f"Downloading a sample tab for: {artist_name}")
    
    if not tabs:
        print("No tabs available to download")
        return None
    
    # Download the first tab
    sample_tab = tabs[0]
    print(f"Downloading: {sample_tab.get('name', 'Unknown')}")
    
    # Note that browser extractor's download_tab has different signature
    tab_content = extractor.download_tab(sample_tab['url'], save_to_file=True)
    
    if tab_content:
        print(f"Tab downloaded successfully")
        
        # Show a sample of the tab content
        lines = tab_content.split("\n")
        preview_lines = lines[:min(15, len(lines))]
        print("\nTab content preview:")
        print("-" * 40)
        for line in preview_lines:
            print(line)
        print("-" * 40)
        print("...")
        
        return True
    else:
        print("Failed to download the tab")
        return False

def batch_download_for_artist(extractor, artist_name, max_tab_count=3, max_chord_count=2):
    """Download multiple tabs and chords for an artist"""
    print_separator(f"Batch download for: {artist_name}")
    
    # Keep track of unique songs
    downloaded_songs = set()
    downloaded_files = []
    
    # First, search for the artist to get all tabs
    results = extractor.search_for_artist(artist_name)
    
    if not results:
        print(f"No results found for {artist_name}")
        return []
    
    # Filter and sort tabs
    tabs_sorted = sorted(
        [t for t in results if t['type'] == 'Tab'], 
        key=lambda x: float(x.get('rating', 0) or 0),
        reverse=True
    )
    
    # Filter and sort chords
    chords_sorted = sorted(
        [c for c in results if c['type'] == 'Chords'],
        key=lambda x: float(x.get('rating', 0) or 0),
        reverse=True
    )
    
    # Get unique top tabs
    top_tabs = []
    for tab in tabs_sorted:
        song_name = tab['name'].lower()
        if len(top_tabs) < max_tab_count and song_name not in downloaded_songs:
            top_tabs.append(tab)
            downloaded_songs.add(song_name)
    
    # Get unique top chords
    top_chords = []
    for chord in chords_sorted:
        song_name = chord['name'].lower()
        if len(top_chords) < max_chord_count and song_name not in downloaded_songs:
            top_chords.append(chord)
            downloaded_songs.add(song_name)
    
    print(f"Selected {len(top_tabs)} tabs and {len(top_chords)} chords for download")
    
    # Download tabs
    for i, tab in enumerate(top_tabs, 1):
        print(f"Downloading tab {i}/{len(top_tabs)}: {tab['name']}")
        try:
            tab_content = extractor.download_tab(tab['url'], save_to_file=True)
            if tab_content:
                print(f"✓ Successfully downloaded tab: {tab['name']}")
                downloaded_files.append(tab['name'])
            else:
                print(f"✗ Failed to download tab: {tab['name']}")
                
            # Delay between downloads
            time.sleep(1.5)
        except Exception as e:
            print(f"Error downloading tab {tab['name']}: {e}")
    
    # Download chords
    for i, chord in enumerate(top_chords, 1):
        print(f"Downloading chord {i}/{len(top_chords)}: {chord['name']}")
        try:
            chord_content = extractor.download_tab(chord['url'], save_to_file=True)
            if chord_content:
                print(f"✓ Successfully downloaded chord: {chord['name']}")
                downloaded_files.append(chord['name'])
            else:
                print(f"✗ Failed to download chord: {chord['name']}")
                
            # Delay between downloads
            time.sleep(1.5)
        except Exception as e:
            print(f"Error downloading chord {chord['name']}: {e}")
    
    return downloaded_files

def main():
    """Main interactive test function"""
    print("\nBrowser-based Ultimate Guitar Tab Extractor - Interactive Test")
    print("============================================================")
    
    # Create debug output directory
    os.makedirs("debug_output", exist_ok=True)
    
    # Browser choice
    print("\nSelect a browser:")
    print("1. Firefox (recommended)")
    print("2. Chrome")
    
    browser_choice = input("\nYour choice [1]: ").strip() or "1"
    browser_type = "firefox" if browser_choice == "1" else "chrome"
    
    # Set up the browser extractor
    try:
        print(f"\nInitializing browser extractor with {browser_type}...")
        extractor = BrowserUGExtractor(browser_type=browser_type)
        print("Browser extractor initialized successfully!")
    except Exception as e:
        print(f"Error initializing browser extractor: {e}")
        return
    
    try:
        while True:
            print("\nSelect an option:")
            print("1. Search for an artist")
            print("2. Get top tabs for an artist")
            print("3. Download a single tab")
            print("4. Run full test sequence on an artist")
            print("5. Batch download for a list of artists")
            print("0. Exit")
            
            choice = input("\nYour choice: ")
            
            try:
                if choice == "0":
                    break
                    
                elif choice == "1":
                    artist_name = input("\nEnter artist name: ")
                    results = search_and_analyze_artist(extractor, artist_name)
                    
                elif choice == "2":
                    artist_name = input("\nEnter artist name: ")
                    tabs, chords = test_top_tabs_extraction(extractor, artist_name)
                    
                elif choice == "3":
                    artist_name = input("\nEnter artist name: ")
                    url = input("Enter tab URL: ")
                    
                    print(f"Downloading tab from {url}...")
                    tab_content = extractor.download_tab(url, save_to_file=True)
                    
                    if tab_content:
                        print("Tab downloaded successfully")
                        lines = tab_content.split("\n")
                        preview_lines = lines[:min(15, len(lines))]
                        print("\nTab content preview:")
                        print("-" * 40)
                        for line in preview_lines:
                            print(line)
                        print("-" * 40)
                    else:
                        print("Failed to download tab")
                    
                elif choice == "4":
                    artist_name = input("\nEnter artist name: ")
                    
                    # Run full sequence
                    print("\nRunning full test sequence...")
                    
                    results = search_and_analyze_artist(extractor, artist_name)
                    if results:
                        tabs, chords = test_top_tabs_extraction(extractor, artist_name)
                        if tabs:
                            download_sample_tab(extractor, artist_name, tabs)
                    else:
                        print("No results found, stopping test sequence")
                    
                elif choice == "5":
                    artists_input = input("\nEnter artist names (comma-separated): ")
                    artists = [a.strip() for a in artists_input.split(",")]
                    
                    tabs_count = input("Number of tabs per artist [3]: ").strip() or "3"
                    chords_count = input("Number of chords per artist [2]: ").strip() or "2"
                    
                    try:
                        tabs_count = int(tabs_count)
                        chords_count = int(chords_count)
                    except ValueError:
                        print("Invalid numbers, using defaults (3 tabs, 2 chords)")
                        tabs_count = 3
                        chords_count = 2
                    
                    total_files = 0
                    
                    for artist in artists:
                        print(f"\nProcessing {artist}...")
                        files = batch_download_for_artist(
                            extractor, 
                            artist, 
                            max_tab_count=tabs_count,
                            max_chord_count=chords_count
                        )
                        print(f"Downloaded {len(files)} files for {artist}")
                        total_files += len(files)
                    
                    print(f"\nBatch download complete! Downloaded {total_files} files total.")
                    print(f"Files are saved in: {TAB_FILES_DIR}")
                        
                else:
                    print("Invalid choice, please try again")
                    
            except Exception as e:
                logger.exception("Error in test")
                print(f"Error: {str(e)}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Always close the browser
        print("\nClosing browser...")
        extractor.close()
    
    print("\nTest completed. Check the debug_output directory for detailed logs.")
    print(f"Downloaded tabs can be found in: {TAB_FILES_DIR}")

if __name__ == "__main__":
    main()
