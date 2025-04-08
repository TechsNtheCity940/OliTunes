#!/usr/bin/env python
"""
Interactive test tool for Ultimate Guitar Tab Extractor.
This allows manual testing and debugging of the tab extractor.
"""

import os
import sys
import json
import logging
from ultimate_guitar_extractor import UltimateGuitarExtractor, DATA_DIR

# Configure logging with more details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("extractor_debug.log")
    ]
)
logger = logging.getLogger("interactive_test")

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
    results = extractor.search_for_artist(artist_name, max_pages=1)
    
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
    
    # Get top tabs and chords
    tabs, chords = extractor.get_top_tabs_for_artist(artist_name, max_tab_count=3, max_chord_count=3)
    
    # Save the results for analysis
    save_debug_data({"tabs": tabs, "chords": chords}, f"{artist_name}_top_tabs.json")
    
    # Display tabs
    print(f"\nTop {len(tabs)} tabs:")
    for i, tab in enumerate(tabs):
        print(f"{i+1}. {tab.get('name', 'Unknown')} - Rating: {tab.get('rating', 0)} - Votes: {tab.get('votes', 0)}")
    
    # Display chords
    print(f"\nTop {len(chords)} chords:")
    for i, chord in enumerate(chords):
        print(f"{i+1}. {chord.get('name', 'Unknown')} - Rating: {chord.get('rating', 0)} - Votes: {chord.get('votes', 0)}")
    
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
    
    filepath = extractor.download_tab(sample_tab['url'], sample_tab['name'], artist_name)
    
    if filepath:
        print(f"Tab downloaded successfully to: {filepath}")
        
        # Show a sample of the file content
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(500)  # Read just the beginning
            print("\nFile preview:")
            print("-" * 40)
            print(content)
            print("-" * 40)
            print("...")
    else:
        print("Failed to download the tab")
    
    return filepath

def main():
    """Main interactive test function"""
    print("\nUltimate Guitar Tab Extractor - Interactive Test")
    print("===============================================")
    
    # Create debug output directory
    os.makedirs("debug_output", exist_ok=True)
    
    # Set up the extractor with a longer delay to avoid blocking
    extractor = UltimateGuitarExtractor(rate_limit=3)
    
    while True:
        print("\nSelect an option:")
        print("1. Search for an artist")
        print("2. Get top tabs for an artist")
        print("3. Download a sample tab")
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
                song_name = input("Enter song name: ")
                
                filepath = extractor.download_tab(url, song_name, artist_name)
                
                if filepath:
                    print(f"Tab downloaded to: {filepath}")
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
                
                for artist in artists:
                    print(f"\nProcessing {artist}...")
                    files = extractor.batch_download_for_artist(artist, max_tab_count=2, max_chord_count=2)
                    print(f"Downloaded {len(files)} files for {artist}")
                    
            else:
                print("Invalid choice, please try again")
                
        except Exception as e:
            logger.exception("Error in test")
            print(f"Error: {str(e)}")
    
    print("\nTest completed. Check the debug_output directory for detailed logs.")

if __name__ == "__main__":
    main()
