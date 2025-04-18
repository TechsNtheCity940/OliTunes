"""
Unified Tab Downloader for OliTunes
This module combines functionality for downloading guitar tabs from Ultimate Guitar
for both individual searches and batch downloads across multiple artists and genres.
"""

import os
import sys
import time
import logging
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
from browser_ug_extractor import BrowserUGExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tab_downloader.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("tab_downloader")

# Song examples by genre/style - for genre-based downloading
GENRE_EXAMPLES = {
    "blues": [
        "B.B. King - The Thrill Is Gone",
        "Stevie Ray Vaughan - Pride And Joy",
        "Muddy Waters - Hoochie Coochie Man"
    ],
    "rock": [
        "Led Zeppelin - Stairway To Heaven",
        "AC/DC - Back In Black",
        "The Rolling Stones - Paint It Black"
    ],
    "metal": [
        "Metallica - Enter Sandman",
        "Black Sabbath - Paranoid",
        "Iron Maiden - The Trooper"
    ],
    "jazz": [
        "Miles Davis - So What",
        "John Coltrane - Giant Steps",
        "Wes Montgomery - Four On Six"
    ],
    "country": [
        "Johnny Cash - Folsom Prison Blues",
        "Willie Nelson - On The Road Again",
        "Dolly Parton - Jolene"
    ],
    "folk": [
        "Bob Dylan - Blowin' In The Wind",
        "Simon & Garfunkel - The Sound Of Silence",
        "Joni Mitchell - Big Yellow Taxi"
    ]
}

def clean_filename(name: str) -> str:
    """
    Clean a filename by removing invalid characters.
    """
    return re.sub(r'[\\/*?:"<>|]', "", name)

def download_tabs_by_genre(tabs_per_genre: int = 3, rate_limit: int = 3) -> Dict[str, List[str]]:
    """
    Download tabs for each genre in the GENRE_EXAMPLES dictionary
    
    Args:
        tabs_per_genre: Number of tabs to download per genre
        rate_limit: Seconds between requests
        
    Returns:
        Dictionary with genres and downloaded tab filepaths
    """
    output_dir = os.path.join(os.getcwd(), "genre_tabs")
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = BrowserUGExtractor(headless=False, browser_type="firefox")
    downloaded_tabs = {}
    
    try:
        for genre, examples in GENRE_EXAMPLES.items():
            logger.info(f"Processing genre: {genre}")
            
            # Create genre directory
            genre_dir = os.path.join(output_dir, genre)
            os.makedirs(genre_dir, exist_ok=True)
            
            downloaded_tabs[genre] = []
            
            # Use example songs to search for similar tabs in the genre
            for example in examples:
                if len(downloaded_tabs[genre]) >= tabs_per_genre:
                    break
                    
                artist, song = example.split(" - ", 1)
                
                try:
                    # Search for tabs by this artist
                    logger.info(f"Searching tabs for artist: {artist}")
                    tabs = extractor.search_tabs(artist)
                    
                    if not tabs:
                        logger.warning(f"No tabs found for {artist}")
                        continue
                        
                    # Download first available tab
                    tab = tabs[0]
                    tab_path = os.path.join(genre_dir, f"{clean_filename(tab['title'])}.txt")
                    
                    if extractor.download_tab(tab["url"], tab_path):
                        downloaded_tabs[genre].append(tab_path)
                        logger.info(f"Downloaded: {tab['title']} to {tab_path}")
                    
                    # Respect rate limit
                    time.sleep(rate_limit)
                    
                except Exception as e:
                    logger.error(f"Error downloading tabs for {artist}: {str(e)}")
    
    finally:
        extractor.close()
    
    return downloaded_tabs

def batch_download(extractor, artists: List[str], tabs_per_artist: int = 5, 
                  chords_per_artist: int = 2, output_dir: Optional[str] = None, 
                  browser_type: str = "firefox") -> Dict[str, List[str]]:
    """
    Download tabs and chords for multiple artists.
    
    Args:
        extractor: Initialized BrowserUGExtractor object
        artists: List of artist names
        tabs_per_artist: Number of tabs to download per artist
        chords_per_artist: Number of chords to download per artist
        output_dir: Directory to save downloaded tabs
        browser_type: Browser type (firefox or chrome)
        
    Returns:
        Dictionary with artists as keys and lists of downloaded files as values
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "downloaded_tabs")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store downloaded filenames per artist
    downloaded_files = {}
    
    # Set of already downloaded song titles to avoid duplicates
    downloaded_titles = set()
    
    # Download tabs for each artist
    for artist in artists:
        logging.info(f"Processing artist: {artist}")
        
        # Create artist directory
        artist_dir = os.path.join(output_dir, clean_filename(artist))
        os.makedirs(artist_dir, exist_ok=True)
        
        downloaded_files[artist] = []
        
        try:
            # Search for tabs
            logging.info(f"Searching for tabs by {artist}")
            tabs = extractor.search_tabs(artist)
            
            if not tabs:
                logging.warning(f"No tabs found for {artist}")
                continue
            
            # Filter for unique tab titles
            unique_tabs = []
            for tab in tabs:
                title = tab["title"]
                # Skip if we already downloaded this title
                if title in downloaded_titles:
                    continue
                unique_tabs.append(tab)
                downloaded_titles.add(title)
            
            # Split tabs into guitar tabs and chord charts
            guitar_tabs = [tab for tab in unique_tabs if tab["type"] == "tab"]
            chord_tabs = [tab for tab in unique_tabs if tab["type"] == "chords"]
            
            # Download guitar tabs
            tab_count = 0
            for tab in guitar_tabs:
                if tab_count >= tabs_per_artist:
                    break
                    
                tab_filename = f"{clean_filename(tab['title'])}.txt"
                tab_path = os.path.join(artist_dir, tab_filename)
                
                logging.info(f"Downloading tab: {tab['title']}")
                if extractor.download_tab(tab["url"], tab_path):
                    downloaded_files[artist].append(tab_path)
                    tab_count += 1
                    
                # Add a small delay between downloads
                time.sleep(1)
            
            # Download chord charts
            chord_count = 0
            for tab in chord_tabs:
                if chord_count >= chords_per_artist:
                    break
                    
                chord_filename = f"{clean_filename(tab['title'])}_chords.txt"
                chord_path = os.path.join(artist_dir, chord_filename)
                
                logging.info(f"Downloading chord chart: {tab['title']}")
                if extractor.download_tab(tab["url"], chord_path):
                    downloaded_files[artist].append(chord_path)
                    chord_count += 1
                    
                # Add a small delay between downloads
                time.sleep(1)
                
        except Exception as e:
            logging.error(f"Error processing artist {artist}: {str(e)}")
            logging.error(traceback.format_exc())
    
    return downloaded_files

def get_artists_input() -> List[str]:
    """
    Get a list of artists from user input.
    """
    print("Enter artist names (one per line). Enter empty line when done:")
    artists = []
    while True:
        artist = input().strip()
        if not artist:
            break
        artists.append(artist)
    return artists

def main():
    """
    Main entry point for the tab downloader.
    """
    parser = argparse.ArgumentParser(description="Download guitar tabs from Ultimate Guitar.")
    parser.add_argument("--artists", nargs="+", help="List of artists to download tabs for")
    parser.add_argument("--tabs-per-artist", type=int, default=5, help="Number of tabs to download per artist")
    parser.add_argument("--chords-per-artist", type=int, default=2, help="Number of chord charts to download per artist")
    parser.add_argument("--output-dir", help="Directory to save downloaded tabs")
    parser.add_argument("--browser", choices=["firefox", "chrome"], default="firefox", help="Browser to use")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--genre-mode", action="store_true", help="Download tabs by genre instead of artist")
    parser.add_argument("--tabs-per-genre", type=int, default=3, help="Number of tabs to download per genre")
    
    args = parser.parse_args()
    
    # Get artists from command line or user input
    artists = args.artists
    if not artists and not args.genre_mode:
        artists = get_artists_input()
        
    if not artists and not args.genre_mode:
        print("No artists specified. Exiting.")
        return
    
    # Create output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), "downloaded_tabs")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if args.genre_mode:
        # Download tabs by genre
        logging.info("Starting genre-based download")
        download_tabs_by_genre(tabs_per_genre=args.tabs_per_genre)
    else:
        # Initialize tab extractor
        extractor = BrowserUGExtractor(headless=args.headless, browser_type=args.browser)
        
        try:
            # Download tabs for each artist
            downloaded = batch_download(
                extractor,
                artists,
                tabs_per_artist=args.tabs_per_artist,
                chords_per_artist=args.chords_per_artist,
                output_dir=output_dir,
                browser_type=args.browser
            )
            
            # Print summary
            print("\nDownload Summary:")
            for artist, files in downloaded.items():
                print(f"{artist}: {len(files)} files")
                for file in files:
                    print(f"  - {os.path.basename(file)}")
        
        finally:
            # Ensure browser is closed
            extractor.close()

if __name__ == "__main__":
    import re
    import traceback
    main()
