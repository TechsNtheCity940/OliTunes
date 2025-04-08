"""
Batch Tab Downloader for Ultimate Guitar
This script allows downloading multiple tabs from multiple artists at once.
It ensures that only unique songs are downloaded by checking for duplicates.
"""

import os
import sys
import time
import logging
import argparse
from browser_ug_extractor import BrowserUGExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tab_batch_download.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("batch_downloader")

def clean_filename(name):
    """Clean a filename by removing invalid characters."""
    return "".join(c for c in name if c.isalnum() or c in " ._-").strip()

def batch_download(extractor, artists, tabs_per_artist=5, chords_per_artist=2, 
                  output_dir=None, browser_type="firefox"):
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
    downloaded_files = {}
    
    for artist in artists:
        logger.info(f"Processing artist: {artist}")
        print(f"\n{'='*50}")
        print(f"Downloading tabs for: {artist}")
        print(f"{'='*50}")
        
        # Keep track of already downloaded songs to avoid duplicates
        downloaded_songs = set()
        artist_files = []
        
        try:
            # Search for tabs by artist
            logger.info(f"Searching for tabs by artist: {artist}")
            results = extractor.search_for_artist(artist)
            
            if not results:
                logger.warning(f"No tabs found for artist: {artist}")
                print(f"No tabs found for artist: {artist}")
                continue
                
            logger.info(f"Found {len(results)} results for {artist}")
            print(f"Found {len(results)} results for {artist}")
            
            # Get unique top tabs (avoid duplicate songs)
            top_tabs = []
            top_chords = []
            
            # Process tabs
            tabs_sorted = sorted(
                [tab for tab in results if tab['type'] == 'Tab'], 
                key=lambda x: float(x.get('rating', 0) or 0), 
                reverse=True
            )
            
            # Process chords
            chords_sorted = sorted(
                [chord for chord in results if chord['type'] == 'Chords'], 
                key=lambda x: float(x.get('rating', 0) or 0), 
                reverse=True
            )
            
            # Get unique top tabs
            for tab in tabs_sorted:
                song_name = tab['name'].lower()
                if len(top_tabs) < tabs_per_artist and song_name not in downloaded_songs:
                    top_tabs.append(tab)
                    downloaded_songs.add(song_name)
            
            # Get unique top chords
            for chord in chords_sorted:
                song_name = chord['name'].lower()
                if len(top_chords) < chords_per_artist and song_name not in downloaded_songs:
                    top_chords.append(chord)
                    downloaded_songs.add(song_name)
            
            logger.info(f"Selected {len(top_tabs)} top tabs and {len(top_chords)} top chords for {artist}")
            print(f"Selected {len(top_tabs)} top tabs and {len(top_chords)} top chords for {artist}")
            
            # Download tabs
            if top_tabs:
                print("\nDownloading tabs:")
                for i, tab in enumerate(top_tabs, 1):
                    print(f"{i}. {tab['name']} - Rating: {tab.get('rating', 'N/A')}")
                    try:
                        logger.info(f"Downloading tab: {tab['name']}")
                        tab_content = extractor.download_tab(tab['url'], save_to_file=True)
                        
                        if tab_content:
                            logger.info(f"Successfully downloaded tab: {tab['name']}")
                            print(f"✓ Successfully downloaded: {tab['name']}")
                            artist_files.append(tab['name'])
                        else:
                            logger.error(f"Failed to download tab: {tab['name']}")
                            print(f"✗ Failed to download: {tab['name']}")
                            
                        # Add delay between downloads to avoid rate limiting
                        time.sleep(1.5)
                    except Exception as e:
                        logger.error(f"Error downloading tab {tab['name']}: {e}")
                        print(f"✗ Error downloading {tab['name']}: {str(e)}")
            
            # Download chords
            if top_chords:
                print("\nDownloading chords:")
                for i, chord in enumerate(top_chords, 1):
                    print(f"{i}. {chord['name']} - Rating: {chord.get('rating', 'N/A')}")
                    try:
                        logger.info(f"Downloading chord: {chord['name']}")
                        chord_content = extractor.download_tab(chord['url'], save_to_file=True)
                        
                        if chord_content:
                            logger.info(f"Successfully downloaded chord: {chord['name']}")
                            print(f"✓ Successfully downloaded: {chord['name']}")
                            artist_files.append(chord['name'])
                        else:
                            logger.error(f"Failed to download chord: {chord['name']}")
                            print(f"✗ Failed to download: {chord['name']}")
                            
                        # Add delay between downloads to avoid rate limiting
                        time.sleep(1.5)
                    except Exception as e:
                        logger.error(f"Error downloading chord {chord['name']}: {e}")
                        print(f"✗ Error downloading {chord['name']}: {str(e)}")
            
            # Add the downloaded files for this artist to the result
            downloaded_files[artist] = artist_files
            logger.info(f"Downloaded {len(artist_files)} files for {artist}")
            print(f"\nDownloaded {len(artist_files)} files for {artist}")
            
            # Wait a bit before processing the next artist
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"Error processing artist {artist}: {e}")
            print(f"Error processing artist {artist}: {str(e)}")
    
    return downloaded_files

def get_artists_input():
    """Get a list of artists from user input."""
    print("\nEnter artist names (one per line). Type 'done' when finished:")
    artists = []
    while True:
        artist = input().strip()
        if artist.lower() == 'done':
            break
        if artist:
            artists.append(artist)
    return artists

def main():
    """Main entry point for the batch downloader."""
    parser = argparse.ArgumentParser(description="Batch download tabs from Ultimate Guitar")
    parser.add_argument('--artists', nargs='+', help='List of artists to download tabs for')
    parser.add_argument('--tabs', type=int, default=5, help='Number of tabs to download per artist')
    parser.add_argument('--chords', type=int, default=2, help='Number of chords to download per artist')
    parser.add_argument('--browser', choices=['firefox', 'chrome'], default='firefox',
                      help='Browser to use for downloading')
    parser.add_argument('--output', help='Output directory for downloaded tabs')
    args = parser.parse_args()
    
    # Initialize the browser extractor
    try:
        print(f"Initializing browser-based Ultimate Guitar extractor with {args.browser}...")
        extractor = BrowserUGExtractor(browser_type=args.browser)
        logger.info(f"Browser-based Ultimate Guitar extractor initialized with {args.browser}")
    except Exception as e:
        logger.error(f"Error initializing browser extractor: {e}")
        print(f"Error initializing browser extractor: {e}")
        return
    
    try:
        # Get artists if not provided as arguments
        artists = args.artists if args.artists else get_artists_input()
        
        if not artists:
            print("No artists specified. Exiting.")
            return
            
        # Get count of tabs and chords per artist
        tabs_per_artist = args.tabs
        chords_per_artist = args.chords
        
        # Download tabs and chords
        downloaded_files = batch_download(
            extractor=extractor,
            artists=artists,
            tabs_per_artist=tabs_per_artist,
            chords_per_artist=chords_per_artist,
            output_dir=args.output,
            browser_type=args.browser
        )
        
        # Print summary
        total_artists = len(downloaded_files)
        total_files = sum(len(files) for files in downloaded_files.values())
        
        print("\n" + "="*50)
        print(f"Batch download complete!")
        print(f"Processed {total_artists} artists")
        print(f"Downloaded {total_files} files total")
        print("="*50)
        
        # Print location of downloaded files
        tab_files_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "data", "tab_data", "tab_files")
        print(f"\nDownloaded files can be found in: {tab_files_dir}")
        
    except Exception as e:
        logger.error(f"Error during batch download: {e}")
        print(f"Error during batch download: {e}")
    finally:
        # Always close the browser
        try:
            extractor.close()
            logger.info("Browser closed")
        except:
            pass

if __name__ == "__main__":
    main()
