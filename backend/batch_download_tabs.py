#!/usr/bin/env python
"""
Batch Tab Downloader for OliTunes

This script automatically downloads tabs from Ultimate-Guitar.com for different genres
to build a comprehensive training dataset for the fretboard position prediction model.
"""

import os
import sys
import logging
from pathlib import Path
from ultimate_guitar_extractor import UltimateGuitarExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("batch_downloader")

# Song examples by genre/style
GENRE_EXAMPLES = {
    "blues": [
        "B.B. King The Thrill Is Gone",
        "Stevie Ray Vaughan Pride And Joy",
        "Muddy Waters Hoochie Coochie Man",
        "Eric Clapton Crossroads",
        "Robert Johnson Sweet Home Chicago"
    ],
    "rock": [
        "AC/DC Back In Black",
        "Led Zeppelin Stairway To Heaven",
        "Queen Bohemian Rhapsody",
        "Guns N Roses Sweet Child O Mine",
        "Pink Floyd Comfortably Numb"
    ],
    "jazz": [
        "Django Reinhardt Minor Swing",
        "Wes Montgomery Round Midnight",
        "Charlie Parker Donna Lee",
        "Pat Metheny Bright Size Life",
        "Joe Pass Satin Doll"
    ],
    "folk": [
        "Bob Dylan Blowin In The Wind",
        "Simon and Garfunkel Scarborough Fair",
        "John Denver Country Roads",
        "James Taylor Fire And Rain",
        "Joni Mitchell Big Yellow Taxi"
    ],
    "metal": [
        "Metallica Enter Sandman",
        "Iron Maiden The Trooper",
        "Black Sabbath Paranoid",
        "Megadeth Symphony Of Destruction",
        "Slayer Raining Blood"
    ]
}

def download_tabs_by_genre(tabs_per_genre=3, rate_limit=3):
    """
    Download tabs for each genre in the GENRE_EXAMPLES dictionary
    
    Args:
        tabs_per_genre: Number of tabs to download per genre
        rate_limit: Seconds between requests
        
    Returns:
        Dictionary with genres and downloaded tab filepaths
    """
    extractor = UltimateGuitarExtractor(rate_limit=rate_limit)
    results = {}
    
    for genre, examples in GENRE_EXAMPLES.items():
        logger.info(f"Downloading tabs for genre: {genre}")
        
        # Create genre directory
        genre_dir = os.path.join("data", "tab_data", "tab_files", genre)
        os.makedirs(genre_dir, exist_ok=True)
        
        paths = []
        for example in examples[:tabs_per_genre]:
            logger.info(f"Searching for: {example}")
            
            # Search for tabs
            search_results = extractor.search_tabs(
                example, 
                tab_type="tab", 
                min_rating=4,  # Only high-rated tabs
                limit=2  # Get top 2 results
            )
            
            if not search_results:
                logger.warning(f"No results found for {example}")
                continue
            
            # Download the first (highest-rated) result
            result = search_results[0]
            logger.info(f"Downloading: {result['artist']} - {result['song']} (Rating: {result['rating']})")
            
            filepath = extractor.download_tab(result['url'], genre)
            if filepath:
                paths.append(filepath)
                logger.info(f"Downloaded to {filepath}")
        
        results[genre] = paths
        logger.info(f"Downloaded {len(paths)} tabs for {genre}")
    
    return results

def main():
    """Main function to download tabs for all genres"""
    logger.info("Starting batch tab download")
    
    # Create necessary directories
    os.makedirs(os.path.join("data", "tab_data", "tab_files"), exist_ok=True)
    
    try:
        # Download tabs for each genre (3 per genre)
        results = download_tabs_by_genre(tabs_per_genre=3)
        
        # Count the total tabs downloaded
        total_tabs = sum(len(paths) for paths in results.values())
        
        if total_tabs > 0:
            print(f"\n✅ Successfully downloaded {total_tabs} tabs across {len(results)} genres!")
            for genre, paths in results.items():
                print(f"  • {genre}: {len(paths)} tabs")
            
            print("\nNext steps:")
            print("  1. Process the tabs with: python tab_data_processor.py")
            print("  2. Train the model with: python train_fretboard_model.py")
        else:
            print("\n❌ No tabs were downloaded. Check the logs for errors.")
    
    except Exception as e:
        logger.error(f"Error in batch download: {e}")
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
