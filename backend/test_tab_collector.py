#!/usr/bin/env python
"""
Test script for the Ultimate-Guitar tab data collector.
This runs a small test to validate the collection process.
"""

import os
import sys
import logging
from tab_data_collector import TabDataCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_collector")

def test_small_collection():
    """Test the collection of a small amount of tab data"""
    logger.info("Testing tab data collection with a small sample")
    
    collector = TabDataCollector(rate_limit=3)
    
    # Test with a single genre, just 2 tabs per genre
    test_genres = ["blues", "rock"]
    tabs_per_genre = 2
    
    total = 0
    for genre in test_genres:
        logger.info(f"Collecting data for genre: {genre}")
        count = collector.collect_data_for_genre(genre, num_tabs=tabs_per_genre)
        total += count
        logger.info(f"Collected {count} positions for {genre}")
    
    # Save the collected data
    collector.save_data()
    
    logger.info(f"Test complete: collected {total} positions")
    return total > 0

def main():
    """Main test function"""
    try:
        success = test_small_collection()
        if success:
            logger.info("✓ Test successful")
            print("\n✓ Data collection test successful!")
            print("  The tab data collector is working correctly.")
            print("  You can now run the full collection with: python tab_data_collector.py")
        else:
            logger.error("✗ Test failed - no data collected")
            print("\n✗ Data collection test failed!")
            print("  No position data was collected. Check the logs for errors.")
    except Exception as e:
        logger.exception("Test error")
        print(f"\n✗ Data collection test error: {str(e)}")
        print("  See the logs for details.")

if __name__ == "__main__":
    main()
