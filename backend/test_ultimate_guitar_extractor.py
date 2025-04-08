#!/usr/bin/env python
"""
Test script for the Ultimate Guitar tab extractor.
This script validates the functionality of the UltimateGuitarExtractor class.
"""

import os
import sys
import unittest
from unittest.mock import patch, Mock
import logging
from ultimate_guitar_extractor import UltimateGuitarExtractor, DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_ug_extractor")

class TestUltimateGuitarExtractor(unittest.TestCase):
    def setUp(self):
        """Set up the test environment"""
        self.extractor = UltimateGuitarExtractor(rate_limit=1)
        
        # Create a test directory within the main data directory
        self.test_download_folder = os.path.join(DATA_DIR, "test_downloads")
        os.makedirs(self.test_download_folder, exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove test downloads
        import shutil
        if os.path.exists(self.test_download_folder):
            shutil.rmtree(self.test_download_folder)
    
    def test_search_for_artist(self):
        """Test that the search_for_artist method returns tab data"""
        # Use a well-known artist to ensure results
        artist_name = "Metallica"
        results = self.extractor.search_for_artist(artist_name, max_pages=1)
        
        # Check if we got any results
        self.assertTrue(len(results) > 0, "No search results returned")
        
        # Check if the results have the expected structure
        for result in results[:1]:  # Just check the first result
            self.assertIn("name", result, "Missing 'name' in search result")
            self.assertIn("url", result, "Missing 'url' in search result")
            self.assertIn("type", result, "Missing 'type' in search result")
            self.assertIn("rating", result, "Missing 'rating' in search result")
            self.assertIn("votes", result, "Missing 'votes' in search result")
    
    def test_get_top_tabs_for_artist(self):
        """Test that get_top_tabs_for_artist returns the top tabs and chords"""
        artist_name = "AC/DC"
        max_tab_count = 3
        max_chord_count = 2
        
        tabs, chords = self.extractor.get_top_tabs_for_artist(
            artist_name, max_tab_count=max_tab_count, max_chord_count=max_chord_count
        )
        
        # Check if we got any results
        self.assertTrue(len(tabs) > 0, "No tabs returned")
        self.assertTrue(len(chords) > 0, "No chords returned")
        
        # Check tab count limits
        self.assertLessEqual(len(tabs), max_tab_count, f"More than {max_tab_count} tabs returned")
        self.assertLessEqual(len(chords), max_chord_count, f"More than {max_chord_count} chords returned")
        
        # Check if tabs are sorted by rating (if multiple tabs)
        if len(tabs) >= 2:
            self.assertTrue(
                tabs[0]['rating'] >= tabs[1]['rating'] or
                (tabs[0]['rating'] == tabs[1]['rating'] and tabs[0]['votes'] >= tabs[1]['votes']),
                "Tabs are not sorted by rating and votes"
            )
    
    @patch('os.path.join', return_value='test_downloads/tabs/Eagles/Hotel California.txt')
    def test_download_tab(self, mock_path):
        """Test that download_tab successfully downloads a tab"""
        # Use a known artist for testing
        artist_name = "Eagles"
        results = self.extractor.search_for_artist(artist_name, max_pages=1)
        
        # Make sure we have results
        self.assertTrue(len(results) > 0, "No search results for Eagles")
        
        # Mock directory creation to avoid file system dependency
        with patch('os.makedirs'):
            # Mock file open to avoid actual file writing
            with patch('builtins.open', unittest.mock.mock_open()) as mocked_file:
                # Download the first tab
                tab_info = results[0]
                saved_path = self.extractor.download_tab(tab_info['url'], tab_info['name'], artist_name)
                
                # Check that the file was opened for writing
                mocked_file.assert_called_once()
    
    @patch('ultimate_guitar_extractor.UltimateGuitarExtractor._make_request')
    def test_handle_multiple_pages(self, mock_make_request):
        """Test that the extractor can handle multiple pages of search results"""
        # Mock the response for the first page
        first_page_response = Mock()
        first_page_response.text = """
        <div class="js-store" data-content='{"store":{"page":{"data":{"results":[
            {"tab_url": "https://tabs.ultimate-guitar.com/tab/metallica/enter-sandman-tabs-79", "song_name": "Enter Sandman", "type": "Tab", "rating": 4.5, "votes": 1000},
            {"tab_url": "https://tabs.ultimate-guitar.com/tab/metallica/nothing-else-matters-tabs-75", "song_name": "Nothing Else Matters", "type": "Tab", "rating": 4.8, "votes": 1500}
        ]}, "pagination": {"current":1, "total":3}}}}'></div>
        """
        
        # Mock the response for the second page
        second_page_response = Mock()
        second_page_response.text = """
        <div class="js-store" data-content='{"store":{"page":{"data":{"results":[
            {"tab_url": "https://tabs.ultimate-guitar.com/tab/metallica/fade-to-black-tabs-65", "song_name": "Fade to Black", "type": "Tab", "rating": 4.6, "votes": 1200},
            {"tab_url": "https://tabs.ultimate-guitar.com/tab/metallica/master-of-puppets-tabs-54", "song_name": "Master of Puppets", "type": "Tab", "rating": 4.7, "votes": 1300}
        ]}, "pagination": {"current":2, "total":3}}}}'></div>
        """
        
        # Set up the mock to return different responses for different URLs
        def side_effect(url):
            if "page=1" in url:
                return first_page_response
            elif "page=2" in url:
                return second_page_response
            return Mock(text="")
        
        mock_make_request.side_effect = side_effect
        
        # Call the method that uses _make_request
        results = self.extractor.search_for_artist("Metallica", max_pages=2)
        
        # Check that we got combined results from both pages
        self.assertEqual(len(results), 4, "Did not get expected number of results from multiple pages")
        
        # Verify _make_request was called twice with different page parameters
        self.assertEqual(mock_make_request.call_count, 2, "_make_request should be called twice")

    @patch('ultimate_guitar_extractor.UltimateGuitarExtractor.download_tab')
    def test_batch_download_for_artist(self, mock_download_tab):
        """Test batch downloading tabs for an artist"""
        artist_name = "Queen"
        max_tab_count = 2
        max_chord_count = 1
        
        # Set up mock to return predictable paths
        test_paths = [
            os.path.join(self.test_download_folder, "Queen", "Bohemian Rhapsody.txt"),
            os.path.join(self.test_download_folder, "Queen", "We Will Rock You.txt"),
            os.path.join(self.test_download_folder, "Queen", "We Are The Champions.txt")
        ]
        mock_download_tab.side_effect = test_paths
        
        # Mock the get_top_tabs_for_artist method
        with patch('ultimate_guitar_extractor.UltimateGuitarExtractor.get_top_tabs_for_artist') as mock_get_tabs:
            # Set up mock to return predictable tab and chord data
            mock_get_tabs.return_value = (
                # Tabs
                [
                    {"name": "Bohemian Rhapsody", "url": "https://example.com/tab1", "type": "Tab"},
                    {"name": "We Will Rock You", "url": "https://example.com/tab2", "type": "Tab"},
                ],
                # Chords
                [
                    {"name": "We Are The Champions", "url": "https://example.com/chord1", "type": "Chords"},
                ]
            )
            
            # Perform the batch download
            downloaded_files = self.extractor.batch_download_for_artist(
                artist_name, max_tab_count=max_tab_count, max_chord_count=max_chord_count
            )
            
            # Check if we got the expected number of files
            self.assertEqual(len(downloaded_files), 3, "Did not download the expected number of files")
            
            # Check if the download_tab method was called the right number of times
            self.assertEqual(mock_download_tab.call_count, 3, "download_tab should be called three times")


def main():
    """Main test function"""
    try:
        # Display welcome message
        print("\nTesting UltimateGuitarExtractor functionality...")
        
        # Run all tests
        unittest.main()
        
    except Exception as e:
        logger.exception("Test error")
        print(f"\n✗ Extractor test error: {str(e)}")
        print("  See the logs for details.")

if __name__ == "__main__":
    main()
