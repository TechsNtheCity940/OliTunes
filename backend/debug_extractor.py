#!/usr/bin/env python
"""
Debug script for Ultimate Guitar Tab Extractor.
This will attempt to extract and analyze the current website structure.
"""

import os
import re
import json
import time
import random
import logging
import requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ug_debug")

# Constants - List of user agents to rotate for requests
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
]

def get_random_user_agent():
    """Get a random user agent from the list"""
    return random.choice(USER_AGENTS)

def make_request(url, headers=None, params=None):
    """Make an HTTP request with proper headers and error handling"""
    if headers is None:
        headers = {
            'User-Agent': get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            logger.error(f"Request failed with status code: {response.status_code}")
            return None
        return response
    except requests.RequestException as e:
        logger.error(f"Request error: {e}")
        return None

def debug_artist_search(artist_name):
    """Debug the artist search functionality"""
    logger.info(f"Debugging search for artist: {artist_name}")
    
    search_url = f"https://www.ultimate-guitar.com/search.php"
    params = {
        'search_type': 'band',
        'value': artist_name
    }
    
    response = make_request(search_url, params=params)
    if not response:
        logger.error("Failed to get search page")
        return
    
    # Save the raw HTML response for analysis
    with open("debug_search_response.html", "w", encoding="utf-8") as f:
        f.write(response.text)
    logger.info("Saved raw search response to debug_search_response.html")
    
    # Try to find the various patterns that might contain tab data
    logger.info("Analyzing HTML structure to find tab data...")
    
    # Method 1: Original pattern search
    store_pattern1 = re.compile(r'window\.UGAPP\.store\.page = (\{.*?\});', re.DOTALL)
    match1 = store_pattern1.search(response.text)
    if match1:
        logger.info("Found tab data using original pattern")
        with open("debug_data_pattern1.json", "w", encoding="utf-8") as f:
            try:
                json_data = json.loads(match1.group(1))
                json.dump(json_data, f, indent=2)
                logger.info("Saved JSON data from pattern 1 to debug_data_pattern1.json")
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON data from pattern 1")
    else:
        logger.info("Original pattern not found")
    
    # Method 2: Look for the js-store div with data-content attribute
    soup = BeautifulSoup(response.text, 'html.parser')
    js_store_div = soup.find('div', class_='js-store')
    if js_store_div and js_store_div.get('data-content'):
        logger.info("Found js-store div with data-content")
        with open("debug_data_jsstore.json", "w", encoding="utf-8") as f:
            try:
                json_data = json.loads(js_store_div['data-content'])
                json.dump(json_data, f, indent=2)
                logger.info("Saved JSON data from js-store div to debug_data_jsstore.json")
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON data from js-store div")
    else:
        logger.info("js-store div not found")
    
    # Method 3: Look for any script tag that might contain search results
    data_patterns = [
        r'data\s*=\s*(\{.*?\});',
        r'store\s*=\s*(\{.*?\});',
        r'results\s*=\s*(\{.*?\});',
        r'results\s*=\s*(\[.*?\]);',
        r'window\.data\s*=\s*(\{.*?\});',
    ]
    
    for i, pattern in enumerate(data_patterns):
        pattern_regex = re.compile(pattern, re.DOTALL)
        matches = pattern_regex.findall(response.text)
        for j, match in enumerate(matches[:3]):  # Limit to first 3 matches
            logger.info(f"Found potential data with pattern {i+1}, match {j+1}")
            with open(f"debug_data_pattern{i+1}_match{j+1}.json", "w", encoding="utf-8") as f:
                try:
                    json_data = json.loads(match)
                    json.dump(json_data, f, indent=2)
                    logger.info(f"Saved JSON data to debug_data_pattern{i+1}_match{j+1}.json")
                except json.JSONDecodeError:
                    logger.warning(f"Data from pattern {i+1}, match {j+1} is not valid JSON")
                    f.write(match)  # Save raw match for inspection
    
    # Method 4: Extract JSON-LD data if available
    json_ld_scripts = soup.find_all('script', type='application/ld+json')
    for i, script in enumerate(json_ld_scripts):
        logger.info(f"Found JSON-LD script {i+1}")
        with open(f"debug_jsonld_{i+1}.json", "w", encoding="utf-8") as f:
            try:
                json_data = json.loads(script.string)
                json.dump(json_data, f, indent=2)
                logger.info(f"Saved JSON-LD data to debug_jsonld_{i+1}.json")
            except json.JSONDecodeError:
                logger.warning(f"JSON-LD script {i+1} is not valid JSON")
    
    # Method 5: Look for tab listings in the HTML structure
    tab_listings = soup.find_all('div', class_=lambda cls: cls and 'js-store' in cls)
    logger.info(f"Found {len(tab_listings)} tab listing elements with js-store class")
    
    tab_listings_alternative = soup.find_all('div', class_=lambda cls: cls and 'result-link' in cls)
    logger.info(f"Found {len(tab_listings_alternative)} tab listing elements with result-link class")
    
    # Check for search results container
    search_results_container = soup.find('main', class_=lambda cls: cls and 'js-content' in cls)
    if search_results_container:
        logger.info("Found search results container")
        result_links = search_results_container.find_all('a', href=lambda href: href and '/tab/' in href)
        logger.info(f"Found {len(result_links)} direct tab links in search results")
    
    # Analyze the page structure
    logger.info("Analyzing page structure")
    print_tag_structure(soup, max_depth=3)

def print_tag_structure(soup, max_depth=3, current_depth=0):
    """Print the tag structure of the HTML to understand layout"""
    if current_depth > max_depth or not soup:
        return
    
    if hasattr(soup, 'name') and soup.name:
        indent = "  " * current_depth
        attrs = " ".join([f"{k}='{v}'" for k, v in soup.attrs.items() if k in ['id', 'class']])
        logger.info(f"{indent}{soup.name} {attrs}")
        
        if current_depth < max_depth:
            for child in list(soup.children):
                if hasattr(child, 'name') and child.name:
                    print_tag_structure(child, max_depth, current_depth + 1)

def main():
    """Main debug function"""
    try:
        print("\nDebugging Ultimate Guitar Tab Extractor")
        print("======================================\n")
        
        # Test with a few well-known artists
        artists = ["Metallica", "AC/DC", "Queen"]
        
        for artist in artists:
            print(f"\nTesting search for: {artist}")
            debug_artist_search(artist)
            time.sleep(5)  # Be nice to the server
        
        print("\nDebug completed. Check the log files and output JSON files.")
        
    except Exception as e:
        logger.exception("Error in debug script")
        print(f"Debug error: {e}")

if __name__ == "__main__":
    main()
