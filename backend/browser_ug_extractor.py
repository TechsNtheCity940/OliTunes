#!/usr/bin/env python
"""
Ultimate Guitar Tab Extractor using Browser Automation

This script uses Selenium to control a web browser for extracting
tabs from Ultimate-Guitar.com, bypassing anti-scraping measures.
"""

import os
import re
import time
import json
import random
import logging
from urllib.parse import quote_plus
import requests
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException,
    StaleElementReferenceException,
    ElementNotInteractableException,
    ElementClickInterceptedException
)

# Install selenium and webdriver-manager if needed:
# pip install selenium webdriver-manager

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ug_browser_extractor")

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "tab_data")
TAB_FILES_DIR = os.path.join(DATA_DIR, "tab_files")
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
]

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TAB_FILES_DIR, exist_ok=True)

class BrowserUGExtractor:
    """Tool for extracting tabs from Ultimate-Guitar.com using browser automation"""
    
    def __init__(self, headless=True, rate_limit=2, browser_type="firefox"):
        """
        Initialize the browser-based Ultimate Guitar extractor
        
        Args:
            headless: Run the browser in headless mode (no UI)
            rate_limit: Rate limit in seconds between requests
            browser_type: Type of browser to use ('chrome' or 'firefox')
        """
        self.rate_limit = rate_limit
        self.headless = headless
        self.browser_type = browser_type
        
        # Initialize directories
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Initialize the browser driver with anti-detection settings
        self.driver = self._initialize_browser()
        
        logger.info(f"Browser-based Ultimate Guitar extractor initialized with {browser_type}")
    
    def _initialize_browser(self):
        """Initialize browser with enhanced stealth settings"""
        try:
            if self.browser_type == "firefox":
                return self._initialize_firefox()
            else:
                return self._initialize_chrome()
        except Exception as e:
            logger.error(f"Error initializing browser: {e}")
            raise
            
    def _initialize_firefox(self):
        """Initialize Firefox browser with enhanced stealth settings"""
        try:
            # Setup Firefox options with anti-detection measures
            firefox_options = FirefoxOptions()
            
            # User agent selection
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36"
            ]
            random_user_agent = random.choice(user_agents)
            firefox_options.set_preference("general.useragent.override", random_user_agent)
            
            # Randomize window size slightly to avoid detection
            width = random.randint(1200, 1600)
            height = random.randint(800, 1000)
            firefox_options.set_preference("width", width)
            firefox_options.set_preference("height", height)
            
            # Set headless mode if requested
            if self.headless:
                firefox_options.add_argument("--headless")
            
            # Add stealth settings
            firefox_options.set_preference("dom.webdriver.enabled", False)
            firefox_options.set_preference('useAutomationExtension', False)
            
            # Performance settings
            firefox_options.set_preference("dom.ipc.processCount", 1)
            firefox_options.set_preference("browser.tabs.remote.autostart", False)
            firefox_options.set_preference("browser.tabs.remote.autostart.2", False)
            
            # Use a random proxy if available
            # You would need to implement a proxy list/rotation system here
            
            # Set a custom Firefox profile to maintain cookies between sessions
            firefox_user_dir = os.path.join(DATA_DIR, "firefox_user_data")
            os.makedirs(firefox_user_dir, exist_ok=True)
            firefox_options.set_preference("profile", firefox_user_dir)
            
            # Use random profile to avoid tracking
            profiles = ['Profile 1', 'Profile 2', 'Profile 3', 'Profile 4', 'Default']
            firefox_options.set_preference("profile.default", random.choice(profiles))
            
            # Initialize Firefox driver with the configured options
            driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)
            
            # Set page load timeout
            driver.set_page_load_timeout(30)
            
            # Execute stealth JavaScript to make automation harder to detect
            stealth_js = """
                // Overwrite the 'webdriver' property to make detection harder
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => false,
                });
                
                // Remove driver property
                if (window.navigator.hasOwnProperty('webdriver')) {
                    delete Object.getPrototypeOf(navigator).webdriver;
                }
                
                // Use plugins array
                Object.defineProperty(navigator, 'plugins', {
                    get: function() {
                        // This just needs to have length > 0
                        return [1, 2, 3, 4, 5];
                    },
                });
                
                // Modify webdriver-related flags
                window.navigator.chrome = {
                    runtime: {},
                };
                
                // Add language settings like a regular browser
                Object.defineProperty(navigator, 'languages', {
                    get: function() {
                        return ['en-US', 'en', 'es'];
                    },
                });
                
                // Add media capability
                Object.defineProperty(navigator, 'permissions', {
                    get: () => ({
                        query: Promise.resolve({ state: 'granted' })
                    })
                });
                
                // Hide Puppeteer/Selenium related properties
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            """
            driver.execute_script(stealth_js)
            
            # Add a random delay to look more human-like
            time.sleep(random.uniform(1, 3))
            
            return driver
            
        except Exception as e:
            logger.error(f"Error initializing Firefox browser: {e}")
            raise

    def _initialize_chrome(self):
        """Initialize Chrome browser with enhanced stealth settings"""
        try:
            # Add Chrome-specific imports
            from selenium.webdriver.chrome.service import Service as ChromeService
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            
            # Setup Chrome options with anti-detection measures
            chrome_options = ChromeOptions()
            
            # User agent selection
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36"
            ]
            random_user_agent = random.choice(user_agents)
            chrome_options.add_argument(f'user-agent={random_user_agent}')
            
            # Randomize window size slightly to avoid detection
            width = random.randint(1200, 1600)
            height = random.randint(800, 1000)
            chrome_options.add_argument(f"--window-size={width},{height}")
            
            # Set headless mode if requested
            if self.headless:
                chrome_options.add_argument("--headless=new")
            
            # Add stealth settings
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Performance settings
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-notifications")
            chrome_options.add_argument("--disable-popup-blocking")
            
            # Set a custom Chrome profile to maintain cookies between sessions
            chrome_user_dir = os.path.join(DATA_DIR, "chrome_user_data")
            os.makedirs(chrome_user_dir, exist_ok=True)
            chrome_options.add_argument(f"--user-data-dir={chrome_user_dir}")
            
            # Use random profile to avoid tracking
            profiles = ['Profile 1', 'Profile 2', 'Profile 3', 'Profile 4', 'Default']
            chrome_options.add_argument(f"--profile-directory={random.choice(profiles)}")
            
            # Initialize Chrome driver with the configured options
            driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
            
            # Set page load timeout
            driver.set_page_load_timeout(30)
            
            # Execute stealth JavaScript to make automation harder to detect
            stealth_js = """
                // Overwrite the 'webdriver' property to make detection harder
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => false,
                });
                
                // Remove driver property
                if (window.navigator.hasOwnProperty('webdriver')) {
                    delete Object.getPrototypeOf(navigator).webdriver;
                }
                
                // Use plugins array
                Object.defineProperty(navigator, 'plugins', {
                    get: function() {
                        // This just needs to have length > 0
                        return [1, 2, 3, 4, 5];
                    },
                });
                
                // Modify webdriver-related flags
                window.navigator.chrome = {
                    runtime: {},
                };
                
                // Add language settings like a regular browser
                Object.defineProperty(navigator, 'languages', {
                    get: function() {
                        return ['en-US', 'en', 'es'];
                    },
                });
                
                // Add media capability
                Object.defineProperty(navigator, 'permissions', {
                    get: () => ({
                        query: Promise.resolve({ state: 'granted' })
                    })
                });
                
                // Hide Puppeteer/Selenium related properties
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            """
            driver.execute_script(stealth_js)
            
            # Add a random delay to look more human-like
            time.sleep(random.uniform(1, 3))
            
            return driver
            
        except Exception as e:
            logger.error(f"Error initializing Chrome browser: {e}")
            raise
    
    def _wait_with_random_delay(self, min_time=1, max_time=None):
        """Wait for a random amount of time to simulate human behavior"""
        if max_time is None:
            max_time = self.rate_limit
        
        delay = min_time + (max_time - min_time) * random.random()
        time.sleep(delay)
        
        # Add random mouse movements to appear more human-like
        if not self.headless:
            try:
                # Move mouse to random positions
                from selenium.webdriver.common.action_chains import ActionChains
                action = ActionChains(self.driver)
                for _ in range(random.randint(1, 3)):
                    x = random.randint(100, 800)
                    y = random.randint(100, 600)
                    action.move_by_offset(x, y).perform()
                    action.reset_actions()
                    time.sleep(random.random() * 0.5)
            except Exception as e:
                logger.debug(f"Random mouse movement failed: {e}")
    
    def _wait_for_element(self, by, value, timeout=15, condition=EC.presence_of_element_located, retries=3):
        """Wait for an element to be present/visible/clickable with retry logic"""
        for attempt in range(retries):
            try:
                element = WebDriverWait(self.driver, timeout).until(
                    condition((by, value))
                )
                return element
            except TimeoutException:
                if attempt < retries - 1:
                    logger.warning(f"Timed out waiting for element: {value}, retrying ({attempt+1}/{retries})")
                    # Add a random scroll to potentially reveal the element
                    try:
                        # Try scrolling to activate any lazy loading
                        self.driver.execute_script("window.scrollBy(0, 500);")
                        time.sleep(1)
                    except Exception as e:
                        logger.debug(f"Error during scrolling: {e}")
                            
            except Exception as e:
                logger.warning(f"Error waiting for element {value}: {e}")
                if attempt < retries - 1:
                    self._wait_with_random_delay(1, 2)
                else:
                    return None
        return None
        
    def search_for_artist(self, artist_name, max_pages=3, max_retries=3):
        """
        Search for an artist and extract tabs
        
        Args:
            artist_name: Name of the artist to search for
            max_pages: Maximum number of pages to extract
            max_retries: Maximum number of retries for search attempts
            
        Returns:
            List of tabs found for the artist
        """
        logger.info(f"Searching for tabs by '{artist_name}'")
        
        search_url = f"https://www.ultimate-guitar.com/search.php?search_type=title&value={quote_plus(artist_name)}"
        all_tabs = []
        current_retry = 0
        
        while current_retry < max_retries and not all_tabs:
            if current_retry > 0:
                logger.info(f"Retry attempt {current_retry} for '{artist_name}'")
                # Add longer delays between retries to avoid detection
                self._wait_with_random_delay(3, 7)
                
            try:
                # Clear cookies to avoid session tracking
                if current_retry > 1:
                    self.driver.delete_all_cookies()
                
                # Navigate to the search URL
                self.driver.get(search_url)
                
                # Wait for initial page load
                self._wait_with_random_delay(3, 5)
                
                # Handle consent popups and other initial overlays
                self._handle_consent_and_popups()
                
                # Add more human-like behavior
                self._perform_human_like_behavior()
                
                # Wait for search results with Firefox-specific timing
                wait_times = [3, 5, 7, 10]  # Firefox may need slightly longer waits
                for wait_time in wait_times:
                    logger.debug(f"Waiting {wait_time}s for search results to load")
                    
                    # Try to wait for different selectors that indicate search results
                    try:
                        # Explicit wait for any of the possible result containers
                        selectors = [
                            "article.js-store", ".js-store div[data-content]", 
                            ".search-results", ".content-results",
                            ".js-search-results", ".js-topresults",
                            "[data-content*='\"tabs\"']"
                        ]
                        
                        for selector in selectors:
                            try:
                                WebDriverWait(self.driver, wait_time).until(
                                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                                )
                                logger.info(f"Found results with selector: {selector}")
                                break
                            except TimeoutException:
                                continue
                                
                        # If we made it here, we found something
                        break
                    except (TimeoutException, StaleElementReferenceException):
                        # For Firefox, interact with the page to trigger any lazy loading
                        try:
                            # Try scrolling to activate any lazy loading
                            self.driver.execute_script("window.scrollBy(0, 500);")
                            time.sleep(1)
                        except Exception as e:
                            logger.debug(f"Error during scrolling: {e}")
                            
                # Now extract the tab data using multiple strategies
                # Strategy 1: Extract from JSON data-content
                logger.info("Attempting to extract results from JSON data-content")
                try:
                    elements_with_data = self.driver.find_elements(By.CSS_SELECTOR, "div[data-content], article.js-store, div.js-store, div.js-content")
                    
                    for element in elements_with_data:
                        try:
                            data_content = element.get_attribute('data-content')
                            if data_content:
                                logger.debug("Found data-content attribute")
                                json_data = json.loads(data_content)
                                
                                # Look for tabs in different JSON structures
                                if "store" in json_data and "page" in json_data["store"]:
                                    page_data = json_data["store"]["page"]
                                    if "data" in page_data and "results" in page_data["data"]:
                                        logger.info("Found tabs in store.page.data.results")
                                        tabs_list = page_data["data"]["results"]
                                        for tab in tabs_list:
                                            tab_data = self._parse_tab_data_from_json(tab)
                                            if tab_data:
                                                all_tabs.append(tab_data)
                                
                                # Try alternative JSON structures
                                if "results" in json_data:
                                    logger.info("Found tabs in root results property")
                                    for tab in json_data["results"]:
                                        tab_data = self._parse_tab_data_from_json(tab)
                                        if tab_data:
                                            all_tabs.append(tab_data)
                                            
                                # Another possible structure
                                if "tabs" in json_data:
                                    logger.info("Found tabs in root tabs property")
                                    for tab in json_data["tabs"]:
                                        tab_data = self._parse_tab_data_from_json(tab)
                                        if tab_data:
                                            all_tabs.append(tab_data)
                                            
                                # Another common structure
                                if "data" in json_data and "tabs" in json_data["data"]:
                                    logger.info("Found tabs in data.tabs property")
                                    for tab in json_data["data"]["tabs"]:
                                        tab_data = self._parse_tab_data_from_json(tab)
                                        if tab_data:
                                            all_tabs.append(tab_data)
                        except Exception as e:
                            logger.debug(f"Error parsing JSON data: {e}")
                            continue
                except Exception as e:
                    logger.debug(f"Error finding data-content elements: {e}")
                
                # Strategy 2: Extract from page text
                if not all_tabs:
                    logger.info("Attempting to extract results from page text")
                    try:
                        # Use JavaScript to extract tab data from the page
                        json_str = self.driver.execute_script("""
                            // Try various methods to find tab data in the page
                            
                            // Method 1: Look for data in js-store elements
                            var storeElements = document.querySelectorAll('.js-store, [data-content]');
                            for (var i = 0; i < storeElements.length; i++) {
                                var content = storeElements[i].getAttribute('data-content');
                                if (content) {
                                    try {
                                        return content;
                                    } catch (e) {}
                                }
                            }
                            
                            // Method 2: Find data in window.UGAPP or similar
                            if (typeof window.UGAPP !== 'undefined') {
                                if (window.UGAPP.store && window.UGAPP.store.page) {
                                    try {
                                        return JSON.stringify(window.UGAPP.store.page);
                                    } catch (e) {}
                                }
                            }
                            
                            // Method 3: Check the newer __REDUX_STATE__ variable
                            if (typeof window.__REDUX_STATE__ !== 'undefined') {
                                try {
                                    return JSON.stringify(window.__REDUX_STATE__);
                                } catch (e) {}
                            }
                            
                            // Method 4: Extract from any script containing tab data
                            var scripts = document.querySelectorAll('script');
                            for (var j = 0; j < scripts.length; j++) {
                                var scriptContent = scripts[j].textContent;
                                if (scriptContent.includes('"tab_url"') || 
                                    scriptContent.includes('"tabs"') || 
                                    scriptContent.includes('"artist_name"')) {
                                    return scriptContent;
                                }
                            }
                            
                            return null;
                        """)
                        
                        if json_str:
                            # Try to extract data from the returned string
                            try:
                                json_data = json.loads(json_str)
                                
                                # Look for tabs in different places in the JSON
                                tab_lists = []
                                
                                # Common paths to tab lists
                                paths = [
                                    ['data', 'results'],
                                    ['data', 'tabs'],
                                    ['page', 'data', 'results'],
                                    ['page', 'data', 'tabs'],
                                    ['results'],
                                    ['tabs'],
                                    ['store', 'page', 'data', 'results'],
                                ]
                                
                                # Try to follow each path
                                for path in paths:
                                    current = json_data
                                    valid_path = True
                                    
                                    for key in path:
                                        if key in current:
                                            current = current[key]
                                        else:
                                            valid_path = False
                                            break
                                    
                                    if valid_path and isinstance(current, list):
                                        tab_lists.append(current)
                                
                                # Process any tab lists we found
                                for tab_list in tab_lists:
                                    for tab in tab_list:
                                        tab_data = self._parse_tab_data_from_json(tab)
                                        if tab_data:
                                            all_tabs.append(tab_data)
                            except Exception as e:
                                logger.debug(f"Error parsing JSON from page text: {e}")
                    except Exception as e:
                        logger.debug(f"Error executing JavaScript to extract tabs: {e}")
                
                # Strategy 3: Extract directly from page elements
                if not all_tabs:
                    logger.info("Attempting to extract results from page elements")
                    try:
                        # Look for tab elements in the DOM using various selectors
                        element_selectors = [
                            ".content article", 
                            ".js-store article", 
                            ".results article",
                            ".tresults article",
                            ".content a[href*='/tabs/']",
                            ".content a[href*='/chords/']",
                            "a[href*='/tabs/']",
                            "a[href*='/chords/']",
                            ".content tr",
                            ".tresults tr"
                        ]
                        
                        for selector in element_selectors:
                            try:
                                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                                if elements:
                                    logger.info(f"Found {len(elements)} tab elements with selector: {selector}")
                                    
                                    for element in elements:
                                        tab_data = self._parse_tab_data_from_element(element)
                                        if tab_data:
                                            all_tabs.append(tab_data)
                                    
                                    if all_tabs:
                                        break
                            except Exception as e:
                                logger.debug(f"Error extracting tab data from elements with selector {selector}: {e}")
                    except Exception as e:
                        logger.debug(f"Error during element extraction: {e}")
                
                # If we still don't have any tabs, try pagination
                if not all_tabs and max_pages > 1:
                    logger.info("No tabs found on first page, trying pagination")
                    try:
                        # Look for pagination elements
                        pagination_selectors = [
                            ".pagination a", ".paging a", ".pages a", 
                            "a[href*='page=']", "a.next", "a.pagination-next"
                        ]
                        
                        for selector in pagination_selectors:
                            try:
                                next_page_links = self.driver.find_elements(By.CSS_SELECTOR, selector)
                                if next_page_links:
                                    logger.info(f"Found pagination links with selector: {selector}")
                                    
                                    # Try to click the next page link
                                    for link in next_page_links:
                                        try:
                                            if link.is_displayed() and link.is_enabled():
                                                # Scroll into view with Firefox-specific behavior
                                                self.driver.execute_script(
                                                    "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", 
                                                    link
                                                )
                                                time.sleep(1)
                                                
                                                # Get the link URL and navigate directly instead of clicking
                                                # This works better for Firefox
                                                href = link.get_attribute('href')
                                                if href:
                                                    logger.info(f"Navigating to next page: {href}")
                                                    self.driver.get(href)
                                                    
                                                    # Wait for page to load
                                                    self._wait_with_random_delay(3, 5)
                                                    
                                                    # Try to extract tabs again
                                                    # (recursive call would be ideal but keeping it simple)
                                                    elements = self.driver.find_elements(By.CSS_SELECTOR, "article")
                                                    for element in elements:
                                                        tab_data = self._parse_tab_data_from_element(element)
                                                        if tab_data:
                                                            all_tabs.append(tab_data)
                                                    
                                                    # Break after checking one page for simplicity
                                                    break
                                        except Exception as e:
                                            logger.debug(f"Error navigating to next page: {e}")
                                            continue
                                    
                                    break
                            except Exception as e:
                                logger.debug(f"Error with pagination selector {selector}: {e}")
                    except Exception as e:
                        logger.debug(f"Error during pagination: {e}")
            
            except Exception as e:
                logger.error(f"Error during search (attempt {current_retry+1}): {e}")
            
            # Add a longer delay between retries
            if not all_tabs:
                time.sleep(3 + current_retry * 2)
                current_retry += 1
        
        # Deduplicate tabs based on URL
        unique_tabs = []
        urls_seen = set()
        
        for tab in all_tabs:
            if tab['url'] not in urls_seen:
                urls_seen.add(tab['url'])
                unique_tabs.append(tab)
        
        logger.info(f"Found {len(unique_tabs)} unique tabs for {artist_name}")
        return unique_tabs
    
    def _perform_human_like_behavior(self):
        """Perform human-like behaviors to avoid detection"""
        try:
            # Random scrolling
            scroll_amount = random.randint(100, 300)
            self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            
            # Small random wait
            time.sleep(random.uniform(0.5, 1.5))
            
            # Maybe move mouse (using JavaScript since we can't use ActionChains in headless)
            if random.random() > 0.5:
                self.driver.execute_script("""
                    const event = new MouseEvent('mousemove', {
                        'view': window,
                        'bubbles': true,
                        'cancelable': true,
                        'clientX': Math.random() * window.innerWidth,
                        'clientY': Math.random() * window.innerHeight
                    });
                    document.dispatchEvent(event);
                """)
            
            # Another small pause
            time.sleep(random.uniform(0.3, 0.8))
            
            # Maybe scroll again
            if random.random() > 0.7:
                scroll_amount = random.randint(100, 500)
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                time.sleep(random.uniform(0.5, 1.2))
        except:
            # Ignore errors in human-like behavior
            pass
    
    def _trigger_search_results_loading(self):
        """Trigger search results to load by scrolling and other actions"""
        try:
            # Scroll down gradually to simulate user behavior and trigger lazy loading
            self.driver.execute_script("""
                // Define a function to scroll gradually
                function scrollGradually(steps, delay) {
                    const height = Math.max(
                        document.body.scrollHeight, 
                        document.documentElement.scrollHeight,
                        800  // Minimum scroll height
                    );
                    const stepSize = height / steps;
                    
                    let currentStep = 0;
                    
                    function step() {
                        if (currentStep < steps) {
                            window.scrollTo(0, stepSize * currentStep);
                            currentStep++;
                            setTimeout(step, delay);
                        }
                    }
                    
                    step();
                }
                
                // Scroll in 5 steps with 300ms delay between each
                scrollGradually(5, 300);
            """)
            
            # Wait for potential lazy loaded content
            self._wait_with_random_delay(2, 3)
            
            # Try clicking "Load more" buttons if they exist
            load_more_selectors = [
                ".load-more", 
                ".js-load-more", 
                "button[data-action='load-more']", 
                "a.more", 
                ".pagination-more",
                "button:contains('Load more')",
                "a:contains('Show more')"
            ]
            
            for selector in load_more_selectors:
                try:
                    load_more_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if load_more_elements:
                        load_more_elements[0].click()
                        self._wait_with_random_delay(1, 2)
                except:
                    pass
            
            # Final scroll to bottom to ensure everything is loaded
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self._wait_with_random_delay(1, 2)
        except:
            # Ignore errors in trigger loading
            pass
    
    def _handle_consent_and_popups(self):
        """Handle consent forms and other popups that might appear"""
        try:
            # Try multiple approaches to handle various popups
            
            # 1. Cookie consent popups
            cookie_consent_selectors = [
                "#onetrust-accept-btn-handler",
                ".accept-cookies-button",
                ".cookie-accept",
                ".cookie-consent-accept",
                ".accept-button",
                "button[data-testid='cookie-policy-dialog-accept-button']",
                "button:contains('Accept')",
                "button:contains('Accept All')",
                "button:contains('I Agree')",
                "button:contains('I Accept')",
                "button:contains('Allow')",
                ".consent button",
                ".consent-popup button",
                ".consent-banner button"
            ]
            
            for selector in cookie_consent_selectors:
                try:
                    # Use JavaScript to click the button, which works better with Firefox
                    self.driver.execute_script(f"""
                        var elements = document.querySelectorAll("{selector}");
                        for(var i=0; i<elements.length; i++) {{
                            if(elements[i].offsetParent !== null) {{  // Check if visible
                                elements[i].click();
                                return true;
                            }}
                        }}
                        return false;
                    """)
                    
                    # Short wait after clicking
                    time.sleep(1)
                except Exception as e:
                    logger.debug(f"Error handling cookie consent with selector {selector}: {e}")
            
            # 2. Generic popups and overlays
            overlay_selectors = [
                ".modal-close", 
                ".popup-close", 
                ".close-btn",
                ".js-close",
                ".modal-overlay button",
                ".overlay button",
                "button.close",
                ".modal .close",
                ".popup .close",
                ".modal-header .close",
                "[data-close]",
                "button[aria-label='Close']",
                ".fa-times",
                ".close-icon"
            ]
            
            for selector in overlay_selectors:
                try:
                    # Use JavaScript for more reliable popup closing
                    self.driver.execute_script(f"""
                        var closeButtons = document.querySelectorAll("{selector}");
                        for(var i=0; i<closeButtons.length; i++) {{
                            if(closeButtons[i].offsetParent !== null) {{  // Check if visible
                                closeButtons[i].click();
                            }}
                        }}
                    """)
                    
                    # Wait a short time after clicking
                    time.sleep(0.5)
                except Exception as e:
                    logger.debug(f"Error handling overlay with selector {selector}: {e}")
            
            # 3. Firefox-specific: Handle notifications permission prompt
            try:
                # Firefox may show a notification permission prompt
                self.driver.execute_script("""
                    try {
                        if (typeof Notification !== 'undefined') {
                            Notification.requestPermission = function() {
                                return Promise.resolve('denied');
                            };
                        }
                    } catch(e) {
                        console.error('Error handling notification permissions:', e);
                    }
                """)
            except Exception as e:
                logger.debug(f"Error handling notification permissions: {e}")
            
            # 4. Use Escape key to close any remaining popups
            try:
                # Find the body element and send escape key
                body = self.driver.find_element(By.TAG_NAME, "body")
                body.send_keys(Keys.ESCAPE)
                time.sleep(0.5)
                
                # Send escape again just to be sure
                body.send_keys(Keys.ESCAPE)
            except Exception as e:
                logger.debug(f"Error sending escape key: {e}")
            
            # 5. Ultimate Guitar specific popups
            ug_popup_selectors = [
                ".welcome-popup button",
                ".welcome-overlay button",
                ".js-welcome-close",
                ".js-gtm--ad-popup-close",
                ".ad-popup-close",
                ".premium-offer-close",
                ".offer-close",
                ".js-in-app-close",
                ".js-alert-close",
                ".alert-close"
            ]
            
            for selector in ug_popup_selectors:
                try:
                    self.driver.execute_script(f"""
                        var closeButtons = document.querySelectorAll("{selector}");
                        for(var i=0; i<closeButtons.length; i++) {{
                            closeButtons[i].click();
                        }}
                    """)
                    time.sleep(0.5)
                except Exception as e:
                    logger.debug(f"Error handling UG popup with selector {selector}: {e}")
            
            # 6. Handle iframes (like ad iframes) that might block interaction
            try:
                self.driver.execute_script("""
                    // Remove any overlay iframes
                    var iframes = document.querySelectorAll('iframe');
                    for (var i = 0; i < iframes.length; i++) {
                        if (iframes[i].style.zIndex > 1000 || 
                            getComputedStyle(iframes[i]).zIndex > 1000) {
                            iframes[i].remove();
                        }
                    }
                    
                    // Remove any fixed position elements that might be overlays
                    var fixed = document.querySelectorAll('div[style*="position: fixed"], div[style*="position:fixed"]');
                    for (var j = 0; j < fixed.length; j++) {
                        if (fixed[j].style.zIndex > 1000 || 
                            getComputedStyle(fixed[j]).zIndex > 1000) {
                            fixed[j].style.display = 'none';
                        }
                    }
                """)
            except Exception as e:
                logger.debug(f"Error handling iframes: {e}")
                
        except Exception as e:
            logger.debug(f"Error in popup handling: {e}")
            # Don't raise the exception, just continue
    
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
        logger.info(f"Searching for tabs by artist: {artist_name}")
        
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
    
    def download_tab(self, url, save_to_file=False):
        """
        Download a tab from Ultimate-Guitar and optionally save it to a file
        
        Args:
            url: URL of the tab to download
            save_to_file: Whether to save the tab to a file
            
        Returns:
            Tab content as a string
        """
        try:
            # Navigate to the tab page
            logger.info(f"Downloading tab from {url}")
            
            # Add a random delay to mimic human behavior
            self._wait_with_random_delay(min_time=2, max_time=4)
            
            # Load the tab page
            self.driver.get(url)
            
            # Wait for the page to load with retry logic
            self._wait_for_page_load(retries=3)
            
            # Handle any consent forms or popups
            self._handle_consent_and_popups()
            
            # Add a short delay after handling popups
            time.sleep(random.uniform(1, 2))
            
            # Try extracting the tab content using multiple approaches
            tab_content = self._extract_tab_content_multi_strategy()
            
            if not tab_content:
                logger.error("Failed to extract tab content after multiple attempts")
                return None
            
            # If save_to_file is True, save the tab to a file
            if save_to_file:
                # Try to extract artist and song name from the page
                artist_name = self._extract_artist_name()
                song_name = self._extract_song_name()
                
                if not song_name:
                    # Try to extract from URL if not found on page
                    parts = url.split('/')
                    if len(parts) > 4:
                        song_name = parts[-1].split('-')[0]
                
                if not artist_name:
                    # Try to extract from URL if not found on page
                    parts = url.split('/')
                    if len(parts) > 4:
                        artist_name = parts[-2]
                
                # Create a safe filename
                if artist_name and song_name:
                    artist_name = re.sub(r'[\\/*?:"<>|]', "_", artist_name)
                    song_name = re.sub(r'[\\/*?:"<>|]', "_", song_name)
                    filename = f"{artist_name}_{song_name}.txt"
                else:
                    # Use the URL as a fallback
                    url_parts = url.split('/')
                    filename = f"tab_{url_parts[-1]}.txt"
                
                # Save the tab to a file
                file_path = os.path.join(TAB_FILES_DIR, filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(tab_content)
                
                logger.info(f"Tab saved to {file_path}")
            
            return tab_content
            
        except Exception as e:
            logger.error(f"Error downloading tab: {e}")
            return None
    
    def _extract_tab_content_multi_strategy(self):
        """
        Extract tab content using multiple strategies to improve reliability
        
        Returns:
            Tab content as a string or None if all strategies fail
        """
        # Try up to 3 times to extract tab content
        for attempt in range(3):
            try:
                logger.info(f"Extracting tab content (attempt {attempt+1}/3)")
                
                # Try each strategy in turn
                content = None
                
                # Strategy 1: Look for the tab content in the js-store div (JSON data)
                if not content:
                    content = self._extract_tab_content_from_json()
                
                # Strategy 2: Extract from pre-formatted tab containers 
                if not content:
                    content = self._extract_tab_content_from_elements()
                
                # Strategy 3: Extract from dedicated content container
                if not content:
                    content = self._extract_tab_content_from_container()
                
                # Strategy 4: Get raw HTML and parse
                if not content:
                    content = self._extract_tab_content_from_html()
                
                # Strategy 5: Extract from page text (last resort)
                if not content:
                    content = self._extract_tab_content_from_text()
                
                # Return content if any strategy succeeded
                if content:
                    return content
                
                # If all strategies failed, wait a bit and retry
                logger.warning(f"All attempts to extract tab content failed {attempt+1}, retrying...")
                time.sleep(2)
                
                # Handle any popups that might have appeared
                self._handle_consent_and_popups()
                
                # Refresh the page to retry
                if attempt < 2:
                    self.driver.refresh()
                    self._wait_for_page_load()
                
            except Exception as e:
                logger.error(f"Error extracting tab content: {e}")
        
        return None
    
    def _extract_tab_content_from_json(self):
        """Extract tab content from JSON data in js-store div"""
        try:
            # Method 1: Using JavaScript to extract the store data
            js_store_data = self.driver.execute_script("""
                try {
                    // Try to find the js-store div
                    var jsStore = document.querySelector('div.js-store');
                    if (jsStore && jsStore.dataset && jsStore.dataset.content) {
                        return jsStore.dataset.content;
                    }
                    
                    // Look for other data containers
                    var dataContainers = document.querySelectorAll('[data-content]');
                    for (var i = 0; i < dataContainers.length; i++) {
                        if (dataContainers[i].dataset.content && 
                            dataContainers[i].dataset.content.includes('content') && 
                            dataContainers[i].dataset.content.includes('tab_view')) {
                            return dataContainers[i].dataset.content;
                        }
                    }
                    
                    // Check for window store data
                    if (window.UGAPP && window.UGAPP.store && window.UGAPP.store.page) {
                        return JSON.stringify(window.UGAPP.store.page);
                    }
                    
                    // Check for newer API data
                    if (window.dataLayer && window.dataLayer.length > 0) {
                        for (var j = 0; j < window.dataLayer.length; j++) {
                            if (window.dataLayer[j].hasOwnProperty('data') && 
                                window.dataLayer[j].data && 
                                window.dataLayer[j].data.hasOwnProperty('tab')) {
                                return JSON.stringify(window.dataLayer[j].data);
                            }
                        }
                    }
                    
                    return null;
                } catch (e) {
                    console.error('Error extracting store data:', e);
                    return null;
                }
            """)
            
            if js_store_data:
                try:
                    # Parse the JSON data
                    data = json.loads(js_store_data)
                    
                    # Method 1: Look for tab_view.wiki_tab.content
                    if 'tab_view' in data and 'wiki_tab' in data['tab_view'] and 'content' in data['tab_view']['wiki_tab']:
                        return data['tab_view']['wiki_tab']['content']
                    
                    # Method 2: Look for data.tab.content
                    if 'data' in data and 'tab' in data['data'] and 'content' in data['data']['tab']:
                        return data['data']['tab']['content']
                    
                    # Method 3: Look for page.data.tab.content
                    if 'page' in data and 'data' in data['page'] and 'tab' in data['page']['data'] and 'content' in data['page']['data']['tab']:
                        return data['page']['data']['tab']['content']
                    
                    # Method 4: Look for data.tab_view.wiki_tab.content
                    if 'data' in data and 'tab_view' in data['data'] and 'wiki_tab' in data['data']['tab_view'] and 'content' in data['data']['tab_view']['wiki_tab']:
                        return data['data']['tab_view']['wiki_tab']['content']
                    
                    # Method 5: Look for content directly
                    if 'content' in data:
                        return data['content']
                    
                    # Method 6: Look for store.page.data.tab.content
                    if 'store' in data and 'page' in data['store'] and 'data' in data['store']['page'] and 'tab' in data['store']['page']['data'] and 'content' in data['store']['page']['data']['tab']:
                        return data['store']['page']['data']['tab']['content']
                    
                    # Advanced recursive search for content field
                    content = self._find_content_in_json(data)
                    if content:
                        return content
                        
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON data from js-store")
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting tab content from JSON: {e}")
            return None
    
    def _find_content_in_json(self, json_obj, max_depth=10, current_depth=0):
        """
        Recursively search for content field in nested JSON
        
        Args:
            json_obj: JSON object to search in
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth
            
        Returns:
            Content string if found, None otherwise
        """
        if current_depth > max_depth:
            return None
            
        if isinstance(json_obj, dict):
            # Check for content field
            if 'content' in json_obj and isinstance(json_obj['content'], str) and len(json_obj['content']) > 100:
                return json_obj['content']
                
            # Check for tab content field patterns
            for key in ['wiki_tab', 'tab', 'tab_view', 'data', 'page']:
                if key in json_obj:
                    # Recurse into nested object
                    result = self._find_content_in_json(json_obj[key], max_depth, current_depth + 1)
                    if result:
                        return result
                        
            # Try all other keys
            for key in json_obj:
                result = self._find_content_in_json(json_obj[key], max_depth, current_depth + 1)
                if result:
                    return result
                    
        elif isinstance(json_obj, list):
            # Try each item in the list
            for item in json_obj:
                result = self._find_content_in_json(item, max_depth, current_depth + 1)
                if result:
                    return result
                    
        return None
    
    def _extract_tab_content_from_elements(self):
        """Extract tab content from DOM elements"""
        try:
            # Try multiple selectors to find the tab content
            selectors = [
                # Main content selectors
                "div.js-tab-content", 
                "div.tb_ct", 
                ".js-tab-content",
                ".tab-content",
                ".tab_content",
                ".js-chord-sheet",
                
                # Legacy selectors
                "pre.js-tab-content",
                "pre.tb_ct",
                "#cont",
                ".content",
                ".ugm-tab-content",
                "[data-content-type='tab']",
                
                # Specific UG selectors
                ".js-tab-wrapper .js-tab",
                "div[data-content-type='tabs']",
                ".tab-content-wrapper"
            ]
            
            for selector in selectors:
                try:
                    # Wait for the element with a timeout
                    wait = WebDriverWait(self.driver, 10)
                    element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                    
                    if element:
                        # Get the content
                        content = element.get_attribute("textContent") or element.text
                        
                        # Check if content is valid
                        if content and len(content.strip()) > 100:
                            return content.strip()
                            
                except TimeoutException:
                    logger.warning(f"Timed out waiting for element: {selector}, trying next selector")
                except Exception as e:
                    logger.debug(f"Error finding element with selector {selector}: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting tab content from elements: {e}")
            return None
    
    def _extract_tab_content_from_container(self):
        """Extract tab content from the dedicated container elements"""
        try:
            # Try to find any pre-formatted text elements that might contain tab data
            js_code = """
                try {
                    // Look for pre-formatted text elements
                    var preElements = document.querySelectorAll('pre');
                    for (var i = 0; i < preElements.length; i++) {
                        var content = preElements[i].textContent || preElements[i].innerText;
                        if (content && content.length > 100 && 
                            (content.includes('|----') || content.includes('e|----') || 
                             content.includes('E|----') || content.includes('Chords used'))) {
                            return content;
                        }
                    }
                    
                    // Look for tab notation in div elements
                    var divElements = document.querySelectorAll('div');
                    for (var j = 0; j < divElements.length; j++) {
                        var divContent = divElements[j].textContent || divElements[j].innerText;
                        if (divContent && divContent.length > 100 && 
                            (divContent.includes('|----') || divContent.includes('e|----') || 
                             divContent.includes('E|----') || divContent.includes('Chords used'))) {
                            return divContent;
                        }
                    }
                    
                    // Look for tab content in any elements with "tab" or "chord" in their class
                    var tabElements = document.querySelectorAll('[class*=tab], [class*=chord], [class*=Tab], [class*=Chord]');
                    for (var k = 0; k < tabElements.length; k++) {
                        var tabContent = tabElements[k].textContent || tabElements[k].innerText;
                        if (tabContent && tabContent.length > 100) {
                            return tabContent;
                        }
                    }
                    
                    return null;
                } catch (e) {
                    console.error('Error extracting tab content from container:', e);
                    return null;
                }
            """
            
            content = self.driver.execute_script(js_code)
            
            if content and len(content.strip()) > 100:
                return content.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting tab content from container: {e}")
            return None
    
    def _extract_tab_content_from_html(self):
        """Extract tab content from the page HTML"""
        try:
            # Get the page HTML
            html = self.driver.page_source
            
            # Parse the HTML
            soup = BeautifulSoup(html, "html.parser")
            
            # Method 1: Find pre tags with tab content
            pre_tags = soup.find_all("pre")
            for pre in pre_tags:
                content = pre.get_text()
                if content and len(content.strip()) > 100 and any(marker in content for marker in ['|----', 'e|----', 'E|----', 'Chords used']):
                    return content.strip()
            
            # Method 2: Look for common tab content containers
            for class_name in ["js-tab-content", "tb_ct", "tab-content", "tab_content", "js-chord-sheet"]:
                element = soup.find(class_=class_name)
                if element:
                    content = element.get_text()
                    if content and len(content.strip()) > 100:
                        return content.strip()
            
            # Method 3: Look for content in data attributes
            js_store = soup.find("div", class_="js-store")
            if js_store and js_store.get("data-content"):
                try:
                    data = json.loads(js_store["data-content"])
                    # Try to find content in the data
                    result = self._find_content_in_json(data)
                    if result:
                        return result
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON data from js-store")
            
            # Method 4: Look for any div containing tab-like content
            divs = soup.find_all("div")
            for div in divs:
                content = div.get_text()
                if content and len(content.strip()) > 100 and any(marker in content for marker in ['|----', 'e|----', 'E|----', 'Chords used']):
                    return content.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting tab content from HTML: {e}")
            return None
    
    def _extract_tab_content_from_text(self):
        """Extract tab content from the page text (last resort)"""
        try:
            # Get all text from the page
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            
            # Look for patterns that indicate tab content
            tab_patterns = [
                r'(e\|[-\d\w\s]*?)\n(B\|[-\d\w\s]*?)\n(G\|[-\d\w\s]*?)\n(D\|[-\d\w\s]*?)\n(A\|[-\d\w\s]*?)\n(E\|[-\d\w\s]*?)',
                r'(E\|[-\d\w\s]*?)\n(A\|[-\d\w\s]*?)\n(D\|[-\d\w\s]*?)\n(G\|[-\d\w\s]*?)\n(B\|[-\d\w\s]*?)\n(e\|[-\d\w\s]*?)',
                r'(Chord diagrams:.*?)(?=\n\n|\Z)',
                r'(Chords used:.*?)(?=\n\n|\Z)',
                r'(\|-----------\|.*?)(?=\n\n|\Z)'
            ]
            
            for pattern in tab_patterns:
                matches = re.findall(pattern, page_text, re.DOTALL)
                if matches:
                    if isinstance(matches[0], tuple):
                        # For multi-line patterns (e.g., guitar tabs)
                        content = "\n".join(matches[0])
                    else:
                        # For single content block
                        content = matches[0]
                    
                    if content and len(content.strip()) > 100:
                        return content.strip()
            
            # If no specific pattern is found, look for large blocks of text
            # that might be lyrics or chord sheets
            paragraphs = page_text.split('\n\n')
            for paragraph in paragraphs:
                if len(paragraph.strip()) > 300:  # Longer content is more likely to be the tab
                    return paragraph.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting tab content from text: {e}")
            return None
    
    def _extract_artist_name(self):
        """Extract artist name from the page"""
        try:
            # Try multiple selectors to find the artist name
            selectors = [
                "h1 .artist-name", 
                ".artist-name",
                ".artist_name",
                ".artist-header__name", 
                "[itemprop='byArtist']",
                ".tabs-title-details .artist",
                ".js-article-metadata .artist",
                ".tab-title__artist a"
            ]
            
            for selector in selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        return elements[0].text.strip()
                except Exception:
                    pass
            
            # Try extracting from the title
            title = self.driver.title
            if " by " in title:
                return title.split(" by ")[1].split(" @ ")[0].strip()
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting artist name: {e}")
            return None
    
    def _extract_song_name(self):
        """Extract song name from the page"""
        try:
            # Try multiple selectors to find the song name
            selectors = [
                "h1 .song-name", 
                ".song-name",
                ".song_name",
                ".tab-title__song", 
                "[itemprop='name']",
                ".tabs-title-details .title",
                ".js-article-metadata .title",
                ".tab-title__title"
            ]
            
            for selector in selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        return elements[0].text.strip()
                except Exception:
                    pass
            
            # Try extracting from the title
            title = self.driver.title
            if " by " in title:
                return title.split(" by ")[0].strip()
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting song name: {e}")
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
    
    def close(self):
        """Close the browser and clean up resources"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Browser closed")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")

    def _parse_tab_data_from_element(self, element):
        """
        Parse tab data from a search result element (article or similar)
        
        Args:
            element: WebElement containing tab data
            
        Returns:
            Dictionary with tab data or None if failed
        """
        try:
            # Initialize tab data with defaults
            tab_data = {
                'name': '',
                'artist': '',
                'type': 'Tab',  # Default type
                'url': '',
                'rating': 0.0,
                'votes': 0
            }
            
            # Try to find the URL (most important piece)
            links = element.find_elements(By.TAG_NAME, 'a')
            for link in links:
                href = link.get_attribute('href')
                if href and ('/tabs/' in href or '/chords/' in href):
                    tab_data['url'] = href
                    break
            
            # If no URL found, this isn't a valid tab
            if not tab_data['url']:
                return None
                
            # Try to find the song name using various selectors
            name_selectors = [
                ".title", 
                ".name", 
                ".song-name", 
                "h2", 
                "h3", 
                ".song", 
                "a.link",
                ".js-title"
            ]
            
            for selector in name_selectors:
                try:
                    name_element = element.find_element(By.CSS_SELECTOR, selector)
                    if name_element and name_element.text.strip():
                        tab_data['name'] = name_element.text.strip()
                        break
                except:
                    pass
            
            # If no name found using selectors, try to infer from URL
            if not tab_data['name'] and tab_data['url']:
                url_parts = tab_data['url'].split('/')
                if len(url_parts) >= 6:
                    tab_data['name'] = url_parts[-2].replace('_', ' ').title()
            
            # Try to get the tab type
            type_selectors = [
                ".type", 
                ".tab-type", 
                "[data-type]", 
                ".js-type"
            ]
            
            for selector in type_selectors:
                try:
                    type_element = element.find_element(By.CSS_SELECTOR, selector)
                    if type_element and type_element.text.strip():
                        type_text = type_element.text.strip().lower()
                        
                        if "chord" in type_text:
                            tab_data['type'] = "Chords"
                        elif "guitar pro" in type_text or "gp" in type_text:
                            tab_data['type'] = "Guitar Pro"
                        elif "bass" in type_text:
                            tab_data['type'] = "Bass Tab"
                        elif "tab" in type_text:
                            tab_data['type'] = "Tab"
                        break
                except:
                    pass
            
            # If type not found from element, try to determine from URL
            if tab_data['type'] == "Tab" and tab_data['url']:
                url = tab_data['url'].lower()
                if "crd" in url:
                    tab_data['type'] = "Chords"
                elif "btab" in url:
                    tab_data['type'] = "Bass Tab"
                elif "guitar_pro" in url or "_gp_" in url:
                    tab_data['type'] = "Guitar Pro"
            
            # Try to get rating
            rating_selectors = [
                ".rating", 
                ".tab-rating", 
                ".js-rating",
                ".rate"
            ]
            
            for selector in rating_selectors:
                try:
                    rating_element = element.find_element(By.CSS_SELECTOR, selector)
                    if rating_element and rating_element.text.strip():
                        # Extract numeric value from text
                        rating_text = rating_element.text.strip()
                        rating_match = re.search(r'(\d+(\.\d+)?)', rating_text)
                        if rating_match:
                            tab_data['rating'] = float(rating_match.group(1))
                        break
                except:
                    pass
            
            # Get artist name if available
            artist_selectors = [
                ".artist", 
                ".js-artist", 
                ".artistname", 
                "[data-artist]"
            ]
            
            for selector in artist_selectors:
                try:
                    artist_element = element.find_element(By.CSS_SELECTOR, selector)
                    if artist_element and artist_element.text.strip():
                        tab_data['artist'] = artist_element.text.strip()
                        break
                except:
                    pass
            
            # If we have a URL but no name, consider this a failure
            if not tab_data['name']:
                return None
                
            return tab_data
            
        except Exception as e:
            logger.debug(f"Error parsing tab data from element: {e}")
            return None
    
    def _parse_tab_data_from_json(self, json_tab):
        """
        Parse tab data from a JSON tab object
        
        Args:
            json_tab: JSON object containing tab data
            
        Returns:
            Dictionary with tab data or None if failed
        """
        try:
            # Initialize with a default structure
            tab_data = {
                'name': '',
                'artist': '',
                'type': 'Tab',  # Default type
                'url': '',
                'rating': 0.0,
                'votes': 0
            }
            
            # Try different key variations that can appear in the JSON
            
            # Names
            for name_key in ['song_name', 'title', 'name', 'song', 'tab_name']:
                if name_key in json_tab and json_tab[name_key]:
                    tab_data['name'] = json_tab[name_key]
                    break
            
            # Artists
            for artist_key in ['artist_name', 'artist', 'author']:
                if artist_key in json_tab and json_tab[artist_key]:
                    tab_data['artist'] = json_tab[artist_key]
                    break
            
            # URLs
            for url_key in ['tab_url', 'url', 'link']:
                if url_key in json_tab and json_tab[url_key]:
                    url = json_tab[url_key]
                    
                    # Fix URLs that don't have the scheme/host
                    if url.startswith('/'):
                        url = f"https://tabs.ultimate-guitar.com{url}"
                    
                    tab_data['url'] = url
                    break
            
            # Types
            for type_key in ['type_name', 'type', 'tab_type']:
                if type_key in json_tab and json_tab[type_key]:
                    tab_type = json_tab[type_key]
                    
                    # Normalize common type values
                    type_lower = str(tab_type).lower()
                    if "chord" in type_lower:
                        tab_data['type'] = "Chords"
                    elif "bass" in type_lower:
                        tab_data['type'] = "Bass Tab"
                    elif "pro" in type_lower or "gp" in type_lower:
                        tab_data['type'] = "Guitar Pro"
                    elif "tab" in type_lower:
                        tab_data['type'] = "Tab"
                    else:
                        tab_data['type'] = tab_type
                    break
            
            # Ratings
            for rating_key in ['rating', 'average', 'average_rating', 'raiting']:
                if rating_key in json_tab:
                    try:
                        tab_data['rating'] = float(json_tab[rating_key])
                        break
                    except:
                        pass
            
            # Votes
            for votes_key in ['votes', 'vote_count', 'votes_count', 'number_of_votes']:
                if votes_key in json_tab:
                    try:
                        tab_data['votes'] = int(json_tab[votes_key])
                        break
                    except:
                        pass
            
            # Validate essential fields exist
            if not tab_data['name'] or not tab_data['url']:
                return None
                
            return tab_data
            
        except Exception as e:
            logger.debug(f"Error parsing tab data from JSON: {e}")
            return None

    def _wait_for_page_load(self, timeout=30, retries=3):
        """
        Wait for the page to fully load with retry logic
        
        Args:
            timeout: Timeout in seconds for each attempt
            retries: Number of retry attempts
            
        Returns:
            True if page loaded successfully, False otherwise
        """
        for attempt in range(retries):
            try:
                # Wait for the document to be in ready state
                wait = WebDriverWait(self.driver, timeout)
                wait.until(lambda d: d.execute_script('return document.readyState') == 'complete')
                
                # Additional check for common page elements to ensure the page is fully loaded
                self.driver.execute_script("""
                    return new Promise((resolve) => {
                        // Check if important resources are loaded
                        if (document.readyState === 'complete') {
                            // Give a small delay for any async resources
                            setTimeout(resolve, 500);
                        } else {
                            // Add event listener for load event
                            window.addEventListener('load', () => {
                                setTimeout(resolve, 500);
                            });
                        }
                    });
                """)
                
                # Add a small delay with randomization to mimic human browsing
                self._wait_with_random_delay(0.5, 1.5)
                
                return True
                
            except TimeoutException:
                logger.warning(f"Timed out waiting for page to load (attempt {attempt+1}/{retries})")
                
                if attempt < retries - 1:
                    # Try to scroll the page to trigger lazy loading
                    try:
                        self.driver.execute_script("window.scrollTo(0, 200);")
                        time.sleep(1)
                    except:
                        pass
                    
                    # Wait a moment before retrying
                    time.sleep(2)
                else:
                    logger.error("All attempts to wait for page load failed")
                    return False
                    
            except Exception as e:
                logger.error(f"Error waiting for page to load: {e}")
                
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    return False
                
        return False

    def search_tabs(self, query):
        """
        Search for tabs using the query string
        
        Args:
            query: String to search for (artist, song, etc.)
            
        Returns:
            List of search results with tab information
        """
        try:
            logger.info(f"Searching for tabs matching: {query}")
            
            # Format query for URL
            search_url = f"https://www.ultimate-guitar.com/search.php?search_type=title&value={quote_plus(query)}"
            
            # Navigate to the search page
            self.driver.get(search_url)
            time.sleep(self.rate_limit)  # Respect rate limit
            
            # Wait for search results to load (modern UG site uses React/JS rendering)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[class*='SearchPage']"))
            )
            
            # Allow extra time for results to render
            time.sleep(2)
            
            # Find and extract all search result items
            results = []
            
            try:
                result_rows = self.driver.find_elements(By.CSS_SELECTOR, 
                                                      "div[class*='result-row'], div[class*='ResultItem']")
                
                for row in result_rows:
                    try:
                        # Skip ads and non-tab results
                        if not row.get_attribute("class") or "adv" in row.get_attribute("class").lower():
                            continue
                            
                        result = {}
                        
                        # Title
                        title_element = row.find_element(By.CSS_SELECTOR, "div[class*='song-name'], a[class*='song']")
                        result["title"] = title_element.text.strip()
                        
                        # URL for tab
                        try:
                            url = title_element.get_attribute("href")
                            result["url"] = url
                        except:
                            # Try to find link in parent elements
                            try:
                                link_element = row.find_element(By.TAG_NAME, "a")
                                result["url"] = link_element.get_attribute("href")
                            except:
                                result["url"] = None
                        
                        # Artist
                        try:
                            artist_element = row.find_element(By.CSS_SELECTOR, "div[class*='artist'], a[class*='artist']")
                            result["artist"] = artist_element.text.strip()
                        except NoSuchElementException:
                            result["artist"] = "Unknown"
                        
                        # Type and rating
                        try:
                            type_element = row.find_element(By.CSS_SELECTOR, "div[class*='type'], span[class*='type']")
                            result["type"] = type_element.text.strip().lower()
                        except NoSuchElementException:
                            result["type"] = "unknown"
                            
                        try:
                            rating_element = row.find_element(By.CSS_SELECTOR, "div[class*='rating'], span[class*='rating']")
                            rating_text = rating_element.text.strip()
                            # Extract numbers from rating text
                            rating_match = re.search(r"(\d+\.?\d*)", rating_text)
                            if rating_match:
                                result["rating"] = float(rating_match.group(1))
                            else:
                                result["rating"] = 0.0
                        except (NoSuchElementException, ValueError):
                            result["rating"] = 0.0
                        
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Error extracting result row: {str(e)}")
                        continue
                
                logger.info(f"Found {len(results)} tab search results for '{query}'")
            except Exception as e:
                logger.error(f"Error finding search results: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search_tabs: {str(e)}")
            return []

def main():
    """Main entry point for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultimate Guitar Tab Extractor with Browser Automation")
    parser.add_argument('--artist', help='Artist name to search for')
    parser.add_argument('--browser', choices=['firefox', 'chrome'], default='firefox',
                      help='Browser to use (default: firefox)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        run_interactive_mode(args.browser, args.headless)
        return
    
    # Command-line mode (non-interactive)
    if args.artist:
        try:
            print("\nUltimate Guitar Tab Extractor (Browser Automation)")
            print("================================================\n")
            
            extractor = BrowserUGExtractor(headless=args.headless, rate_limit=2, browser_type=args.browser)
            
            try:
                # Search for tabs by artist
                print(f"Searching for tabs by '{args.artist}'...")
                results = extractor.search_for_artist(args.artist)
                
                if results:
                    print(f"Found {len(results)} tabs for {args.artist}")
                    
                    # Display top 5 results
                    for i, tab in enumerate(results[:5]):
                        print(f"{i+1}. {tab['name']} - {tab['type']} - Rating: {tab.get('rating', 'N/A')}")
                else:
                    print(f"No tabs found for {args.artist}")
            
            finally:
                extractor.close()
                
        except Exception as e:
            print(f"Error: {e}")
    else:
        parser.print_help()

def run_interactive_mode(browser_type='firefox', headless=False):
    """Run the extractor in interactive mode with menu options"""
    import time
    
    print("\nBrowser-based Ultimate Guitar Tab Extractor - Interactive Mode")
    print("============================================================")
    
    # Set up the browser extractor
    try:
        print(f"\nInitializing browser extractor with {browser_type}...")
        extractor = BrowserUGExtractor(browser_type=browser_type, headless=headless)
        print("Browser extractor initialized successfully!")
    except Exception as e:
        print(f"Error initializing browser extractor: {e}")
        return
    
    try:
        while True:
            print("\nSelect an option:")
            print("1. Search for an artist")
            print("2. Get top tabs for an artist")
            print("3. Download a specific tab")
            print("4. Run full test sequence on an artist")
            print("5. Batch download for multiple artists")
            print("0. Exit")
            
            choice = input("\nYour choice: ")
            
            try:
                if choice == "0":
                    break
                    
                elif choice == "1":
                    artist_name = input("\nEnter artist name: ")
                    print(f"\nSearching for tabs by '{artist_name}'...")
                    results = extractor.search_for_artist(artist_name)
                    
                    if results:
                        print(f"Found {len(results)} tabs for {artist_name}")
                        
                        # Display top 5 results
                        for i, tab in enumerate(results[:5]):
                            print(f"{i+1}. {tab['name']} - {tab['type']} - Rating: {tab.get('rating', 'N/A')}")
                    else:
                        print(f"No tabs found for {artist_name}")
                    
                elif choice == "2":
                    artist_name = input("\nEnter artist name: ")
                    
                    # First, search for the artist to get all tabs
                    print(f"\nSearching for tabs by '{artist_name}'...")
                    results = extractor.search_for_artist(artist_name)
                    
                    if not results:
                        print(f"No results found for {artist_name}")
                        continue
                    
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
                    
                    # Display tabs
                    print(f"\nTop {len(tabs)} tabs:")
                    for i, tab in enumerate(tabs):
                        print(f"{i+1}. {tab.get('name', 'Unknown')} - Rating: {tab.get('rating', 0)}")
                    
                    # Display chords
                    print(f"\nTop {len(chords)} chords:")
                    for i, chord in enumerate(chords):
                        print(f"{i+1}. {chord.get('name', 'Unknown')} - Rating: {chord.get('rating', 0)}")
                    
                elif choice == "3":
                    url = input("\nEnter tab URL: ")
                    
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
                    print(f"\nRunning full test sequence for '{artist_name}'...")
                    
                    # Step 1: Search for artist
                    print(f"\nSearching for tabs by '{artist_name}'...")
                    results = extractor.search_for_artist(artist_name)
                    
                    if not results:
                        print(f"No tabs found for artist: {artist_name}")
                        continue
                        
                    print(f"Found {len(results)} results for {artist_name}")
                    
                    # Step 2: Get top tabs and chords
                    # Filter and sort tabs
                    tabs_sorted = sorted(
                        [tab for tab in results if tab['type'] == 'Tab'], 
                        key=lambda x: float(x.get('rating', 0) or 0), 
                        reverse=True
                    )[:3]
                    
                    # Filter and sort chords
                    chords_sorted = sorted(
                        [chord for chord in results if chord['type'] == 'Chords'], 
                        key=lambda x: float(x.get('rating', 0) or 0), 
                        reverse=True
                    )[:3]
                    
                    # Display tabs
                    print(f"\nTop {len(tabs_sorted)} tabs:")
                    for i, tab in enumerate(tabs_sorted):
                        print(f"{i+1}. {tab.get('name', 'Unknown')} - Rating: {tab.get('rating', 0)}")
                    
                    # Display chords
                    print(f"\nTop {len(chords_sorted)} chords:")
                    for i, chord in enumerate(chords_sorted):
                        print(f"{i+1}. {chord.get('name', 'Unknown')} - Rating: {chord.get('rating', 0)}")
                    
                    # Step 3: Download a sample tab
                    if tabs_sorted:
                        sample_tab = tabs_sorted[0]
                        print(f"\nDownloading sample tab: {sample_tab['name']}")
                        
                        tab_content = extractor.download_tab(sample_tab['url'], save_to_file=True)
                        
                        if tab_content:
                            print("Sample tab downloaded successfully")
                            lines = tab_content.split("\n")
                            preview_lines = lines[:min(15, len(lines))]
                            print("\nTab content preview:")
                            print("-" * 40)
                            for line in preview_lines:
                                print(line)
                            print("-" * 40)
                        else:
                            print("Failed to download sample tab")
                    
                elif choice == "5":
                    artists_input = input("\nEnter artist names (comma-separated): ")
                    artists = [a.strip() for a in artists_input.split(",")]
                    
                    if not artists:
                        print("No artists specified")
                        continue
                        
                    tabs_count = input("Number of tabs per artist [5]: ").strip() or "5"
                    chords_count = input("Number of chords per artist [2]: ").strip() or "2"
                    
                    try:
                        tabs_count = int(tabs_count)
                        chords_count = int(chords_count)
                    except ValueError:
                        print("Invalid numbers, using defaults (5 tabs, 2 chords)")
                        tabs_count = 5
                        chords_count = 2
                    
                    total_downloaded = 0
                    
                    for artist in artists:
                        print(f"\n{'='*50}")
                        print(f"Processing artist: {artist}")
                        print(f"{'='*50}")
                        
                        # Keep track of already downloaded songs to avoid duplicates
                        downloaded_songs = set()
                        artist_files = []
                        
                        try:
                            # Search for tabs by artist
                            print(f"Searching for tabs by '{artist}'...")
                            results = extractor.search_for_artist(artist)
                            
                            if not results:
                                print(f"No tabs found for artist: {artist}")
                                continue
                                
                            print(f"Found {len(results)} results for {artist}")
                            
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
                            top_tabs = []
                            for tab in tabs_sorted:
                                song_name = tab['name'].lower()
                                if len(top_tabs) < tabs_count and song_name not in downloaded_songs:
                                    top_tabs.append(tab)
                                    downloaded_songs.add(song_name)
                            
                            # Get unique top chords
                            top_chords = []
                            for chord in chords_sorted:
                                song_name = chord['name'].lower()
                                if len(top_chords) < chords_count and song_name not in downloaded_songs:
                                    top_chords.append(chord)
                                    downloaded_songs.add(song_name)
                            
                            print(f"Selected {len(top_tabs)} top tabs and {len(top_chords)} top chords for {artist}")
                            
                            # Download tabs
                            if top_tabs:
                                print("\nDownloading tabs:")
                                for i, tab in enumerate(top_tabs, 1):
                                    print(f"{i}. {tab['name']} - Rating: {tab.get('rating', 'N/A')}")
                                    try:
                                        tab_content = extractor.download_tab(tab['url'], save_to_file=True)
                                        
                                        if tab_content:
                                            print(f"✓ Successfully downloaded: {tab['name']}")
                                            artist_files.append(tab['name'])
                                            total_downloaded += 1
                                        else:
                                            print(f"✗ Failed to download: {tab['name']}")
                                            
                                        # Add delay between downloads to avoid rate limiting
                                        time.sleep(1.5)
                                    except Exception as e:
                                        print(f"✗ Error downloading {tab['name']}: {str(e)}")
                            
                            # Download chords
                            if top_chords:
                                print("\nDownloading chords:")
                                for i, chord in enumerate(top_chords, 1):
                                    print(f"{i}. {chord['name']} - Rating: {chord.get('rating', 'N/A')}")
                                    try:
                                        chord_content = extractor.download_tab(chord['url'], save_to_file=True)
                                        
                                        if chord_content:
                                            print(f"✓ Successfully downloaded: {chord['name']}")
                                            artist_files.append(chord['name'])
                                            total_downloaded += 1
                                        else:
                                            print(f"✗ Failed to download: {chord['name']}")
                                            
                                        # Add delay between downloads to avoid rate limiting
                                        time.sleep(1.5)
                                    except Exception as e:
                                        print(f"✗ Error downloading {chord['name']}: {str(e)}")
                            
                            print(f"\nDownloaded {len(artist_files)} files for {artist}")
                            
                            # Wait a bit before processing the next artist
                            time.sleep(2)
                            
                        except Exception as e:
                            print(f"Error processing artist {artist}: {str(e)}")
                    
                    print(f"\n{'='*50}")
                    print(f"Batch download complete!")
                    print(f"Downloaded {total_downloaded} files total")
                    print(f"Files are saved in: {TAB_FILES_DIR}")
                    print(f"{'='*50}")
                    
                else:
                    print("Invalid choice, please try again")
                    
            except Exception as e:
                import traceback
                print(f"Error: {str(e)}")
                print(traceback.format_exc())
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Always close the browser
        print("\nClosing browser...")
        extractor.close()
    
    print("\nExtractor closed. Downloaded tabs can be found in:")
    print(TAB_FILES_DIR)

if __name__ == "__main__":
    main()
