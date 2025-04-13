"""
Unified Tablature Data Loader

Provides a consistent interface for loading tablature data from multiple sources.
"""
import os
import logging
from typing import Dict, List, Optional
from pathlib import Path

from .guitarset_loader import GuitarSetLoader
from .dadagp_loader import DadaGPLoader
from .ultimate_guitar_extractor import UltimateGuitarExtractor

logger = logging.getLogger(__name__)

class UnifiedTabLoader:
    """Unified interface for loading tablature from multiple sources"""
    
    def __init__(self, config: Dict):
        """
        Initialize with configuration for all data sources
        
        Args:
            config: Dictionary containing paths and settings for each data source
                Expected keys:
                - guitarset_path
                - dadagp_path
                - ug_cache_dir
        """
        self.config = config
        
        # Initialize loaders for each source
        self.guitarset_loader = GuitarSetLoader(config.get('guitarset_path'))
        self.dadagp_loader = DadaGPLoader(config.get('dadagp_path'))
        self.ug_extractor = UltimateGuitarExtractor()
        
    def load_guitarset(self, max_batches: int = None) -> Optional[Dict]:
        """Load data from GuitarSet"""
        if not self.config.get('guitarset_path'):
            logger.warning("No GuitarSet path configured")
            return None
            
        return self.guitarset_loader.load_all_batches(max_batches)
    
    def load_dadagp(self) -> Optional[List[Dict]]:
        """Load data from DadaGP"""
        if not self.config.get('dadagp_path'):
            logger.warning("No DadaGP path configured")
            return None
            
        return self.dadagp_loader.load_all()
    
    def load_ultimate_guitar(self, artist: str, num_tabs: int = 10) -> Optional[List[Dict]]:
        """Load tabs from Ultimate Guitar"""
        if not self.config.get('ug_cache_dir'):
            logger.warning("No Ultimate Guitar cache directory configured")
            return None
            
        # Ensure cache directory exists
        os.makedirs(self.config['ug_cache_dir'], exist_ok=True)
        
        # Search and download tabs
        tabs = self.ug_extractor.search_for_artist(artist, max_pages=2)
        if not tabs:
            return None
            
        # Download top N tabs
        downloaded = []
        for tab in tabs[:num_tabs]:
            tab_data = self.ug_extractor.download_tab(
                tab['url'], 
                save_dir=self.config['ug_cache_dir']
            )
            if tab_data:
                downloaded.append(tab_data)
                
        return downloaded if downloaded else None
    
    def load_all(self, sources: List[str] = None) -> Dict:
        """
        Load data from all configured sources
        
        Args:
            sources: List of sources to load ('guitarset', 'dadagp', 'ultimate_guitar')
                    If None, loads all configured sources
        """
        if sources is None:
            sources = ['guitarset', 'dadagp', 'ultimate_guitar']
            
        data = {}
        
        if 'guitarset' in sources:
            data['guitarset'] = self.load_guitarset()
            
        if 'dadagp' in sources:
            data['dadagp'] = self.load_dadagp()
            
        if 'ultimate_guitar' in sources:
            data['ultimate_guitar'] = self.load_ultimate_guitar('The Beatles')  # Example artist
            
        return data
