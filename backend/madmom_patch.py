"""
Patch module for madmom compatibility with Python 3.12

This module patches the collections import in madmom to make it compatible with Python 3.12+
"""
import sys
import logging
import importlib
import importlib.util

logger = logging.getLogger(__name__)

def patch_madmom():
    """
    Apply compatibility patches to madmom to make it work with Python 3.12+
    Primarily fixes the MutableSequence import from collections which moved to collections.abc
    """
    if sys.version_info >= (3, 12):
        logger.info("Applying Python 3.12+ compatibility patches to madmom")
        
        # Find madmom in site-packages
        try:
            madmom_spec = importlib.util.find_spec('madmom')
            if madmom_spec is None:
                logger.warning("Could not find madmom module to patch")
                return False
                
            madmom_path = madmom_spec.submodule_search_locations[0]
            logger.info(f"Found madmom at: {madmom_path}")
            
            # Patch madmom.audio.signal module - a key module that uses MutableSequence
            signal_path = f"{madmom_path}/audio/signal.py"
            
            # Read the file
            with open(signal_path, 'r') as f:
                content = f.read()
            
            # Check if file contains the problematic import
            if 'from collections import MutableSequence' in content:
                # Replace the import
                patched_content = content.replace(
                    'from collections import MutableSequence', 
                    'try:\n    from collections.abc import MutableSequence\nexcept ImportError:\n    from collections import MutableSequence'
                )
                
                # Write the patched file
                with open(signal_path, 'w') as f:
                    f.write(patched_content)
                    
                logger.info(f"Successfully patched {signal_path}")
                
                # Force reload of the module if it's already been imported
                if 'madmom.audio.signal' in sys.modules:
                    del sys.modules['madmom.audio.signal']
                    
                return True
            else:
                logger.info(f"The file {signal_path} does not need patching")
                return True
                
        except Exception as e:
            logger.error(f"Error patching madmom: {e}")
            return False
    else:
        logger.info("Python version is below 3.12, no patching needed")
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = patch_madmom()
    print(f"Patch result: {'Success' if result else 'Failed'}")
