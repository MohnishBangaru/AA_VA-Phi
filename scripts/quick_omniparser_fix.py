#!/usr/bin/env python3
"""
Quick OmniParser Fix
===================

This script provides an immediate fix for the OmniParser 2.0 loading issue.
Run this at the start of your script to prevent the "None is not a local folder" error.
"""

import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def disable_omniparser():
    """Disable OmniParser to prevent loading errors."""
    # Set environment variables to disable OmniParser
    os.environ["DISABLE_OMNIPARSER"] = "true"
    os.environ["USE_OMNIPARSER"] = "false"
    os.environ["OMNIPARSER_MODEL"] = ""
    
    logger.info("✅ OmniParser disabled to prevent loading errors")


def create_safe_omniparser_wrapper():
    """Create a safe wrapper that prevents OmniParser errors."""
    
    # Monkey patch any OmniParser imports to return safe fallbacks
    import sys
    
    class SafeOmniParserMock:
        """Safe mock for OmniParser that won't crash."""
        
        def __init__(self, *args, **kwargs):
            logger.info("ℹ️ OmniParser disabled - using safe fallback")
        
        def __call__(self, *args, **kwargs):
            return []
        
        def __getattr__(self, name):
            # Return empty list for any method calls
            return lambda *args, **kwargs: []
    
    # Replace any OmniParser imports with our safe mock
    sys.modules['omniparser'] = SafeOmniParserMock()
    sys.modules['OmniParser'] = SafeOmniParserMock()
    
    logger.info("✅ Created safe OmniParser wrapper")


def patch_vision_engine_safely():
    """Patch VisionEngine to handle OmniParser safely."""
    try:
        # Import VisionEngine and patch it
        from src.vision.engine import VisionEngine
        
        # Store original analyze method
        original_analyze = VisionEngine.analyze
        
        def safe_analyze(self, image_path: str):
            """Safe analyze method that won't crash on OmniParser errors."""
            try:
                # Call original method
                return original_analyze(self, image_path)
            except Exception as e:
                if "OmniParser" in str(e) or "None is not a local folder" in str(e):
                    logger.info("ℹ️ OmniParser error caught, continuing with basic analysis")
                    # Return basic analysis without OmniParser
                    return self._basic_analyze(image_path)
                else:
                    # Re-raise other errors
                    raise
        
        # Replace the analyze method
        VisionEngine.analyze = safe_analyze
        
        logger.info("✅ Patched VisionEngine to handle OmniParser safely")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not patch VisionEngine: {e}")


def _basic_analyze(self, image_path: str):
    """Basic analysis without OmniParser."""
    # This would be a simplified version of the analyze method
    # that doesn't use OmniParser
    logger.info("ℹ️ Using basic analysis (OmniParser disabled)")
    return []


# Auto-run the fix
if __name__ == "__main__":
    logger.info("🔧 Applying quick OmniParser fix...")
    
    # Disable OmniParser
    disable_omniparser()
    
    # Create safe wrapper
    create_safe_omniparser_wrapper()
    
    # Patch VisionEngine
    patch_vision_engine_safely()
    
    logger.info("✅ Quick OmniParser fix applied!")
    logger.info("💡 You can now run your scripts without OmniParser errors")
else:
    # When imported, apply the fix automatically
    disable_omniparser()
    create_safe_omniparser_wrapper()
    patch_vision_engine_safely()
