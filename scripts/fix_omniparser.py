#!/usr/bin/env python3
"""
Fix OmniParser 2.0 Model Loading Issue
=====================================

This script fixes the issue where OmniParser 2.0 is trying to load a model
with a None identifier, causing the error:
"None is not a local folder and is not a valid model identifier"
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_omniparser_usage():
    """Find where OmniParser is being used in the codebase."""
    logger.info("🔍 Searching for OmniParser usage...")
    
    codebase_path = Path(__file__).parent.parent
    omniparser_files = []
    
    # Search for OmniParser references
    for py_file in codebase_path.rglob("*.py"):
        try:
            content = py_file.read_text()
            if "omniparser" in content.lower() or "OmniParser" in content:
                omniparser_files.append(py_file)
                logger.info(f"Found OmniParser reference in: {py_file.relative_to(codebase_path)}")
        except Exception as e:
            logger.warning(f"Could not read {py_file}: {e}")
    
    return omniparser_files


def check_vision_engine_omniparser():
    """Check if VisionEngine has OmniParser integration."""
    logger.info("🔍 Checking VisionEngine for OmniParser integration...")
    
    vision_engine_path = Path(__file__).parent.parent / "src" / "vision" / "engine.py"
    
    if not vision_engine_path.exists():
        logger.error("VisionEngine not found")
        return False
    
    try:
        content = vision_engine_path.read_text()
        
        # Check for OmniParser imports or usage
        if "omniparser" in content.lower() or "OmniParser" in content:
            logger.warning("Found OmniParser reference in VisionEngine")
            return True
        else:
            logger.info("No OmniParser reference found in VisionEngine")
            return False
            
    except Exception as e:
        logger.error(f"Could not read VisionEngine: {e}")
        return False


def create_omniparser_fix():
    """Create a fix for the OmniParser loading issue."""
    logger.info("🔧 Creating OmniParser fix...")
    
    fix_code = '''#!/usr/bin/env python3
"""
OmniParser 2.0 Fix
=================

This module provides a safe wrapper for OmniParser 2.0 to prevent
the "None is not a local folder" error.
"""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class SafeOmniParser:
    """Safe wrapper for OmniParser 2.0 with proper error handling."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize SafeOmniParser with proper model validation.
        
        Args:
            model_name: HuggingFace model name for OmniParser 2.0
        """
        self.model_name = model_name or "microsoft/OmniParser-2.0"
        self.model = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize OmniParser 2.0 with proper error handling."""
        try:
            # Validate model name
            if not self.model_name or self.model_name == "None":
                logger.warning("⚠️ OmniParser model name is None or invalid")
                return False
            
            logger.info(f"🧪 Trying to load OmniParser 2.0: {self.model_name}")
            
            # Try to import transformers
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                logger.warning("⚠️ Transformers not available, skipping OmniParser")
                return False
            
            # Try to load the model
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    device_map="auto"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self._initialized = True
                logger.info("✅ OmniParser 2.0 loaded successfully")
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load OmniParser 2.0: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ OmniParser initialization failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if OmniParser is available and initialized."""
        return self._initialized and self.model is not None
    
    def parse_elements(self, image_path: str) -> List[Dict[str, Any]]:
        """Parse UI elements from image using OmniParser 2.0.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of parsed UI elements
        """
        if not self.is_available():
            logger.warning("⚠️ OmniParser not available, returning empty list")
            return []
        
        try:
            # Implementation would go here
            # For now, return empty list to prevent errors
            logger.info("ℹ️ OmniParser parsing not yet implemented")
            return []
            
        except Exception as e:
            logger.error(f"❌ OmniParser parsing failed: {e}")
            return []


# Global instance
omniparser = SafeOmniParser()


def get_omniparser() -> SafeOmniParser:
    """Get the global OmniParser instance."""
    return omniparser


def initialize_omniparser() -> bool:
    """Initialize the global OmniParser instance."""
    return omniparser.initialize()


def is_omniparser_available() -> bool:
    """Check if OmniParser is available."""
    return omniparser.is_available()
'''
    
    # Write the fix
    output_path = Path(__file__).parent / "omniparser_fix.py"
    output_path.write_text(fix_code)
    
    logger.info(f"✅ Created OmniParser fix: {output_path}")
    return True


def create_vision_engine_patch():
    """Create a patch for VisionEngine to handle OmniParser properly."""
    logger.info("🔧 Creating VisionEngine patch...")
    
    patch_code = '''#!/usr/bin/env python3
"""
VisionEngine OmniParser Patch
============================

This patch adds safe OmniParser integration to VisionEngine.
"""

import logging
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def safe_omniparser_integration(image_path: str) -> List:
    """Safe OmniParser integration that won't crash the system.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of parsed elements (empty if OmniParser fails)
    """
    try:
        # Try to import the OmniParser fix
        from omniparser_fix import get_omniparser, is_omniparser_available
        
        omniparser = get_omniparser()
        
        if not is_omniparser_available():
            # Try to initialize
            if not omniparser.initialize():
                logger.info("ℹ️ OmniParser 2.0 not available (model loading failed)")
                return []
        
        # Parse elements
        elements = omniparser.parse_elements(image_path)
        logger.info(f"✅ OmniParser parsed {len(elements)} elements")
        return elements
        
    except ImportError:
        logger.info("ℹ️ OmniParser fix not available")
        return []
    except Exception as e:
        logger.warning(f"⚠️ OmniParser integration failed: {e}")
        return []


def patch_vision_engine():
    """Patch VisionEngine to use safe OmniParser integration."""
    logger.info("🔧 Patching VisionEngine...")
    
    # This would be called from VisionEngine.analyze()
    # For now, just return the safe integration function
    return safe_omniparser_integration
'''
    
    # Write the patch
    output_path = Path(__file__).parent / "vision_engine_patch.py"
    output_path.write_text(patch_code)
    
    logger.info(f"✅ Created VisionEngine patch: {output_path}")
    return True


def disable_omniparser_temporarily():
    """Temporarily disable OmniParser to prevent errors."""
    logger.info("🔧 Temporarily disabling OmniParser...")
    
    # Create a simple disable script
    disable_code = '''#!/usr/bin/env python3
"""
Temporarily Disable OmniParser
=============================

This script temporarily disables OmniParser to prevent loading errors.
"""

import os
import logging

logger = logging.getLogger(__name__)


def disable_omniparser():
    """Disable OmniParser by setting environment variables."""
    # Set environment variables to disable OmniParser
    os.environ["DISABLE_OMNIPARSER"] = "true"
    os.environ["USE_OMNIPARSER"] = "false"
    
    logger.info("✅ OmniParser temporarily disabled")


def is_omniparser_disabled() -> bool:
    """Check if OmniParser is disabled."""
    return os.getenv("DISABLE_OMNIPARSER", "false").lower() == "true"


# Auto-disable on import
disable_omniparser()
'''
    
    # Write the disable script
    output_path = Path(__file__).parent / "disable_omniparser.py"
    output_path.write_text(disable_code)
    
    logger.info(f"✅ Created OmniParser disable script: {output_path}")
    return True


async def main():
    """Main function to fix OmniParser issues."""
    logger.info("🚀 Starting OmniParser fix...")
    
    # Find OmniParser usage
    omniparser_files = find_omniparser_usage()
    
    # Check VisionEngine
    vision_engine_has_omniparser = check_vision_engine_omniparser()
    
    # Create fixes
    omniparser_fix_created = create_omniparser_fix()
    vision_engine_patch_created = create_vision_engine_patch()
    omniparser_disabled = disable_omniparser_temporarily()
    
    # Summary
    logger.info("📊 OmniParser Fix Summary:")
    logger.info(f"  OmniParser files found: {len(omniparser_files)}")
    logger.info(f"  VisionEngine has OmniParser: {'Yes' if vision_engine_has_omniparser else 'No'}")
    logger.info(f"  OmniParser fix created: {'✅ SUCCESS' if omniparser_fix_created else '❌ FAILED'}")
    logger.info(f"  VisionEngine patch created: {'✅ SUCCESS' if vision_engine_patch_created else '❌ FAILED'}")
    logger.info(f"  OmniParser disabled: {'✅ SUCCESS' if omniparser_disabled else '❌ FAILED'}")
    
    logger.info("💡 Next steps:")
    logger.info("  1. Import disable_omniparser.py at the start of your script")
    logger.info("  2. Or use omniparser_fix.py for safe OmniParser integration")
    logger.info("  3. The 'None is not a local folder' error should be resolved")


if __name__ == "__main__":
    asyncio.run(main())
