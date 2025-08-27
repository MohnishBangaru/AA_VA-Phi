#!/usr/bin/env python3
"""
Find OmniParser Source
=====================

This script helps find where the OmniParser loading error is coming from.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_omniparser_loading_code():
    """Find code that might be loading OmniParser."""
    logger.info("🔍 Searching for OmniParser loading code...")
    
    codebase_path = Path(__file__).parent.parent
    potential_sources = []
    
    # Search for patterns that might load OmniParser
    patterns = [
        ("from_pretrained", "Model loading with from_pretrained"),
        ("AutoModel", "AutoModel usage"),
        ("AutoTokenizer", "AutoTokenizer usage"),
        ("transformers", "Transformers import"),
        ("huggingface", "HuggingFace usage"),
        ("model_name", "Model name variable"),
        ("config", "Config usage"),
        ("trust_remote_code", "Trust remote code usage"),
        ("device_map", "Device map usage"),
    ]
    
    for pattern, description in patterns:
        for py_file in codebase_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                if pattern in content:
                    # Check if this file also has OmniParser references
                    if "omniparser" in content.lower() or "OmniParser" in content:
                        potential_sources.append(f"{description} + OmniParser in {py_file.relative_to(codebase_path)}")
                        logger.warning(f"Found: {description} + OmniParser in {py_file.relative_to(codebase_path)}")
            except Exception as e:
                logger.warning(f"Could not read {py_file}: {e}")
    
    return potential_sources


def check_for_recent_omniparser_additions():
    """Check for recent additions that might include OmniParser."""
    logger.info("🔍 Checking for recent OmniParser additions...")
    
    codebase_path = Path(__file__).parent.parent
    
    # Look for files modified in the last 2 hours
    import time
    current_time = time.time()
    two_hours_ago = current_time - 7200
    
    recent_files = []
    for py_file in codebase_path.rglob("*.py"):
        try:
            if py_file.stat().st_mtime > two_hours_ago:
                recent_files.append(py_file)
                logger.info(f"Recent file: {py_file.relative_to(codebase_path)}")
        except Exception as e:
            logger.warning(f"Could not check {py_file}: {e}")
    
    return recent_files


def create_omniparser_tracker():
    """Create a tracker to catch OmniParser loading attempts."""
    logger.info("🔧 Creating OmniParser tracker...")
    
    tracker_code = '''#!/usr/bin/env python3
"""
OmniParser Loading Tracker
=========================

This script tracks and catches OmniParser loading attempts.
"""

import logging
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


class OmniParserTracker:
    """Track OmniParser loading attempts."""
    
    def __init__(self):
        self.loading_attempts = []
        self.none_model_attempts = []
    
    def track_loading(self, model_name: Optional[str], source: str = "unknown"):
        """Track a model loading attempt."""
        attempt = {
            "model_name": model_name,
            "source": source,
            "timestamp": time.time()
        }
        
        self.loading_attempts.append(attempt)
        
        if model_name is None:
            self.none_model_attempts.append(attempt)
            logger.error(f"❌ OmniParser loading attempt with None model name from: {source}")
            logger.error("   Stack trace:")
            import traceback
            traceback.print_stack()
    
    def get_none_attempts(self):
        """Get attempts with None model names."""
        return self.none_model_attempts


# Global tracker
tracker = OmniParserTracker()


def patch_from_pretrained():
    """Patch from_pretrained to track OmniParser loading."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Store original methods
        original_model_from_pretrained = AutoModelForCausalLM.from_pretrained
        original_tokenizer_from_pretrained = AutoTokenizer.from_pretrained
        
        def tracked_model_from_pretrained(*args, **kwargs):
            """Tracked wrapper for AutoModelForCausalLM.from_pretrained."""
            model_name = args[0] if args else None
            
            # Check if this might be OmniParser
            if model_name and ("omniparser" in str(model_name).lower() or "microsoft" in str(model_name).lower()):
                tracker.track_loading(model_name, "AutoModelForCausalLM.from_pretrained")
            
            return original_model_from_pretrained(*args, **kwargs)
        
        def tracked_tokenizer_from_pretrained(*args, **kwargs):
            """Tracked wrapper for AutoTokenizer.from_pretrained."""
            model_name = args[0] if args else None
            
            # Check if this might be OmniParser
            if model_name and ("omniparser" in str(model_name).lower() or "microsoft" in str(model_name).lower()):
                tracker.track_loading(model_name, "AutoTokenizer.from_pretrained")
            
            return original_tokenizer_from_pretrained(*args, **kwargs)
        
        # Replace methods
        AutoModelForCausalLM.from_pretrained = tracked_model_from_pretrained
        AutoTokenizer.from_pretrained = tracked_tokenizer_from_pretrained
        
        logger.info("✅ Patched from_pretrained to track OmniParser loading")
        
    except ImportError:
        logger.warning("⚠️ Transformers not available, skipping patch")
    except Exception as e:
        logger.error(f"❌ Failed to patch from_pretrained: {e}")


# Auto-patch on import
patch_from_pretrained()
'''
    
    # Write the tracker
    output_path = Path(__file__).parent / "omniparser_tracker.py"
    output_path.write_text(tracker_code)
    
    logger.info(f"✅ Created OmniParser tracker: {output_path}")
    return True


def main():
    """Main function to find OmniParser source."""
    logger.info("🚀 Starting OmniParser source search...")
    
    # Find potential OmniParser loading code
    potential_sources = find_omniparser_loading_code()
    
    # Check recent additions
    recent_files = check_for_recent_omniparser_additions()
    
    # Create tracker
    tracker_created = create_omniparser_tracker()
    
    # Summary
    logger.info("📊 OmniParser Source Search Summary:")
    logger.info(f"  Potential sources: {len(potential_sources)}")
    logger.info(f"  Recent files: {len(recent_files)}")
    logger.info(f"  Tracker created: {'✅ SUCCESS' if tracker_created else '❌ FAILED'}")
    
    if potential_sources:
        logger.warning("⚠️ Potential OmniParser sources found:")
        for source in potential_sources:
            logger.warning(f"  - {source}")
    
    logger.info("💡 Next steps:")
    logger.info("  1. Import omniparser_tracker.py at the start of your script")
    logger.info("  2. This will track and catch OmniParser loading attempts")
    logger.info("  3. It will show you exactly where the None model name is coming from")


if __name__ == "__main__":
    main()
