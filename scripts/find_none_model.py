#!/usr/bin/env python3
"""
Find None Model Identifier
=========================

This script helps find where a None model identifier is being passed to
from_pretrained, causing the "None is not a local folder" error.
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


def find_none_model_usage():
    """Find where None might be passed as a model identifier."""
    logger.info("🔍 Searching for potential None model usage...")
    
    codebase_path = Path(__file__).parent.parent
    potential_issues = []
    
    # Search for patterns that might lead to None model names
    patterns = [
        ("from_pretrained(None", "Direct None passed to from_pretrained"),
        ("from_pretrained(model_name", "Variable model_name might be None"),
        ("from_pretrained(config", "Config variable might be None"),
        ("model_name = None", "Model name explicitly set to None"),
        ("model_name=None", "Model name parameter set to None"),
        ("model_name: Optional[str] = None", "Optional model name with None default"),
        ("model_name: str = None", "String model name with None default"),
    ]
    
    for pattern, description in patterns:
        for py_file in codebase_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                if pattern in content:
                    potential_issues.append(f"{description} in {py_file.relative_to(codebase_path)}")
                    logger.warning(f"Found: {description} in {py_file.relative_to(codebase_path)}")
            except Exception as e:
                logger.warning(f"Could not read {py_file}: {e}")
    
    return potential_issues


def check_vision_engine_for_omniparser():
    """Check if VisionEngine has any OmniParser integration."""
    logger.info("🔍 Checking VisionEngine for OmniParser integration...")
    
    vision_engine_path = Path(__file__).parent.parent / "src" / "vision" / "engine.py"
    
    if not vision_engine_path.exists():
        logger.error("VisionEngine not found")
        return False
    
    try:
        content = vision_engine_path.read_text()
        
        # Check for any model loading patterns
        model_loading_patterns = [
            "from_pretrained",
            "AutoModel",
            "AutoTokenizer",
            "transformers",
            "huggingface",
            "model_name",
            "config"
        ]
        
        found_patterns = []
        for pattern in model_loading_patterns:
            if pattern in content:
                found_patterns.append(pattern)
        
        if found_patterns:
            logger.warning(f"Found model loading patterns in VisionEngine: {found_patterns}")
            return True
        else:
            logger.info("No model loading patterns found in VisionEngine")
            return False
            
    except Exception as e:
        logger.error(f"Could not read VisionEngine: {e}")
        return False


def check_recent_changes():
    """Check for recent changes that might have added OmniParser."""
    logger.info("🔍 Checking for recent changes...")
    
    codebase_path = Path(__file__).parent.parent
    
    # Look for files modified in the last hour
    import time
    current_time = time.time()
    one_hour_ago = current_time - 3600
    
    recent_files = []
    for py_file in codebase_path.rglob("*.py"):
        try:
            if py_file.stat().st_mtime > one_hour_ago:
                recent_files.append(py_file)
                logger.info(f"Recent file: {py_file.relative_to(codebase_path)}")
        except Exception as e:
            logger.warning(f"Could not check {py_file}: {e}")
    
    return recent_files


def create_monkey_patch_fix():
    """Create a monkey patch to catch None model names."""
    logger.info("🔧 Creating monkey patch to catch None model names...")
    
    patch_code = '''#!/usr/bin/env python3
"""
Monkey Patch for None Model Names
================================

This script monkey patches from_pretrained to catch None model names
and provide better error messages.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def patch_from_pretrained():
    """Patch from_pretrained to catch None model names."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Store original methods
        original_model_from_pretrained = AutoModelForCausalLM.from_pretrained
        original_tokenizer_from_pretrained = AutoTokenizer.from_pretrained
        
        def safe_model_from_pretrained(*args, **kwargs):
            """Safe wrapper for AutoModelForCausalLM.from_pretrained."""
            if args and args[0] is None:
                logger.error("❌ Attempted to load model with None identifier!")
                logger.error("   This is likely an OmniParser or other integration issue.")
                logger.error("   Stack trace:")
                import traceback
                traceback.print_stack()
                raise ValueError("Model identifier cannot be None")
            return original_model_from_pretrained(*args, **kwargs)
        
        def safe_tokenizer_from_pretrained(*args, **kwargs):
            """Safe wrapper for AutoTokenizer.from_pretrained."""
            if args and args[0] is None:
                logger.error("❌ Attempted to load tokenizer with None identifier!")
                logger.error("   This is likely an OmniParser or other integration issue.")
                logger.error("   Stack trace:")
                import traceback
                traceback.print_stack()
                raise ValueError("Tokenizer identifier cannot be None")
            return original_tokenizer_from_pretrained(*args, **kwargs)
        
        # Replace methods
        AutoModelForCausalLM.from_pretrained = safe_model_from_pretrained
        AutoTokenizer.from_pretrained = safe_tokenizer_from_pretrained
        
        logger.info("✅ Patched from_pretrained to catch None identifiers")
        
    except ImportError:
        logger.warning("⚠️ Transformers not available, skipping patch")
    except Exception as e:
        logger.error(f"❌ Failed to patch from_pretrained: {e}")


# Auto-patch on import
patch_from_pretrained()
'''
    
    # Write the patch
    output_path = Path(__file__).parent / "monkey_patch_none_model.py"
    output_path.write_text(patch_code)
    
    logger.info(f"✅ Created monkey patch: {output_path}")
    return True


def main():
    """Main diagnostic function."""
    logger.info("🚀 Starting None model identifier diagnostic...")
    
    # Find potential None model usage
    potential_issues = find_none_model_usage()
    
    # Check VisionEngine
    vision_engine_has_model_loading = check_vision_engine_for_omniparser()
    
    # Check recent changes
    recent_files = check_recent_changes()
    
    # Create monkey patch
    monkey_patch_created = create_monkey_patch_fix()
    
    # Summary
    logger.info("📊 Diagnostic Summary:")
    logger.info(f"  Potential None model issues: {len(potential_issues)}")
    logger.info(f"  VisionEngine has model loading: {'Yes' if vision_engine_has_model_loading else 'No'}")
    logger.info(f"  Recent files modified: {len(recent_files)}")
    logger.info(f"  Monkey patch created: {'✅ SUCCESS' if monkey_patch_created else '❌ FAILED'}")
    
    if potential_issues:
        logger.warning("⚠️ Potential issues found:")
        for issue in potential_issues:
            logger.warning(f"  - {issue}")
    
    logger.info("💡 Next steps:")
    logger.info("  1. Import monkey_patch_none_model.py at the start of your script")
    logger.info("  2. This will catch the exact location where None is passed to from_pretrained")
    logger.info("  3. The stack trace will show you exactly where the issue is occurring")


if __name__ == "__main__":
    main()
