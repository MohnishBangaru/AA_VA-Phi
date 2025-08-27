#!/usr/bin/env python3
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
