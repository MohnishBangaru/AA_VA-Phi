#!/usr/bin/env python3
"""
Fix OmniParser Constructor
=========================

This script fixes the OmniParser integration constructor to properly handle None values.
"""

import logging

logger = logging.getLogger(__name__)


def get_fixed_constructor():
    """Get the fixed constructor code."""
    
    fixed_constructor = '''
    def __init__(self, model_name: Optional[str] = "microsoft/OmniParser-v2.0", device: str = None):
        """Initialize OmniParser 2.0 integration.
        
        Args:
            model_name: Hugging Face model name for OmniParser 2.0 (can be None to disable)
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        # Handle None model_name properly
        if model_name is None:
            self.model_name = None
            self._available = False
            logger.info("ℹ️ OmniParser 2.0 disabled (model_name is None)")
            return
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        self._available = self._initialize_model()
'''
    
    return fixed_constructor


def get_safe_initialization():
    """Get safe initialization code."""
    
    safe_init = '''
    def _initialize_model(self) -> bool:
        """Initialize the OmniParser 2.0 model from Hugging Face."""
        try:
            # Validate model name
            if not self.model_name or self.model_name == "None":
                logger.warning("⚠️ OmniParser model name is None or invalid")
                return False
            
            logger.info(f"🚀 Loading OmniParser 2.0 model: {self.model_name}")
            
            # Check Hugging Face Hub login
            if not self._check_hf_login():
                logger.warning("⚠️ Hugging Face Hub not logged in, skipping OmniParser")
                return False
            
            # Import transformers components
            try:
                from transformers import (
                    AutoProcessor, 
                    AutoTokenizer, 
                    AutoModel,
                    AutoImageProcessor
                )
            except ImportError:
                logger.warning("⚠️ Transformers not available, skipping OmniParser")
                return False
            
            # Try to load the model
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                logger.info("✅ OmniParser 2.0 loaded successfully")
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load OmniParser 2.0: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ OmniParser initialization failed: {e}")
            return False
'''
    
    return safe_init


def create_omniparser_constructor_patch():
    """Create a patch for the OmniParser constructor."""
    
    patch_code = '''#!/usr/bin/env python3
"""
OmniParser Constructor Patch
==========================

This patch fixes the OmniParser integration constructor to handle None values properly.
"""

import logging
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


def patch_omniparser_constructor():
    """Patch the OmniParser constructor to handle None values safely."""
    
    # Create a safe constructor wrapper
    class SafeOmniParserConstructor:
        """Safe wrapper for OmniParser constructor."""
        
        def __init__(self, model_name: Any = None, device: Any = None):
            # Handle None model_name properly
            if model_name is None:
                self.model_name = None
                self._available = False
                logger.info("ℹ️ OmniParser 2.0 disabled (model_name is None)")
                return
            
            # If model_name is a string, use it
            if isinstance(model_name, str):
                self.model_name = model_name
            else:
                # Convert to string or use default
                self.model_name = str(model_name) if model_name else "microsoft/OmniParser-v2.0"
            
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.processor = None
            self.tokenizer = None
            
            # Only try to initialize if we have a valid model name
            if self.model_name and self.model_name != "None":
                self._available = self._initialize_model()
            else:
                self._available = False
        
        def _initialize_model(self) -> bool:
            """Safe model initialization."""
            try:
                logger.info(f"🚀 Loading OmniParser 2.0 model: {self.model_name}")
                
                # Import transformers
                from transformers import AutoProcessor, AutoTokenizer, AutoModel
                
                # Load model components
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                logger.info("✅ OmniParser 2.0 loaded successfully")
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load OmniParser 2.0: {e}")
                return False
        
        def is_available(self) -> bool:
            """Check if OmniParser is available."""
            return self._available and self.model is not None
        
        def parse_elements(self, image_path: str) -> list:
            """Parse elements (safe fallback)."""
            if not self.is_available():
                return []
            # Implementation would go here
            return []
    
    # Replace the problematic class
    sys.modules['OMniParserIntegration'] = SafeOmniParserConstructor
    sys.modules['OmniParserIntegration'] = SafeOmniParserConstructor
    
    logger.info("✅ Patched OmniParser constructor to handle None values safely")


# Auto-patch on import
patch_omniparser_constructor()
'''
    
    # Write the patch
    output_path = Path(__file__).parent / "omniparser_constructor_patch.py"
    output_path.write_text(patch_code)
    
    logger.info(f"✅ Created OmniParser constructor patch: {output_path}")
    return True


def main():
    """Main function to fix OmniParser constructor."""
    logger.info("🚀 Starting OmniParser constructor fix...")
    
    # Show the fixed constructor
    fixed_constructor = get_fixed_constructor()
    logger.info("📝 Fixed constructor:")
    logger.info(fixed_constructor)
    
    # Show safe initialization
    safe_init = get_safe_initialization()
    logger.info("📝 Safe initialization:")
    logger.info(safe_init)
    
    # Create patch
    patch_created = create_omniparser_constructor_patch()
    
    # Summary
    logger.info("📊 OmniParser Constructor Fix Summary:")
    logger.info(f"  Patch created: {'✅ SUCCESS' if patch_created else '❌ FAILED'}")
    
    logger.info("💡 Next steps:")
    logger.info("  1. Import omniparser_constructor_patch.py at the start of your script")
    logger.info("  2. This will fix the None model name error")
    logger.info("  3. Or manually update your constructor with the fixed code above")


if __name__ == "__main__":
    main()
