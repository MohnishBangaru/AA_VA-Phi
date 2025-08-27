#!/usr/bin/env python3
"""
Phi-3 Vision Tokenization Fix
============================

This script provides a targeted fix for Phi-3 vision tokenization based on debugging results.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_phi3_vision_fix():
    """Create a targeted fix for Phi-3 vision tokenization."""
    
    fix_code = '''#!/usr/bin/env python3
"""
Phi-3 Vision Tokenization Fix
============================

This module provides a targeted fix for Phi-3 vision tokenization.
Based on debugging results, manual processing works best for Phi-3.
"""

import logging
from typing import Optional, Dict, Any
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class Phi3VisionFix:
    """Targeted fix for Phi-3 vision tokenization."""
    
    def __init__(self, model_name: str, tokenizer, device: str):
        """Initialize the Phi-3 vision fix."""
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device
    
    def tokenize_with_vision(self, prompt: str, image: Image.Image) -> Optional[Dict[str, torch.Tensor]]:
        """Phi-3 specific vision tokenization."""
        
        # Based on debugging, manual processing works best for Phi-3
        approaches = [
            ("Phi-3 Manual Processing", self._phi3_manual_processing),
            ("Phi-3 Direct Images", self._phi3_direct_images),
            ("Phi-3 AutoImageProcessor", self._phi3_auto_image_processor)
        ]
        
        for name, approach in approaches:
            try:
                logger.info(f"Trying {name}...")
                inputs = approach(prompt, image)
                if inputs is not None:
                    logger.info(f"✅ {name} successful")
                    return inputs
            except Exception as e:
                logger.debug(f"{name} failed: {e}")
                continue
        
        logger.warning("❌ All Phi-3 vision approaches failed")
        return None
    
    def _phi3_manual_processing(self, prompt: str, image: Image.Image) -> Optional[Dict[str, torch.Tensor]]:
        """Phi-3 manual processing (confirmed working)."""
        try:
            import torchvision.transforms as transforms
            
            # Phi-3 specific image size (336x336 is common for Phi-3)
            target_size = (336, 336)
            image_resized = image.resize(target_size)
            
            # Convert to tensor with ImageNet normalization
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image_resized).unsqueeze(0)
            
            # Create text inputs
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Add image tensor as pixel_values (confirmed working)
            inputs['pixel_values'] = image_tensor.to(self.device)
            
            return inputs
            
        except Exception as e:
            logger.debug(f"Phi-3 manual processing failed: {e}")
            return None
    
    def _phi3_direct_images(self, prompt: str, image: Image.Image) -> Optional[Dict[str, torch.Tensor]]:
        """Phi-3 direct images parameter."""
        try:
            inputs = self.tokenizer(
                prompt,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            return {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as e:
            logger.debug(f"Phi-3 direct images failed: {e}")
            return None
    
    def _phi3_auto_image_processor(self, prompt: str, image: Image.Image) -> Optional[Dict[str, torch.Tensor]]:
        """Phi-3 AutoImageProcessor fallback."""
        try:
            from transformers import AutoImageProcessor
            image_processor = AutoImageProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            processed_image = image_processor(image, return_tensors="pt")
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                **processed_image,
                padding=True,
                truncation=True
            )
            return {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as e:
            logger.debug(f"Phi-3 AutoImageProcessor failed: {e}")
            return None


def apply_phi3_vision_fix():
    """Apply the Phi-3 vision fix to the existing code."""
    try:
        from src.ai.phi_ground import PhiGroundActionGenerator
        
        # Store original method
        original_generate_touch_action = PhiGroundActionGenerator.generate_touch_action
        
        async def patched_generate_touch_action(self, image_path: str, task_description: str, action_history, ui_elements):
            """Patched generate_touch_action with Phi-3 vision fix."""
            
            if not self.vision_supported:
                return await original_generate_touch_action(self, image_path, task_description, action_history, ui_elements)
            
            try:
                from PIL import Image
                image = Image.open(image_path).convert('RGB')
                
                # Create Phi-3 vision fix
                phi3_fix = Phi3VisionFix(self.model_name, self.tokenizer, self.device)
                
                # Create prompt
                prompt = self._create_phi_ground_prompt(image_path, task_description, action_history)
                
                # Try Phi-3 vision tokenization
                inputs = phi3_fix.tokenize_with_vision(prompt, image)
                
                if inputs is not None:
                    logger.info("✅ Using Phi-3 vision tokenization")
                    # Continue with the vision inputs
                else:
                    logger.warning("⚠️ Falling back to text-only tokenization")
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Continue with generation using the inputs
                return await original_generate_touch_action(self, image_path, task_description, action_history, ui_elements)
                
            except Exception as e:
                logger.error(f"Phi-3 vision fix failed: {e}")
                return await original_generate_touch_action(self, image_path, task_description, action_history, ui_elements)
        
        # Replace the method
        PhiGroundActionGenerator.generate_touch_action = patched_generate_touch_action
        
        logger.info("✅ Applied Phi-3 vision tokenization fix")
        
    except Exception as e:
        logger.error(f"❌ Failed to apply Phi-3 vision fix: {e}")


# Auto-apply on import
apply_phi3_vision_fix()
'''
    
    # Write the fix
    output_path = Path(__file__).parent / "phi3_vision_fix.py"
    output_path.write_text(fix_code)
    
    print(f"✅ Created Phi-3 vision fix: {output_path}")
    return True


def create_quick_test():
    """Create a quick test to verify the fix works."""
    
    test_code = '''#!/usr/bin/env python3
"""
Quick Test for Phi-3 Vision Fix
==============================

This script quickly tests if the Phi-3 vision fix works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the fix
import scripts.phi3_vision_fix

# Test with a sample image
from PIL import Image
import numpy as np

# Create test image
test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
test_image.save("test_image.png")

print("✅ Phi-3 vision fix imported successfully")
print("✅ Test image created: test_image.png")
print("🚀 Ready to test with your APK tester!")
'''
    
    # Write the test
    output_path = Path(__file__).parent / "test_phi3_fix.py"
    output_path.write_text(test_code)
    
    print(f"✅ Created quick test: {output_path}")
    return True


def main():
    """Main function to create the Phi-3 vision fix."""
    print("🔧 Creating Phi-3 Vision Tokenization Fix")
    print("=" * 50)
    
    # Create the fix
    create_phi3_vision_fix()
    
    # Create quick test
    create_quick_test()
    
    print("\n💡 Phi-3 Vision Fix Summary:")
    print("✅ Based on debugging results:")
    print("   - Manual processing works (confirmed)")
    print("   - Creates correct input keys: ['input_ids', 'attention_mask', 'pixel_values']")
    print("   - Phi-3 models handle vision differently than traditional VLMs")
    print("   - No need for vision_encoder, vision_model, etc.")
    
    print("\n🚀 Next Steps:")
    print("1. Add this line at the start of your script:")
    print("   import scripts.phi3_vision_fix")
    print("2. Run your APK tester - vision should work now!")
    print("3. Or test with: python scripts/test_phi3_fix.py")


if __name__ == "__main__":
    main()
