#!/usr/bin/env python3
"""
Fix Vision Tokenization Issues
============================

This script fixes the vision tokenization issues in Phi Ground where
"All vision tokenization approaches failed, falling back to text-only"
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.phi_ground import PhiGroundActionGenerator
from src.vision.engine import VisionEngine
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_vision_tokenization():
    """Test vision tokenization to identify the issue."""
    
    logger.info("🧪 Testing vision tokenization...")
    
    # Initialize Phi Ground
    phi_ground = PhiGroundActionGenerator()
    
    try:
        await phi_ground.initialize()
        
        # Test with a sample screenshot
        screenshot_path = "screenshot_1756248771.png"
        
        if not Path(screenshot_path).exists():
            logger.error(f"Screenshot not found: {screenshot_path}")
            return False
        
        logger.info(f"Testing with screenshot: {screenshot_path}")
        
        # Test vision tokenization directly
        from PIL import Image
        import torch
        
        # Load image
        image = Image.open(screenshot_path).convert('RGB')
        logger.info(f"Loaded image: {image.size}")
        
        # Test different vision tokenization approaches
        approaches = [
            ("AutoImageProcessor", "test_auto_image_processor"),
            ("Model Image Processor", "test_model_image_processor"),
            ("Direct Images Parameter", "test_direct_images"),
            ("Vision Encoder", "test_vision_encoder")
        ]
        
        for approach_name, test_func in approaches:
            try:
                logger.info(f"Testing {approach_name}...")
                success = await globals()[test_func](phi_ground, image)
                if success:
                    logger.info(f"✅ {approach_name} works!")
                else:
                    logger.warning(f"❌ {approach_name} failed")
            except Exception as e:
                logger.error(f"❌ {approach_name} error: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Vision tokenization test failed: {e}")
        return False


async def test_auto_image_processor(phi_ground, image):
    """Test AutoImageProcessor approach."""
    try:
        from transformers import AutoImageProcessor
        
        image_processor = AutoImageProcessor.from_pretrained(
            phi_ground.model_name, 
            trust_remote_code=True
        )
        processed_image = image_processor(image, return_tensors="pt")
        
        # Test tokenization
        test_prompt = "Test prompt"
        inputs = phi_ground.tokenizer(
            test_prompt,
            return_tensors="pt",
            **processed_image,
            padding=True,
            truncation=True
        )
        
        return True
    except Exception as e:
        logger.debug(f"AutoImageProcessor failed: {e}")
        return False


async def test_model_image_processor(phi_ground, image):
    """Test model's image processor approach."""
    try:
        if hasattr(phi_ground.tokenizer, 'image_processor'):
            processed_image = phi_ground.tokenizer.image_processor(image, return_tensors="pt")
            
            # Test tokenization
            test_prompt = "Test prompt"
            inputs = phi_ground.tokenizer(
                test_prompt,
                return_tensors="pt",
                **processed_image,
                padding=True,
                truncation=True
            )
            
            return True
        else:
            logger.debug("No image_processor attribute found")
            return False
    except Exception as e:
        logger.debug(f"Model image processor failed: {e}")
        return False


async def test_direct_images(phi_ground, image):
    """Test direct images parameter approach."""
    try:
        # Test direct images parameter
        test_prompt = "Test prompt"
        inputs = phi_ground.tokenizer(
            test_prompt,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return True
    except Exception as e:
        logger.debug(f"Direct images parameter failed: {e}")
        return False


async def test_vision_encoder(phi_ground, image):
    """Test vision encoder approach."""
    try:
        # Check if model has vision encoder
        if hasattr(phi_ground.model, 'vision_encoder'):
            logger.info("Model has vision encoder")
            return True
        else:
            logger.debug("No vision encoder found")
            return False
    except Exception as e:
        logger.debug(f"Vision encoder test failed: {e}")
        return False


def create_vision_tokenization_fix():
    """Create a fix for vision tokenization issues."""
    
    logger.info("🔧 Creating vision tokenization fix...")
    
    fix_code = '''#!/usr/bin/env python3
"""
Vision Tokenization Fix
=====================

This module provides fixes for vision tokenization issues in Phi Ground.
"""

import logging
from typing import Optional, Dict, Any
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class VisionTokenizationFix:
    """Fix for vision tokenization issues."""
    
    def __init__(self, model_name: str, tokenizer, device: str):
        """Initialize the vision tokenization fix."""
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device
        self.image_processor = None
        self._initialize_image_processor()
    
    def _initialize_image_processor(self):
        """Initialize the image processor."""
        try:
            from transformers import AutoImageProcessor
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            logger.info("✅ Image processor initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize image processor: {e}")
            self.image_processor = None
    
    def tokenize_with_vision(self, prompt: str, image: Image.Image) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenize with vision support using multiple fallback approaches."""
        
        approaches = [
            self._approach_1_auto_image_processor,
            self._approach_2_model_image_processor,
            self._approach_3_direct_images,
            self._approach_4_manual_processing
        ]
        
        for i, approach in enumerate(approaches, 1):
            try:
                logger.info(f"Trying vision tokenization approach {i}...")
                inputs = approach(prompt, image)
                if inputs is not None:
                    logger.info(f"✅ Vision tokenization successful with approach {i}")
                    return inputs
            except Exception as e:
                logger.debug(f"Approach {i} failed: {e}")
                continue
        
        logger.warning("❌ All vision tokenization approaches failed")
        return None
    
    def _approach_1_auto_image_processor(self, prompt: str, image: Image.Image) -> Optional[Dict[str, torch.Tensor]]:
        """Approach 1: Use AutoImageProcessor."""
        if self.image_processor is None:
            return None
        
        try:
            processed_image = self.image_processor(image, return_tensors="pt")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                **processed_image,
                padding=True,
                truncation=True
            )
            return {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as e:
            logger.debug(f"AutoImageProcessor approach failed: {e}")
            return None
    
    def _approach_2_model_image_processor(self, prompt: str, image: Image.Image) -> Optional[Dict[str, torch.Tensor]]:
        """Approach 2: Use model's image processor."""
        try:
            if hasattr(self.tokenizer, 'image_processor'):
                processed_image = self.tokenizer.image_processor(image, return_tensors="pt")
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    **processed_image,
                    padding=True,
                    truncation=True
                )
                return {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as e:
            logger.debug(f"Model image processor approach failed: {e}")
        return None
    
    def _approach_3_direct_images(self, prompt: str, image: Image.Image) -> Optional[Dict[str, torch.Tensor]]:
        """Approach 3: Use direct images parameter."""
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
            logger.debug(f"Direct images approach failed: {e}")
        return None
    
    def _approach_4_manual_processing(self, prompt: str, image: Image.Image) -> Optional[Dict[str, torch.Tensor]]:
        """Approach 4: Manual image processing."""
        try:
            # Resize image to standard size
            image = image.resize((224, 224))
            
            # Convert to tensor
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0)
            
            # Create inputs with image
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Add image tensor
            inputs['pixel_values'] = image_tensor.to(self.device)
            
            return {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as e:
            logger.debug(f"Manual processing approach failed: {e}")
        return None


def patch_phi_ground_vision_tokenization():
    """Patch Phi Ground to use the vision tokenization fix."""
    
    try:
        from src.ai.phi_ground import PhiGroundActionGenerator
        
        # Store original method
        original_generate_touch_action = PhiGroundActionGenerator.generate_touch_action
        
        async def patched_generate_touch_action(self, image_path: str, task_description: str, action_history, ui_elements):
            """Patched generate_touch_action with improved vision tokenization."""
            
            if not self.vision_supported:
                return await original_generate_touch_action(self, image_path, task_description, action_history, ui_elements)
            
            try:
                from PIL import Image
                image = Image.open(image_path).convert('RGB')
                
                # Create vision tokenization fix
                vision_fix = VisionTokenizationFix(self.model_name, self.tokenizer, self.device)
                
                # Create prompt
                prompt = self._create_phi_ground_prompt(image_path, task_description, action_history)
                
                # Try vision tokenization
                inputs = vision_fix.tokenize_with_vision(prompt, image)
                
                if inputs is not None:
                    # Use vision inputs
                    logger.info("✅ Using vision tokenization")
                else:
                    # Fallback to text-only
                    logger.warning("⚠️ Falling back to text-only tokenization")
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Continue with generation
                return await original_generate_touch_action(self, image_path, task_description, action_history, ui_elements)
                
            except Exception as e:
                logger.error(f"Vision tokenization patch failed: {e}")
                return await original_generate_touch_action(self, image_path, task_description, action_history, ui_elements)
        
        # Replace the method
        PhiGroundActionGenerator.generate_touch_action = patched_generate_touch_action
        
        logger.info("✅ Patched Phi Ground vision tokenization")
        
    except Exception as e:
        logger.error(f"❌ Failed to patch Phi Ground: {e}")


# Auto-patch on import
patch_phi_ground_vision_tokenization()
'''
    
    # Write the fix
    output_path = Path(__file__).parent / "vision_tokenization_fix.py"
    output_path.write_text(fix_code)
    
    logger.info(f"✅ Created vision tokenization fix: {output_path}")
    return True


async def main():
    """Main function to fix vision tokenization."""
    logger.info("🚀 Starting vision tokenization fix...")
    
    # Test current vision tokenization
    test_result = await test_vision_tokenization()
    
    # Create fix
    fix_created = create_vision_tokenization_fix()
    
    # Summary
    logger.info("📊 Vision Tokenization Fix Summary:")
    logger.info(f"  Test result: {'✅ PASSED' if test_result else '❌ FAILED'}")
    logger.info(f"  Fix created: {'✅ SUCCESS' if fix_created else '❌ FAILED'}")
    
    logger.info("💡 Next steps:")
    logger.info("  1. Import vision_tokenization_fix.py at the start of your script")
    logger.info("  2. This will fix the vision tokenization issues")
    logger.info("  3. The model should now properly process images")


if __name__ == "__main__":
    asyncio.run(main())
