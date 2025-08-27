#!/usr/bin/env python3
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
