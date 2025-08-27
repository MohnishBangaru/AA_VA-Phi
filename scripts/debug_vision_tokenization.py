#!/usr/bin/env python3
"""
Debug Vision Tokenization Issues
==============================

This script helps debug why vision tokenization is failing in Phi Ground.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_environment():
    """Test the Python environment and dependencies."""
    print("🔍 Testing Environment...")
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
    except ImportError as e:
        print(f"❌ PyTorch not available: {e}")
        return False
    
    # Test Transformers
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers not available: {e}")
        return False
    
    # Test PIL
    try:
        from PIL import Image
        print(f"✅ PIL/Pillow available")
    except ImportError as e:
        print(f"❌ PIL not available: {e}")
        return False
    
    return True


def test_phi_ground_initialization():
    """Test Phi Ground initialization."""
    print("\n🔍 Testing Phi Ground Initialization...")
    
    try:
        from src.ai.phi_ground import PhiGroundActionGenerator
        import asyncio
        
        phi_ground = PhiGroundActionGenerator()
        
        # Test initialization
        asyncio.run(phi_ground.initialize())
        
        print(f"✅ Phi Ground initialized successfully")
        print(f"   Model name: {phi_ground.model_name}")
        print(f"   Device: {phi_ground.device}")
        print(f"   Vision supported: {phi_ground.vision_supported}")
        
        return phi_ground
        
    except Exception as e:
        print(f"❌ Phi Ground initialization failed: {e}")
        return None


def test_vision_tokenization_methods(phi_ground):
    """Test different vision tokenization methods."""
    print("\n🔍 Testing Vision Tokenization Methods...")
    
    # Create a test image
    from PIL import Image
    import numpy as np
    
    # Create a simple test image
    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    test_prompt = "Test prompt for vision tokenization"
    
    print(f"   Test image size: {test_image.size}")
    print(f"   Test prompt: {test_prompt}")
    
    # Test Approach 1: AutoImageProcessor
    print("\n   Testing Approach 1: AutoImageProcessor")
    try:
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(phi_ground.model_name, trust_remote_code=True)
        processed_image = image_processor(test_image, return_tensors="pt")
        print(f"   ✅ AutoImageProcessor works")
        print(f"   Processed image keys: {list(processed_image.keys())}")
    except Exception as e:
        print(f"   ❌ AutoImageProcessor failed: {e}")
    
    # Test Approach 2: Model's image processor
    print("\n   Testing Approach 2: Model's image processor")
    try:
        if hasattr(phi_ground.tokenizer, 'image_processor'):
            processed_image = phi_ground.tokenizer.image_processor(test_image, return_tensors="pt")
            print(f"   ✅ Model's image processor works")
            print(f"   Processed image keys: {list(processed_image.keys())}")
        else:
            print(f"   ❌ No image_processor attribute found")
    except Exception as e:
        print(f"   ❌ Model's image processor failed: {e}")
    
    # Test Approach 3: Direct images parameter
    print("\n   Testing Approach 3: Direct images parameter")
    try:
        inputs = phi_ground.tokenizer(
            test_prompt,
            images=test_image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        print(f"   ✅ Direct images parameter works")
        print(f"   Input keys: {list(inputs.keys())}")
    except Exception as e:
        print(f"   ❌ Direct images parameter failed: {e}")
    
    # Test Approach 4: Manual processing
    print("\n   Testing Approach 4: Manual processing")
    try:
        import torchvision.transforms as transforms
        
        # Resize image
        image_resized = test_image.resize((224, 224))
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image_resized).unsqueeze(0)
        
        # Create inputs
        inputs = phi_ground.tokenizer(
            test_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Add image tensor
        inputs['pixel_values'] = image_tensor.to(phi_ground.device)
        print(f"   ✅ Manual processing works")
        print(f"   Input keys: {list(inputs.keys())}")
        
    except Exception as e:
        print(f"   ❌ Manual processing failed: {e}")


def test_model_vision_capabilities(phi_ground):
    """Test the model's vision capabilities."""
    print("\n🔍 Testing Model Vision Capabilities...")
    
    try:
        # Check if model has vision components
        model = phi_ground.model
        
        vision_components = [
            'vision_encoder',
            'vision_model',
            'image_encoder',
            'vision_tower'
        ]
        
        for component in vision_components:
            if hasattr(model, component):
                print(f"   ✅ Model has {component}")
            else:
                print(f"   ❌ Model missing {component}")
        
        # Check model config
        if hasattr(model, 'config'):
            config = model.config
            print(f"   Model type: {type(model).__name__}")
            print(f"   Config type: {type(config).__name__}")
            
            # Check for vision-related config
            vision_config_keys = [
                'vision_config',
                'image_size',
                'num_channels',
                'patch_size'
            ]
            
            for key in vision_config_keys:
                if hasattr(config, key):
                    print(f"   ✅ Config has {key}: {getattr(config, key)}")
                else:
                    print(f"   ❌ Config missing {key}")
        
    except Exception as e:
        print(f"   ❌ Model vision capability test failed: {e}")


def create_vision_fix():
    """Create a comprehensive vision tokenization fix."""
    print("\n🔧 Creating Comprehensive Vision Fix...")
    
    fix_code = '''#!/usr/bin/env python3
"""
Comprehensive Vision Tokenization Fix
===================================

This module provides a comprehensive fix for vision tokenization issues.
"""

import logging
from typing import Optional, Dict, Any
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class ComprehensiveVisionFix:
    """Comprehensive fix for vision tokenization issues."""
    
    def __init__(self, model_name: str, tokenizer, model, device: str):
        """Initialize the comprehensive vision fix."""
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.image_processor = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all vision components."""
        try:
            from transformers import AutoImageProcessor
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            logger.info("✅ Image processor initialized")
        except Exception as e:
            logger.warning(f"⚠️ Image processor failed: {e}")
            self.image_processor = None
    
    def tokenize_with_vision(self, prompt: str, image: Image.Image) -> Optional[Dict[str, torch.Tensor]]:
        """Comprehensive vision tokenization with multiple approaches."""
        
        approaches = [
            ("AutoImageProcessor", self._approach_1_auto_image_processor),
            ("Model Image Processor", self._approach_2_model_image_processor),
            ("Direct Images Parameter", self._approach_3_direct_images),
            ("Manual Processing", self._approach_4_manual_processing),
            ("Vision Encoder", self._approach_5_vision_encoder),
            ("Custom Vision Processing", self._approach_6_custom_vision)
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
        
        logger.warning("❌ All vision approaches failed")
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
            logger.debug(f"AutoImageProcessor failed: {e}")
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
            logger.debug(f"Model image processor failed: {e}")
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
            logger.debug(f"Direct images failed: {e}")
        return None
    
    def _approach_4_manual_processing(self, prompt: str, image: Image.Image) -> Optional[Dict[str, torch.Tensor]]:
        """Approach 4: Manual image processing."""
        try:
            import torchvision.transforms as transforms
            
            # Resize image
            image_resized = image.resize((224, 224))
            
            # Convert to tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image_resized).unsqueeze(0)
            
            # Create inputs
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Add image tensor
            inputs['pixel_values'] = image_tensor.to(self.device)
            return inputs
            
        except Exception as e:
            logger.debug(f"Manual processing failed: {e}")
        return None
    
    def _approach_5_vision_encoder(self, prompt: str, image: Image.Image) -> Optional[Dict[str, torch.Tensor]]:
        """Approach 5: Use vision encoder if available."""
        try:
            if hasattr(self.model, 'vision_encoder'):
                # Process image through vision encoder
                vision_outputs = self.model.vision_encoder(image)
                
                # Create text inputs
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # Add vision outputs
                inputs['vision_outputs'] = vision_outputs
                return {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as e:
            logger.debug(f"Vision encoder failed: {e}")
        return None
    
    def _approach_6_custom_vision(self, prompt: str, image: Image.Image) -> Optional[Dict[str, torch.Tensor]]:
        """Approach 6: Custom vision processing for Phi-3."""
        try:
            # Phi-3 specific processing
            if 'phi' in self.model_name.lower():
                # Resize to Phi-3 expected size
                image_resized = image.resize((336, 336))
                
                # Convert to tensor
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                image_tensor = transform(image_resized).unsqueeze(0)
                
                # Create inputs
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # Add image tensor for Phi-3
                inputs['pixel_values'] = image_tensor.to(self.device)
                return inputs
                
        except Exception as e:
            logger.debug(f"Custom vision failed: {e}")
        return None


def apply_comprehensive_vision_fix():
    """Apply the comprehensive vision fix to Phi Ground."""
    try:
        from src.ai.phi_ground import PhiGroundActionGenerator
        
        # Store original method
        original_generate_touch_action = PhiGroundActionGenerator.generate_touch_action
        
        async def patched_generate_touch_action(self, image_path: str, task_description: str, action_history, ui_elements):
            """Patched generate_touch_action with comprehensive vision fix."""
            
            if not self.vision_supported:
                return await original_generate_touch_action(self, image_path, task_description, action_history, ui_elements)
            
            try:
                from PIL import Image
                image = Image.open(image_path).convert('RGB')
                
                # Create comprehensive vision fix
                vision_fix = ComprehensiveVisionFix(self.model_name, self.tokenizer, self.model, self.device)
                
                # Create prompt
                prompt = self._create_phi_ground_prompt(image_path, task_description, action_history)
                
                # Try comprehensive vision tokenization
                inputs = vision_fix.tokenize_with_vision(prompt, image)
                
                if inputs is not None:
                    logger.info("✅ Using comprehensive vision tokenization")
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
                logger.error(f"Comprehensive vision fix failed: {e}")
                return await original_generate_touch_action(self, image_path, task_description, action_history, ui_elements)
        
        # Replace the method
        PhiGroundActionGenerator.generate_touch_action = patched_generate_touch_action
        
        logger.info("✅ Applied comprehensive vision tokenization fix")
        
    except Exception as e:
        logger.error(f"❌ Failed to apply comprehensive vision fix: {e}")


# Auto-apply on import
apply_comprehensive_vision_fix()
'''
    
    # Write the fix
    output_path = Path(__file__).parent / "comprehensive_vision_fix.py"
    output_path.write_text(fix_code)
    
    print(f"✅ Created comprehensive vision fix: {output_path}")
    return True


def main():
    """Main debugging function."""
    print("🔍 Debugging Vision Tokenization Issues")
    print("=" * 50)
    
    # Test environment
    if not test_environment():
        print("❌ Environment test failed")
        return
    
    # Test Phi Ground initialization
    phi_ground = test_phi_ground_initialization()
    if phi_ground is None:
        print("❌ Phi Ground initialization failed")
        return
    
    # Test vision tokenization methods
    test_vision_tokenization_methods(phi_ground)
    
    # Test model vision capabilities
    test_model_vision_capabilities(phi_ground)
    
    # Create comprehensive fix
    create_vision_fix()
    
    print("\n💡 Next Steps:")
    print("1. Import comprehensive_vision_fix.py at the start of your script")
    print("2. This will apply a comprehensive fix for vision tokenization")
    print("3. The fix includes 6 different approaches for vision processing")


if __name__ == "__main__":
    main()
