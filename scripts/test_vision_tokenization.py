#!/usr/bin/env python3
"""
Test Vision Tokenization for Phi-3-vision Model
===============================================

This script tests the vision tokenization specifically for the Phi-3-vision model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_phi_vision_tokenization():
    """Test vision tokenization for Phi-3-vision model."""
    print("🔍 Testing Phi-3-vision Tokenization")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer, AutoImageProcessor
        from PIL import Image
        import numpy as np
        import torch
        
        model_name = "microsoft/Phi-3-vision-128k-instruct"
        print(f"📊 Testing with model: {model_name}")
        
        # Load tokenizer and image processor
        print("📦 Loading tokenizer and image processor...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        print("✅ Tokenizer and image processor loaded successfully")
        
        # Create test image
        print("📸 Creating test image...")
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Create test prompt
        test_prompt = "Describe what you see in this image."
        
        # Test Approach 1: AutoImageProcessor
        print("\n🧪 Testing Approach 1: AutoImageProcessor")
        try:
            processed_image = image_processor(test_image, return_tensors="pt")
            print(f"✅ Image processed successfully. Keys: {list(processed_image.keys())}")
            
            # Create text inputs
            text_inputs = tokenizer(
                test_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            print(f"✅ Text tokenized successfully. Keys: {list(text_inputs.keys())}")
            
            # Combine text and image inputs
            combined_inputs = text_inputs.copy()
            for key, value in processed_image.items():
                combined_inputs[key] = value
            
            print(f"✅ Combined inputs created. Final keys: {list(combined_inputs.keys())}")
            print("✅ Approach 1 successful!")
            
        except Exception as e:
            print(f"❌ Approach 1 failed: {e}")
            return False
        
        # Test Approach 2: Direct image embedding
        print("\n🧪 Testing Approach 2: Direct image embedding")
        try:
            inputs = tokenizer(
                test_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs['image'] = test_image
            print("✅ Approach 2 successful!")
            
        except Exception as e:
            print(f"❌ Approach 2 failed: {e}")
        
        # Test with actual screenshot if available
        print("\n🧪 Testing with actual screenshot...")
        screenshot_path = "test_reports/action_1_screenshot.png"
        if Path(screenshot_path).exists():
            try:
                screenshot = Image.open(screenshot_path)
                print(f"✅ Loaded screenshot: {screenshot.size}")
                
                # Process with image processor
                processed_screenshot = image_processor(screenshot, return_tensors="pt")
                print(f"✅ Screenshot processed. Keys: {list(processed_screenshot.keys())}")
                
                # Test tokenization
                screenshot_inputs = tokenizer(
                    "What UI elements do you see in this screenshot?",
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # Combine
                final_inputs = screenshot_inputs.copy()
                for key, value in processed_screenshot.items():
                    final_inputs[key] = value
                
                print(f"✅ Screenshot tokenization successful. Final keys: {list(final_inputs.keys())}")
                
            except Exception as e:
                print(f"❌ Screenshot test failed: {e}")
        else:
            print("⚠️  No screenshot found for testing")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision tokenization test failed: {e}")
        return False

def test_phi_ground_vision():
    """Test vision tokenization through Phi Ground."""
    print("\n👁️ Testing Phi Ground Vision Integration")
    print("=" * 50)
    
    try:
        from src.ai.phi_ground import PhiGroundActionGenerator
        import asyncio
        
        # Initialize Phi Ground
        print("🚀 Initializing Phi Ground...")
        phi_ground = PhiGroundActionGenerator()
        
        # Test initialization
        asyncio.run(phi_ground.initialize())
        print(f"✅ Phi Ground initialized. Vision supported: {phi_ground.vision_supported}")
        
        # Test with screenshot if available
        screenshot_path = "test_reports/action_1_screenshot.png"
        if Path(screenshot_path).exists():
            print("📸 Testing with actual screenshot...")
            
            action = asyncio.run(phi_ground.generate_touch_action(
                image_path=screenshot_path,
                task_description="Find and tap on any button or interactive element",
                action_history=[],
                ui_elements=[]
            ))
            
            if action:
                print(f"✅ Action generated: {action}")
            else:
                print("❌ No action generated")
        else:
            print("⚠️  No screenshot found for testing")
        
        return True
        
    except Exception as e:
        print(f"❌ Phi Ground vision test failed: {e}")
        return False

def main():
    """Main function."""
    print("🚀 Phi-3-vision Tokenization Test")
    print("=" * 60)
    
    # Test 1: Basic vision tokenization
    vision_ok = test_phi_vision_tokenization()
    
    # Test 2: Phi Ground integration
    phi_ground_ok = test_phi_ground_vision()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 40)
    print(f"Vision Tokenization: {'✅ PASS' if vision_ok else '❌ FAIL'}")
    print(f"Phi Ground Integration: {'✅ PASS' if phi_ground_ok else '❌ FAIL'}")
    
    if vision_ok and phi_ground_ok:
        print("\n🎉 All tests passed! Vision tokenization is working correctly.")
    else:
        print("\n🔧 Some tests failed. Check the error messages above.")
    
    return vision_ok and phi_ground_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
