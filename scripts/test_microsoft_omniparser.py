#!/usr/bin/env python3
"""
Test Microsoft OmniParser-v2.0 Integration
==========================================

This script tests the Microsoft OmniParser-v2.0 integration for local UI element detection.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_microsoft_omniparser_setup():
    """Test Microsoft OmniParser-v2.0 setup and configuration."""
    print("🔍 Testing Microsoft OmniParser-v2.0 Setup")
    print("=" * 50)
    
    # Test imports
    try:
        from src.vision.microsoft_omniparser import MicrosoftOmniParserEngine, OmniParserConfig
        print("✅ Microsoft OmniParser-v2.0 engine imports successful")
    except ImportError as e:
        print(f"❌ Microsoft OmniParser-v2.0 imports failed: {e}")
        return False
    
    # Test engine creation
    try:
        config = OmniParserConfig()
        engine = MicrosoftOmniParserEngine(config)
        print("✅ Microsoft OmniParser-v2.0 engine created successfully")
        return True
    except Exception as e:
        print(f"❌ Microsoft OmniParser-v2.0 engine creation failed: {e}")
        return False

def test_microsoft_omniparser_model():
    """Test Microsoft OmniParser-v2.0 model loading."""
    print("\n🤖 Testing Microsoft OmniParser-v2.0 Model Loading")
    print("=" * 50)
    
    try:
        from src.vision.microsoft_omniparser import MicrosoftOmniParserEngine, OmniParserConfig
        
        config = OmniParserConfig()
        engine = MicrosoftOmniParserEngine(config)
        
        print("🚀 Initializing Microsoft OmniParser-v2.0 model...")
        engine.initialize()
        
        print(f"✅ Model loaded successfully on device: {engine.device}")
        print(f"✅ Model: {engine.config.model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Microsoft OmniParser-v2.0 model loading failed: {e}")
        print("💡 This might be due to:")
        print("   - Model not available on HuggingFace")
        print("   - Network connectivity issues")
        print("   - Insufficient memory/GPU")
        return False

def test_microsoft_omniparser_inference():
    """Test Microsoft OmniParser-v2.0 inference."""
    print("\n🧪 Testing Microsoft OmniParser-v2.0 Inference")
    print("=" * 50)
    
    try:
        from src.vision.microsoft_omniparser import MicrosoftOmniParserEngine, OmniParserConfig
        from PIL import Image
        import numpy as np
        
        config = OmniParserConfig()
        engine = MicrosoftOmniParserEngine(config)
        
        # Create test image
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        test_path = "test_microsoft_omniparser_image.png"
        test_image.save(test_path)
        
        print(f"📸 Created test image: {test_path}")
        
        # Test inference
        print("🔍 Testing Microsoft OmniParser-v2.0 inference...")
        elements = engine.analyze_screenshot(test_path)
        
        if elements:
            print(f"✅ Microsoft OmniParser-v2.0 inference successful! Detected {len(elements)} elements")
            
            # Show element details
            for i, element in enumerate(elements[:3]):  # Show first 3 elements
                print(f"  Element {i+1}: '{element.text}' ({element.metadata.get('omniparser_type', 'unknown')}) "
                      f"confidence: {element.confidence:.2f}")
            
            # Get summary
            summary = engine.get_element_summary(elements)
            print(f"📊 Summary: {summary['total_elements']} total, {summary['clickable_elements']} clickable")
            
        else:
            print("⚠️  Microsoft OmniParser-v2.0 returned no elements")
        
        # Cleanup
        os.remove(test_path)
        return True
        
    except Exception as e:
        print(f"❌ Microsoft OmniParser-v2.0 inference test failed: {e}")
        return False

def test_vision_engine_integration():
    """Test Microsoft OmniParser-v2.0 integration with VisionEngine."""
    print("\n👁️ Testing VisionEngine Integration")
    print("=" * 50)
    
    try:
        from src.vision.engine import VisionEngine
        
        # Initialize VisionEngine with Microsoft OmniParser-v2.0
        print("🚀 Initializing VisionEngine with Microsoft OmniParser-v2.0...")
        vision_engine = VisionEngine(use_microsoft_omniparser=True)
        
        # Check if Microsoft OmniParser-v2.0 is available
        if hasattr(vision_engine, '_microsoft_omniparser_available') and vision_engine._microsoft_omniparser_available:
            print("✅ Microsoft OmniParser-v2.0 is available in VisionEngine")
        else:
            print("❌ Microsoft OmniParser-v2.0 is not available in VisionEngine")
            return False
        
        # Test with actual screenshot if available
        screenshot_path = "test_reports/action_1_screenshot.png"
        if Path(screenshot_path).exists():
            print(f"📸 Testing with actual screenshot: {screenshot_path}")
            
            elements = vision_engine.analyze(screenshot_path)
            print(f"✅ VisionEngine with Microsoft OmniParser-v2.0 detected {len(elements)} elements")
            
            # Show some elements
            for i, element in enumerate(elements[:5]):
                print(f"  Element {i+1}: '{element.text}' at {element.bbox}")
            
            return True
        else:
            print("⚠️  No screenshot found for testing")
            return False
            
    except Exception as e:
        print(f"❌ VisionEngine integration test failed: {e}")
        return False

def test_fallback_behavior():
    """Test fallback behavior when Microsoft OmniParser-v2.0 is not available."""
    print("\n🔄 Testing Fallback Behavior")
    print("=" * 50)
    
    try:
        from src.vision.engine import VisionEngine
        
        # Test without Microsoft OmniParser-v2.0
        print("🚀 Testing VisionEngine without Microsoft OmniParser-v2.0...")
        vision_engine = VisionEngine(use_microsoft_omniparser=False)
        
        screenshot_path = "test_reports/action_1_screenshot.png"
        if Path(screenshot_path).exists():
            elements = vision_engine.analyze(screenshot_path)
            print(f"✅ Fallback OCR detected {len(elements)} elements")
            return True
        else:
            print("⚠️  No screenshot found for fallback testing")
            return False
            
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False

def main():
    """Main function."""
    print("🚀 Microsoft OmniParser-v2.0 Integration Test")
    print("=" * 60)
    
    # Test 1: Setup
    setup_ok = test_microsoft_omniparser_setup()
    
    # Test 2: Model loading
    model_ok = test_microsoft_omniparser_model() if setup_ok else False
    
    # Test 3: Inference
    inference_ok = test_microsoft_omniparser_inference() if model_ok else False
    
    # Test 4: VisionEngine integration
    integration_ok = test_vision_engine_integration() if inference_ok else False
    
    # Test 5: Fallback behavior
    fallback_ok = test_fallback_behavior()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 40)
    print(f"Setup: {'✅ PASS' if setup_ok else '❌ FAIL'}")
    print(f"Model Loading: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"Inference: {'✅ PASS' if inference_ok else '❌ FAIL'}")
    print(f"VisionEngine Integration: {'✅ PASS' if integration_ok else '❌ FAIL'}")
    print(f"Fallback Behavior: {'✅ PASS' if fallback_ok else '❌ FAIL'}")
    
    if setup_ok and model_ok and inference_ok and integration_ok:
        print("\n🎉 Microsoft OmniParser-v2.0 integration is working correctly!")
        print("💡 Benefits:")
        print("   - Local UI element detection (no API calls)")
        print("   - Advanced element classification")
        print("   - Better accuracy than basic OCR")
        print("   - No external dependencies or costs")
    else:
        print("\n🔧 Some tests failed. Check the error messages above.")
        print("💡 Microsoft OmniParser-v2.0 might not be available yet.")
        print("   The system will automatically fall back to OCR.")
    
    return setup_ok and model_ok and inference_ok and integration_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
