#!/usr/bin/env python3
"""
Test OmniParser Integration
===========================

This script tests the OmniParser integration for advanced UI element detection.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_omniparser_setup():
    """Test OmniParser setup and configuration."""
    print("🔍 Testing OmniParser Setup")
    print("=" * 40)
    
    # Check API key
    api_key = os.getenv("OMNIPARSER_API_KEY")
    if api_key:
        print(f"✅ OmniParser API key found: {api_key[:8]}...")
    else:
        print("❌ OmniParser API key not found")
        print("   Set OMNIPARSER_API_KEY environment variable")
        return False
    
    # Test imports
    try:
        from src.vision.omniparser_engine import OmniParserEngine, OmniParserConfig
        print("✅ OmniParser engine imports successful")
    except ImportError as e:
        print(f"❌ OmniParser imports failed: {e}")
        return False
    
    # Test engine creation
    try:
        config = OmniParserConfig(api_key=api_key)
        engine = OmniParserEngine(config)
        print("✅ OmniParser engine created successfully")
        return True
    except Exception as e:
        print(f"❌ OmniParser engine creation failed: {e}")
        return False

def test_omniparser_api():
    """Test OmniParser API connectivity."""
    print("\n🌐 Testing OmniParser API Connectivity")
    print("=" * 40)
    
    try:
        from src.vision.omniparser_engine import OmniParserEngine, OmniParserConfig
        
        api_key = os.getenv("OMNIPARSER_API_KEY")
        if not api_key:
            print("❌ No API key available for testing")
            return False
        
        config = OmniParserConfig(api_key=api_key)
        engine = OmniParserEngine(config)
        
        # Test with a simple image
        from PIL import Image
        import numpy as np
        
        # Create test image
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        test_path = "test_omniparser_image.png"
        test_image.save(test_path)
        
        print(f"📸 Created test image: {test_path}")
        
        # Test API call
        print("🔍 Testing OmniParser API call...")
        elements = engine.analyze_screenshot(test_path)
        
        if elements:
            print(f"✅ OmniParser API successful! Detected {len(elements)} elements")
            
            # Show element details
            for i, element in enumerate(elements[:3]):  # Show first 3 elements
                print(f"  Element {i+1}: '{element.text}' ({element.metadata.get('omniparser_type', 'unknown')})")
            
            # Get summary
            summary = engine.get_element_summary(elements)
            print(f"📊 Summary: {summary['total_elements']} total, {summary['clickable_elements']} clickable")
            
        else:
            print("⚠️  OmniParser API returned no elements")
        
        # Cleanup
        os.remove(test_path)
        return True
        
    except Exception as e:
        print(f"❌ OmniParser API test failed: {e}")
        return False

def test_vision_engine_integration():
    """Test OmniParser integration with VisionEngine."""
    print("\n👁️ Testing VisionEngine Integration")
    print("=" * 40)
    
    try:
        from src.vision.engine import VisionEngine
        
        # Initialize VisionEngine with OmniParser
        print("🚀 Initializing VisionEngine with OmniParser...")
        vision_engine = VisionEngine(use_omniparser=True)
        
        # Check if OmniParser is available
        if hasattr(vision_engine, '_omniparser_available') and vision_engine._omniparser_available:
            print("✅ OmniParser is available in VisionEngine")
        else:
            print("❌ OmniParser is not available in VisionEngine")
            return False
        
        # Test with actual screenshot if available
        screenshot_path = "test_reports/action_1_screenshot.png"
        if Path(screenshot_path).exists():
            print(f"📸 Testing with actual screenshot: {screenshot_path}")
            
            elements = vision_engine.analyze(screenshot_path)
            print(f"✅ VisionEngine with OmniParser detected {len(elements)} elements")
            
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
    """Test fallback behavior when OmniParser is not available."""
    print("\n🔄 Testing Fallback Behavior")
    print("=" * 40)
    
    try:
        from src.vision.engine import VisionEngine
        
        # Test without OmniParser
        print("🚀 Testing VisionEngine without OmniParser...")
        vision_engine = VisionEngine(use_omniparser=False)
        
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
    print("🚀 OmniParser Integration Test")
    print("=" * 60)
    
    # Test 1: Setup
    setup_ok = test_omniparser_setup()
    
    # Test 2: API connectivity
    api_ok = test_omniparser_api() if setup_ok else False
    
    # Test 3: VisionEngine integration
    integration_ok = test_vision_engine_integration() if api_ok else False
    
    # Test 4: Fallback behavior
    fallback_ok = test_fallback_behavior()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 40)
    print(f"Setup: {'✅ PASS' if setup_ok else '❌ FAIL'}")
    print(f"API Connectivity: {'✅ PASS' if api_ok else '❌ FAIL'}")
    print(f"VisionEngine Integration: {'✅ PASS' if integration_ok else '❌ FAIL'}")
    print(f"Fallback Behavior: {'✅ PASS' if fallback_ok else '❌ FAIL'}")
    
    if setup_ok and api_ok and integration_ok:
        print("\n🎉 OmniParser integration is working correctly!")
        print("💡 Benefits:")
        print("   - Advanced UI element detection")
        print("   - Better text extraction")
        print("   - Element classification (buttons, inputs, icons)")
        print("   - Improved automation accuracy")
    else:
        print("\n🔧 Some tests failed. Check the error messages above.")
        print("💡 To use OmniParser:")
        print("   1. Get an API key from https://omniparser.ai")
        print("   2. Set OMNIPARSER_API_KEY environment variable")
        print("   3. Restart your Python process")
    
    return setup_ok and api_ok and integration_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
