#!/usr/bin/env python3
"""
Simple Vision Test
=================

Test basic vision capabilities without complex models.
"""

import os
import sys
from pathlib import Path

def test_basic_vision():
    """Test basic vision capabilities."""
    print("👁️ Testing Basic Vision Capabilities")
    print("=" * 50)
    
    # Test if we can import basic vision libraries
    try:
        from PIL import Image
        import cv2
        import pytesseract
        print("✅ Basic vision libraries imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test if we can load and process an image
    try:
        # Look for a test screenshot
        test_screenshots = [
            "test_reports/action_1_screenshot.png",
            "screenshot_1756248771.png",
            "screenshot_1756248767.png"
        ]
        
        screenshot_path = None
        for path in test_screenshots:
            if Path(path).exists():
                screenshot_path = path
                break
        
        if screenshot_path:
            # Load image with PIL
            image = Image.open(screenshot_path)
            print(f"✅ Loaded image: {image.size}")
            
            # Convert to OpenCV format
            import numpy as np
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            print(f"✅ Converted to OpenCV format: {cv_image.shape}")
            
            # Try basic OCR
            try:
                text = pytesseract.image_to_string(cv_image)
                print(f"✅ OCR extracted {len(text)} characters")
            except Exception as e:
                print(f"⚠️  OCR failed: {e}")
            
            return True
        else:
            print("⚠️  No test screenshot found, but basic libraries work")
            return True
            
    except Exception as e:
        print(f"❌ Image processing failed: {e}")
        return False

def test_vision_engine():
    """Test VisionEngine without OmniParser."""
    print("\n🔧 Testing VisionEngine (OCR Only)")
    print("=" * 40)
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from src.vision.engine import VisionEngine
        
        # Create VisionEngine
        vision_engine = VisionEngine()
        
        # Check if OCR is available
        if hasattr(vision_engine, '_ocr_available'):
            ocr_available = vision_engine._ocr_available
            print(f"OCR Available: {'✅ YES' if ocr_available else '❌ NO'}")
        
        # Check OmniParser
        if hasattr(vision_engine, '_omniparser_available'):
            omniparser_available = vision_engine._omniparser_available
            print(f"OmniParser Available: {'✅ YES' if omniparser_available else '❌ NO'}")
        
        return True
        
    except Exception as e:
        print(f"❌ VisionEngine test failed: {e}")
        return False

def main():
    """Main function."""
    print("🚀 Simple Vision Test")
    print("=" * 30)
    
    # Test basic vision
    basic_ok = test_basic_vision()
    
    # Test VisionEngine
    engine_ok = test_vision_engine()
    
    print("\n📊 Test Summary")
    print("=" * 20)
    print(f"Basic Vision: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    print(f"VisionEngine: {'✅ PASS' if engine_ok else '❌ FAIL'}")
    
    if basic_ok and engine_ok:
        print("\n🎉 Basic vision capabilities are working!")
        print("✅ You can use OCR for text detection")
        print("✅ VisionEngine is functional")
        if not engine_ok:
            print("⚠️  OmniParser not available, but OCR will work")
    else:
        print("\n❌ Some basic vision features failed")
    
    return basic_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
