#!/usr/bin/env python3
"""
Test OCR Functionality
======================

This script tests Tesseract OCR installation and functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_tesseract_installation():
    """Test if Tesseract is properly installed."""
    print("🔍 Testing Tesseract Installation")
    print("=" * 40)
    
    try:
        import pytesseract
        print("✅ pytesseract imported successfully")
    except ImportError as e:
        print(f"❌ pytesseract import failed: {e}")
        return False
    
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
    except Exception as e:
        print(f"❌ Tesseract version check failed: {e}")
        return False
    
    return True

def test_ocr_functionality():
    """Test OCR functionality with a test image."""
    print("\n🧪 Testing OCR Functionality")
    print("=" * 40)
    
    try:
        import pytesseract
        from PIL import Image
        import numpy as np
        
        # Create a simple test image with text
        print("📸 Creating test image...")
        test_image = Image.fromarray(np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8))
        
        # Test OCR
        print("🔍 Running OCR test...")
        text = pytesseract.image_to_string(test_image)
        print(f"✅ OCR test successful, extracted text length: {len(text)}")
        
        return True
        
    except Exception as e:
        print(f"❌ OCR functionality test failed: {e}")
        return False

def test_vision_engine_ocr():
    """Test OCR through the VisionEngine."""
    print("\n👁️ Testing VisionEngine OCR")
    print("=" * 40)
    
    try:
        from src.vision.engine import VisionEngine
        
        # Initialize VisionEngine
        print("🚀 Initializing VisionEngine...")
        vision_engine = VisionEngine()
        
        # Check OCR availability
        if vision_engine._ocr_available:
            print("✅ OCR is available in VisionEngine")
            
            # Test with a dummy image
            print("📸 Testing VisionEngine OCR...")
            elements = vision_engine.analyze("test_reports/action_1_screenshot.png")
            print(f"✅ VisionEngine detected {len(elements)} UI elements")
            
            # Show some detected elements
            for i, element in enumerate(elements[:5]):  # Show first 5 elements
                print(f"  Element {i+1}: '{element.text}' at {element.bbox}")
            
            return True
        else:
            print("❌ OCR is not available in VisionEngine")
            return False
            
    except Exception as e:
        print(f"❌ VisionEngine OCR test failed: {e}")
        return False

def diagnose_tesseract_paths():
    """Diagnose Tesseract installation paths."""
    print("\n🔍 Diagnosing Tesseract Paths")
    print("=" * 40)
    
    import subprocess
    import os
    
    # Check common Tesseract paths
    tesseract_paths = [
        "/opt/homebrew/bin/tesseract",  # macOS Homebrew
        "/usr/local/bin/tesseract",     # macOS/Linux
        "/usr/bin/tesseract",           # Linux
        "tesseract"                     # PATH
    ]
    
    for path in tesseract_paths:
        try:
            result = subprocess.run([path, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ Tesseract found at: {path}")
                print(f"   Version: {result.stdout.strip()}")
                return path
        except Exception as e:
            print(f"❌ Tesseract not found at: {path}")
    
    print("❌ Tesseract not found in any common location")
    return None

def main():
    """Main function."""
    print("🚀 OCR Diagnostic Tool")
    print("=" * 50)
    
    # Test 1: Tesseract installation
    tesseract_ok = test_tesseract_installation()
    
    # Test 2: Diagnose paths
    tesseract_path = diagnose_tesseract_paths()
    
    # Test 3: OCR functionality
    ocr_ok = test_ocr_functionality()
    
    # Test 4: VisionEngine integration
    vision_ok = test_vision_engine_ocr()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 40)
    print(f"Tesseract Installation: {'✅ PASS' if tesseract_ok else '❌ FAIL'}")
    print(f"Tesseract Path Found: {'✅ YES' if tesseract_path else '❌ NO'}")
    print(f"OCR Functionality: {'✅ PASS' if ocr_ok else '❌ FAIL'}")
    print(f"VisionEngine Integration: {'✅ PASS' if vision_ok else '❌ FAIL'}")
    
    if not tesseract_ok or not tesseract_path:
        print("\n🔧 Recommendations:")
        print("1. Install Tesseract: ./scripts/install_tesseract_runpod.sh")
        print("2. Or install manually: sudo apt-get install tesseract-ocr")
        print("3. Restart your Python process after installation")
    
    return tesseract_ok and ocr_ok and vision_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
