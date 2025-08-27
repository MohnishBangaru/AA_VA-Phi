#!/usr/bin/env python3
"""
Setup OmniParser 2.0 Integration
================================

This script helps set up OmniParser 2.0 integration for enhanced UI element detection.
"""

import os
import sys
from pathlib import Path

def setup_omniparser():
    """Set up OmniParser 2.0 integration."""
    print("🚀 Setting up OmniParser 2.0 Integration")
    print("=" * 50)
    
    # Check system requirements
    print("🔍 Checking system requirements...")
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} available")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available, will use CPU")
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Install with: pip install torch torchvision")
        return False
    
    # Check Transformers
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__} available")
    except ImportError:
        print("❌ Transformers not installed")
        print("   Install with: pip install transformers")
        return False
    
    # Test OmniParser 2.0 integration
    print("\n🧪 Testing OmniParser 2.0 integration...")
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from src.vision.omniparser_integration import create_omniparser_integration
        
        # Create integration instance
        omniparser = create_omniparser_integration()
        
        if omniparser.is_available():
            print("✅ OmniParser 2.0 integration successful!")
            print("✅ Ready to use enhanced UI element detection")
            return True
        else:
            print("❌ OmniParser 2.0 integration failed")
            print("   Check your internet connection and model download")
            return False
            
    except Exception as e:
        print(f"❌ OmniParser 2.0 test failed: {e}")
        return False

def test_omniparser_with_screenshot():
    """Test OmniParser 2.0 with a sample screenshot."""
    print("\n📸 Testing OmniParser 2.0 with screenshot...")
    
    try:
        from src.vision.omniparser_integration import create_omniparser_integration
        
        omniparser = create_omniparser_integration()
        
        if not omniparser.is_available():
            print("❌ OmniParser 2.0 not available for testing")
            return False
        
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
            print(f"📸 Using screenshot: {screenshot_path}")
            
            # Test analysis
            elements = omniparser.analyze_screenshot(screenshot_path)
            print(f"✅ OmniParser 2.0 detected {len(elements)} UI elements")
            
            # Show some examples
            for i, element in enumerate(elements[:5]):
                print(f"  Element {i+1}: {element.element_type} - '{element.text}' at {element.bbox}")
            
            return True
        else:
            print("⚠️  No test screenshot found")
            return False
            
    except Exception as e:
        print(f"❌ OmniParser 2.0 screenshot test failed: {e}")
        return False

def main():
    """Main function."""
    print("🔧 OmniParser 2.0 Setup and Test")
    print("=" * 40)
    
    # Setup OmniParser 2.0
    setup_success = setup_omniparser()
    
    if setup_success:
        # Test with screenshot
        test_success = test_omniparser_with_screenshot()
        
        print("\n📊 Setup Summary")
        print("=" * 20)
        print(f"Setup: {'✅ PASS' if setup_success else '❌ FAIL'}")
        print(f"Test: {'✅ PASS' if test_success else '❌ FAIL'}")
        
        if setup_success and test_success:
            print("\n🎉 OmniParser 2.0 integration is ready!")
            print("✅ Enhanced UI element detection will be used")
            print("✅ Better text extraction and button detection")
            print("✅ Improved automation accuracy")
        else:
            print("\n⚠️  OmniParser 2.0 setup incomplete")
            print("   The system will fall back to OCR")
    else:
        print("\n⚠️  OmniParser 2.0 setup skipped")
        print("   The system will use OCR for text detection")

if __name__ == "__main__":
    main()
