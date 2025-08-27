#!/usr/bin/env python3
"""
Setup OMniParser Integration
============================

This script helps set up OMniParser integration for enhanced UI element detection.
"""

import os
import sys
from pathlib import Path

def setup_omniparser():
    """Set up OMniParser integration."""
    print("🚀 Setting up OMniParser Integration")
    print("=" * 50)
    
    # Check if API key is already set
    api_key = os.getenv('OMNIPARSER_API_KEY')
    if api_key:
        print(f"✅ OMniParser API key found: {api_key[:8]}...")
    else:
        print("❌ OMniParser API key not found")
        print("\n📋 To get an OMniParser API key:")
        print("1. Visit https://omniparser.ai")
        print("2. Sign up for an account")
        print("3. Get your API key from the dashboard")
        print("4. Set the environment variable:")
        print("   export OMNIPARSER_API_KEY='your_api_key_here'")
        
        # Ask user to input API key
        user_key = input("\n🔑 Enter your OMniParser API key (or press Enter to skip): ").strip()
        if user_key:
            # Add to .env file
            env_file = Path(".env")
            if env_file.exists():
                with open(env_file, "a") as f:
                    f.write(f"\n# OMniParser Configuration\nOMNIPARSER_API_KEY={user_key}\n")
            else:
                with open(env_file, "w") as f:
                    f.write(f"# OMniParser Configuration\nOMNIPARSER_API_KEY={user_key}\n")
            
            print("✅ API key saved to .env file")
            api_key = user_key
        else:
            print("⚠️  Skipping OMniParser setup")
            return False
    
    # Test OMniParser integration
    print("\n🧪 Testing OMniParser integration...")
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from src.vision.omniparser_integration import create_omniparser_integration
        
        # Create integration instance
        omniparser = create_omniparser_integration(api_key)
        
        if omniparser.is_available():
            print("✅ OMniParser integration successful!")
            print("✅ Ready to use enhanced UI element detection")
            return True
        else:
            print("❌ OMniParser integration failed")
            print("   Check your API key and internet connection")
            return False
            
    except Exception as e:
        print(f"❌ OMniParser test failed: {e}")
        return False

def test_omniparser_with_screenshot():
    """Test OMniParser with a sample screenshot."""
    print("\n📸 Testing OMniParser with screenshot...")
    
    try:
        from src.vision.omniparser_integration import create_omniparser_integration
        
        omniparser = create_omniparser_integration()
        
        if not omniparser.is_available():
            print("❌ OMniParser not available for testing")
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
            print(f"✅ OMniParser detected {len(elements)} UI elements")
            
            # Show some examples
            for i, element in enumerate(elements[:5]):
                print(f"  Element {i+1}: {element.element_type} - '{element.text}' at {element.bbox}")
            
            return True
        else:
            print("⚠️  No test screenshot found")
            return False
            
    except Exception as e:
        print(f"❌ OMniParser screenshot test failed: {e}")
        return False

def main():
    """Main function."""
    print("🔧 OMniParser Setup and Test")
    print("=" * 40)
    
    # Setup OMniParser
    setup_success = setup_omniparser()
    
    if setup_success:
        # Test with screenshot
        test_success = test_omniparser_with_screenshot()
        
        print("\n📊 Setup Summary")
        print("=" * 20)
        print(f"Setup: {'✅ PASS' if setup_success else '❌ FAIL'}")
        print(f"Test: {'✅ PASS' if test_success else '❌ FAIL'}")
        
        if setup_success and test_success:
            print("\n🎉 OMniParser integration is ready!")
            print("✅ Enhanced UI element detection will be used")
            print("✅ Better text extraction and button detection")
            print("✅ Improved automation accuracy")
        else:
            print("\n⚠️  OMniParser setup incomplete")
            print("   The system will fall back to OCR")
    else:
        print("\n⚠️  OMniParser setup skipped")
        print("   The system will use OCR for text detection")

if __name__ == "__main__":
    main()
