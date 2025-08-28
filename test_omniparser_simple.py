#!/usr/bin/env python3
"""Simple test script for OmniParser v2 integration."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_omniparser_v2():
    """Test OmniParser v2 integration."""
    print("Testing OmniParser v2 integration...")
    
    try:
        # Test imports
        print("\n1. Testing imports...")
        from vision.omniparser_v2 import get_omniparser_v2_engine
        from vision.engine import VisionEngine
        from core.config import config
        print("✓ All imports successful")
        
        # Test OmniParser v2 engine initialization
        print("\n2. Testing OmniParser v2 engine initialization...")
        omniparser_engine = get_omniparser_v2_engine()
        
        if omniparser_engine.is_available():
            print("✓ OmniParser v2 engine is available")
            
            # Initialize the model
            print("Initializing model (this may take a few minutes on first run)...")
            asyncio.run(omniparser_engine.initialize())
            
            if omniparser_engine._initialized:
                print("✓ OmniParser v2 model initialized successfully")
            else:
                print("✗ Failed to initialize OmniParser v2 model")
        else:
            print("✗ OmniParser v2 engine is not available")
            print("  This could be due to:")
            print("  - Missing dependencies (transformers, torch)")
            print("  - OCR disabled in configuration")
            print("  - OmniParser v2 disabled in configuration")
        
        # Test VisionEngine integration
        print("\n3. Testing VisionEngine integration...")
        vision_engine = VisionEngine()
        
        if vision_engine._ocr_available:
            print("✓ VisionEngine OCR is available")
            
            if vision_engine._omniparser_engine and vision_engine._omniparser_engine.is_available():
                print("✓ VisionEngine is using OmniParser v2")
            elif vision_engine._tesseract_available:
                print("✓ VisionEngine is using Tesseract fallback")
            else:
                print("✗ VisionEngine OCR available but no engine detected")
        else:
            print("✗ VisionEngine OCR is not available")
        
        # Test with a sample image if available
        print("\n4. Testing with sample image...")
        sample_images = [
            "screenshot_1756396025.png",
            "test_reports/action_1_screenshot.png",
            "ocr_images/action_1_screenshot_debug.png"
        ]
        
        for image_path in sample_images:
            if os.path.exists(image_path):
                print(f"Found sample image: {image_path}")
                
                try:
                    # Test OmniParser v2 analysis
                    if omniparser_engine.is_available():
                        elements = omniparser_engine.analyze(image_path)
                        print(f"  OmniParser v2 detected {len(elements)} elements")
                        
                        # Show first few elements
                        for i, element in enumerate(elements[:3]):
                            print(f"    {i+1}. '{element.text}' (confidence: {element.confidence:.2f})")
                    
                    # Test VisionEngine analysis
                    if vision_engine._ocr_available:
                        elements = vision_engine.analyze(image_path)
                        print(f"  VisionEngine detected {len(elements)} elements")
                        
                        # Show first few elements
                        for i, element in enumerate(elements[:3]):
                            print(f"    {i+1}. '{element.text}' (confidence: {element.confidence:.2f})")
                    
                    break
                    
                except Exception as e:
                    print(f"  Error analyzing image: {e}")
        
        print("\n5. Configuration summary:")
        print(f"  USE_OCR: {config.use_ocr}")
        print(f"  USE_OMNIPARSER_V2: {config.use_omniparser_v2}")
        print(f"  OMNIPARSER_V2_MODEL: {config.omniparser_v2_model}")
        print(f"  USE_GPU: {config.use_gpu}")
        print(f"  HUGGING_FACE_HUB_TOKEN: {'Set' if os.getenv('HUGGING_FACE_HUB_TOKEN') else 'Not set'}")
        
        print("\n✅ OmniParser v2 integration test completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_omniparser_v2()
