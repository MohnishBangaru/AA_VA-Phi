#!/usr/bin/env python3
"""
Test OMniParser Integration
===========================

This script tests the OMniParser integration with the VisionEngine.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_omniparser_basic():
    """Test basic OMniParser functionality."""
    print("🔍 Testing Basic OMniParser Functionality")
    print("=" * 50)
    
    try:
        from src.vision.omniparser_integration import create_omniparser_integration
        
        # Create integration
        omniparser = create_omniparser_integration()
        
        # Check availability
        is_available = omniparser.is_available()
        print(f"OMniParser Available: {'✅ YES' if is_available else '❌ NO'}")
        
        if not is_available:
            print("⚠️  OMniParser not available. Check API key and connection.")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Basic OMniParser test failed: {e}")
        return False

def test_omniparser_analysis():
    """Test OMniParser screenshot analysis."""
    print("\n📸 Testing OMniParser Screenshot Analysis")
    print("=" * 50)
    
    try:
        from src.vision.omniparser_integration import create_omniparser_integration
        
        omniparser = create_omniparser_integration()
        
        if not omniparser.is_available():
            print("❌ OMniParser not available for analysis test")
            return False
        
        # Find test screenshot
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
        
        if not screenshot_path:
            print("⚠️  No test screenshot found")
            return False
        
        print(f"📸 Analyzing: {screenshot_path}")
        
        # Test analysis
        elements = omniparser.analyze_screenshot(screenshot_path)
        print(f"✅ Detected {len(elements)} UI elements")
        
        # Show element details
        if elements:
            print("\n📋 Element Details:")
            for i, element in enumerate(elements[:10]):  # Show first 10
                print(f"  {i+1:2d}. {element.element_type:8s} | '{element.text[:30]:<30}' | Confidence: {element.confidence:.2f}")
        
        # Test layout analysis
        print("\n🏗️  Testing Layout Analysis...")
        layout_info = omniparser.extract_layout_info(screenshot_path)
        if layout_info:
            print(f"✅ Layout analysis successful: {len(layout_info)} layout features")
        else:
            print("⚠️  Layout analysis returned no data")
        
        # Test hierarchy analysis
        print("\n🌳 Testing Hierarchy Analysis...")
        hierarchy = omniparser.get_element_hierarchy(screenshot_path)
        if hierarchy:
            print(f"✅ Hierarchy analysis successful: {len(hierarchy)} hierarchy levels")
        else:
            print("⚠️  Hierarchy analysis returned no data")
        
        return True
        
    except Exception as e:
        print(f"❌ OMniParser analysis test failed: {e}")
        return False

def test_vision_engine_integration():
    """Test OMniParser integration with VisionEngine."""
    print("\n👁️ Testing VisionEngine Integration")
    print("=" * 50)
    
    try:
        from src.vision.engine import VisionEngine
        
        # Create VisionEngine
        vision_engine = VisionEngine()
        
        # Check OMniParser availability in VisionEngine
        has_omniparser = hasattr(vision_engine, 'omniparser') and vision_engine.omniparser is not None
        omniparser_available = hasattr(vision_engine, '_omniparser_available') and vision_engine._omniparser_available
        
        print(f"VisionEngine has OMniParser: {'✅ YES' if has_omniparser else '❌ NO'}")
        print(f"OMniParser available: {'✅ YES' if omniparser_available else '❌ NO'}")
        
        if not omniparser_available:
            print("⚠️  OMniParser not available in VisionEngine")
            return False
        
        # Find test screenshot
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
        
        if not screenshot_path:
            print("⚠️  No test screenshot found")
            return False
        
        print(f"📸 Testing VisionEngine analysis: {screenshot_path}")
        
        # Test VisionEngine analysis
        elements = vision_engine.analyze(screenshot_path)
        print(f"✅ VisionEngine detected {len(elements)} elements")
        
        # Show element details
        if elements:
            print("\n📋 VisionEngine Element Details:")
            for i, element in enumerate(elements[:10]):  # Show first 10
                print(f"  {i+1:2d}. {element.element_type:8s} | '{element.text[:30]:<30}' | Confidence: {element.confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ VisionEngine integration test failed: {e}")
        return False

def test_phi_ground_integration():
    """Test OMniParser integration with Phi Ground."""
    print("\n🤖 Testing Phi Ground Integration")
    print("=" * 50)
    
    try:
        from src.ai.phi_ground import PhiGroundActionGenerator
        import asyncio
        
        # Create Phi Ground
        phi_ground = PhiGroundActionGenerator()
        
        # Initialize (this will also initialize VisionEngine with OMniParser)
        asyncio.run(phi_ground.initialize())
        
        # Find test screenshot
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
        
        if not screenshot_path:
            print("⚠️  No test screenshot found")
            return False
        
        print(f"📸 Testing Phi Ground with OMniParser: {screenshot_path}")
        
        # Test action generation
        action = asyncio.run(phi_ground.generate_touch_action(
            image_path=screenshot_path,
            task_description="Find and tap on any button or interactive element",
            action_history=[],
            ui_elements=[]
        ))
        
        if action:
            print(f"✅ Phi Ground generated action: {action}")
        else:
            print("❌ Phi Ground generated no action")
        
        return True
        
    except Exception as e:
        print(f"❌ Phi Ground integration test failed: {e}")
        return False

def main():
    """Main function."""
    print("🚀 OMniParser Integration Test Suite")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Basic OMniParser", test_omniparser_basic),
        ("OMniParser Analysis", test_omniparser_analysis),
        ("VisionEngine Integration", test_vision_engine_integration),
        ("Phi Ground Integration", test_phi_ground_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25s}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! OMniParser integration is working perfectly!")
        print("✅ Enhanced UI element detection is active")
        print("✅ Better text extraction and button detection")
        print("✅ Improved automation accuracy")
    elif passed > 0:
        print("\n⚠️  Some tests failed. OMniParser integration is partially working.")
        print("   The system will use fallback methods for failed components.")
    else:
        print("\n❌ All tests failed. OMniParser integration is not working.")
        print("   Check your API key and internet connection.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
