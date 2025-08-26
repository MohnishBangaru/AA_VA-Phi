#!/usr/bin/env python3
"""
Test Phi Ground Vision Capabilities
==================================

This script tests whether Phi Ground can properly process screenshots.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.phi_ground import PhiGroundActionGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_phi_vision():
    """Test Phi Ground vision capabilities."""
    print("🔍 Testing Phi Ground Vision Capabilities")
    print("=" * 50)
    
    # Initialize Phi Ground
    phi_ground = PhiGroundActionGenerator()
    
    try:
        print("🚀 Initializing Phi Ground...")
        await phi_ground.initialize()
        
        print(f"📊 Model: {phi_ground.model_name}")
        print(f"👁️  Vision Supported: {phi_ground.vision_supported}")
        
        # Test with an existing screenshot
        test_image_path = "screenshot_1756248771.png"  # Use an existing screenshot
        
        if not Path(test_image_path).exists():
            # Try to find any screenshot file
            import os
            screenshot_files = [f for f in os.listdir(".") if f.startswith("screenshot_") and f.endswith(".png")]
            if screenshot_files:
                test_image_path = screenshot_files[0]
                print(f"📸 Using existing screenshot: {test_image_path}")
            else:
                print(f"⚠️  No screenshot files found")
                print("   Create a test screenshot or use an existing one")
                test_image_path = None
        
        if test_image_path and Path(test_image_path).exists():
            print(f"📸 Testing with screenshot: {test_image_path}")
            
            # Test action generation
            action = await phi_ground.generate_touch_action(
                image_path=test_image_path,
                task_description="Test task - find and tap on any button or interactive element",
                action_history=[],
                ui_elements=[]
            )
            
            if action:
                print(f"✅ Action generated: {action}")
            else:
                print("❌ No action generated")
        else:
            # Test with vision support check only
            print("\n🔍 Vision Support Details:")
            print(f"   Model: {phi_ground.model_name}")
            print(f"   Vision Supported: {phi_ground.vision_supported}")
            
            if "vision" in phi_ground.model_name.lower():
                print("   ✅ Using vision-capable model")
            else:
                print("   ⚠️  Using text-only model")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

def main():
    """Main function."""
    asyncio.run(test_phi_vision())

if __name__ == "__main__":
    main()
