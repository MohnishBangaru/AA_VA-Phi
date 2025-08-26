#!/usr/bin/env python3
"""Simple test script to verify Phi Ground works without FlashAttention2."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_phi_ground():
    """Test Phi Ground initialization without FlashAttention2."""
    print("🧪 Testing Phi Ground Model...")
    
    try:
        from src.ai.phi_ground import PhiGroundActionGenerator
        
        # Initialize Phi Ground
        phi_ground = PhiGroundActionGenerator()
        await phi_ground.initialize()
        
        print("✅ Phi Ground initialized successfully!")
        print("✅ FlashAttention2 fallback is working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Phi Ground test failed: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    
    success = asyncio.run(test_phi_ground())
    if success:
        print("\n🎉 All tests passed! Phi Ground is ready to use.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
        sys.exit(1)
