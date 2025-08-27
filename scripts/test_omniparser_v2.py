#!/usr/bin/env python3
"""
Test OmniParser 2.0 Specifically
===============================

This script tests OmniParser 2.0 with multiple loading approaches.
"""

import os
import sys
from pathlib import Path

def test_omniparser_v2_direct():
    """Test OmniParser 2.0 directly."""
    print("🧪 Testing OmniParser 2.0 Direct Loading")
    print("=" * 50)
    
    try:
        from transformers import AutoProcessor, AutoTokenizer, AutoModel
        
        model_name = "microsoft/OmniParser-v2.0"
        
        # Approach 1: Standard loading
        print("📦 Approach 1: Standard loading...")
        try:
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=True
            )
            print("✅ Processor loaded")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=True
            )
            print("✅ Tokenizer loaded")
            
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=True
            )
            print("✅ Model loaded")
            
            return True
            
        except Exception as e:
            print(f"❌ Standard loading failed: {e}")
        
        # Approach 2: Minimal loading
        print("\n📦 Approach 2: Minimal loading...")
        try:
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=True,
                local_files_only=False
            )
            print("✅ Processor loaded")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=True,
                local_files_only=False
            )
            print("✅ Tokenizer loaded")
            
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=True,
                local_files_only=False
            )
            print("✅ Model loaded")
            
            return True
            
        except Exception as e:
            print(f"❌ Minimal loading failed: {e}")
        
        # Approach 3: Force download
        print("\n📦 Approach 3: Force download...")
        try:
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=True,
                force_download=True
            )
            print("✅ Processor loaded")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=True,
                force_download=True
            )
            print("✅ Tokenizer loaded")
            
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=True,
                force_download=True
            )
            print("✅ Model loaded")
            
            return True
            
        except Exception as e:
            print(f"❌ Force download failed: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        return False

def test_omniparser_v2_integration():
    """Test OmniParser 2.0 through our integration."""
    print("\n🔧 Testing OmniParser 2.0 Integration")
    print("=" * 50)
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from src.vision.omniparser_integration import create_omniparser_integration
        
        # Create integration
        omniparser = create_omniparser_integration()
        
        # Check availability
        is_available = omniparser.is_available()
        print(f"OmniParser 2.0 Available: {'✅ YES' if is_available else '❌ NO'}")
        
        if is_available:
            print("✅ OmniParser 2.0 integration successful!")
            return True
        else:
            print("❌ OmniParser 2.0 integration failed")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_vision_engine_with_omniparser_v2():
    """Test VisionEngine with OmniParser 2.0."""
    print("\n👁️ Testing VisionEngine with OmniParser 2.0")
    print("=" * 50)
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from src.vision.engine import VisionEngine
        
        # Create VisionEngine
        vision_engine = VisionEngine()
        
        # Check OmniParser availability
        if hasattr(vision_engine, '_omniparser_available'):
            omniparser_available = vision_engine._omniparser_available
            print(f"VisionEngine OmniParser 2.0: {'✅ YES' if omniparser_available else '❌ NO'}")
            
            if omniparser_available:
                print("✅ VisionEngine will use OmniParser 2.0")
                return True
            else:
                print("❌ VisionEngine will fall back to OCR")
                return False
        
        return False
        
    except Exception as e:
        print(f"❌ VisionEngine test failed: {e}")
        return False

def main():
    """Main function."""
    print("🚀 OmniParser 2.0 Specific Test")
    print("=" * 50)
    
    # Check token
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if token:
        print(f"✅ Token found: {token[:10]}...")
    else:
        print("❌ No token found")
        print("Set with: export HUGGING_FACE_HUB_TOKEN='your_token'")
        return False
    
    # Test direct loading
    direct_ok = test_omniparser_v2_direct()
    
    # Test integration
    integration_ok = test_omniparser_v2_integration()
    
    # Test VisionEngine
    vision_ok = test_vision_engine_with_omniparser_v2()
    
    print("\n📊 Test Results")
    print("=" * 30)
    print(f"Direct Loading: {'✅ PASS' if direct_ok else '❌ FAIL'}")
    print(f"Integration: {'✅ PASS' if integration_ok else '❌ FAIL'}")
    print(f"VisionEngine: {'✅ PASS' if vision_ok else '❌ FAIL'}")
    
    if direct_ok and integration_ok and vision_ok:
        print("\n🎉 OmniParser 2.0 is fully working!")
        print("✅ Direct model loading successful")
        print("✅ Integration working properly")
        print("✅ VisionEngine will use OmniParser 2.0")
    elif integration_ok or vision_ok:
        print("\n⚠️  Partial success")
        print("✅ Some components are working")
        print("⚠️  Check the failed components above")
    else:
        print("\n❌ OmniParser 2.0 is not working")
        print("⚠️  The system will fall back to OCR")
    
    return direct_ok and integration_ok and vision_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
