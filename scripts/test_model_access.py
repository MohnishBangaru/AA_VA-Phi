#!/usr/bin/env python3
"""
Test Hugging Face Model Access
============================

Simple script to test if we can access the OmniParser model.
"""

import os
import sys

def test_hf_login():
    """Test Hugging Face login."""
    print("🔑 Testing Hugging Face Login")
    print("=" * 40)
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        user = api.whoami()
        print(f"✅ Logged in as: {user}")
        return True
        
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False

def test_model_access():
    """Test model access."""
    print("\n📦 Testing Model Access")
    print("=" * 40)
    
    # Check token
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if token:
        print(f"✅ Token found: {token[:10]}...")
    else:
        print("❌ No token found in environment")
        return False
    
    # Test different model names (including simpler alternatives)
    models_to_test = [
        "microsoft/OmniParser",
        "microsoft/OmniParser-v2.0"
    ]
    
    for model_name in models_to_test:
        print(f"\n🧪 Testing model: {model_name}")
        try:
            from transformers import AutoTokenizer
            
            # Try to load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=True
            )
            print(f"✅ Tokenizer loaded successfully for {model_name}")
            
            # Try to load processor
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=True
            )
            print(f"✅ Processor loaded successfully for {model_name}")
            
            return model_name
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
    
    return False

def main():
    """Main function."""
    print("🚀 Hugging Face Model Access Test")
    print("=" * 50)
    
    # Test login
    login_ok = test_hf_login()
    
    if login_ok:
        # Test model access
        working_model = test_model_access()
        
        if working_model:
            print(f"\n🎉 Success! Working model: {working_model}")
            print("✅ You can use this model in your integration")
        else:
            print("\n❌ No working model found")
    else:
        print("\n❌ Login failed, cannot test models")
    
    return login_ok and working_model

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
