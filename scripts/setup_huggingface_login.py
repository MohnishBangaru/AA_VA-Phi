#!/usr/bin/env python3
"""
Setup Hugging Face Hub Login for OmniParser 2.0
==============================================

This script helps set up Hugging Face Hub login for accessing OmniParser 2.0.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_hf_installed():
    """Check if huggingface_hub is installed."""
    try:
        import huggingface_hub
        print(f"✅ huggingface_hub {huggingface_hub.__version__} is installed")
        return True
    except ImportError:
        print("❌ huggingface_hub not installed")
        print("   Install with: pip install huggingface-hub")
        return False

def check_hf_login():
    """Check if user is logged in to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        user = api.whoami()
        print(f"✅ Logged in to Hugging Face Hub as: {user}")
        return True
        
    except Exception as e:
        print(f"❌ Not logged in to Hugging Face Hub: {e}")
        return False

def setup_hf_login():
    """Set up Hugging Face Hub login."""
    print("🔑 Setting up Hugging Face Hub Login")
    print("=" * 50)
    
    # Check if already logged in
    if check_hf_login():
        print("✅ Already logged in to Hugging Face Hub")
        return True
    
    # Get token from user
    print("\n📋 To get your Hugging Face token:")
    print("1. Visit https://huggingface.co/settings/tokens")
    print("2. Click 'New token'")
    print("3. Give it a name (e.g., 'AA_VA-Phi')")
    print("4. Select 'Read' role")
    print("5. Copy the token")
    
    token = input("\n🔑 Enter your Hugging Face token (or press Enter to skip): ").strip()
    
    if not token:
        print("⚠️  Skipping Hugging Face login setup")
        return False
    
    # Try to login with the token
    try:
        from huggingface_hub import login
        
        login(token=token)
        print("✅ Successfully logged in to Hugging Face Hub")
        
        # Verify login
        if check_hf_login():
            print("✅ Login verified successfully")
            return True
        else:
            print("❌ Login verification failed")
            return False
            
    except Exception as e:
        print(f"❌ Failed to login: {e}")
        return False

def test_omniparser_access():
    """Test access to OmniParser 2.0 model."""
    print("\n🧪 Testing OmniParser 2.0 Model Access")
    print("=" * 50)
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from src.vision.omniparser_integration import create_omniparser_integration
        
        # Create integration instance
        omniparser = create_omniparser_integration()
        
        if omniparser.is_available():
            print("✅ OmniParser 2.0 model access successful!")
            print("✅ Ready to use enhanced UI element detection")
            return True
        else:
            print("❌ OmniParser 2.0 model access failed")
            print("   Check your Hugging Face login and model permissions")
            return False
            
    except Exception as e:
        print(f"❌ OmniParser 2.0 test failed: {e}")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing Dependencies")
    print("=" * 30)
    
    dependencies = [
        "huggingface-hub>=0.19.0",
        "transformers>=4.36.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "accelerate>=0.24.0"
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {dep}")
            return False
    
    return True

def main():
    """Main function."""
    print("🚀 Hugging Face Hub Login Setup for OmniParser 2.0")
    print("=" * 60)
    
    # Check dependencies
    if not check_hf_installed():
        print("\n📦 Installing dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            return False
    
    # Setup login
    login_success = setup_hf_login()
    
    if login_success:
        # Test model access
        model_success = test_omniparser_access()
        
        print("\n📊 Setup Summary")
        print("=" * 20)
        print(f"Login: {'✅ PASS' if login_success else '❌ FAIL'}")
        print(f"Model Access: {'✅ PASS' if model_success else '❌ FAIL'}")
        
        if login_success and model_success:
            print("\n🎉 Hugging Face Hub setup is complete!")
            print("✅ You can now use OmniParser 2.0 for enhanced UI detection")
            print("✅ The model will be downloaded automatically on first use")
        else:
            print("\n⚠️  Setup incomplete")
            print("   Check your Hugging Face token and model permissions")
    else:
        print("\n⚠️  Hugging Face login setup failed")
        print("   You can still use the system with OCR fallback")
    
    return login_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
