#!/usr/bin/env python3
"""
Fix Vision Support for Phi Ground
=================================

This script diagnoses and fixes vision tokenization issues.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def diagnose_vision_support():
    """Diagnose vision support issues."""
    print("🔍 Diagnosing Vision Support for Phi Ground")
    print("=" * 50)
    
    # Check required packages
    print("📦 Checking required packages...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not found")
        return False
    
    try:
        from PIL import Image
        print(f"✅ Pillow: {Image.__version__}")
    except ImportError:
        print("❌ Pillow not found")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy not found")
        return False
    
    # Test vision tokenization
    print("\n🔍 Testing vision tokenization...")
    
    try:
        from transformers import AutoTokenizer
        
        # Test with Phi-3-vision model
        model_name = "microsoft/Phi-3-vision-128k-instruct"
        print(f"📊 Testing with: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Tokenizer loaded successfully")
        
        # Create test image
        import numpy as np
        from PIL import Image
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        print("✅ Test image created")
        
        # Test vision tokenization
        try:
            inputs = tokenizer("test", return_tensors="pt", images=test_image)
            print("✅ Vision tokenization successful!")
            print(f"📊 Input keys: {list(inputs.keys())}")
            return True
        except Exception as e:
            print(f"❌ Vision tokenization failed: {e}")
            
            # Try alternative approach
            print("🔄 Trying alternative vision approach...")
            try:
                # Try with image processing
                inputs = tokenizer("test", return_tensors="pt")
                print("✅ Text-only tokenization works")
                print("⚠️  Vision requires additional setup")
                return False
            except Exception as e2:
                print(f"❌ Text tokenization also failed: {e2}")
                return False
                
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        return False

def install_vision_dependencies():
    """Install vision processing dependencies."""
    print("\n📦 Installing vision dependencies...")
    
    import subprocess
    import sys
    
    packages = [
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "transformers>=4.36.0"
    ]
    
    for package in packages:
        print(f"📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def main():
    """Main function."""
    print("🚀 Phi Ground Vision Support Fix")
    print("=" * 40)
    
    # Diagnose current state
    vision_works = diagnose_vision_support()
    
    if vision_works:
        print("\n🎉 Vision support is working!")
        return True
    
    # Try to fix
    print("\n🔧 Attempting to fix vision support...")
    
    if install_vision_dependencies():
        print("\n🔄 Re-testing vision support...")
        if diagnose_vision_support():
            print("\n🎉 Vision support fixed!")
            return True
    
    print("\n⚠️  Vision support may require manual setup")
    print("💡 The model will work in text-only mode")
    return False

if __name__ == "__main__":
    main()
