#!/usr/bin/env python3
"""
Phi Ground Tokenizer Fix Script
===============================

This script helps fix tokenizer compatibility issues with Phi Ground models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_phi_ground_compatibility():
    """Check Phi Ground model and tokenizer compatibility."""
    print("🔍 Checking Phi Ground compatibility...")
    
    try:
        # Test with a known Phi Ground model
        model_name = "microsoft/Phi-3-vision-128k-instruct"
        
        print(f"Testing with model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Tokenizer loaded successfully")
        
        # Test tokenizer with images
        from PIL import Image
        import numpy as np
        
        # Create a dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Test text-only tokenization
        text = "Hello, world!"
        try:
            inputs = tokenizer(text, return_tensors="pt")
            print("✅ Text-only tokenization works")
        except Exception as e:
            print(f"❌ Text-only tokenization failed: {e}")
            return False
        
        # Test vision tokenization
        try:
            inputs = tokenizer(text, return_tensors="pt", images=dummy_image)
            print("✅ Vision tokenization works")
            return True
        except TypeError as e:
            if "unexpected keyword argument 'images'" in str(e):
                print("⚠️  Vision tokenization not supported - will use text-only mode")
                return False
            else:
                print(f"❌ Vision tokenization failed: {e}")
                return False
        except Exception as e:
            print(f"❌ Vision tokenization failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_alternative_models():
    """Test alternative models that might work better."""
    print("\n🔍 Testing alternative models...")
    
    alternative_models = [
        "microsoft/Phi-3-vision-128k-instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-mini-128k-instruct",
        "microsoft/Phi-2",
        "microsoft/Phi-1_5"
    ]
    
    for model_name in alternative_models:
        print(f"\nTesting: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print(f"✅ {model_name} tokenizer loaded")
            
            # Test basic tokenization
            inputs = tokenizer("Test", return_tensors="pt")
            print(f"✅ {model_name} basic tokenization works")
            
            # Test vision if supported
            try:
                from PIL import Image
                import numpy as np
                dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                inputs = tokenizer("Test", return_tensors="pt", images=dummy_image)
                print(f"✅ {model_name} vision tokenization works")
                return model_name
            except:
                print(f"⚠️  {model_name} vision tokenization not supported")
                
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
    
    return None

def fix_tokenizer_usage():
    """Provide guidance on fixing tokenizer usage."""
    print("\n🔧 Tokenizer Usage Fixes:")
    print("=" * 40)
    
    print("""
1. **For Vision-Language Models (Phi-3-vision):**
   - Use: tokenizer(text, images=image, return_tensors="pt")
   - Requires: transformers >= 4.36.0

2. **For Text-Only Models (Phi-2, Phi-1.5):**
   - Use: tokenizer(text, return_tensors="pt")
   - No image support

3. **Fallback Strategy:**
   - Try vision tokenization first
   - Fall back to text-only if not supported
   - Use UI element detection for visual context

4. **Recommended Approach:**
   - Use Phi-3-vision for best results
   - Fall back to Phi-2 for text-only mode
   - Combine with computer vision for UI element detection
""")

def main():
    """Main function."""
    print("🔧 Phi Ground Tokenizer Fix Script")
    print("=" * 40)
    
    # Check current compatibility
    vision_supported = check_phi_ground_compatibility()
    
    if not vision_supported:
        print("\n🔍 Looking for alternative models...")
        best_model = test_alternative_models()
        
        if best_model:
            print(f"\n✅ Recommended model: {best_model}")
        else:
            print("\n❌ No compatible models found")
    
    # Provide fixes
    fix_tokenizer_usage()
    
    print("\n📝 Next Steps:")
    print("1. Update the Phi Ground initialization to use the recommended model")
    print("2. Implement fallback tokenization as shown in the code")
    print("3. Test with both vision and text-only modes")

if __name__ == "__main__":
    main()
