#!/usr/bin/env python3
"""
Model Caching Fix Script
========================

This script helps fix caching issues with different model types and transformers versions.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_generation(model_name: str = "microsoft/Phi-3-mini-128k-instruct"):
    """Test model generation with different caching strategies."""
    print(f"🔍 Testing model generation: {model_name}")
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Test text
        test_text = "Hello, how are you?"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            model = model.to("cuda")
        
        print("✅ Model loaded successfully")
        
        # Test different generation strategies
        strategies = [
            {
                "name": "Full generation with cache",
                "params": {
                    "max_new_tokens": 50,
                    "temperature": 0.7,
                    "do_sample": True,
                    "use_cache": True
                }
            },
            {
                "name": "Generation without cache",
                "params": {
                    "max_new_tokens": 50,
                    "temperature": 0.7,
                    "do_sample": True,
                    "use_cache": False
                }
            },
            {
                "name": "Minimal generation",
                "params": {
                    "max_new_tokens": 50,
                    "do_sample": False,
                    "use_cache": False
                }
            },
            {
                "name": "Basic generation",
                "params": {
                    "max_new_tokens": 50
                }
            }
        ]
        
        for strategy in strategies:
            try:
                print(f"\nTesting: {strategy['name']}")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        **strategy['params']
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"✅ {strategy['name']} - Success")
                print(f"   Response: {response[:100]}...")
                
            except Exception as e:
                print(f"❌ {strategy['name']} - Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def check_transformers_version():
    """Check transformers version and compatibility."""
    try:
        import transformers
        version = transformers.__version__
        print(f"Transformers version: {version}")
        
        # Check for known issues
        if version.startswith("4.3"):
            print("⚠️  Transformers 4.3.x may have caching issues")
        elif version.startswith("4.4"):
            print("✅ Transformers 4.4.x should work well")
        elif version.startswith("4.5"):
            print("✅ Transformers 4.5.x should work well")
        else:
            print(f"ℹ️  Transformers {version} - compatibility unknown")
            
        return version
        
    except ImportError:
        print("❌ Transformers not installed")
        return None

def suggest_fixes():
    """Suggest fixes for caching issues."""
    print("\n🔧 Suggested Fixes for Caching Issues:")
    print("=" * 50)
    
    print("""
1. **Update Transformers:**
   pip install --upgrade transformers

2. **Use Minimal Generation Parameters:**
   - Disable cache: use_cache=False
   - Disable sampling: do_sample=False
   - Use only essential parameters

3. **Model-Specific Fixes:**
   - For Phi models: Use trust_remote_code=True
   - For newer models: May need different caching strategies

4. **Environment Issues:**
   - Check PyTorch version compatibility
   - Ensure sufficient GPU memory
   - Try CPU fallback if GPU issues persist

5. **Code Fixes:**
   - Implement multiple fallback strategies
   - Handle different model types gracefully
   - Add proper error handling for caching
""")

def main():
    """Main function."""
    print("🔧 Model Caching Fix Script")
    print("=" * 40)
    
    # Check transformers version
    version = check_transformers_version()
    
    # Test model generation
    print("\n🧪 Testing model generation...")
    success = test_model_generation()
    
    if success:
        print("\n✅ Model generation tests completed")
    else:
        print("\n❌ Model generation tests failed")
    
    # Provide suggestions
    suggest_fixes()
    
    print("\n📝 Next Steps:")
    print("1. Update transformers if needed")
    print("2. Use the fallback generation strategies in the code")
    print("3. Test with different model parameters")

if __name__ == "__main__":
    main()
