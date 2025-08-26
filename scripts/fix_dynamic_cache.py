#!/usr/bin/env python3
"""
DynamicCache Fix Script
=======================

This script specifically addresses the 'DynamicCache' object has no attribute 'seen_tokens' error.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dynamic_cache_fix():
    """Test different approaches to fix DynamicCache issues."""
    print("🔍 Testing DynamicCache fixes...")
    
    model_name = "microsoft/Phi-3-mini-128k-instruct"
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        test_text = "Hello, how are you?"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            model = model.to("cuda")
        
        print("✅ Model loaded successfully")
        
        # Test different approaches
        approaches = [
            {
                "name": "Minimal Generation (Recommended)",
                "params": {
                    "input_ids": inputs.get('input_ids'),
                    "max_new_tokens": 50,
                    "do_sample": False,
                    "use_cache": False
                }
            },
            {
                "name": "Basic Generation",
                "params": {
                    "input_ids": inputs.get('input_ids'),
                    "attention_mask": inputs.get('attention_mask'),
                    "max_new_tokens": 50,
                    "temperature": 0.7,
                    "do_sample": True,
                    "use_cache": False
                }
            },
            {
                "name": "Direct Forward Pass",
                "method": "forward"
            },
            {
                "name": "Manual Token Generation",
                "method": "manual"
            }
        ]
        
        for approach in approaches:
            try:
                print(f"\nTesting: {approach['name']}")
                
                if approach.get("method") == "forward":
                    # Direct forward pass
                    with torch.no_grad():
                        outputs = model.forward(**inputs)
                        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                        response = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                        print(f"✅ {approach['name']} - Success")
                        print(f"   Next token: {response}")
                        
                elif approach.get("method") == "manual":
                    # Manual token generation
                    current_input = inputs.get('input_ids')
                    generated_tokens = []
                    
                    with torch.no_grad():
                        for _ in range(10):  # Generate 10 tokens
                            outputs = model.forward(input_ids=current_input)
                            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                            generated_tokens.append(next_token.item())
                            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
                            
                            if next_token.item() == tokenizer.eos_token_id:
                                break
                    
                    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    print(f"✅ {approach['name']} - Success")
                    print(f"   Generated: {response}")
                    
                else:
                    # Standard generation
                    with torch.no_grad():
                        outputs = model.generate(**approach['params'])
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        print(f"✅ {approach['name']} - Success")
                        print(f"   Response: {response[:100]}...")
                
            except Exception as e:
                print(f"❌ {approach['name']} - Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def check_model_compatibility():
    """Check model compatibility and suggest fixes."""
    print("\n🔍 Checking model compatibility...")
    
    try:
        import transformers
        version = transformers.__version__
        print(f"Transformers version: {version}")
        
        # Check for known DynamicCache issues
        if version.startswith("4.3"):
            print("⚠️  Transformers 4.3.x has known DynamicCache issues")
            print("   Recommendation: Update to 4.4+ or use minimal generation")
        elif version.startswith("4.4"):
            print("✅ Transformers 4.4.x should handle DynamicCache better")
        elif version.startswith("4.5"):
            print("✅ Transformers 4.5.x should handle DynamicCache well")
        else:
            print(f"ℹ️  Transformers {version} - DynamicCache compatibility unknown")
            
        return version
        
    except ImportError:
        print("❌ Transformers not installed")
        return None

def suggest_dynamic_cache_fixes():
    """Suggest specific fixes for DynamicCache issues."""
    print("\n🔧 DynamicCache Fixes:")
    print("=" * 40)
    
    print("""
1. **Immediate Fixes:**
   - Use use_cache=False in all generation calls
   - Start with minimal generation parameters
   - Avoid complex caching strategies

2. **Code Changes:**
   - Implement multiple fallback strategies
   - Use direct forward pass as ultimate fallback
   - Handle different model types gracefully

3. **Environment Fixes:**
   - Update transformers: pip install --upgrade transformers
   - Check PyTorch compatibility
   - Ensure sufficient memory

4. **Model-Specific:**
   - For Phi models: Use trust_remote_code=True
   - For newer models: May need different approaches
   - Consider model-specific generation methods

5. **Recommended Approach:**
   - Start with minimal generation (most reliable)
   - Fall back to manual token generation if needed
   - Always disable cache to avoid DynamicCache issues
""")

def main():
    """Main function."""
    print("🔧 DynamicCache Fix Script")
    print("=" * 40)
    
    # Check compatibility
    version = check_model_compatibility()
    
    # Test fixes
    print("\n🧪 Testing DynamicCache fixes...")
    success = test_dynamic_cache_fix()
    
    if success:
        print("\n✅ DynamicCache tests completed")
    else:
        print("\n❌ DynamicCache tests failed")
    
    # Provide suggestions
    suggest_dynamic_cache_fixes()
    
    print("\n📝 Next Steps:")
    print("1. Use minimal generation strategy (most reliable)")
    print("2. Always set use_cache=False")
    print("3. Implement fallback to direct forward pass")
    print("4. Update transformers if using older version")

if __name__ == "__main__":
    main()
