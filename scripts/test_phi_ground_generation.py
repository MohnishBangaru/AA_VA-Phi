#!/usr/bin/env python3
"""
Phi Ground Generation Test Script
================================

This script tests different generation strategies to find what works with the current model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_generation_strategies():
    """Test different generation strategies."""
    print("🔍 Testing Phi Ground generation strategies...")
    
    try:
        # Load model and tokenizer
        model_name = "microsoft/Phi-3-mini-128k-instruct"
        print(f"Testing with model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Test prompt
        prompt = "Analyze this Android screenshot and suggest the next action:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Strategy 1: Basic generation
        print("\nStrategy 1: Basic generation")
        try:
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ Success: {response[:100]}...")
            return "basic"
        except Exception as e:
            print(f"❌ Failed: {e}")
        
        # Strategy 2: No cache
        print("\nStrategy 2: No cache")
        try:
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ Success: {response[:100]}...")
            return "no_cache"
        except Exception as e:
            print(f"❌ Failed: {e}")
        
        # Strategy 3: Greedy decoding
        print("\nStrategy 3: Greedy decoding")
        try:
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ Success: {response[:100]}...")
            return "greedy"
        except Exception as e:
            print(f"❌ Failed: {e}")
        
        # Strategy 4: Minimal parameters
        print("\nStrategy 4: Minimal parameters")
        try:
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=20
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ Success: {response[:100]}...")
            return "minimal"
        except Exception as e:
            print(f"❌ Failed: {e}")
        
        print("\n❌ All strategies failed")
        return None
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_alternative_models():
    """Test alternative models that might work better."""
    print("\n🔍 Testing alternative models...")
    
    alternative_models = [
        "microsoft/Phi-2",
        "microsoft/Phi-1_5",
        "microsoft/Phi-3-mini-4k-instruct"
    ]
    
    for model_name in alternative_models:
        print(f"\nTesting: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Test basic generation
            prompt = "Hello, world!"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=20,
                use_cache=False
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ {model_name} works: {response}")
            return model_name
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
    
    return None

def main():
    """Main function."""
    print("🔧 Phi Ground Generation Test")
    print("=" * 40)
    
    # Test current model
    working_strategy = test_generation_strategies()
    
    if working_strategy:
        print(f"\n✅ Working strategy: {working_strategy}")
    else:
        print("\n❌ No working strategy found for current model")
        
        # Try alternative models
        working_model = test_alternative_models()
        if working_model:
            print(f"\n✅ Recommended alternative model: {working_model}")
        else:
            print("\n❌ No working models found")
    
    print("\n📝 Recommendations:")
    print("1. Use the working strategy in the code")
    print("2. Consider switching to a more compatible model")
    print("3. Implement multiple fallback strategies")

if __name__ == "__main__":
    main()
