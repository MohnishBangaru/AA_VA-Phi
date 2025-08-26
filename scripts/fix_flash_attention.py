#!/usr/bin/env python3
"""
Flash Attention 2 Compatibility Fix Script for RunPod

This script helps resolve Flash Attention 2 compatibility issues with PyTorch/CUDA versions.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(f"Output: {result.stdout}")
    return True

def check_environment():
    """Check the current environment."""
    print("=== Environment Check ===")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("PyTorch not installed")
        return False
    
    # Check if flash-attn is installed
    try:
        import flash_attn
        print(f"Flash Attention version: {flash_attn.__version__}")
    except ImportError:
        print("Flash Attention not installed")
    except Exception as e:
        print(f"Flash Attention import error: {e}")
    
    return True

def fix_flash_attention():
    """Fix Flash Attention 2 compatibility issues."""
    print("\n=== Fixing Flash Attention 2 ===")
    
    # Option 1: Reinstall flash-attn with the correct version
    print("Option 1: Reinstalling flash-attn...")
    if not run_command("pip uninstall flash-attn -y", check=False):
        print("Flash-attn not installed or already removed")
    
    # Try installing the latest compatible version
    if not run_command("pip install flash-attn --no-build-isolation"):
        print("Failed to install flash-attn with --no-build-isolation")
        
        # Try with specific version that's known to work with PyTorch 2.8
        if not run_command("pip install flash-attn==2.6.3 --no-build-isolation"):
            print("Failed to install flash-attn 2.6.3")
            
            # Try without CUDA extensions
            if not run_command("pip install flash-attn --no-build-isolation --no-cache-dir"):
                print("Failed to install flash-attn without cache")
                return False
    
    return True

def test_flash_attention():
    """Test if Flash Attention 2 is working."""
    print("\n=== Testing Flash Attention 2 ===")
    
    test_code = """
import torch
import flash_attn

# Test basic import
print("Flash Attention imported successfully")

# Test with a simple tensor
if torch.cuda.is_available():
    x = torch.randn(2, 4, 8, 16, device='cuda', dtype=torch.float16)
    print("Created test tensor on GPU")
    
    try:
        # Test flash attention function
        output = flash_attn.flash_attn_func(x, x, x)
        print("Flash Attention test successful!")
        print("SUCCESS")
    except Exception as e:
        print(f"Flash Attention test failed: {e}")
        print("FAILED")
else:
    print("CUDA not available for testing")
    print("FAILED")
"""
    
    # Write test code to a temporary file
    test_file = "test_flash_attn.py"
    with open(test_file, "w") as f:
        f.write(test_code)
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}")
        return "SUCCESS" in result.stdout
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def main():
    """Main function."""
    print("Flash Attention 2 Compatibility Fix Script")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("Environment check failed")
        return
    
    # Ask user what they want to do
    print("\nOptions:")
    print("1. Fix Flash Attention 2 (reinstall)")
    print("2. Test current Flash Attention 2 installation")
    print("3. Both")
    print("4. Skip Flash Attention 2 (use standard attention)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice in ["1", "3"]:
        if fix_flash_attention():
            print("Flash Attention 2 fix completed")
        else:
            print("Flash Attention 2 fix failed")
    
    if choice in ["2", "3"]:
        if test_flash_attention():
            print("Flash Attention 2 is working correctly")
        else:
            print("Flash Attention 2 test failed")
    
    if choice == "4":
        print("Skipping Flash Attention 2. The model will use standard attention.")
        print("This is slower but more compatible.")
    
    print("\n=== Recommendations ===")
    print("1. If Flash Attention 2 works, you'll get better performance")
    print("2. If it doesn't work, the model will fall back to standard attention")
    print("3. You can also try updating PyTorch to a stable release:")
    print("   pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    main()
