#!/usr/bin/env python3
"""
Comprehensive Flash Attention 2 Fix Script for RunPod

This script handles the undefined symbol error and ensures Flash Attention 2 works
with PyTorch 2.8.0.dev + CUDA 12.8.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

def run_command(cmd, check=True, capture_output=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False, result.stderr
    if capture_output:
        print(f"Output: {result.stdout}")
    return True, result.stdout

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
    
    # Check CUDA toolkit
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"CUDA Toolkit: {result.stdout.split('release ')[1].split(',')[0]}")
    except:
        print("CUDA Toolkit not found in PATH")
    
    return True

def remove_flash_attention():
    """Completely remove Flash Attention 2."""
    print("\n=== Removing Flash Attention 2 ===")
    
    # Uninstall via pip
    run_command("pip uninstall flash-attn -y", check=False)
    
    # Remove any remaining files
    flash_attn_paths = [
        "/usr/local/lib/python3.11/dist-packages/flash_attn",
        "/usr/local/lib/python3.11/dist-packages/flash_attn_2_cuda*",
        "/usr/local/lib/python3.11/dist-packages/flash_attn_cuda*"
    ]
    
    for path in flash_attn_paths:
        if os.path.exists(path):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                print(f"Removed: {path}")
            except Exception as e:
                print(f"Could not remove {path}: {e}")

def install_compatible_flash_attention():
    """Install a compatible version of Flash Attention 2."""
    print("\n=== Installing Compatible Flash Attention 2 ===")
    
    # Strategy 1: Try installing with specific version known to work with PyTorch 2.8
    print("Strategy 1: Installing flash-attn 2.6.3...")
    success, output = run_command("pip install flash-attn==2.6.3 --no-build-isolation --no-cache-dir")
    if success:
        return True
    
    # Strategy 2: Try installing with force reinstall
    print("Strategy 2: Force reinstalling flash-attn...")
    success, output = run_command("pip install flash-attn --force-reinstall --no-build-isolation --no-cache-dir")
    if success:
        return True
    
    # Strategy 3: Try installing from source
    print("Strategy 3: Installing from source...")
    success, output = run_command("pip install flash-attn --no-build-isolation --no-cache-dir --force-reinstall --no-deps")
    if success:
        return True
    
    # Strategy 4: Try with specific PyTorch version compatibility
    print("Strategy 4: Installing with PyTorch compatibility...")
    success, output = run_command("pip install flash-attn --no-build-isolation --no-cache-dir --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu121")
    if success:
        return True
    
    return False

def test_flash_attention():
    """Test if Flash Attention 2 is working."""
    print("\n=== Testing Flash Attention 2 ===")
    
    test_code = """
import torch
import sys

try:
    import flash_attn
    print("Flash Attention imported successfully")
    print(f"Flash Attention version: {flash_attn.__version__}")
except Exception as e:
    print(f"Flash Attention import failed: {e}")
    sys.exit(1)

# Test with a simple tensor
if torch.cuda.is_available():
    try:
        x = torch.randn(2, 4, 8, 16, device='cuda', dtype=torch.float16)
        print("Created test tensor on GPU")
        
        # Test flash attention function
        output = flash_attn.flash_attn_func(x, x, x)
        print("Flash Attention test successful!")
        print("SUCCESS")
    except Exception as e:
        print(f"Flash Attention test failed: {e}")
        print("FAILED")
        sys.exit(1)
else:
    print("CUDA not available for testing")
    print("FAILED")
    sys.exit(1)
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
        success = result.returncode == 0 and "SUCCESS" in result.stdout
        return success
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def update_pytorch():
    """Update PyTorch to a stable version if needed."""
    print("\n=== Checking PyTorch Version ===")
    
    try:
        import torch
        version = torch.__version__
        print(f"Current PyTorch version: {version}")
        
        # If using dev version, suggest updating to stable
        if "dev" in version:
            print("Detected PyTorch dev version. Consider updating to stable version:")
            print("pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121")
            
            choice = input("Do you want to update PyTorch to stable version? (y/n): ").strip().lower()
            if choice == 'y':
                print("Updating PyTorch...")
                success, output = run_command("pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall")
                if success:
                    print("PyTorch updated successfully. Please restart your Python environment.")
                    return True
    except ImportError:
        print("PyTorch not installed")
    
    return False

def main():
    """Main function."""
    print("Comprehensive Flash Attention 2 Fix Script")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        print("Environment check failed")
        return
    
    # Ask user what they want to do
    print("\nOptions:")
    print("1. Fix Flash Attention 2 (recommended)")
    print("2. Update PyTorch to stable version")
    print("3. Both")
    print("4. Test current installation only")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice in ["1", "3"]:
        # Remove existing flash-attn
        remove_flash_attention()
        
        # Install compatible version
        if install_compatible_flash_attention():
            print("Flash Attention 2 installation completed")
        else:
            print("Flash Attention 2 installation failed")
            return
    
    if choice in ["2", "3"]:
        update_pytorch()
    
    if choice in ["1", "3", "4"]:
        # Test the installation
        if test_flash_attention():
            print("✅ Flash Attention 2 is working correctly!")
        else:
            print("❌ Flash Attention 2 test failed")
            print("\nRecommendations:")
            print("1. The model will fall back to standard attention (slower but works)")
            print("2. Try updating PyTorch to a stable release")
            print("3. Check if your CUDA drivers are up to date")
    
    print("\n=== Summary ===")
    print("If Flash Attention 2 works, you'll get better performance")
    print("If it doesn't work, the model will automatically fall back to standard attention")
    print("The enhanced initialization code will handle this gracefully")

if __name__ == "__main__":
    main()
