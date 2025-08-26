#!/usr/bin/env python3
"""Fix FlashAttention2 compatibility issues.

This script addresses common FlashAttention2 compatibility problems:
1. Symbol linking errors with CUDA
2. PyTorch version mismatches
3. CUDA version incompatibilities
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command: str, description: str, silent: bool = False) -> bool:
    """Run a command and return success status."""
    if not silent:
        print(f"🔧 {description}...")
    try:
        # Use python -m pip to ensure we use the correct Python interpreter
        if command.startswith("pip install"):
            command = command.replace("pip install", "python -m pip install")
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            if not silent:
                print(f"✅ {description} - Success")
            return True
        else:
            if not silent:
                print(f"❌ {description} - Failed")
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        if not silent:
            print(f"❌ {description} - Exception: {e}")
        return False

def get_system_info():
    """Get system information for debugging."""
    print("🔍 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {sys.version}")
    
    # Check PyTorch and CUDA
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("   PyTorch: Not installed")

def check_flash_attention_status():
    """Check current FlashAttention2 status."""
    print("\n🔍 Checking FlashAttention2 Status...")
    
    try:
        import flash_attn
        print(f"✅ FlashAttention2 is installed (version: {flash_attn.__version__})")
        
        # Test basic import
        try:
            from flash_attn import flash_attn_func
            print("✅ FlashAttention2 functions can be imported")
            return True
        except Exception as e:
            print(f"❌ FlashAttention2 functions import failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ FlashAttention2 not installed: {e}")
        return False
    except Exception as e:
        print(f"❌ FlashAttention2 check failed: {e}")
        return False

def uninstall_flash_attention():
    """Uninstall FlashAttention2 completely."""
    print("\n🗑️ Uninstalling FlashAttention2...")
    
    uninstall_commands = [
        "python -m pip uninstall flash-attn -y",
        "python -m pip uninstall flash-attn --force -y",
        "conda remove flash-attn -y" if run_command("which conda", "Checking conda", silent=True) else None
    ]
    
    for cmd in uninstall_commands:
        if cmd:
            run_command(cmd, f"Uninstalling FlashAttention2: {cmd}")

def reinstall_pytorch_compatible():
    """Reinstall PyTorch with compatible version for FlashAttention2."""
    print("\n🔄 Reinstalling PyTorch with compatible version...")
    
    # Get current PyTorch version
    try:
        import torch
        current_version = torch.__version__
        print(f"   Current PyTorch version: {current_version}")
    except ImportError:
        current_version = "unknown"
    
    # Install compatible PyTorch version
    pytorch_commands = [
        "python -m pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121",
        "python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118",
        "python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu"
    ]
    
    for cmd in pytorch_commands:
        if run_command(cmd, f"Installing PyTorch: {cmd}"):
            print("✅ PyTorch installed successfully")
            return True
    
    return False

def install_flash_attention_compatible():
    """Install FlashAttention2 with compatible version."""
    print("\n🚀 Installing FlashAttention2 with compatible version...")
    
    # Try different FlashAttention2 versions
    flash_commands = [
        "python -m pip install flash-attn==2.3.0 --no-build-isolation",
        "python -m pip install flash-attn==2.2.0 --no-build-isolation",
        "python -m pip install flash-attn==2.1.0 --no-build-isolation",
        "python -m pip install flash-attn --no-build-isolation"
    ]
    
    for cmd in flash_commands:
        if run_command(cmd, f"Installing FlashAttention2: {cmd}"):
            print("✅ FlashAttention2 installed successfully")
            return True
    
    return False

def test_flash_attention():
    """Test FlashAttention2 functionality."""
    print("\n🧪 Testing FlashAttention2...")
    
    test_code = """
import torch
import flash_attn

# Test basic functionality
try:
    from flash_attn import flash_attn_func
    print("✅ FlashAttention2 import successful")
    
    # Test with dummy data
    if torch.cuda.is_available():
        device = torch.device("cuda")
        q = torch.randn(1, 8, 64, 64, device=device, dtype=torch.float16)
        k = torch.randn(1, 8, 64, 64, device=device, dtype=torch.float16)
        v = torch.randn(1, 8, 64, 64, device=device, dtype=torch.float16)
        
        output = flash_attn_func(q, k, v)
        print("✅ FlashAttention2 computation successful")
    else:
        print("⚠️ CUDA not available, skipping computation test")
        
except Exception as e:
    print(f"❌ FlashAttention2 test failed: {e}")
    return False

return True
"""
    
    # Write test code to temporary file
    test_file = "test_flash_attn.py"
    with open(test_file, "w") as f:
        f.write(test_code)
    
    # Run test
    success = run_command(f"python {test_file}", "Running FlashAttention2 test")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    
    return success

def provide_manual_solutions():
    """Provide manual solutions for FlashAttention2 issues."""
    print("\n💡 Manual Solutions for FlashAttention2 Issues:")
    print("=" * 60)
    print("1. Symbol Linking Errors:")
    print("   - This usually indicates PyTorch/CUDA version mismatch")
    print("   - Try: pip install torch==2.1.0+cu121 --force-reinstall")
    print()
    print("2. CUDA Version Issues:")
    print("   - Check CUDA version: nvidia-smi")
    print("   - Install matching PyTorch version")
    print()
    print("3. Alternative Installation:")
    print("   - Use conda: conda install -c conda-forge flash-attn")
    print("   - Try specific version: pip install flash-attn==2.2.0")
    print()
    print("4. Manual Compilation:")
    print("   - Clone: git clone https://github.com/Dao-AILab/flash-attention.git")
    print("   - Follow manual build instructions")
    print()
    print("5. Skip FlashAttention2:")
    print("   - The system works perfectly without FlashAttention2")
    print("   - Performance will be slower but functionality is preserved")

def main():
    """Main function to fix FlashAttention2 compatibility."""
    print("🔧 FlashAttention2 Compatibility Fix")
    print("=" * 50)
    
    # Get system information
    get_system_info()
    
    # Check current status
    current_status = check_flash_attention_status()
    
    if current_status:
        print("\n✅ FlashAttention2 appears to be working correctly")
        if test_flash_attention():
            print("🎉 FlashAttention2 is fully functional!")
            return
        else:
            print("⚠️ FlashAttention2 has issues, attempting to fix...")
    
    # Fix process
    print("\n🔧 Starting FlashAttention2 compatibility fix...")
    
    # Step 1: Uninstall current FlashAttention2
    uninstall_flash_attention()
    
    # Step 2: Reinstall PyTorch with compatible version
    if not reinstall_pytorch_compatible():
        print("❌ Failed to reinstall PyTorch")
        provide_manual_solutions()
        return
    
    # Step 3: Install FlashAttention2
    if not install_flash_attention_compatible():
        print("❌ Failed to install FlashAttention2")
        provide_manual_solutions()
        return
    
    # Step 4: Test FlashAttention2
    if test_flash_attention():
        print("\n🎉 FlashAttention2 compatibility fix successful!")
        print("You can now use Phi Ground with GPU acceleration.")
    else:
        print("\n⚠️ FlashAttention2 still has issues after fix attempt.")
        provide_manual_solutions()
        print("\n💡 The system will work without FlashAttention2 using standard attention.")

if __name__ == "__main__":
    main()
