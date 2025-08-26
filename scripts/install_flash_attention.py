#!/usr/bin/env python3
"""Install FlashAttention2 with proper error handling and fallback methods.

This script attempts to install FlashAttention2 using multiple approaches:
1. Try pre-compiled wheel first
2. Try building from source with specific flags
3. Try alternative installation methods
4. Provide clear error messages and solutions
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
    
    # Check CUDA
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
    except ImportError:
        print("   PyTorch: Not installed")

def try_precompiled_wheel():
    """Try to install pre-compiled wheel."""
    print("\n🚀 Attempting to install pre-compiled FlashAttention2 wheel...")
    
    # Try different wheel sources
    wheel_sources = [
        "flash-attn --no-build-isolation",
        "flash-attn --find-links https://github.com/Dao-AILab/flash-attention/releases",
        "pip install flash-attn --index-url https://download.pytorch.org/whl/cu121"
    ]
    
    for source in wheel_sources:
        if run_command(f"pip install {source}", f"Installing from {source}"):
            return True
    
    return False

def try_build_from_source():
    """Try to build FlashAttention2 from source with specific flags."""
    print("\n🔨 Attempting to build FlashAttention2 from source...")
    
    # Set environment variables for compilation
    env_vars = {
        "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6;8.9;9.0",  # Support multiple GPU architectures
        "MAX_JOBS": "4",  # Limit parallel jobs
        "CMAKE_BUILD_TYPE": "Release"
    }
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Try building with specific flags
    build_commands = [
        "pip install flash-attn --no-cache-dir --verbose",
        "pip install flash-attn --no-cache-dir --build-option='--verbose'",
        "pip install flash-attn --no-cache-dir --global-option='--verbose'"
    ]
    
    for cmd in build_commands:
        if run_command(cmd, f"Building with: {cmd}"):
            return True
    
    return False

def try_alternative_installation():
    """Try alternative installation methods."""
    print("\n🔄 Trying alternative installation methods...")
    
    # Try installing from git
    git_commands = [
        "pip install git+https://github.com/Dao-AILab/flash-attention.git",
        "pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.3.0",
        "pip install git+https://github.com/Dao-AILab/flash-attention.git@main"
    ]
    
    for cmd in git_commands:
        if run_command(cmd, f"Installing from git: {cmd}"):
            return True
    
    return False

def install_prerequisites():
    """Install prerequisites for FlashAttention2."""
    print("\n📦 Installing prerequisites...")
    
    prerequisites = [
        "ninja",
        "packaging",
        "wheel",
        "setuptools",
        "build"
    ]
    
    for pkg in prerequisites:
        run_command(f"pip install {pkg}", f"Installing {pkg}")

def verify_installation():
    """Verify FlashAttention2 installation."""
    print("\n✅ Verifying FlashAttention2 installation...")
    
    try:
        import flash_attn
        print(f"✅ FlashAttention2 imported successfully")
        print(f"   Version: {flash_attn.__version__}")
        
        # Test basic functionality
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA is available for FlashAttention2")
        else:
            print("⚠️ CUDA not available - FlashAttention2 will use CPU")
        
        return True
    except ImportError as e:
        print(f"❌ FlashAttention2 import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ FlashAttention2 verification failed: {e}")
        return False

def provide_solutions():
    """Provide solutions for common FlashAttention2 installation issues."""
    print("\n💡 Common Solutions for FlashAttention2 Installation Issues:")
    print("=" * 60)
    print("1. CUDA Version Mismatch:")
    print("   - Ensure PyTorch and CUDA versions are compatible")
    print("   - Try: pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html")
    print()
    print("2. Compiler Issues:")
    print("   - Install build tools: sudo apt-get install build-essential")
    print("   - Install ninja: pip install ninja")
    print()
    print("3. Memory Issues:")
    print("   - Reduce MAX_JOBS: export MAX_JOBS=1")
    print("   - Use swap space if needed")
    print()
    print("4. Alternative Installation:")
    print("   - Try conda: conda install -c conda-forge flash-attn")
    print("   - Try specific version: pip install flash-attn==2.3.0")
    print()
    print("5. Manual Compilation:")
    print("   - Clone repository: git clone https://github.com/Dao-AILab/flash-attention.git")
    print("   - Follow manual build instructions in the repository")

def main():
    """Main installation function."""
    print("🚀 FlashAttention2 Installation Script")
    print("=" * 50)
    
    # Get system information
    get_system_info()
    
    # Install prerequisites
    install_prerequisites()
    
    # Try different installation methods
    success = False
    
    # Method 1: Pre-compiled wheel
    if not success:
        success = try_precompiled_wheel()
    
    # Method 2: Build from source
    if not success:
        success = try_build_from_source()
    
    # Method 3: Alternative installation
    if not success:
        success = try_alternative_installation()
    
    # Verify installation
    if success:
        if verify_installation():
            print("\n🎉 FlashAttention2 installed successfully!")
            print("You can now use Phi Ground with GPU acceleration.")
        else:
            print("\n⚠️ Installation completed but verification failed.")
            print("FlashAttention2 may not work correctly.")
    else:
        print("\n❌ All installation methods failed.")
        provide_solutions()
        print("\n💡 The system will work without FlashAttention2 using standard attention.")
        print("Performance may be slower but functionality will be preserved.")

if __name__ == "__main__":
    main()
