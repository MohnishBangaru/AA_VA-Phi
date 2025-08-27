#!/usr/bin/env python3
"""
Memory Cleanup and Optimization Script
====================================

This script helps free up memory and optimize the system for AI model loading.
"""

import os
import gc
import psutil
import subprocess
import sys
from pathlib import Path

def get_memory_info():
    """Get current memory usage information."""
    memory = psutil.virtual_memory()
    return {
        'total': memory.total / 1024**3,
        'available': memory.available / 1024**3,
        'used': memory.used / 1024**3,
        'percent': memory.percent
    }

def print_memory_status():
    """Print current memory status."""
    mem_info = get_memory_info()
    print(f"📊 Memory Status:")
    print(f"   Total: {mem_info['total']:.1f}GB")
    print(f"   Used: {mem_info['used']:.1f}GB ({mem_info['percent']:.1f}%)")
    print(f"   Available: {mem_info['available']:.1f}GB")

def cleanup_python_memory():
    """Clean up Python memory."""
    print("🧹 Cleaning up Python memory...")
    
    # Force garbage collection
    collected = gc.collect()
    print(f"   Garbage collected: {collected} objects")
    
    # Clear Python cache
    cache_dirs = [
        Path.home() / ".cache" / "pip",
        Path.home() / ".cache" / "python",
        Path.cwd() / "__pycache__",
        Path.cwd() / ".pytest_cache"
    ]
    
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            try:
                import shutil
                shutil.rmtree(cache_dir)
                print(f"   Cleared cache: {cache_dir}")
            except Exception as e:
                print(f"   Could not clear {cache_dir}: {e}")

def cleanup_system_memory():
    """Clean up system memory."""
    print("🧹 Cleaning up system memory...")
    
    # Clear pip cache
    try:
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], 
                      capture_output=True, text=True)
        print("   Cleared pip cache")
    except Exception as e:
        print(f"   Could not clear pip cache: {e}")
    
    # Clear temporary files
    temp_dirs = ["/tmp", "/var/tmp"]
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                subprocess.run(["find", temp_dir, "-name", "*.tmp", "-delete"], 
                              capture_output=True)
                print(f"   Cleared temp files in {temp_dir}")
            except Exception as e:
                print(f"   Could not clear {temp_dir}: {e}")

def optimize_for_ai_models():
    """Optimize system for AI model loading."""
    print("🚀 Optimizing for AI models...")
    
    # Set environment variables for memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Disable gradient computation for inference
    os.environ['TORCH_DISABLE_GRAD'] = '1'
    
    print("   Set memory optimization environment variables")

def create_memory_efficient_config():
    """Create memory-efficient configuration."""
    print("⚙️ Creating memory-efficient configuration...")
    
    config_content = '''# Memory-efficient configuration for AI models
import os
import torch

# Memory optimization settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_DISABLE_GRAD'] = '1'

# Set memory fraction for GPU
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    torch.cuda.empty_cache()

# Disable gradient computation for inference
torch.set_grad_enabled(False)

print("✅ Memory-efficient configuration loaded")
'''
    
    config_path = Path("memory_config.py")
    config_path.write_text(config_content)
    print(f"   Created: {config_path}")

def install_lightweight_pytorch():
    """Install lightweight PyTorch version."""
    print("📦 Installing lightweight PyTorch...")
    
    # Try to install CPU-only version first (much smaller)
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], check=True)
        print("   ✅ Installed CPU-only PyTorch (lightweight)")
        return True
    except subprocess.CalledProcessError:
        print("   ❌ Failed to install CPU-only PyTorch")
        return False

def main():
    """Main memory cleanup function."""
    print("🧠 Memory Cleanup and Optimization")
    print("=" * 40)
    
    # Show initial memory status
    print_memory_status()
    
    # Clean up memory
    cleanup_python_memory()
    cleanup_system_memory()
    
    # Show memory after cleanup
    print("\nAfter cleanup:")
    print_memory_status()
    
    # Optimize for AI models
    optimize_for_ai_models()
    
    # Create memory-efficient config
    create_memory_efficient_config()
    
    # Try lightweight PyTorch installation
    print("\n📦 PyTorch Installation Options:")
    print("1. CPU-only PyTorch (recommended for memory-constrained systems)")
    print("2. CUDA PyTorch (requires more memory)")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        install_lightweight_pytorch()
    elif choice == "2":
        print("⚠️ CUDA PyTorch requires significant memory")
        print("   Consider using CPU-only version for now")
    else:
        print("❌ Invalid choice")
    
    print("\n💡 Memory Optimization Tips:")
    print("   - Use CPU-only PyTorch for memory-constrained systems")
    print("   - Load models with torch.load(..., map_location='cpu')")
    print("   - Use model.eval() to disable training mode")
    print("   - Clear cache with torch.cuda.empty_cache() if using GPU")
    print("   - Import memory_config.py at the start of your scripts")

if __name__ == "__main__":
    main()
