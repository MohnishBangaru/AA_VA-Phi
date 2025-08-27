#!/usr/bin/env python3
"""
Enable GPU Usage
==============

This script configures the system to use GPU instead of CPU for better performance.
"""

import os
import sys
from pathlib import Path

def check_gpu_availability():
    """Check if GPU is available."""
    print("🔍 Checking GPU availability...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            return True
        else:
            print("❌ CUDA not available")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def configure_gpu_environment():
    """Configure environment variables for GPU usage."""
    print("\n⚙️ Configuring GPU environment...")
    
    # Set environment variables for GPU usage
    gpu_env_vars = {
        'CUDA_VISIBLE_DEVICES': '0',  # Use first GPU
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
        'CUDA_LAUNCH_BLOCKING': '1',
        'TORCH_USE_CUDA_DSA': '1',  # Enable CUDA dynamic shared arrays
    }
    
    for var, value in gpu_env_vars.items():
        os.environ[var] = value
        print(f"   Set {var}={value}")
    
    print("✅ GPU environment variables configured")

def patch_phi_ground_for_gpu():
    """Patch Phi Ground to use GPU by default."""
    print("\n🔧 Patching Phi Ground for GPU usage...")
    
    phi_ground_path = Path("src/ai/phi_ground.py")
    
    if not phi_ground_path.exists():
        print("❌ Phi Ground file not found")
        return False
    
    # Read the current file
    with open(phi_ground_path, 'r') as f:
        content = f.read()
    
    # Find and replace device selection logic
    patterns = [
        # Replace CPU fallback with GPU preference
        (r"self\.device = 'cpu'", "self.device = 'cuda' if torch.cuda.is_available() else 'cpu'"),
        # Replace device selection in initialization
        (r"device = 'cpu'", "device = 'cuda' if torch.cuda.is_available() else 'cpu'"),
        # Add GPU memory optimization
        (r"# Move to correct device", "# Move to correct device\n                # GPU memory optimization\n                if torch.cuda.is_available():\n                    torch.cuda.empty_cache()"),
    ]
    
    modified = False
    for pattern, replacement in patterns:
        if pattern in content:
            content = content.replace(pattern, replacement)
            modified = True
            print(f"   Applied GPU optimization: {pattern}")
    
    if modified:
        # Write the modified content
        with open(phi_ground_path, 'w') as f:
            f.write(content)
        print("✅ Phi Ground patched for GPU usage")
        return True
    else:
        print("⚠️ No changes needed - GPU configuration already present")
        return True

def create_gpu_config():
    """Create a GPU configuration file."""
    print("\n📝 Creating GPU configuration...")
    
    gpu_config_content = '''#!/usr/bin/env python3
"""
GPU Configuration
================

This module provides GPU configuration for optimal performance.
"""

import os
import torch

def configure_gpu():
    """Configure GPU for optimal performance."""
    
    # Set GPU environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    if torch.cuda.is_available():
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        print("✅ GPU configured successfully")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        return True
    else:
        print("⚠️ GPU not available, using CPU")
        return False

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

# Auto-configure on import
configure_gpu()
'''
    
    config_path = Path("gpu_config.py")
    config_path.write_text(gpu_config_content)
    print(f"✅ Created GPU configuration: {config_path}")

def create_gpu_launcher():
    """Create a GPU launcher script."""
    print("\n🚀 Creating GPU launcher...")
    
    launcher_content = '''#!/usr/bin/env python3
"""
GPU Launcher
===========

This script launches your application with GPU optimization.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure GPU environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Import GPU configuration
try:
    import gpu_config
    print("✅ GPU configuration loaded")
except ImportError:
    print("⚠️ GPU configuration not found")

# Import and run your main script
if __name__ == "__main__":
    # Example: Run APK tester with GPU
    import subprocess
    
    cmd = [
        sys.executable, "scripts/distributed_apk_tester.py",
        "--apk", "com.Dominos_12.1.16-299_minAPI23(arm64-v8a,armeabi-v7a,x86,x86_64)(nodpi)_apkmirror.com.apk",
        "--local-server", "https://dominos-test.loca.lt",
        "--actions", "10"
    ]
    
    print("🚀 Launching with GPU optimization...")
    subprocess.run(cmd)
'''
    
    launcher_path = Path("run_with_gpu.py")
    launcher_path.write_text(launcher_content)
    print(f"✅ Created GPU launcher: {launcher_path}")

def main():
    """Main function to enable GPU usage."""
    print("🚀 Enabling GPU Usage")
    print("=" * 40)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    if not gpu_available:
        print("\n❌ GPU not available. Please install CUDA-enabled PyTorch:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return
    
    # Configure GPU environment
    configure_gpu_environment()
    
    # Patch Phi Ground for GPU
    patch_phi_ground_for_gpu()
    
    # Create GPU configuration
    create_gpu_config()
    
    # Create GPU launcher
    create_gpu_launcher()
    
    print("\n✅ GPU Configuration Complete!")
    print("\n💡 Usage:")
    print("1. Run with GPU optimization:")
    print("   python run_with_gpu.py")
    print("2. Or import GPU config in your scripts:")
    print("   import gpu_config")
    print("3. Or set environment variables manually:")
    print("   export CUDA_VISIBLE_DEVICES=0")
    print("   python scripts/distributed_apk_tester.py ...")

if __name__ == "__main__":
    main()
