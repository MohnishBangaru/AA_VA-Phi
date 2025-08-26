#!/bin/bash
# Install FlashAttention2 on RunPod environment
# This script should be run on the RunPod instance with CUDA support

echo "🚀 Installing FlashAttention2 on RunPod..."

# Check if we're on RunPod with CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ NVCC not found. This script requires CUDA development tools."
    echo "   Make sure you're running on a RunPod instance with CUDA support."
    exit 1
fi

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
echo "📊 CUDA Version: $CUDA_VERSION"

# Check PyTorch version
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "📊 PyTorch Version: $PYTORCH_VERSION"

# Uninstall existing flash-attn if present
echo "🧹 Uninstalling existing flash-attn..."
pip uninstall -y flash-attn

# Install flash-attn with specific version for compatibility
echo "📦 Installing flash-attn==2.6.3..."
pip install flash-attn==2.6.3 --no-build-isolation

# Verify installation
echo "✅ Verifying FlashAttention2 installation..."
python -c "
import flash_attn
print(f'✅ FlashAttention2 version: {flash_attn.__version__}')
try:
    from flash_attn import flash_attn_func
    print('✅ FlashAttention2 functions imported successfully')
except ImportError as e:
    print(f'❌ FlashAttention2 import failed: {e}')
"

echo "🎉 FlashAttention2 installation complete!"
echo ""
echo "Next steps:"
echo "1. Restart your Python process"
echo "2. Run the Phi Ground test again"
echo "3. The model should now initialize with FlashAttention2"
