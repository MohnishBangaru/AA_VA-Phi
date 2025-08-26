#!/bin/bash
# Quick FlashAttention2 installation for RunPod

echo "🚀 Installing FlashAttention2 on RunPod..."

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ NVCC not found. This requires CUDA support."
    exit 1
fi

# Check PyTorch version
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "📊 PyTorch: $PYTORCH_VERSION"

# Clear cache first
echo "🧹 Clearing cache..."
pip cache purge
pip uninstall -y flash-attn

# Check PyTorch and CUDA compatibility
echo "🔍 Checking compatibility..."
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
echo "📊 CUDA: $CUDA_VERSION"

# Install compatible FlashAttention2 version
if [[ "$PYTORCH_VERSION" == 2.1* ]]; then
    echo "📦 Installing flash-attn==2.3.6 for PyTorch 2.1.x..."
    pip install flash-attn==2.3.6 --no-build-isolation
elif [[ "$PYTORCH_VERSION" == 2.0* ]]; then
    echo "📦 Installing flash-attn==2.2.4 for PyTorch 2.0.x..."
    pip install flash-attn==2.2.4 --no-build-isolation
else
    echo "📦 Installing compatible flash-attn version..."
    pip install flash-attn==2.5.8 --no-build-isolation
fi

# Verify installation
echo "✅ Verifying FlashAttention2..."
python -c "
import flash_attn
print(f'✅ FlashAttention2 {flash_attn.__version__} installed')
from flash_attn import flash_attn_func
print('✅ FlashAttention2 functions ready')
"

echo "🎉 FlashAttention2 enabled! Restart your Python process."
