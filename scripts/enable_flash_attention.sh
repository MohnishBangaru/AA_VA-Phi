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

# Install FlashAttention2 (pre-compiled wheel for speed)
echo "📦 Installing FlashAttention2..."
pip install flash-attn --find-links https://github.com/Dao-AILab/flash-attention/releases

# Verify installation
echo "✅ Verifying FlashAttention2..."
python -c "
import flash_attn
print(f'✅ FlashAttention2 {flash_attn.__version__} installed')
from flash_attn import flash_attn_func
print('✅ FlashAttention2 functions ready')
"

echo "🎉 FlashAttention2 enabled! Restart your Python process."
