#!/bin/bash
# Install stable FlashAttention2 version

echo "📦 Installing stable FlashAttention2..."

# Check PyTorch version
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "📊 PyTorch Version: $PYTORCH_VERSION"

# Check CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "📊 CUDA Version: $CUDA_VERSION"
else
    echo "⚠️  CUDA not found - FlashAttention2 requires CUDA"
    exit 1
fi

# Uninstall existing flash-attn
echo "🧹 Uninstalling existing flash-attn..."
pip uninstall -y flash-attn

# Install based on PyTorch version
if [[ "$PYTORCH_VERSION" == 2.1* ]]; then
    echo "📦 Installing flash-attn==2.3.6 for PyTorch 2.1.x..."
    pip install flash-attn==2.3.6 --no-build-isolation
elif [[ "$PYTORCH_VERSION" == 2.0* ]]; then
    echo "📦 Installing flash-attn==2.2.4 for PyTorch 2.0.x..."
    pip install flash-attn==2.2.4 --no-build-isolation
elif [[ "$PYTORCH_VERSION" == 2.2* ]]; then
    echo "📦 Installing flash-attn==2.4.2 for PyTorch 2.2.x..."
    pip install flash-attn==2.4.2 --no-build-isolation
else
    echo "📦 Installing latest stable flash-attn==2.5.8..."
    pip install flash-attn==2.5.8 --no-build-isolation
fi

# Verify installation
echo "✅ Verifying installation..."
python -c "
import flash_attn
print(f'✅ FlashAttention2 version: {flash_attn.__version__}')
try:
    from flash_attn import flash_attn_func
    print('✅ FlashAttention2 functions imported successfully')
except ImportError as e:
    print(f'❌ FlashAttention2 import failed: {e}')
"

echo "🎉 Stable FlashAttention2 installation complete!"
