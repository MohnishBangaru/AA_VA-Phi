#!/bin/bash

# Quick Flash Attention 2 Fix Script for RunPod
# Run this script on your RunPod instance to fix compatibility issues

echo "=== Flash Attention 2 Quick Fix ==="
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Check if flash-attn is causing issues
if python -c "import flash_attn" 2>/dev/null; then
    echo "Flash Attention 2 is installed, testing compatibility..."
    
    # Test if it works
    if python -c "
import torch
import flash_attn
x = torch.randn(2, 4, 8, 16, device='cuda', dtype=torch.float16)
output = flash_attn.flash_attn_func(x, x, x)
print('Flash Attention 2 is working!')
" 2>/dev/null | grep -q "Flash Attention 2 is working!"; then
        echo "✅ Flash Attention 2 is working correctly"
        exit 0
    else
        echo "❌ Flash Attention 2 has compatibility issues"
    fi
else
    echo "Flash Attention 2 is not installed"
fi

echo ""
echo "Attempting to fix Flash Attention 2..."

# Uninstall current flash-attn
echo "Uninstalling current flash-attn..."
pip uninstall flash-attn -y

# Try installing a compatible version
echo "Installing compatible flash-attn version..."
pip install flash-attn==2.6.3 --no-build-isolation

# Test again
echo "Testing installation..."
if python -c "
import torch
import flash_attn
x = torch.randn(2, 4, 8, 16, device='cuda', dtype=torch.float16)
output = flash_attn.flash_attn_func(x, x, x)
print('Flash Attention 2 is working!')
" 2>/dev/null | grep -q "Flash Attention 2 is working!"; then
    echo "✅ Flash Attention 2 fixed successfully!"
else
    echo "❌ Flash Attention 2 still has issues"
    echo ""
    echo "Recommendations:"
    echo "1. The model will fall back to standard attention (slower but works)"
    echo "2. Try updating PyTorch to a stable release:"
    echo "   pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121"
    echo "3. Or use the Python fix script: python scripts/fix_flash_attention.py"
fi
