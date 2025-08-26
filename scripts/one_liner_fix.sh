#!/bin/bash

# One-liner Flash Attention 2 Fix for RunPod
# This script completely removes and reinstalls Flash Attention 2

echo "=== One-liner Flash Attention 2 Fix ==="

# Step 1: Remove existing flash-attn completely
echo "Step 1: Removing existing Flash Attention 2..."
pip uninstall flash-attn -y
rm -rf /usr/local/lib/python3.11/dist-packages/flash_attn*
rm -rf /usr/local/lib/python3.11/dist-packages/flash_attn_2_cuda*

# Step 2: Install compatible version
echo "Step 2: Installing compatible Flash Attention 2..."
pip install flash-attn==2.6.3 --no-build-isolation --no-cache-dir --force-reinstall

# Step 3: Test installation
echo "Step 3: Testing installation..."
python -c "
import torch
import flash_attn
x = torch.randn(2, 4, 8, 16, device='cuda', dtype=torch.float16)
output = flash_attn.flash_attn_func(x, x, x)
print('✅ Flash Attention 2 is working!')
" 2>/dev/null && echo "SUCCESS" || echo "FAILED - will use standard attention fallback"
