#!/bin/bash
# Quick fix to skip FlashAttention2 and use faster model loading

echo "⚡ Quick Fix: Skipping FlashAttention2 for faster loading..."

# Check if we're on RunPod
if command -v nvcc &> /dev/null; then
    echo "📊 RunPod detected with CUDA support"
    echo "🚀 Using optimized model loading without FlashAttention2"
    
    # Set environment variable to skip FlashAttention2
    export USE_FLASH_ATTENTION_2=false
    export TRANSFORMERS_OFFLINE=0
    
    echo "✅ Environment configured for fast loading"
    echo ""
    echo "Next steps:"
    echo "1. Restart your Python process"
    echo "2. Run your distributed testing"
    echo "3. Model will load with standard attention (faster)"
    
else
    echo "📊 Local environment detected"
    echo "✅ Already configured for fast loading without FlashAttention2"
fi

echo ""
echo "💡 Tip: If you need FlashAttention2 later, run:"
echo "   ./scripts/install_flash_attention_runpod.sh"
