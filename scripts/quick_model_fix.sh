#!/bin/bash
# Quick fix to make model work without FlashAttention2

echo "🔧 Quick Model Fix - No FlashAttention2 Required"

# Set environment variables to disable FlashAttention2
export USE_FLASH_ATTENTION_2=false
export TRANSFORMERS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0

echo "✅ Environment configured for standard attention"
echo "📊 Model will load with CPU or CUDA standard attention"
echo ""
echo "🚀 Ready to run your distributed test!"
echo ""
echo "💡 If you want FlashAttention2 later:"
echo "   ./scripts/enable_flash_attention.sh"
