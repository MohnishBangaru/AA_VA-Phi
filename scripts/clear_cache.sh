#!/bin/bash
# Clear pip cache and stop FlashAttention2 installation

echo "🧹 Clearing pip cache to speed up installation..."

# Stop any running pip processes
echo "🛑 Stopping any running pip processes..."
pkill -f "pip install" 2>/dev/null || true

# Clear pip cache
echo "🗑️  Clearing pip cache..."
pip cache purge

# Clear build cache
echo "🗑️  Clearing build cache..."
rm -rf ~/.cache/pip/build/* 2>/dev/null || true
rm -rf /tmp/pip-* 2>/dev/null || true

# Clear PyTorch cache
echo "🗑️  Clearing PyTorch cache..."
rm -rf ~/.cache/torch/* 2>/dev/null || true

# Clear HuggingFace cache (optional - be careful!)
echo "🗑️  Clearing HuggingFace cache..."
rm -rf ~/.cache/huggingface/hub/* 2>/dev/null || true

echo "✅ Cache cleared!"
echo ""
echo "💡 Now run the quick fix instead:"
echo "   ./scripts/quick_flash_fix.sh"
echo ""
echo "🚀 This will skip FlashAttention2 and load the model immediately!"
