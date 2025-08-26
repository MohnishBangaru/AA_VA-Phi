#!/bin/bash
# Install Tesseract OCR on RunPod environment

echo "🔍 Installing Tesseract OCR on RunPod..."

# Check if we're on a Linux system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "📊 Linux system detected"
    
    # Update package list
    echo "📦 Updating package list..."
    apt-get update
    
    # Install Tesseract and language packs
    echo "📦 Installing Tesseract OCR..."
    apt-get install -y tesseract-ocr tesseract-ocr-eng
    
    # Install additional language packs if needed
    echo "📦 Installing additional language packs..."
    apt-get install -y tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-spa
    
    # Verify installation
    echo "✅ Verifying Tesseract installation..."
    if command -v tesseract &> /dev/null; then
        VERSION=$(tesseract --version | head -n 1)
        echo "✅ Tesseract installed: $VERSION"
        
        # Test OCR
        echo "🧪 Testing OCR functionality..."
        python -c "
import pytesseract
from PIL import Image
import numpy as np

# Create a test image with text
test_image = Image.fromarray(np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8))
try:
    text = pytesseract.image_to_string(test_image)
    print('✅ OCR test successful')
except Exception as e:
    print(f'❌ OCR test failed: {e}')
"
    else
        echo "❌ Tesseract installation failed"
        exit 1
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📊 macOS system detected"
    
    # Install via Homebrew
    if command -v brew &> /dev/null; then
        echo "📦 Installing Tesseract via Homebrew..."
        brew install tesseract tesseract-lang
        
        # Verify installation
        if command -v tesseract &> /dev/null; then
            VERSION=$(tesseract --version | head -n 1)
            echo "✅ Tesseract installed: $VERSION"
        else
            echo "❌ Tesseract installation failed"
            exit 1
        fi
    else
        echo "❌ Homebrew not found. Please install Homebrew first."
        exit 1
    fi
    
else
    echo "❌ Unsupported operating system: $OSTYPE"
    exit 1
fi

echo "🎉 Tesseract OCR installation complete!"
echo ""
echo "Next steps:"
echo "1. Restart your Python process"
echo "2. Run your distributed testing again"
echo "3. OCR should now be available for text detection"
