# Hugging Face Hub Setup for OmniParser 2.0

This guide explains how to set up Hugging Face Hub login to access the OmniParser 2.0 model for enhanced UI element detection.

## Overview

OmniParser 2.0 is a powerful vision-language model that provides advanced UI element detection capabilities. To access this model, you need to authenticate with Hugging Face Hub.

## Prerequisites

- Python 3.8 or higher
- Internet connection for model download
- Hugging Face account (free)

## Step 1: Get Your Hugging Face Token

1. **Visit Hugging Face**: Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

2. **Create New Token**:
   - Click "New token"
   - Give it a name (e.g., "AA_VA-Phi")
   - Select "Read" role (sufficient for model access)
   - Click "Generate token"

3. **Copy Token**: Copy the generated token (it starts with `hf_`)

## Step 2: Install Dependencies

```bash
# Install required packages
pip install huggingface-hub transformers torch torchvision accelerate

# Or install from requirements.txt
pip install -r requirements.txt
```

## Step 3: Setup Hugging Face Login

### Option A: Automated Setup (Recommended)

```bash
# Run the automated setup script
python scripts/setup_huggingface_login.py
```

This script will:
- Check if dependencies are installed
- Guide you through token input
- Verify login status
- Test model access

### Option B: Manual Setup

```bash
# Login using the CLI
huggingface-cli login

# Enter your token when prompted
```

### Option C: Environment Variable

```bash
# Set environment variable
export HUGGING_FACE_HUB_TOKEN="your_token_here"

# Or add to .env file
echo "HUGGING_FACE_HUB_TOKEN=your_token_here" >> .env
```

## Step 4: Verify Setup

```bash
# Test the complete setup
python scripts/setup_omniparser.py

# Test OmniParser integration
python scripts/test_omniparser_integration.py
```

## Usage

Once set up, OmniParser 2.0 will be automatically used by the VisionEngine:

```python
from src.vision.engine import VisionEngine

# Create vision engine (automatically uses OmniParser 2.0 if available)
vision_engine = VisionEngine()

# Analyze screenshot
elements = vision_engine.analyze("screenshot.png")
```

## Troubleshooting

### "Not logged in to Hugging Face Hub"

**Solution**: Run the setup script again:
```bash
python scripts/setup_huggingface_login.py
```

### "Model access failed"

**Possible causes**:
- Invalid token
- Network connectivity issues
- Model permissions

**Solutions**:
1. Verify your token at https://huggingface.co/settings/tokens
2. Check internet connection
3. Ensure you have read access to the model

### "CUDA out of memory"

**Solution**: The model will automatically fall back to CPU if GPU memory is insufficient.

### "Model download failed"

**Solutions**:
1. Check internet connection
2. Verify Hugging Face Hub access
3. Try downloading with a different network

## Model Details

- **Model**: `microsoft/OmniParser-v2.0`
- **Size**: ~2GB (downloaded once)
- **Device**: Auto-detects CUDA/CPU
- **Memory**: Optimized with mixed precision

## Benefits

- ✅ **Enhanced UI Detection**: Better text, button, and input field detection
- ✅ **Confidence Scoring**: Know how reliable each detection is
- ✅ **Context Awareness**: Understand UI relationships
- ✅ **No API Costs**: Completely free to use
- ✅ **Offline Capable**: Works after initial download

## Fallback Behavior

If OmniParser 2.0 is not available, the system automatically falls back to:
1. Tesseract OCR
2. Template matching
3. Clickable element detection

## Security Notes

- Your Hugging Face token is stored locally
- Only read permissions are required
- No sensitive data is sent to external services
- Model runs completely locally after download

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your Hugging Face account and token
3. Ensure all dependencies are installed
4. Check network connectivity

The system will continue to work with OCR fallback even if OmniParser 2.0 setup fails.
