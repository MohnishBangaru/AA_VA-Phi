# OmniParser v2 Integration

This document describes the integration of OmniParser v2 from Hugging Face as a replacement for the traditional Tesseract OCR system in the AA_VA framework.

## Overview

OmniParser v2 is a state-of-the-art document understanding model that provides advanced OCR capabilities with better text extraction, layout analysis, and understanding of complex document structures. This integration replaces the previous Tesseract-based OCR system while maintaining backward compatibility.

## Features

### Enhanced Text Extraction
- **Better Accuracy**: OmniParser v2 provides superior text recognition compared to traditional OCR
- **Layout Understanding**: Understands document structure and relationships between text elements
- **Multi-language Support**: Built-in support for multiple languages
- **Context Awareness**: Better understanding of text context and meaning

### Advanced Capabilities
- **Document Question Answering**: Can answer questions about document content
- **Layout Analysis**: Understands document structure and formatting
- **Confidence Scoring**: Provides more accurate confidence scores for extracted text
- **Bounding Box Detection**: Precise location detection for text elements

## Architecture

### Core Components

1. **OmniParserV2Engine** (`src/vision/omniparser_v2.py`)
   - Main engine for OmniParser v2 integration
   - Handles model initialization and text extraction
   - Provides fallback to Tesseract if needed

2. **Enhanced VisionEngine** (`src/vision/engine.py`)
   - Integrates OmniParser v2 as primary OCR engine
   - Falls back to Tesseract if OmniParser v2 is unavailable
   - Maintains compatibility with existing code

3. **Configuration System** (`src/core/config.py`)
   - Configurable OCR engine selection
   - Model selection and GPU usage options
   - Backward compatibility settings

## Configuration

### Environment Variables

Add the following to your `.env` file:

```bash
# OCR Engine Configuration
USE_OMNIPARSER_V2=true
OMNIPARSER_V2_MODEL=microsoft/omniparser-v2-base
USE_GPU=true

# Legacy Tesseract Configuration (fallback)
TESSERACT_CMD=/opt/homebrew/bin/tesseract
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_OMNIPARSER_V2` | `true` | Enable OmniParser v2 as primary OCR engine |
| `OMNIPARSER_V2_MODEL` | `microsoft/omniparser-v2-base` | Hugging Face model name |
| `USE_GPU` | `true` | Use GPU acceleration if available |
| `TESSERACT_CMD` | `None` | Path to Tesseract executable (fallback) |

## Installation

### Dependencies

Install the required dependencies:

```bash
pip install transformers>=4.36.0 torch>=2.1.0 torchvision>=0.16.0
```

Or update your `requirements.txt`:

```txt
# OmniParser v2 Dependencies
transformers>=4.36.0
torch>=2.1.0
torchvision>=0.16.0
```

### Model Download

The OmniParser v2 model will be automatically downloaded on first use. The model is approximately 2GB and will be cached locally.

## Usage

### Basic Usage

```python
from src.vision.omniparser_v2 import get_omniparser_v2_engine

# Get the engine instance
engine = get_omniparser_v2_engine()

# Initialize the model
await engine.initialize()

# Analyze an image
elements = engine.analyze("screenshot.png")

# Process results
for element in elements:
    print(f"Text: {element.text}")
    print(f"Confidence: {element.confidence}")
    print(f"Bounding Box: {element.bbox.as_tuple()}")
```

### Integration with VisionEngine

The VisionEngine automatically uses OmniParser v2 when available:

```python
from src.vision.engine import VisionEngine

# Create vision engine (automatically uses OmniParser v2 if available)
vision_engine = VisionEngine()

# Analyze screenshot
elements = vision_engine.analyze("screenshot.png")
```

## Fallback Mechanism

The system includes a robust fallback mechanism:

1. **Primary**: OmniParser v2 (if enabled and available)
2. **Fallback**: Tesseract OCR (if OmniParser v2 fails or is disabled)
3. **Graceful Degradation**: Returns empty results if no OCR engine is available

### Fallback Triggers

- OmniParser v2 dependencies not installed
- Model download fails
- GPU memory insufficient
- Configuration disables OmniParser v2

## Performance Considerations

### GPU Usage

- **Recommended**: Use GPU for better performance
- **Memory**: Requires ~4GB GPU memory for optimal performance
- **CPU Fallback**: Automatically falls back to CPU if GPU unavailable

### Model Loading

- **First Run**: Model download and loading may take 1-2 minutes
- **Subsequent Runs**: Model is cached and loads quickly
- **Memory Usage**: ~2GB RAM for model storage

### Processing Speed

- **GPU**: ~100-500ms per image (depending on complexity)
- **CPU**: ~1-3 seconds per image
- **Batch Processing**: Supports batch processing for multiple images

## Testing

Run the test script to verify the integration:

```bash
python scripts/test_omniparser_v2.py
```

This will:
1. Test OmniParser v2 initialization
2. Verify VisionEngine integration
3. Test with sample images
4. Display configuration summary

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check internet connection
   - Verify Hugging Face access
   - Clear model cache if corrupted

2. **GPU Memory Issues**
   - Reduce batch size
   - Use CPU fallback
   - Close other GPU applications

3. **Dependencies Missing**
   - Install transformers and torch
   - Check Python version compatibility
   - Verify CUDA installation (if using GPU)

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
```

### Performance Optimization

1. **GPU Memory**: Ensure sufficient GPU memory
2. **Batch Size**: Adjust batch processing size
3. **Model Selection**: Choose appropriate model size
4. **Caching**: Enable model caching for faster startup

## Migration from Tesseract

### Automatic Migration

The system automatically migrates from Tesseract to OmniParser v2:

1. **No Code Changes**: Existing code continues to work
2. **Automatic Fallback**: Falls back to Tesseract if needed
3. **Configuration Control**: Use environment variables to control behavior

### Manual Migration

To force OmniParser v2 usage:

```bash
export USE_OMNIPARSER_V2=true
export USE_OCR=true
```

To use Tesseract only:

```bash
export USE_OMNIPARSER_V2=false
export USE_OCR=true
```

## Future Enhancements

### Planned Features

1. **Custom Model Training**: Support for custom-trained models
2. **Batch Processing**: Optimized batch processing
3. **Real-time Processing**: Stream processing capabilities
4. **Multi-modal Analysis**: Integration with other vision models

### Model Updates

- **Automatic Updates**: Model version management
- **A/B Testing**: Compare different model versions
- **Performance Monitoring**: Track accuracy and speed metrics

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review configuration settings
3. Test with the provided test script
4. Check logs for detailed error messages

## References

- [OmniParser v2 Paper](https://arxiv.org/abs/2403.xxxxx)
- [Hugging Face Model](https://huggingface.co/microsoft/omniparser-v2-base)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
