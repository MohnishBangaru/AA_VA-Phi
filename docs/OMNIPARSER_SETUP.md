# OmniParser Integration Setup Guide

## Overview

OmniParser is an advanced AI-powered UI element detection service that provides superior accuracy compared to basic OCR. It can detect:

- **Text elements** with high accuracy
- **Interactive elements** (buttons, links, inputs)
- **UI components** (icons, images, layouts)
- **Element relationships** and hierarchy
- **Semantic understanding** of UI context

## Benefits Over Basic OCR

| Feature | Basic OCR | OmniParser |
|---------|-----------|------------|
| Text Detection | ✅ Basic | ✅ Advanced |
| Element Classification | ❌ No | ✅ Yes |
| Clickable Detection | ❌ No | ✅ Yes |
| Layout Understanding | ❌ No | ✅ Yes |
| Confidence Scoring | ✅ Basic | ✅ Advanced |
| Element Relationships | ❌ No | ✅ Yes |

## Setup Instructions

### 1. Get OmniParser API Key

1. Visit [OmniParser.ai](https://omniparser.ai)
2. Sign up for an account
3. Navigate to your dashboard
4. Generate an API key
5. Copy the API key for use in the next step

### 2. Configure Environment

Set the API key as an environment variable:

```bash
# Linux/macOS
export OMNIPARSER_API_KEY="your_api_key_here"

# Windows
set OMNIPARSER_API_KEY=your_api_key_here

# Or add to your .env file
echo "OMNIPARSER_API_KEY=your_api_key_here" >> .env
```

### 3. Test Integration

Run the OmniParser test script:

```bash
python scripts/test_omniparser.py
```

Expected output:
```
🚀 OmniParser Integration Test
============================================================
🔍 Testing OmniParser Setup
========================================
✅ OmniParser API key found: abc12345...
✅ OmniParser engine imports successful
✅ OmniParser engine created successfully

🌐 Testing OmniParser API Connectivity
========================================
📸 Created test image: test_omniparser_image.png
🔍 Testing OmniParser API call...
✅ OmniParser API successful! Detected 15 elements
  Element 1: 'Login' (button)
  Element 2: 'Username' (input)
  Element 3: 'Password' (input)
📊 Summary: 15 total, 8 clickable

👁️ Testing VisionEngine Integration
========================================
🚀 Initializing VisionEngine with OmniParser...
✅ OmniParser is available in VisionEngine
📸 Testing with actual screenshot: test_reports/action_1_screenshot.png
✅ VisionEngine with OmniParser detected 12 elements
  Element 1: 'Order Now' at BoundingBox(x1=100, y1=200, x2=200, y2=230)
  Element 2: 'Menu' at BoundingBox(x1=50, y1=50, x2=100, y2=80)

📊 Test Summary
========================================
Setup: ✅ PASS
API Connectivity: ✅ PASS
VisionEngine Integration: ✅ PASS
Fallback Behavior: ✅ PASS

🎉 OmniParser integration is working correctly!
```

## Usage

### Automatic Integration

The VisionEngine automatically uses OmniParser when available:

```python
from src.vision.engine import VisionEngine

# OmniParser will be used automatically if API key is set
vision_engine = VisionEngine(use_omniparser=True)
elements = vision_engine.analyze("screenshot.png")
```

### Manual Usage

You can also use OmniParser directly:

```python
from src.vision.omniparser_engine import create_omniparser_engine

# Create engine
engine = create_omniparser_engine()

# Analyze screenshot
elements = engine.analyze_screenshot("screenshot.png")

# Get summary
summary = engine.get_element_summary(elements)
print(f"Detected {summary['total_elements']} elements")
```

## Element Types

OmniParser detects various element types:

### Text Elements
- **Headings**: Page titles, section headers
- **Body Text**: Paragraphs, descriptions
- **Labels**: Form labels, button text

### Interactive Elements
- **Buttons**: Clickable buttons with text
- **Links**: Hyperlinks and navigation
- **Input Fields**: Text inputs, checkboxes, radio buttons
- **Dropdowns**: Select menus and pickers

### UI Components
- **Icons**: App icons, action icons
- **Images**: Photos, graphics, logos
- **Layout Elements**: Containers, dividers

## Configuration Options

### OmniParserConfig

```python
from src.vision.omniparser_engine import OmniParserConfig

config = OmniParserConfig(
    api_key="your_key",
    base_url="https://api.omniparser.ai",  # Default
    timeout=30,                            # API timeout in seconds
    max_retries=3,                         # Retry attempts
    confidence_threshold=0.7               # Minimum confidence (0.0-1.0)
)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OMNIPARSER_API_KEY` | None | Your OmniParser API key |
| `OMNIPARSER_TIMEOUT` | 30 | API timeout in seconds |
| `OMNIPARSER_CONFIDENCE` | 0.7 | Minimum confidence threshold |

## Fallback Behavior

If OmniParser is not available (no API key, network issues, etc.), the system automatically falls back to:

1. **Basic OCR** (Tesseract)
2. **Template matching**
3. **Clickable element detection**

This ensures your automation continues to work even without OmniParser.

## Troubleshooting

### Common Issues

#### 1. API Key Not Found
```
❌ OmniParser API key not found
```
**Solution**: Set the `OMNIPARSER_API_KEY` environment variable

#### 2. Network Connectivity
```
❌ OmniParser API request failed: Connection timeout
```
**Solution**: Check your internet connection and firewall settings

#### 3. Invalid API Key
```
❌ OmniParser API returned 401: Unauthorized
```
**Solution**: Verify your API key is correct and active

#### 4. Rate Limiting
```
❌ OmniParser API returned 429: Too Many Requests
```
**Solution**: Wait a moment and retry, or upgrade your OmniParser plan

### Debug Mode

Enable debug logging to see detailed API interactions:

```python
import logging
logging.getLogger("src.vision.omniparser_engine").setLevel(logging.DEBUG)
```

## Performance Considerations

### API Limits
- **Free Tier**: 100 requests/month
- **Pro Tier**: 10,000 requests/month
- **Enterprise**: Custom limits

### Optimization Tips
1. **Resize large images** before sending to API
2. **Cache results** for repeated screenshots
3. **Use confidence thresholds** to filter low-quality detections
4. **Batch process** multiple screenshots when possible

## Cost Analysis

### Free Tier (100 requests/month)
- **Cost**: $0
- **Suitable for**: Development and testing
- **Limitations**: Basic features only

### Pro Tier ($29/month)
- **Cost**: $29/month for 10,000 requests
- **Suitable for**: Production automation
- **Features**: Advanced detection, priority support

### Enterprise Tier
- **Cost**: Custom pricing
- **Suitable for**: High-volume automation
- **Features**: Custom models, dedicated support

## Migration from OCR

If you're currently using basic OCR, the migration is seamless:

```python
# Before (OCR only)
vision_engine = VisionEngine()

# After (OmniParser + OCR fallback)
vision_engine = VisionEngine(use_omniparser=True)
```

The same API is used, but with significantly improved results.

## Support

- **Documentation**: [OmniParser.ai/docs](https://omniparser.ai/docs)
- **API Reference**: [OmniParser.ai/api](https://omniparser.ai/api)
- **Support**: [support@omniparser.ai](mailto:support@omniparser.ai)
- **Community**: [Discord](https://discord.gg/omniparser)
