# Vision Templates Directory

This directory is used for storing template images for computer vision template matching.

## Purpose

Template matching allows the vision engine to detect specific UI elements by comparing screenshots against pre-defined template images.

## Usage

Place PNG, JPG, or JPEG template images in this directory. The template matcher will automatically load them and use them for detection.

## Template Naming

- Use descriptive names for your template files
- The filename (without extension) will be used as the template name
- Supported formats: `.png`, `.jpg`, `.jpeg`

## Example

```
templates/
├── button_continue.png
├── icon_settings.jpg
└── logo_app.png
```

## Configuration

Template matching behavior can be configured in the main config file:
- `cv_template_matching_threshold`: Minimum confidence threshold
- `cv_template_max_scales`: Maximum number of scales to try
- `cv_template_scale_step`: Scale step size
- `cv_template_threads`: Number of threads for processing
