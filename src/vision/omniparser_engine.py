"""
OmniParser Integration for Advanced UI Element Detection
=======================================================

This module integrates OmniParser for superior UI element detection,
text extraction, and element classification compared to basic OCR.
"""

import logging
import requests
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import base64
import io

from .ui_elements import UIElement, BoundingBox

logger = logging.getLogger(__name__)

@dataclass
class OmniParserConfig:
    """Configuration for OmniParser integration."""
    api_key: str
    base_url: str = "https://api.omniparser.ai"
    timeout: int = 30
    max_retries: int = 3
    confidence_threshold: float = 0.7

class OmniParserEngine:
    """Advanced UI element detection using OmniParser."""
    
    def __init__(self, config: OmniParserConfig):
        """Initialize OmniParser engine."""
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        })
    
    def analyze_screenshot(self, image_path: str) -> List[UIElement]:
        """
        Analyze screenshot using OmniParser for advanced UI element detection.
        
        Args:
            image_path: Path to the screenshot image
            
        Returns:
            List of detected UI elements with enhanced information
        """
        try:
            logger.info(f"Analyzing screenshot with OmniParser: {image_path}")
            
            # Prepare image for API
            image_data = self._prepare_image(image_path)
            if not image_data:
                logger.error("Failed to prepare image for OmniParser")
                return []
            
            # Call OmniParser API
            elements_data = self._call_omniparser_api(image_data)
            if not elements_data:
                logger.error("Failed to get response from OmniParser")
                return []
            
            # Convert to UIElement objects
            ui_elements = self._convert_to_ui_elements(elements_data)
            
            logger.info(f"OmniParser detected {len(ui_elements)} UI elements")
            return ui_elements
            
        except Exception as e:
            logger.error(f"OmniParser analysis failed: {e}")
            return []
    
    def _prepare_image(self, image_path: str) -> Optional[str]:
        """Prepare image for OmniParser API."""
        try:
            # Load and optimize image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (OmniParser has size limits)
                max_size = 2048
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return image_base64
                
        except Exception as e:
            logger.error(f"Failed to prepare image: {e}")
            return None
    
    def _call_omniparser_api(self, image_base64: str) -> Optional[List[Dict[str, Any]]]:
        """Call OmniParser API for UI element detection."""
        payload = {
            "image": image_base64,
            "options": {
                "detect_text": True,
                "detect_buttons": True,
                "detect_inputs": True,
                "detect_icons": True,
                "detect_layout": True,
                "confidence_threshold": self.config.confidence_threshold,
                "output_format": "detailed"
            }
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    f"{self.config.base_url}/v1/analyze",
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("elements", [])
                else:
                    logger.warning(f"OmniParser API returned {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"OmniParser API request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        return None
    
    def _convert_to_ui_elements(self, elements_data: List[Dict[str, Any]]) -> List[UIElement]:
        """Convert OmniParser response to UIElement objects."""
        ui_elements = []
        
        for element in elements_data:
            try:
                # Extract bounding box
                bbox_data = element.get("bounding_box", {})
                bbox = BoundingBox(
                    x1=bbox_data.get("x1", 0),
                    y1=bbox_data.get("y1", 0),
                    x2=bbox_data.get("x2", 0),
                    y2=bbox_data.get("y2", 0)
                )
                
                # Extract text content
                text = element.get("text", "").strip()
                
                # Extract confidence
                confidence = element.get("confidence", 0.0)
                
                # Extract element type
                element_type = element.get("type", "unknown")
                
                # Create UIElement with enhanced information
                ui_element = UIElement(
                    bbox=bbox,
                    text=text,
                    confidence=confidence,
                    element_type=element_type,
                    metadata={
                        "omniparser_type": element_type,
                        "clickable": element.get("clickable", False),
                        "input_type": element.get("input_type"),
                        "placeholder": element.get("placeholder"),
                        "icon_type": element.get("icon_type"),
                        "layout_role": element.get("layout_role")
                    }
                )
                
                ui_elements.append(ui_element)
                
            except Exception as e:
                logger.warning(f"Failed to convert element: {e}")
                continue
        
        return ui_elements
    
    def get_element_summary(self, elements: List[UIElement]) -> Dict[str, Any]:
        """Get summary of detected elements."""
        summary = {
            "total_elements": len(elements),
            "text_elements": len([e for e in elements if e.text]),
            "clickable_elements": len([e for e in elements if e.metadata.get("clickable", False)]),
            "input_elements": len([e for e in elements if e.metadata.get("input_type")]),
            "button_elements": len([e for e in elements if e.metadata.get("omniparser_type") == "button"]),
            "icon_elements": len([e for e in elements if e.metadata.get("icon_type")]),
            "average_confidence": sum(e.confidence for e in elements) / len(elements) if elements else 0
        }
        
        return summary

class OmniParserFallback:
    """Fallback implementation when OmniParser is not available."""
    
    def __init__(self):
        """Initialize fallback engine."""
        logger.info("Using OmniParser fallback (basic OCR)")
        from .engine import VisionEngine
        self.fallback_engine = VisionEngine()
    
    def analyze_screenshot(self, image_path: str) -> List[UIElement]:
        """Fallback to basic OCR when OmniParser is not available."""
        return self.fallback_engine.analyze(image_path)

def create_omniparser_engine(api_key: str = None) -> OmniParserEngine:
    """
    Create OmniParser engine with configuration.
    
    Args:
        api_key: OmniParser API key. If None, will try to get from environment.
        
    Returns:
        Configured OmniParser engine or fallback
    """
    if not api_key:
        import os
        api_key = os.getenv("OMNIPARSER_API_KEY")
    
    if not api_key:
        logger.warning("OmniParser API key not found. Using fallback OCR.")
        return OmniParserFallback()
    
    config = OmniParserConfig(api_key=api_key)
    return OmniParserEngine(config)
