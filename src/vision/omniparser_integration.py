"""
OMniParser Integration for Enhanced UI Element Detection
=======================================================

This module integrates OMniParser for advanced UI element detection,
text extraction, and layout analysis in Android screenshots.
"""

import logging
import json
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
import base64
from PIL import Image
import io

from .ui_elements import UIElement, BoundingBox

logger = logging.getLogger(__name__)


class OMniParserIntegration:
    """Integration with OMniParser for advanced UI analysis."""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.omniparser.ai"):
        """Initialize OMniParser integration.
        
        Args:
            api_key: OMniParser API key (can be set via environment variable OMNIPARSER_API_KEY)
            base_url: OMniParser API base URL
        """
        self.api_key = api_key or self._get_api_key()
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
        
        self._available = self._test_connection()
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variable."""
        import os
        return os.getenv('OMNIPARSER_API_KEY')
    
    def _test_connection(self) -> bool:
        """Test connection to OMniParser API."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                logger.info("✅ OMniParser connection successful")
                return True
            else:
                logger.warning(f"OMniParser connection failed: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"OMniParser connection test failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if OMniParser is available."""
        return self._available and self.api_key is not None
    
    def analyze_screenshot(self, image_path: str) -> List[UIElement]:
        """Analyze screenshot using OMniParser.
        
        Args:
            image_path: Path to screenshot image
            
        Returns:
            List of detected UI elements
        """
        if not self.is_available():
            logger.warning("OMniParser not available, skipping analysis")
            return []
        
        try:
            # Load and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare request payload
            payload = {
                "image": image_b64,
                "options": {
                    "extract_text": True,
                    "detect_buttons": True,
                    "detect_input_fields": True,
                    "detect_icons": True,
                    "layout_analysis": True,
                    "confidence_threshold": 0.5
                }
            }
            
            # Send request to OMniParser
            logger.info("🔍 Analyzing screenshot with OMniParser...")
            response = self.session.post(
                f"{self.base_url}/analyze",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_omniparser_result(result)
            else:
                logger.error(f"OMniParser analysis failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"OMniParser analysis error: {e}")
            return []
    
    def _parse_omniparser_result(self, result: Dict[str, Any]) -> List[UIElement]:
        """Parse OMniParser analysis result into UIElement objects.
        
        Args:
            result: OMniParser analysis result
            
        Returns:
            List of UIElement objects
        """
        elements = []
        
        try:
            # Extract text elements
            if 'text_elements' in result:
                for text_elem in result['text_elements']:
                    bbox = BoundingBox(
                        x1=text_elem.get('x', 0),
                        y1=text_elem.get('y', 0),
                        x2=text_elem.get('x', 0) + text_elem.get('width', 0),
                        y2=text_elem.get('y', 0) + text_elem.get('height', 0)
                    )
                    
                    element = UIElement(
                        bbox=bbox,
                        text=text_elem.get('text', ''),
                        confidence=text_elem.get('confidence', 0.0),
                        element_type='text'
                    )
                    elements.append(element)
            
            # Extract button elements
            if 'buttons' in result:
                for button in result['buttons']:
                    bbox = BoundingBox(
                        x1=button.get('x', 0),
                        y1=button.get('y', 0),
                        x2=button.get('x', 0) + button.get('width', 0),
                        y2=button.get('y', 0) + button.get('height', 0)
                    )
                    
                    element = UIElement(
                        bbox=bbox,
                        text=button.get('text', ''),
                        confidence=button.get('confidence', 0.0),
                        element_type='button',
                        clickable=True
                    )
                    elements.append(element)
            
            # Extract input fields
            if 'input_fields' in result:
                for input_field in result['input_fields']:
                    bbox = BoundingBox(
                        x1=input_field.get('x', 0),
                        y1=input_field.get('y', 0),
                        x2=input_field.get('x', 0) + input_field.get('width', 0),
                        y2=input_field.get('y', 0) + input_field.get('height', 0)
                    )
                    
                    element = UIElement(
                        bbox=bbox,
                        text=input_field.get('placeholder', ''),
                        confidence=input_field.get('confidence', 0.0),
                        element_type='input',
                        input_type=input_field.get('input_type', 'text')
                    )
                    elements.append(element)
            
            # Extract icons
            if 'icons' in result:
                for icon in result['icons']:
                    bbox = BoundingBox(
                        x1=icon.get('x', 0),
                        y1=icon.get('y', 0),
                        x2=icon.get('x', 0) + icon.get('width', 0),
                        y2=icon.get('y', 0) + icon.get('height', 0)
                    )
                    
                    element = UIElement(
                        bbox=bbox,
                        text=icon.get('description', ''),
                        confidence=icon.get('confidence', 0.0),
                        element_type='icon',
                        clickable=icon.get('clickable', False)
                    )
                    elements.append(element)
            
            logger.info(f"✅ OMniParser detected {len(elements)} UI elements")
            
        except Exception as e:
            logger.error(f"Error parsing OMniParser result: {e}")
        
        return elements
    
    def extract_layout_info(self, image_path: str) -> Dict[str, Any]:
        """Extract layout information from screenshot.
        
        Args:
            image_path: Path to screenshot image
            
        Returns:
            Layout information dictionary
        """
        if not self.is_available():
            return {}
        
        try:
            # Load and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Request layout analysis
            payload = {
                "image": image_b64,
                "options": {
                    "layout_analysis": True,
                    "extract_text": False,
                    "detect_buttons": False,
                    "detect_input_fields": False,
                    "detect_icons": False
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/analyze",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('layout_info', {})
            else:
                logger.error(f"OMniParser layout analysis failed: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"OMniParser layout analysis error: {e}")
            return {}
    
    def get_element_hierarchy(self, image_path: str) -> Dict[str, Any]:
        """Get UI element hierarchy and relationships.
        
        Args:
            image_path: Path to screenshot image
            
        Returns:
            Element hierarchy information
        """
        if not self.is_available():
            return {}
        
        try:
            # Load and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Request hierarchy analysis
            payload = {
                "image": image_b64,
                "options": {
                    "hierarchy_analysis": True,
                    "extract_text": True,
                    "detect_buttons": True,
                    "detect_input_fields": True,
                    "detect_icons": True
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/analyze",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('hierarchy', {})
            else:
                logger.error(f"OMniParser hierarchy analysis failed: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"OMniParser hierarchy analysis error: {e}")
            return {}


# Convenience function for easy usage
def create_omniparser_integration(api_key: str = None) -> OMniParserIntegration:
    """Create OMniParser integration instance.
    
    Args:
        api_key: OMniParser API key (optional, can use environment variable)
        
    Returns:
        OMniParserIntegration instance
    """
    return OMniParserIntegration(api_key=api_key)
