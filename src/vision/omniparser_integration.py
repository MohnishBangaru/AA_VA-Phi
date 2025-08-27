"""
OMniParser 2.0 Integration for Enhanced UI Element Detection
===========================================================

This module integrates OmniParser 2.0 via Hugging Face for advanced UI element detection,
text extraction, and layout analysis in Android screenshots.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image
import torch
import numpy as np

from .ui_elements import UIElement, BoundingBox

logger = logging.getLogger(__name__)


class OMniParserIntegration:
    """Integration with OmniParser 2.0 via Hugging Face for advanced UI analysis."""
    
    def __init__(self, model_name: str = "microsoft/OmniParser-v2.0", device: str = None):
        """Initialize OmniParser 2.0 integration.
        
        Args:
            model_name: Hugging Face model name for OmniParser 2.0
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        self._available = self._initialize_model()
    
    def _initialize_model(self) -> bool:
        """Initialize the OmniParser 2.0 model from Hugging Face."""
        try:
            logger.info(f"🚀 Loading OmniParser 2.0 model: {self.model_name}")
            
            # Check Hugging Face Hub login
            self._check_hf_login()
            
            # Import transformers components
            from transformers import (
                AutoProcessor, 
                AutoTokenizer, 
                AutoModelForVisionTextGeneration,
                AutoImageProcessor
            )
            
            # Load processor (handles both text and image)
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                use_auth_token=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                use_auth_token=True
            )
            
            # Load model
            self.model = AutoModelForVisionTextGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
                use_auth_token=True
            )
            
            logger.info(f"✅ OmniParser 2.0 model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load OmniParser 2.0 model: {e}")
            return False
    
    def _check_hf_login(self) -> bool:
        """Check if user is logged in to Hugging Face Hub."""
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            user = api.whoami()
            logger.info(f"✅ Logged in to Hugging Face Hub as: {user}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Not logged in to Hugging Face Hub: {e}")
            logger.info("🔑 Please login with: huggingface-cli login")
            return False
    
    def is_available(self) -> bool:
        """Check if OmniParser 2.0 is available."""
        return self._available and self.model is not None
    
    def analyze_screenshot(self, image_path: str) -> List[UIElement]:
        """Analyze screenshot using OmniParser 2.0.
        
        Args:
            image_path: Path to screenshot image
            
        Returns:
            List of detected UI elements
        """
        if not self.is_available():
            logger.warning("OmniParser 2.0 not available, skipping analysis")
            return []
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            logger.info(f"📸 Loaded image: {image.size}")
            
            # Create analysis prompt
            prompt = """Analyze this Android app screenshot and extract UI elements. 
            For each element, provide:
            - Type (button, text, input, icon)
            - Text content
            - Bounding box coordinates (x, y, width, height)
            - Confidence score
            - Whether it's clickable
            
            Format the response as JSON with this structure:
            {
                "elements": [
                    {
                        "type": "button",
                        "text": "Login",
                        "x": 100,
                        "y": 200,
                        "width": 80,
                        "height": 40,
                        "confidence": 0.95,
                        "clickable": true
                    }
                ]
            }"""
            
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate analysis
            logger.info("🔍 Running OmniParser 2.0 analysis...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                try:
                    result = json.loads(json_str)
                    return self._parse_omniparser_result(result, image.size)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Response text: {response_text}")
            
            # Fallback: try to extract elements from text response
            return self._parse_text_response(response_text, image.size)
            
        except Exception as e:
            logger.error(f"OmniParser 2.0 analysis error: {e}")
            return []
    
    def _parse_omniparser_result(self, result: Dict[str, Any], image_size: tuple) -> List[UIElement]:
        """Parse OmniParser 2.0 analysis result into UIElement objects.
        
        Args:
            result: OmniParser 2.0 analysis result
            image_size: Original image size (width, height)
            
        Returns:
            List of UIElement objects
        """
        elements = []
        
        try:
            if 'elements' in result:
                for elem in result['elements']:
                    # Create bounding box
                    x = elem.get('x', 0)
                    y = elem.get('y', 0)
                    width = elem.get('width', 0)
                    height = elem.get('height', 0)
                    
                    bbox = BoundingBox(
                        x1=x,
                        y1=y,
                        x2=x + width,
                        y2=y + height
                    )
                    
                    # Create UI element
                    element = UIElement(
                        bbox=bbox,
                        text=elem.get('text', ''),
                        confidence=elem.get('confidence', 0.5),
                        element_type=elem.get('type', 'unknown'),
                        clickable=elem.get('clickable', False)
                    )
                    
                    # Add input type if available
                    if elem.get('type') == 'input':
                        element.input_type = elem.get('input_type', 'text')
                    
                    elements.append(element)
            
            logger.info(f"✅ OmniParser 2.0 detected {len(elements)} UI elements")
            
        except Exception as e:
            logger.error(f"Error parsing OmniParser 2.0 result: {e}")
        
        return elements
    
    def _parse_text_response(self, response_text: str, image_size: tuple) -> List[UIElement]:
        """Parse text response when JSON parsing fails.
        
        Args:
            response_text: Raw text response from model
            image_size: Original image size (width, height)
            
        Returns:
            List of UIElement objects
        """
        elements = []
        
        try:
            # Simple text-based parsing as fallback
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Try to extract element information from text
                if any(keyword in line.lower() for keyword in ['button', 'text', 'input', 'icon']):
                    # Create a basic element
                    bbox = BoundingBox(
                        x1=0,
                        y1=0,
                        x2=image_size[0] // 4,  # Default size
                        y2=image_size[1] // 8
                    )
                    
                    element = UIElement(
                        bbox=bbox,
                        text=line[:50],  # First 50 chars as text
                        confidence=0.5,
                        element_type='text',
                        clickable='button' in line.lower()
                    )
                    
                    elements.append(element)
            
            logger.info(f"✅ Text parsing detected {len(elements)} elements")
            
        except Exception as e:
            logger.error(f"Error parsing text response: {e}")
        
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
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Create layout analysis prompt
            prompt = """Analyze the layout structure of this Android app screenshot.
            Identify:
            - Main sections/regions
            - Navigation elements
            - Content areas
            - Button placement patterns
            
            Return as JSON:
            {
                "layout": {
                    "sections": [],
                    "navigation": {},
                    "content_areas": []
                }
            }"""
            
            # Process and generate
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.1
                )
            
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                try:
                    result = json.loads(response_text[json_start:json_end])
                    return result.get('layout', {})
                except json.JSONDecodeError:
                    pass
            
            return {}
            
        except Exception as e:
            logger.error(f"OmniParser 2.0 layout analysis error: {e}")
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
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Create hierarchy analysis prompt
            prompt = """Analyze the UI element hierarchy in this Android app screenshot.
            Identify parent-child relationships between UI elements.
            
            Return as JSON:
            {
                "hierarchy": {
                    "root_elements": [],
                    "child_elements": {},
                    "relationships": []
                }
            }"""
            
            # Process and generate
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.1
                )
            
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                try:
                    result = json.loads(response_text[json_start:json_end])
                    return result.get('hierarchy', {})
                except json.JSONDecodeError:
                    pass
            
            return {}
            
        except Exception as e:
            logger.error(f"OmniParser 2.0 hierarchy analysis error: {e}")
            return {}


# Convenience function for easy usage
def create_omniparser_integration(model_name: str = None, device: str = None) -> OMniParserIntegration:
    """Create OmniParser 2.0 integration instance.
    
    Args:
        model_name: Hugging Face model name (optional)
        device: Device to run on (optional)
        
    Returns:
        OMniParserIntegration instance
    """
    return OMniParserIntegration(model_name=model_name, device=device)
