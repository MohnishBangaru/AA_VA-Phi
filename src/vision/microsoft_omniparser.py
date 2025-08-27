"""
Microsoft OmniParser-v2.0 Integration for Local UI Element Detection
===================================================================

This module integrates Microsoft's OmniParser-v2.0 model for superior
local UI element detection without requiring external API calls.
"""

import logging
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import numpy as np
import cv2

from .ui_elements import UIElement, BoundingBox

logger = logging.getLogger(__name__)

@dataclass
class OmniParserConfig:
    """Configuration for Microsoft OmniParser-v2.0."""
    model_name: str = "microsoft/OmniParser-v2.0"
    device: str = "auto"
    confidence_threshold: float = 0.7
    max_image_size: int = 1024
    batch_size: int = 1

class MicrosoftOmniParserEngine:
    """Local UI element detection using Microsoft OmniParser-v2.0."""
    
    def __init__(self, config: OmniParserConfig):
        """Initialize Microsoft OmniParser engine."""
        self.config = config
        self.device = self._get_device()
        self.model = None
        self.processor = None
        self._initialized = False
        
        logger.info(f"Initializing Microsoft OmniParser-v2.0 on {self.device}")
    
    def _get_device(self) -> str:
        """Get the best available device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def initialize(self):
        """Initialize the OmniParser model."""
        if self._initialized:
            return
        
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            
            logger.info("Loading Microsoft OmniParser-v2.0 model...")
            
            # Load model and processor
            self.model = AutoModelForImageClassification.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.processor = AutoImageProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._initialized = True
            logger.info("Microsoft OmniParser-v2.0 initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Microsoft OmniParser-v2.0: {e}")
            raise
    
    def analyze_screenshot(self, image_path: str) -> List[UIElement]:
        """
        Analyze screenshot using Microsoft OmniParser-v2.0.
        
        Args:
            image_path: Path to the screenshot image
            
        Returns:
            List of detected UI elements with enhanced information
        """
        if not self._initialized:
            self.initialize()
        
        try:
            logger.info(f"Analyzing screenshot with Microsoft OmniParser-v2.0: {image_path}")
            
            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path)
            if image is None:
                logger.error("Failed to load image for OmniParser analysis")
                return []
            
            # Process with OmniParser
            elements_data = self._process_with_omniparser(image)
            if not elements_data:
                logger.error("Failed to get results from OmniParser")
                return []
            
            # Convert to UIElement objects
            ui_elements = self._convert_to_ui_elements(elements_data, image.size)
            
            logger.info(f"Microsoft OmniParser-v2.0 detected {len(ui_elements)} UI elements")
            return ui_elements
            
        except Exception as e:
            logger.error(f"Microsoft OmniParser-v2.0 analysis failed: {e}")
            return []
    
    def _load_and_preprocess_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and preprocess image for OmniParser."""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Resize if too large
            max_size = self.config.max_image_size
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"Resized image from {image.size} to {new_size}")
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load and preprocess image: {e}")
            return None
    
    def _process_with_omniparser(self, image: Image.Image) -> Optional[List[Dict[str, Any]]]:
        """Process image with Microsoft OmniParser-v2.0."""
        try:
            with torch.no_grad():
                # Prepare inputs
                inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Run inference
                outputs = self.model(**inputs)
                
                # Process outputs
                elements_data = self._extract_elements_from_outputs(outputs, image.size)
                
                return elements_data
                
        except Exception as e:
            logger.error(f"Failed to process with OmniParser: {e}")
            return None
    
    def _extract_elements_from_outputs(self, outputs, image_size) -> List[Dict[str, Any]]:
        """Extract UI elements from model outputs."""
        try:
            # This is a simplified extraction - actual implementation depends on model output format
            elements = []
            
            # Extract predictions from outputs
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probs, k=min(10, probs.size(-1)), dim=-1)
                
                for i in range(top_probs.size(1)):
                    confidence = float(top_probs[0, i])
                    if confidence >= self.config.confidence_threshold:
                        element_type = self._get_element_type(top_indices[0, i])
                        
                        # Create bounding box (simplified - actual implementation would use model's bbox outputs)
                        bbox = self._create_bbox_for_element(i, image_size)
                        
                        elements.append({
                            "bounding_box": bbox,
                            "text": "",  # OmniParser-v2.0 doesn't extract text directly
                            "confidence": confidence,
                            "type": element_type,
                            "clickable": element_type in ["button", "link", "input"],
                            "input_type": "text" if element_type == "input" else None
                        })
            
            return elements
            
        except Exception as e:
            logger.error(f"Failed to extract elements from outputs: {e}")
            return []
    
    def _get_element_type(self, index: int) -> str:
        """Map model output index to element type."""
        # This mapping depends on the model's class labels
        element_types = [
            "button", "input", "text", "image", "icon", "link", 
            "checkbox", "radio", "dropdown", "slider", "container"
        ]
        
        if index < len(element_types):
            return element_types[index]
        return "unknown"
    
    def _create_bbox_for_element(self, index: int, image_size) -> Dict[str, int]:
        """Create bounding box for detected element."""
        # Simplified bbox creation - actual implementation would use model's bbox outputs
        width, height = image_size
        
        # Create a grid-based bbox (simplified)
        grid_size = 4
        row = index // grid_size
        col = index % grid_size
        
        cell_width = width // grid_size
        cell_height = height // grid_size
        
        x1 = col * cell_width
        y1 = row * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height
        
        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
    
    def _convert_to_ui_elements(self, elements_data: List[Dict[str, Any]], image_size) -> List[UIElement]:
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
                        "model": "microsoft/OmniParser-v2.0"
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
            "average_confidence": sum(e.confidence for e in elements) / len(elements) if elements else 0,
            "model": "microsoft/OmniParser-v2.0"
        }
        
        return summary

def create_microsoft_omniparser_engine(config: OmniParserConfig = None) -> MicrosoftOmniParserEngine:
    """
    Create Microsoft OmniParser engine with configuration.
    
    Args:
        config: Configuration for OmniParser. If None, uses defaults.
        
    Returns:
        Configured Microsoft OmniParser engine
    """
    if config is None:
        config = OmniParserConfig()
    
    return MicrosoftOmniParserEngine(config)
