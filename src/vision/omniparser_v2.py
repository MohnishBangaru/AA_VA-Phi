"""OmniParser v2 integration for advanced OCR and document understanding.

This module provides OmniParser v2 integration for enhanced text extraction,
layout analysis, and UI element detection from screenshots.
"""

from __future__ import annotations

import os
import concurrent.futures
from typing import List, Dict, Any, Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from loguru import logger

try:
    from transformers import AutoProcessor, AutoModelForDocumentQuestionAnswering
    from transformers import pipeline
    OMNIPARSER_AVAILABLE = True
except ImportError:
    OMNIPARSER_AVAILABLE = False
    logger.warning("OmniParser v2 dependencies not available. Install transformers and torch.")

from core.config import config
from vision.models import BoundingBox, UIElement


class OmniParserV2Engine:
    """Advanced OCR and document understanding using OmniParser v2."""
    
    def __init__(self, model_name: str = None):
        """Initialize OmniParser v2 engine.
        
        Args:
            model_name: Hugging Face model name for OmniParser v2 (optional, uses config if None)
        """
        self.model_name = model_name or config.omniparser_v2_model
        self.processor = None
        self.model = None
        self.ocr_pipeline = None
        self._initialized = False
        
        # Use thread pool for async processing
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Check if OmniParser should be enabled
        self._omniparser_available: bool = False
        if not config.use_ocr:
            logger.warning("OmniParserV2Engine: OCR disabled via configuration (USE_OCR=false)")
        elif not config.use_omniparser_v2:
            logger.warning("OmniParserV2Engine: OmniParser v2 disabled via configuration")
        elif not OMNIPARSER_AVAILABLE:
            logger.warning("OmniParserV2Engine: OmniParser v2 dependencies not available")
        else:
            self._omniparser_available = True
            logger.info(f"OmniParserV2Engine: Initializing with model {self.model_name}")
    
    async def initialize(self) -> None:
        """Initialize the OmniParser v2 model."""
        if self._initialized or not self._omniparser_available:
            return
            
        try:
            logger.info(f"Loading OmniParser v2 model: {self.model_name}")
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForDocumentQuestionAnswering.from_pretrained(self.model_name)
            
            # Create OCR pipeline
            self.ocr_pipeline = pipeline(
                "document-question-answering",
                model=self.model,
                tokenizer=self.processor,
                device="cuda" if config.use_gpu else "cpu"
            )
            
            self._initialized = True
            logger.info("OmniParser v2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OmniParser v2: {e}")
            self._omniparser_available = False
    
    def analyze(self, image_path: str) -> List[UIElement]:
        """Analyze screenshot using OmniParser v2 and return detected UI elements.
        
        Args:
            image_path: Path to the screenshot image
            
        Returns:
            List of detected UIElement objects
        """
        if not self._omniparser_available or not self._initialized:
            logger.warning("OmniParser v2 not available, returning empty list")
            return []
        
        if not os.path.exists(image_path):
            logger.error(f"Screenshot not found: {image_path}")
            return []
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Run OmniParser v2 analysis
            elements = self._run_omniparser_analysis(image, image_path)
            
            logger.debug(f"OmniParser v2 detected {len(elements)} elements")
            return elements
            
        except Exception as e:
            logger.error(f"OmniParser v2 analysis failed: {e}")
            return []
    
    def _run_omniparser_analysis(self, image: Image.Image, image_path: str) -> List[UIElement]:
        """Run OmniParser v2 analysis on the image.
        
        Args:
            image: PIL Image object
            image_path: Path to the image for reference
            
        Returns:
            List of UIElement objects
        """
        elements = []
        
        try:
            # Use OmniParser v2 for text extraction and layout analysis
            # We'll use a general question to extract all text elements
            question = "What text elements are visible in this image?"
            
            # Run the pipeline
            result = self.ocr_pipeline(
                image=image,
                question=question,
                max_answer_len=1000,
                return_overflowing_tokens=True
            )
            
            # Process results to extract text and bounding boxes
            if result and isinstance(result, list) and len(result) > 0:
                for item in result:
                    if 'answer' in item and item['answer']:
                        # Extract text
                        text = item['answer'].strip()
                        
                        # Extract bounding box if available
                        bbox = self._extract_bbox_from_result(item, image.size)
                        
                        if bbox and text:
                            confidence = item.get('score', 0.8)
                            element = UIElement(
                                bbox=bbox,
                                text=text,
                                confidence=float(confidence),
                                element_type="text"
                            )
                            elements.append(element)
            
            # If no results from question-answering, try direct OCR
            if not elements:
                elements = self._fallback_ocr_analysis(image, image_path)
            
        except Exception as e:
            logger.warning(f"OmniParser v2 analysis failed, using fallback: {e}")
            elements = self._fallback_ocr_analysis(image, image_path)
        
        return elements
    
    def _extract_bbox_from_result(self, result: Dict[str, Any], image_size: tuple) -> Optional[BoundingBox]:
        """Extract bounding box from OmniParser v2 result.
        
        Args:
            result: Result dictionary from OmniParser v2
            image_size: Tuple of (width, height) of the image
            
        Returns:
            BoundingBox object or None
        """
        try:
            # Try to extract coordinates from the result
            # OmniParser v2 may provide coordinates in different formats
            if 'start' in result and 'end' in result:
                # Convert token positions to pixel coordinates
                # This is a simplified approach - actual implementation may vary
                start_pos = result['start']
                end_pos = result['end']
                
                # Estimate bounding box based on token positions
                # This is a placeholder - actual coordinate extraction depends on model output
                img_width, img_height = image_size
                
                # Simple estimation (this would need to be refined based on actual model output)
                x = int((start_pos / 100) * img_width)  # Simplified mapping
                y = int((start_pos / 100) * img_height)
                w = int(((end_pos - start_pos) / 100) * img_width)
                h = 30  # Default height for text
                
                return BoundingBox(x, y, x + w, y + h)
            
            # If no coordinates available, return None
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract bounding box: {e}")
            return None
    
    def _fallback_ocr_analysis(self, image: Image.Image, image_path: str) -> List[UIElement]:
        """Fallback OCR analysis using traditional methods.
        
        Args:
            image: PIL Image object
            image_path: Path to the image
            
        Returns:
            List of UIElement objects
        """
        elements = []
        
        try:
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold for better text detection
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # Use pytesseract as fallback
            try:
                import pytesseract
                
                # Run OCR
                ocr_data = pytesseract.image_to_data(
                    thresh,
                    lang='eng',
                    output_type=pytesseract.Output.DICT
                )
                
                # Process results
                n_boxes = len(ocr_data["level"])
                for i in range(n_boxes):
                    text = ocr_data["text"][i].strip()
                    conf = float(ocr_data["conf"][i])
                    
                    if text and conf > 0:
                        x, y, w, h = (
                            ocr_data["left"][i],
                            ocr_data["top"][i],
                            ocr_data["width"][i],
                            ocr_data["height"][i],
                        )
                        bbox = BoundingBox(x, y, x + w, y + h)
                        element = UIElement(
                            bbox=bbox,
                            text=text,
                            confidence=conf / 100.0,
                            element_type="text"
                        )
                        elements.append(element)
                        
            except ImportError:
                logger.warning("pytesseract not available for fallback OCR")
            
        except Exception as e:
            logger.error(f"Fallback OCR analysis failed: {e}")
        
        return elements
    
    def extract_text_with_layout(self, image_path: str) -> Dict[str, Any]:
        """Extract text with layout information using OmniParser v2.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary containing text and layout information
        """
        if not self._omniparser_available or not self._initialized:
            return {"text": "", "layout": [], "error": "OmniParser v2 not available"}
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Use OmniParser v2 for layout analysis
            # This is a placeholder - actual implementation would depend on model capabilities
            result = {
                "text": "",
                "layout": [],
                "confidence": 0.0
            }
            
            # Extract text elements
            elements = self._run_omniparser_analysis(image, image_path)
            
            # Build result
            result["text"] = " ".join([elem.text for elem in elements])
            result["layout"] = [
                {
                    "text": elem.text,
                    "bbox": elem.bbox.as_tuple(),
                    "confidence": elem.confidence,
                    "type": elem.element_type
                }
                for elem in elements
            ]
            
            if elements:
                result["confidence"] = sum(elem.confidence for elem in elements) / len(elements)
            
            return result
            
        except Exception as e:
            logger.error(f"Layout extraction failed: {e}")
            return {"text": "", "layout": [], "error": str(e)}
    
    def is_available(self) -> bool:
        """Check if OmniParser v2 is available and initialized.
        
        Returns:
            True if available and initialized, False otherwise
        """
        return self._omniparser_available and self._initialized


# Global instance for reuse
_omniparser_v2_engine = None


def get_omniparser_v2_engine() -> OmniParserV2Engine:
    """Get the global OmniParser v2 engine instance.
    
    Returns:
        OmniParserV2Engine instance
    """
    global _omniparser_v2_engine
    if _omniparser_v2_engine is None:
        _omniparser_v2_engine = OmniParserV2Engine()
    return _omniparser_v2_engine
