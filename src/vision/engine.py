"""Vision engine for screenshot analysis and UI element detection."""

from __future__ import annotations

import concurrent.futures
import os

import cv2  # type: ignore
from loguru import logger

from core.config import config
from vision.models import BoundingBox, UIElement
from vision.omniparser_v2 import get_omniparser_v2_engine


class VisionEngine:
    """Analyze screenshots and return detected UI elements using OCR."""

    def __init__(self) -> None:
        """Initialize VisionEngine with OmniParser v2."""
        # Use a thread pool for OCR to avoid blocking event loop.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # Check if OCR should be enabled and OmniParser v2 is available
        self._ocr_available: bool = False
        self._omniparser_engine = None
        self._tesseract_available = False
        
        if not config.use_ocr:
            logger.warning("VisionEngine: OCR disabled via configuration (USE_OCR=false)")
        else:
            # Try OmniParser v2 first if enabled
            if config.use_omniparser_v2:
                self._omniparser_engine = get_omniparser_v2_engine()
                if self._omniparser_engine.is_available():
                    self._ocr_available = True
                    logger.info("VisionEngine: OmniParser v2 initialized successfully")
                else:
                    logger.warning("VisionEngine: OmniParser v2 not available, trying Tesseract fallback")
            
            # Fallback to Tesseract if OmniParser v2 is not available or disabled
            if not self._ocr_available:
                try:
                    import pytesseract
                    pytesseract.get_tesseract_version()
                    self._tesseract_available = True
                    self._ocr_available = True
                    logger.info("VisionEngine: Tesseract fallback initialized successfully")
                except Exception as e:
                    logger.warning(f"VisionEngine: Tesseract fallback not available: {e}")
                    logger.warning("VisionEngine: No OCR engine available. OCR will be skipped.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, image_path: str) -> list[UIElement]:
        """Return list of detected `UIElement` objects from the screenshot using OmniParser v2."""
        if not self._ocr_available:
            return []  # OCR disabled, return empty list

        if not os.path.exists(image_path):
            logger.error(f"Screenshot not found: {image_path}")
            return []

        # Use OmniParser v2 or Tesseract for text extraction
        elements = []
        if self._omniparser_engine and self._omniparser_engine.is_available():
            try:
                elements = self._omniparser_engine.analyze(image_path)
                logger.debug(f"OmniParser v2 detected {len(elements)} text elements")
            except Exception as e:
                logger.error(f"OmniParser v2 analysis failed: {e}")
                elements = []
        elif self._tesseract_available:
            try:
                elements = self._run_tesseract_analysis(image_path)
                logger.debug(f"Tesseract detected {len(elements)} text elements")
            except Exception as e:
                logger.error(f"Tesseract analysis failed: {e}")
                elements = []

        # Load image for additional processing
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return elements

        # Pre-process: convert to grayscale for template matching
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Template matching (non-text elements)
        try:
            from .template_matcher import match_templates
            tmpl_elements = match_templates(gray)
            elements.extend(tmpl_elements)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Template matching failed: {exc}")

        # Clickable element detection
        try:
            from .clickable_detector import detect_clickable_elements
            clickable_elements = detect_clickable_elements(image_path)
            elements.extend(clickable_elements)
            logger.debug(f"VisionEngine detected {len(clickable_elements)} clickable elements")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Clickable detection failed: {exc}")

        # Filter for interactive elements only
        interactive_elements = self._filter_interactive_elements(elements)
        
        # Save debug overlay if enabled
        try:
            from .debug import save_debug_overlay
            save_debug_overlay(image_path, interactive_elements)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to save debug overlay: {exc}")

        logger.debug(f"VisionEngine total interactive elements: {len(interactive_elements)}")
        return interactive_elements
    
    def _filter_interactive_elements(self, elements: list[UIElement]) -> list[UIElement]:
        """Filter elements to only include interactive/clickable ones.
        
        Parameters
        ----------
        elements : list[UIElement]
            List of all detected elements
            
        Returns
        -------
        list[UIElement]
            List of interactive elements only
        """
        interactive_elements = []
        
        for element in elements:
            # Include elements that are likely interactive
            if element.element_type in ["button", "colored_button", "edge_bounded", "template"]:
                # Only include if they meet size criteria
                bbox = element.bbox
                if 30 <= bbox.width() <= 300 and 30 <= bbox.height() <= 200:
                    interactive_elements.append(element)
            elif element.element_type == "text":
                # For text elements, check if they might be clickable
                # Look for common interactive text patterns
                text_lower = element.text.lower()
                interactive_keywords = [
                    "button", "click", "tap", "press", "select", "choose",
                    "ok", "cancel", "yes", "no", "save", "delete", "edit",
                    "add", "remove", "next", "back", "continue", "skip",
                    "login", "sign", "register", "submit", "send", "search",
                    "menu", "settings", "profile", "help", "close", "exit"
                ]
                
                # Check if text contains interactive keywords
                if any(keyword in text_lower for keyword in interactive_keywords):
                    interactive_elements.append(element)
                # Check if text is short and in button-like size (likely a button label)
                elif len(element.text) <= 15 and element.confidence > 0.8:
                    # Check if it's in a reasonable size range for a button
                    bbox = element.bbox
                    if 30 <= bbox.width() <= 200 and 20 <= bbox.height() <= 80:
                        interactive_elements.append(element)
        
        # Remove duplicates and overlapping elements
        interactive_elements = self._remove_overlapping_elements(interactive_elements)
        
        logger.debug(f"Filtered {len(elements)} total elements to {len(interactive_elements)} interactive elements")
        return interactive_elements
    
    def _remove_overlapping_elements(self, elements: list[UIElement]) -> list[UIElement]:
        """Remove overlapping elements, keeping the most likely interactive ones."""
        if not elements:
            return []
        
        # Sort by confidence and element type priority
        def element_priority(element: UIElement) -> int:
            priority_map = {
                "button": 1,
                "colored_button": 2,
                "text": 3,
                "edge_bounded": 4,
                "template": 5
            }
            return priority_map.get(element.element_type, 6)
        
        elements = sorted(elements, key=lambda e: (element_priority(e), e.confidence), reverse=True)
        kept = []
        
        for element in elements:
            # Check if this element overlaps significantly with any kept element
            is_duplicate = False
            for kept_element in kept:
                iou = self._calculate_iou(element.bbox, kept_element.bbox)
                if iou > 0.3:  # More than 30% overlap
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append(element)
        
        return kept
    
    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        # Calculate intersection
        x_left = max(bbox1.left, bbox2.left)
        y_top = max(bbox1.top, bbox2.top)
        x_right = min(bbox1.right, bbox2.right)
        y_bottom = min(bbox1.bottom, bbox2.bottom)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        bbox1_area = bbox1.width() * bbox1.height()
        bbox2_area = bbox2.width() * bbox2.height()
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _run_tesseract_analysis(self, image_path: str) -> list[UIElement]:
        """Run Tesseract OCR analysis on the image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            List of UIElement objects
        """
        try:
            import pytesseract
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold for better text detection
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # Run OCR
            ocr_data = pytesseract.image_to_data(
                thresh,
                lang='eng',
                output_type=pytesseract.Output.DICT
            )
            
            # Process results
            elements = []
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
            
            return elements
            
        except Exception as e:
            logger.error(f"Tesseract analysis failed: {e}")
            return []
    
   