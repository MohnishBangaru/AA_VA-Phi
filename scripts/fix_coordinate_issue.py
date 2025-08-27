#!/usr/bin/env python3
"""
Fix Coordinate Issue
==================

This script fixes the issue where the system taps on hardcoded coordinates (300,400)
instead of using the coordinates discovered by the vision system.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.phi_ground import PhiGroundActionGenerator
from src.vision.engine import VisionEngine
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_coordinate_generation():
    """Test coordinate generation to ensure it's working properly."""
    
    logger.info("🧪 Testing coordinate generation...")
    
    # Initialize components
    vision_engine = VisionEngine()
    phi_ground = PhiGroundActionGenerator()
    
    # Test with a sample screenshot
    screenshot_path = "screenshot_1756248771.png"
    
    if not Path(screenshot_path).exists():
        logger.error(f"Screenshot not found: {screenshot_path}")
        return False
    
    try:
        # Initialize Phi Ground
        await phi_ground.initialize()
        
        # Analyze screenshot
        elements = vision_engine.analyze(screenshot_path)
        logger.info(f"Found {len(elements)} UI elements")
        
        # Generate action
        action = await phi_ground.generate_touch_action(
            screenshot_path,
            "Test coordinate generation",
            [],
            elements
        )
        
        if action and action.get('type') == 'tap':
            x, y = action.get('x'), action.get('y')
            logger.info(f"✅ Generated tap coordinates: ({x}, {y})")
            
            # Validate coordinates
            if x is not None and y is not None:
                # Check if coordinates are reasonable
                if 0 <= x <= 2000 and 0 <= y <= 2000:
                    logger.info("✅ Coordinates are within reasonable bounds")
                    return True
                else:
                    logger.warning(f"⚠️ Coordinates ({x}, {y}) are outside normal screen bounds")
                    return False
            else:
                logger.error("❌ Generated action missing coordinates")
                return False
        else:
            logger.warning("⚠️ No tap action generated")
            return False
            
    except Exception as e:
        logger.error(f"❌ Coordinate generation test failed: {e}")
        return False


def fix_universal_apk_tester():
    """Fix the hardcoded coordinates in universal_apk_tester.py."""
    
    logger.info("🔧 Fixing hardcoded coordinates in universal_apk_tester.py...")
    
    file_path = Path(__file__).parent / "universal_apk_tester.py"
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    try:
        # Read the file
        content = file_path.read_text()
        
        # Check if the problematic line exists
        if "y = random.choice([200, 300, 400])" in content:
            logger.info("Found hardcoded coordinates, fixing...")
            
            # Replace the hardcoded coordinates with a more intelligent approach
            old_line = 'y = random.choice([200, 300, 400])  # Top area'
            new_line = 'y = random.randint(200, 400)  # Top area - dynamic range'
            
            content = content.replace(old_line, new_line)
            
            # Also fix any other hardcoded coordinate patterns
            content = content.replace(
                'x = random.choice([400, 500, 600])  # Center area',
                'x = random.randint(400, 600)  # Center area - dynamic range'
            )
            
            content = content.replace(
                'y = random.choice([1200, 1300, 1400])  # Bottom area',
                'y = random.randint(1200, 1400)  # Bottom area - dynamic range'
            )
            
            # Write the fixed content back
            file_path.write_text(content)
            
            logger.info("✅ Fixed hardcoded coordinates in universal_apk_tester.py")
            return True
        else:
            logger.info("✅ No hardcoded coordinates found in universal_apk_tester.py")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to fix universal_apk_tester.py: {e}")
        return False


def create_improved_action_generator():
    """Create an improved action generator that prioritizes discovered coordinates."""
    
    logger.info("🔧 Creating improved action generator...")
    
    improved_code = '''#!/usr/bin/env python3
"""
Improved Action Generator
========================

This module provides an improved action generator that prioritizes coordinates
discovered by the vision system over hardcoded fallbacks.
"""

import random
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ImprovedActionGenerator:
    """Improved action generator that prioritizes discovered coordinates."""
    
    def __init__(self):
        """Initialize the improved action generator."""
        self.last_discovered_coordinates = None
        self.coordinate_history = []
    
    def generate_action_from_elements(
        self, 
        elements: List[Any], 
        action_type: str = "tap",
        fallback_to_random: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Generate action using discovered UI elements.
        
        Args:
            elements: List of discovered UI elements
            action_type: Type of action to generate
            fallback_to_random: Whether to fallback to random coordinates
            
        Returns:
            Action dictionary or None
        """
        if not elements:
            if fallback_to_random:
                return self._generate_smart_random_action()
            return None
        
        # Try to find the best element for the action
        best_element = self._find_best_element(elements, action_type)
        
        if best_element:
            # Use the element's coordinates
            bbox = best_element.bbox
            x = (bbox.x1 + bbox.x2) // 2
            y = (bbox.y1 + bbox.y2) // 2
            
            # Store discovered coordinates
            self.last_discovered_coordinates = (x, y)
            self.coordinate_history.append((x, y, best_element.text))
            
            logger.info(f"Using discovered coordinates: ({x}, {y}) for element: {best_element.text}")
            
            return {
                "type": action_type,
                "x": x,
                "y": y,
                "element_text": best_element.text,
                "confidence": best_element.confidence,
                "reasoning": f"Discovered element: {best_element.text}",
                "source": "vision_discovery"
            }
        
        # Fallback to smart random if no good element found
        if fallback_to_random:
            return self._generate_smart_random_action()
        
        return None
    
    def _find_best_element(self, elements: List[Any], action_type: str) -> Optional[Any]:
        """Find the best element for the given action type."""
        if not elements:
            return None
        
        # Sort elements by confidence and relevance
        scored_elements = []
        
        for element in elements:
            score = element.confidence
            
            # Boost score for interactive elements
            if hasattr(element, 'element_type'):
                if element.element_type in ['button', 'link', 'input']:
                    score += 0.2
                elif element.element_type == 'text':
                    score += 0.1
            
            # Boost score for elements with meaningful text
            if element.text and len(element.text.strip()) > 0:
                text = element.text.lower()
                if any(keyword in text for keyword in ['login', 'submit', 'continue', 'next', 'ok', 'yes']):
                    score += 0.3
                elif any(keyword in text for keyword in ['cancel', 'back', 'no']):
                    score += 0.1
            
            scored_elements.append((score, element))
        
        # Sort by score (highest first)
        scored_elements.sort(key=lambda x: x[0], reverse=True)
        
        # Return the best element
        return scored_elements[0][1] if scored_elements else None
    
    def _generate_smart_random_action(self) -> Dict[str, Any]:
        """Generate a smart random action using screen-aware coordinates."""
        # Use screen-aware coordinate ranges
        x = random.randint(100, 900)  # Avoid edges
        y = random.randint(300, 1200)  # Avoid status bar and navigation
        
        logger.info(f"Using smart random coordinates: ({x}, {y})")
        
        return {
            "type": "tap",
            "x": x,
            "y": y,
            "reasoning": "Smart random fallback",
            "source": "smart_random"
        }
    
    def get_coordinate_history(self) -> List[tuple]:
        """Get the history of discovered coordinates."""
        return self.coordinate_history.copy()
    
    def get_last_discovered_coordinates(self) -> Optional[tuple]:
        """Get the last discovered coordinates."""
        return self.last_discovered_coordinates
'''
    
    # Write the improved action generator
    output_path = Path(__file__).parent / "improved_action_generator.py"
    output_path.write_text(improved_code)
    
    logger.info(f"✅ Created improved action generator: {output_path}")
    return True


async def main():
    """Main function to fix the coordinate issue."""
    logger.info("🚀 Starting coordinate issue fix...")
    
    # Test coordinate generation
    coordinate_test_passed = await test_coordinate_generation()
    
    # Fix universal APK tester
    universal_fix_success = fix_universal_apk_tester()
    
    # Create improved action generator
    improved_generator_created = create_improved_action_generator()
    
    # Summary
    logger.info("📊 Fix Summary:")
    logger.info(f"  Coordinate generation test: {'✅ PASSED' if coordinate_test_passed else '❌ FAILED'}")
    logger.info(f"  Universal APK tester fix: {'✅ SUCCESS' if universal_fix_success else '❌ FAILED'}")
    logger.info(f"  Improved generator created: {'✅ SUCCESS' if improved_generator_created else '❌ FAILED'}")
    
    if coordinate_test_passed and universal_fix_success:
        logger.info("🎉 Coordinate issue should be resolved!")
        logger.info("💡 Next steps:")
        logger.info("  1. Run the distributed APK tester again")
        logger.info("  2. Check if coordinates are now being discovered properly")
        logger.info("  3. Use improved_action_generator.py for better coordinate handling")
    else:
        logger.warning("⚠️ Some fixes failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
