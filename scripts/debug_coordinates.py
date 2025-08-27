#!/usr/bin/env python3
"""
Debug Coordinate Issue
=====================

This script helps debug why the system is tapping on hardcoded coordinates (300,400)
instead of the coordinates discovered by the vision system.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.phi_ground import PhiGroundActionGenerator
from src.vision.engine import VisionEngine
from src.core.distributed_device_manager import DistributedDeviceManager
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def debug_coordinate_flow():
    """Debug the coordinate flow from vision to action execution."""
    
    logger.info("🔍 Starting coordinate debugging...")
    
    # Initialize components
    vision_engine = VisionEngine()
    phi_ground = PhiGroundActionGenerator()
    
    # Test with a sample screenshot
    screenshot_path = "screenshot_1756248771.png"  # Use existing screenshot
    
    if not Path(screenshot_path).exists():
        logger.error(f"Screenshot not found: {screenshot_path}")
        return
    
    logger.info(f"📸 Analyzing screenshot: {screenshot_path}")
    
    # Step 1: Analyze screenshot with vision engine
    try:
        elements = vision_engine.analyze(screenshot_path)
        logger.info(f"✅ Vision engine found {len(elements)} elements:")
        
        for i, element in enumerate(elements[:5]):  # Show first 5 elements
            bbox = element.bbox
            logger.info(f"  Element {i+1}: '{element.text}' at ({bbox.x1}, {bbox.y1}) to ({bbox.x2}, {bbox.y2})")
            
    except Exception as e:
        logger.error(f"❌ Vision engine analysis failed: {e}")
        return
    
    # Step 2: Generate action with Phi Ground
    try:
        await phi_ground.initialize()
        
        action = await phi_ground.generate_touch_action(
            screenshot_path,
            "Debug coordinate testing",
            [],  # Empty action history
            elements
        )
        
        if action:
            logger.info(f"✅ Phi Ground generated action:")
            logger.info(f"  Type: {action.get('type')}")
            logger.info(f"  Coordinates: ({action.get('x')}, {action.get('y')})")
            logger.info(f"  Reasoning: {action.get('reasoning', 'N/A')}")
            logger.info(f"  Confidence: {action.get('confidence', 'N/A')}")
            
            # Validate coordinates
            if action.get('type') == 'tap':
                x, y = action.get('x'), action.get('y')
                if x is not None and y is not None:
                    logger.info(f"  📍 Tap coordinates: ({x}, {y})")
                    
                    # Check if coordinates match any detected elements
                    matching_element = None
                    for element in elements:
                        bbox = element.bbox
                        if (bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2):
                            matching_element = element
                            break
                    
                    if matching_element:
                        logger.info(f"  ✅ Coordinates match element: '{matching_element.text}'")
                    else:
                        logger.warning(f"  ⚠️ Coordinates ({x}, {y}) don't match any detected element")
                else:
                    logger.error(f"  ❌ Missing coordinates in tap action")
        else:
            logger.warning("⚠️ Phi Ground did not generate an action")
            
    except Exception as e:
        logger.error(f"❌ Phi Ground action generation failed: {e}")
        return
    
    # Step 3: Test action execution (if coordinates look good)
    if action and action.get('type') == 'tap' and action.get('x') is not None:
        logger.info("🧪 Testing action execution...")
        
        # Note: This would require a connected device
        # For now, just log what would be executed
        logger.info(f"  Would execute: tap({action.get('x')}, {action.get('y')})")
        
        # Check if coordinates are reasonable
        x, y = action.get('x'), action.get('y')
        if 0 <= x <= 2000 and 0 <= y <= 2000:  # Reasonable screen bounds
            logger.info(f"  ✅ Coordinates are within reasonable bounds")
        else:
            logger.warning(f"  ⚠️ Coordinates ({x}, {y}) seem outside normal screen bounds")


def check_for_hardcoded_coordinates():
    """Check the codebase for any hardcoded coordinate usage."""
    logger.info("🔍 Checking for hardcoded coordinates in codebase...")
    
    # Common patterns to check
    patterns = [
        ("300, 400", "Hardcoded coordinates 300,400"),
        ("300,400", "Hardcoded coordinates 300,400"),
        ("x=300", "Hardcoded x=300"),
        ("y=400", "Hardcoded y=400"),
        ("random.randint(300", "Random range starting at 300"),
        ("random.choice([300", "Random choice including 300"),
    ]
    
    codebase_path = Path(__file__).parent.parent
    found_issues = []
    
    for pattern, description in patterns:
        for py_file in codebase_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                if pattern in content:
                    found_issues.append(f"{description} in {py_file.relative_to(codebase_path)}")
            except Exception as e:
                logger.warning(f"Could not read {py_file}: {e}")
    
    if found_issues:
        logger.warning("⚠️ Found potential hardcoded coordinate issues:")
        for issue in found_issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("✅ No obvious hardcoded coordinate issues found")


async def main():
    """Main debugging function."""
    logger.info("🚀 Starting coordinate debugging session")
    
    # Check for hardcoded coordinates
    check_for_hardcoded_coordinates()
    
    # Debug the coordinate flow
    await debug_coordinate_flow()
    
    logger.info("🏁 Coordinate debugging completed")


if __name__ == "__main__":
    asyncio.run(main())
