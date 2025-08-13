#!/usr/bin/env python3
"""Example script demonstrating Phi Ground integration with AA_VA.

This script shows how to use Phi Ground for generating touch actions
instead of mouse actions, following the paper's methodology.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.ai.phi_ground import get_phi_ground_generator
from src.vision.models import UIElement, BoundingBox
from src.core.config import config


async def example_phi_ground_usage():
    """Example of using Phi Ground for action generation."""
    
    logger.info("=== Phi Ground Integration Example ===")
    
    # Check if Phi Ground is enabled
    if not config.use_phi_ground:
        logger.warning("Phi Ground is disabled. Set USE_PHI_GROUND=true in your .env file")
        return
    
    try:
        # Initialize Phi Ground
        logger.info("Initializing Phi Ground...")
        phi_ground = get_phi_ground_generator()
        await phi_ground.initialize()
        logger.info("✓ Phi Ground initialized successfully")
        
        # Create sample UI elements (simulating detected elements)
        sample_elements = [
            UIElement(
                bbox=BoundingBox(100, 200, 300, 250),
                text="Login",
                confidence=0.9,
                element_type="button"
            ),
            UIElement(
                bbox=BoundingBox(100, 300, 400, 350),
                text="Email",
                confidence=0.8,
                element_type="input"
            ),
            UIElement(
                bbox=BoundingBox(100, 400, 400, 450),
                text="Password",
                confidence=0.8,
                element_type="input"
            ),
            UIElement(
                bbox=BoundingBox(100, 500, 200, 550),
                text="Forgot Password?",
                confidence=0.7,
                element_type="link"
            )
        ]
        
        # Sample task and action history
        task_description = "Login to the application with test credentials"
        action_history = [
            {"type": "tap", "element_text": "Login", "x": 200, "y": 225}
        ]
        
        logger.info(f"Task: {task_description}")
        logger.info(f"Available elements: {[elem.text for elem in sample_elements]}")
        
        # Example 1: Generate action with screenshot (if available)
        screenshot_path = "example_screenshot.png"
        if os.path.exists(screenshot_path):
            logger.info(f"\n--- Example 1: With Screenshot ---")
            logger.info(f"Using screenshot: {screenshot_path}")
            
            action = await phi_ground.generate_touch_action(
                screenshot_path, task_description, action_history, sample_elements
            )
            
            if action:
                logger.info("✓ Phi Ground generated action:")
                logger.info(f"  Type: {action.get('type')}")
                logger.info(f"  Reasoning: {action.get('reasoning')}")
                logger.info(f"  Confidence: {action.get('confidence', 0.5):.2f}")
                
                if action.get('type') == 'tap':
                    logger.info(f"  Coordinates: ({action.get('x')}, {action.get('y')})")
                elif action.get('type') == 'text_input':
                    logger.info(f"  Text: {action.get('text')}")
                    logger.info(f"  Field: {action.get('field_hint')}")
                
                # Validate coordinates
                if phi_ground.validate_action_coordinates(action):
                    logger.info("  ✓ Coordinates are valid")
                else:
                    logger.warning("  ⚠ Coordinates are invalid")
            else:
                logger.warning("⚠ Phi Ground did not generate an action")
        
        # Example 2: Generate action without screenshot (fallback)
        logger.info(f"\n--- Example 2: Without Screenshot (Fallback) ---")
        logger.info("Testing fallback behavior when no screenshot is available")
        
        action = await phi_ground.generate_touch_action(
            "", task_description, action_history, sample_elements
        )
        
        if action is None:
            logger.info("✓ Correctly returned None when no screenshot available")
        else:
            logger.warning("⚠ Generated action without screenshot (unexpected)")
        
        # Example 3: Different task scenarios
        logger.info(f"\n--- Example 3: Different Task Scenarios ---")
        
        scenarios = [
            {
                "task": "Fill out the registration form",
                "elements": [
                    UIElement(BoundingBox(100, 200, 400, 250), "Full Name", 0.9, "input"),
                    UIElement(BoundingBox(100, 300, 400, 350), "Email", 0.9, "input"),
                    UIElement(BoundingBox(100, 400, 400, 450), "Password", 0.9, "input"),
                    UIElement(BoundingBox(100, 500, 300, 550), "Register", 0.9, "button")
                ]
            },
            {
                "task": "Search for products",
                "elements": [
                    UIElement(BoundingBox(50, 100, 350, 150), "Search", 0.8, "input"),
                    UIElement(BoundingBox(400, 100, 450, 150), "🔍", 0.9, "button"),
                    UIElement(BoundingBox(100, 200, 300, 250), "Categories", 0.7, "button")
                ]
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"\nScenario {i}: {scenario['task']}")
            logger.info(f"Elements: {[elem.text for elem in scenario['elements']]}")
            
            # Simulate Phi Ground analysis (without actual screenshot)
            logger.info("(Simulating Phi Ground analysis...)")
            
            # In a real scenario, this would generate an action
            # For demonstration, we'll show what the action might look like
            if "registration" in scenario['task'].lower():
                logger.info("Expected action: Fill 'Full Name' field with test data")
            elif "search" in scenario['task'].lower():
                logger.info("Expected action: Tap 'Search' field and enter query")
        
        logger.info(f"\n=== Example Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise


async def example_integration_with_action_determiner():
    """Example of using Phi Ground with the action determiner."""
    
    logger.info("\n=== Integration with Action Determiner Example ===")
    
    try:
        from src.ai.action_determiner import get_enhanced_action_determiner
        
        # Get action determiner
        determiner = get_enhanced_action_determiner()
        
        # Sample UI elements
        ui_elements = [
            UIElement(BoundingBox(100, 200, 300, 250), "Login", 0.9, "button"),
            UIElement(BoundingBox(100, 300, 400, 350), "Email", 0.8, "input"),
            UIElement(BoundingBox(100, 400, 400, 450), "Password", 0.8, "input")
        ]
        
        task_description = "Login to the application"
        action_history = []
        device_info = {"screen_width": 1080, "screen_height": 1920}
        
        # Determine next action (Phi Ground will be used if screenshot available)
        screenshot_path = "example_screenshot.png" if os.path.exists("example_screenshot.png") else None
        
        logger.info("Determining next action...")
        action = await determiner.determine_next_action(
            ui_elements=ui_elements,
            task_description=task_description,
            action_history=action_history,
            device_info=device_info,
            screenshot_path=screenshot_path
        )
        
        if action:
            logger.info("✓ Action determined:")
            logger.info(f"  Type: {action.get('type')}")
            logger.info(f"  Reasoning: {action.get('reasoning')}")
            
            if action.get('phi_ground_generated'):
                logger.info("  ✓ Generated by Phi Ground")
            else:
                logger.info("  ✓ Generated by traditional method")
        else:
            logger.warning("⚠ No action determined")
        
        logger.info("=== Integration Example Completed ===")
        
    except Exception as e:
        logger.error(f"❌ Integration example failed: {e}")
        raise


async def main():
    """Main example function."""
    logger.info("Starting Phi Ground integration examples...")
    
    # Example 1: Basic Phi Ground usage
    await example_phi_ground_usage()
    
    # Example 2: Integration with action determiner
    await example_integration_with_action_determiner()
    
    logger.info("\n🎉 All examples completed successfully!")
    logger.info("\nTo use Phi Ground in your own scripts:")
    logger.info("1. Set USE_PHI_GROUND=true in your .env file")
    logger.info("2. Ensure you have sufficient RAM (8-16GB recommended)")
    logger.info("3. GPU recommended for better performance")
    logger.info("4. Pass screenshot_path to action determination methods")


if __name__ == "__main__":
    asyncio.run(main())
