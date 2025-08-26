#!/usr/bin/env python3
"""
Distributed Universal APK Tester for RunPod + Local Emulator
============================================================

This script runs on RunPod and communicates with your local laptop's ADB server
to test Android APKs. The AI processing happens on RunPod while device interaction
happens on your local emulator.

Usage:
    python scripts/distributed_apk_tester.py --apk /path/to/app.apk --local-server http://YOUR_LAPTOP_IP:8000
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.distributed_device_manager import DistributedDeviceManager, test_distributed_connection
from src.core.distributed_config import distributed_config
from src.vision.engine import VisionEngine
from src.ai.phi_ground import PhiGroundActionGenerator
from src.ai.openai_client import OpenAIClient
from src.automation.action_executor import ActionExecutor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DistributedAPKTester:
    """Distributed APK tester for RunPod + Local Emulator setup."""
    
    def __init__(self, apk_path: str, local_server_url: str, output_dir: str = "test_reports"):
        """Initialize distributed APK tester."""
        self.apk_path = apk_path
        self.local_server_url = local_server_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.device_manager = None
        self.vision_engine = None
        self.phi_ground = None
        self.openai_client = None
        self.action_executor = None
        
        # Test results
        self.test_results = {
            "start_time": None,
            "end_time": None,
            "actions_performed": 0,
            "screenshots_taken": 0,
            "errors": [],
            "success": False
        }
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Distributed APK Tester...")
        
        # Test connection to local ADB server
        if not await test_distributed_connection(self.local_server_url):
            raise ConnectionError(f"Cannot connect to local ADB server at {self.local_server_url}")
        
        # Initialize device manager
        self.device_manager = DistributedDeviceManager(self.local_server_url)
        
        # Initialize vision engine
        self.vision_engine = VisionEngine()
        
        # Initialize AI components
        try:
            self.phi_ground = PhiGroundActionGenerator()
            await self.phi_ground.initialize()
            logger.info("Phi Ground initialized successfully")
        except Exception as e:
            logger.warning(f"Phi Ground initialization failed: {e}")
            self.phi_ground = None
        
        try:
            self.openai_client = OpenAIClient()
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}")
            self.openai_client = None
        
        # Initialize action executor
        self.action_executor = ActionExecutor(self.device_manager)
        
        logger.info("Distributed APK Tester initialized successfully")
    
    async def run_test(self, num_actions: int = 10):
        """Run the distributed APK test."""
        logger.info(f"Starting distributed APK test with {num_actions} actions")
        self.test_results["start_time"] = time.time()
        
        try:
            # Wait for device
            if not self.device_manager.wait_for_device():
                raise RuntimeError("Device not available")
            
            # Optimize device settings
            self.device_manager.optimize_device_settings()
            
            # Install APK
            if not self.device_manager.install_apk(self.apk_path):
                raise RuntimeError("Failed to install APK")
            
            # Get package name from APK
            package_name = self._extract_package_name()
            
            # Launch app
            if not self.device_manager.launch_app(package_name):
                raise RuntimeError("Failed to launch app")
            
            # Wait for app to stabilize
            await asyncio.sleep(3)
            
            # Take initial screenshot
            initial_screenshot = self.device_manager.take_screenshot(
                str(self.output_dir / "initial_screenshot.png")
            )
            if initial_screenshot:
                self.test_results["screenshots_taken"] += 1
            
            # Perform actions
            for i in range(num_actions):
                logger.info(f"Performing action {i+1}/{num_actions}")
                
                try:
                    # Take screenshot for analysis
                    screenshot_path = str(self.output_dir / f"action_{i+1}_screenshot.png")
                    screenshot = self.device_manager.take_screenshot(screenshot_path)
                    
                    if screenshot:
                        self.test_results["screenshots_taken"] += 1
                        
                        # Analyze screenshot with vision engine
                        elements = self.vision_engine.analyze_screenshot(screenshot_path)
                        
                        # Generate action using AI
                        action = await self._generate_action(screenshot_path, elements)
                        
                        if action:
                            # Execute action
                            success = await self.action_executor.execute_action(action)
                            if success:
                                self.test_results["actions_performed"] += 1
                            
                            # Wait for action to complete
                            await asyncio.sleep(1)
                        else:
                            logger.warning("No action generated, skipping")
                    else:
                        logger.error("Failed to take screenshot")
                        
                except Exception as e:
                    logger.error(f"Error during action {i+1}: {e}")
                    self.test_results["errors"].append(f"Action {i+1}: {str(e)}")
            
            # Take final screenshot
            final_screenshot = self.device_manager.take_screenshot(
                str(self.output_dir / "final_screenshot.png")
            )
            if final_screenshot:
                self.test_results["screenshots_taken"] += 1
            
            # Uninstall app
            self.device_manager.uninstall_app(package_name)
            
            self.test_results["success"] = True
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            self.test_results["errors"].append(f"Test failure: {str(e)}")
        
        finally:
            self.test_results["end_time"] = time.time()
            await self._save_results()
    
    async def _generate_action(self, screenshot_path: str, elements: list) -> Optional[dict]:
        """Generate action using AI."""
        try:
            if self.phi_ground:
                # Use Phi Ground for action generation
                action = await self.phi_ground.generate_action(screenshot_path, elements)
                return action
            elif self.openai_client:
                # Fallback to OpenAI
                action = await self.openai_client.generate_action(screenshot_path, elements)
                return action
            else:
                # Random action as last resort
                return self._generate_random_action()
        except Exception as e:
            logger.error(f"Failed to generate action: {e}")
            return self._generate_random_action()
    
    def _generate_random_action(self) -> dict:
        """Generate a random action."""
        import random
        
        action_types = ["tap", "swipe", "keyevent"]
        action_type = random.choice(action_types)
        
        if action_type == "tap":
            return {
                "type": "tap",
                "x": random.randint(100, 800),
                "y": random.randint(200, 1200)
            }
        elif action_type == "swipe":
            return {
                "type": "swipe",
                "start_x": random.randint(100, 400),
                "start_y": random.randint(200, 600),
                "end_x": random.randint(500, 800),
                "end_y": random.randint(700, 1200),
                "duration": random.randint(300, 800)
            }
        else:  # keyevent
            return {
                "type": "keyevent",
                "key_code": random.choice([4, 24, 25])  # Back, Volume Up, Volume Down
            }
    
    def _extract_package_name(self) -> str:
        """Extract package name from APK path."""
        # This is a simplified version - you might want to use a proper APK parser
        apk_name = Path(self.apk_path).stem
        if "com." in apk_name:
            # Extract package name from filename
            parts = apk_name.split("_")
            for part in parts:
                if part.startswith("com."):
                    return part
        return "com.example.app"  # Fallback
    
    async def _save_results(self):
        """Save test results."""
        import json
        
        results_file = self.output_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Generate summary
        duration = self.test_results["end_time"] - self.test_results["start_time"]
        logger.info(f"Test completed in {duration:.2f} seconds")
        logger.info(f"Actions performed: {self.test_results['actions_performed']}")
        logger.info(f"Screenshots taken: {self.test_results['screenshots_taken']}")
        logger.info(f"Errors: {len(self.test_results['errors'])}")
        logger.info(f"Success: {self.test_results['success']}")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Distributed Universal APK Tester")
    parser.add_argument("--apk", required=True, help="Path to APK file")
    parser.add_argument("--local-server", required=True, help="Local ADB server URL (e.g., http://192.168.1.100:8000)")
    parser.add_argument("--output-dir", default="test_reports", help="Output directory for results")
    parser.add_argument("--actions", type=int, default=10, help="Number of actions to perform")
    
    args = parser.parse_args()
    
    # Validate APK file
    if not Path(args.apk).exists():
        logger.error(f"APK file not found: {args.apk}")
        sys.exit(1)
    
    # Create and run tester
    tester = DistributedAPKTester(args.apk, args.local_server, args.output_dir)
    
    try:
        await tester.initialize()
        await tester.run_test(args.actions)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
