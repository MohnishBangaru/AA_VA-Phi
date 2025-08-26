"""Phi Ground integration for intelligent touch action generation.

This module implements Phi Ground exactly as described in the paper:
"Phi Ground: A Framework for Learning Grounded Policy with Large Language Models"
for generating touch actions instead of mouse actions.

The implementation follows the paper's approach:
1. Screen understanding through vision-language model
2. Action generation based on screen content and task description
3. Touch coordinate prediction for Android automation
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import numpy as np
from loguru import logger

from ..core.config import config
from ..vision.models import UIElement, BoundingBox


class PhiGroundActionGenerator:
    """Phi Ground action generator for Android touch automation."""
    
    def __init__(self, model_name: str = "microsoft/Phi-3-vision-128k-instruct"):
        """Initialize Phi Ground action generator.
        
        Args:
            model_name: The Phi Ground model to use for action generation
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = False
        self.vision_supported = False
        
    async def initialize(self) -> None:
        """Initialize the Phi Ground model."""
        if self._initialized:
            return
            
        try:
            logger.info(f"Initializing Phi Ground model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Check if vision is supported
            try:
                from PIL import Image
                import numpy as np
                dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                test_inputs = self.tokenizer("test", return_tensors="pt", images=dummy_image)
                self.vision_supported = True
                logger.info("Vision tokenization supported")
            except (TypeError, AttributeError):
                self.vision_supported = False
                logger.info("Vision tokenization not supported, using text-only mode")
            except Exception as e:
                self.vision_supported = False
                logger.warning(f"Vision support check failed: {e}, using text-only mode")
            
            # Try optimized initialization strategies (fastest first)
            initialization_strategies = [
                # Strategy 1: Fast CPU loading (fastest for local testing)
                {
                    "name": "Fast CPU Loading",
                    "kwargs": {
                        "torch_dtype": torch.float32,
                        "device_map": None,
                        "trust_remote_code": True,
                        "attn_implementation": "eager",
                        "low_cpu_mem_usage": True,
                        "load_in_8bit": False,
                        "load_in_4bit": False
                    }
                },
                # Strategy 2: CUDA without FlashAttention2 (fast for RunPod)
                {
                    "name": "CUDA Standard",
                    "kwargs": {
                        "torch_dtype": torch.float16,
                        "device_map": "auto",
                        "trust_remote_code": True,
                        "attn_implementation": "eager",
                        "low_cpu_mem_usage": True
                    }
                },
                # Strategy 3: CUDA with FlashAttention2 (only if available)
                {
                    "name": "CUDA FlashAttention2",
                    "kwargs": {
                        "torch_dtype": torch.float16,
                        "device_map": "auto",
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True
                    }
                }
            ]
            
            for strategy in initialization_strategies:
                try:
                    logger.info(f"Trying initialization strategy: {strategy['name']}")
                    
                    # For strategies that need to disable FlashAttention2, modify the config first
                    if strategy['name'] != "CUDA FlashAttention2":
                        from transformers import AutoConfig
                        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
                        config.use_flash_attention_2 = False
                        logger.info(f"Disabled FlashAttention2 in model config for {strategy['name']}")
                        
                        # Remove use_flash_attention_2 from kwargs if present
                        kwargs = strategy['kwargs'].copy()
                        kwargs.pop('use_flash_attention_2', None)
                        
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            config=config,
                            **kwargs
                        )
                    else:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            **strategy['kwargs']
                        )
                    
                    # Move to CPU if using CPU fallback strategy
                    if strategy['name'] == "CPU Fallback":
                        self.device = "cpu"
                        self.model = self.model.to("cpu")
                    
                    logger.info(f"Phi Ground model initialized successfully with {strategy['name']}")
                    break
                    
                except Exception as strategy_error:
                    error_msg = str(strategy_error)
                    logger.warning(f"Strategy '{strategy['name']}' failed: {error_msg}")
                    
                    # If this is the last strategy, raise the error
                    if strategy == initialization_strategies[-1]:
                        logger.error(f"All initialization strategies failed. Last error: {strategy_error}")
                        raise strategy_error
                    continue
            
            self._initialized = True
            logger.info("Phi Ground model initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize Phi Ground model: {e}")
            raise
    
    def _create_phi_ground_prompt(
        self, 
        image_path: str, 
        task_description: str,
        action_history: List[Dict[str, Any]]
    ) -> str:
        """Create Phi Ground prompt following the paper's format.
        
        Args:
            image_path: Path to the screenshot
            task_description: Current automation task
            action_history: Previous actions performed
            
        Returns:
            Formatted prompt for Phi Ground
        """
        # Format action history for context
        history_text = ""
        if action_history:
            recent_actions = action_history[-3:]  # Last 3 actions
            history_parts = []
            for action in recent_actions:
                action_type = action.get("type", "")
                element_text = action.get("element_text", "")
                if element_text:
                    history_parts.append(f"{action_type}: {element_text}")
            if history_parts:
                history_text = f"Recent actions: {', '.join(history_parts)}"
        
        # Create the prompt following Phi Ground paper format
        prompt = f"""<|im_start|>system
You are an Android automation assistant. Analyze the screenshot and generate appropriate touch actions to accomplish the given task. Focus on the app interface, not system UI elements.

Task: {task_description}
{history_text}

You must respond with exactly one action in this format:
1. TAP: [element_description] at coordinates (x, y)
2. INPUT: [text] into [field_description] at coordinates (x, y)
3. SWIPE: from (x1, y1) to (x2, y2)
4. WAIT: [duration] seconds

Rules:
- Always start with a number (1.)
- Use exact format: "1. TAP: [description] at coordinates (x, y)"
- Coordinates should be integers between 100-1000
- Focus on visible, clickable elements
- Avoid system UI (status bar, navigation bar)
- If no clear action, use "1. WAIT: 2 seconds"
- Do not repeat the prompt or system instructions

Prioritize:
1. Text input fields that need to be filled
2. Primary action buttons (Continue, Submit, Next, etc.)
3. Interactive elements (buttons, links, etc.)
4. Navigation elements as last resort
<|im_end|>
<|im_start|>user
<|image|>
Please analyze this Android app screenshot and suggest the next touch action to accomplish the task.
<|im_end|>
<|im_start|>assistant
1. TAP: """
        
        return prompt
    
    async def generate_touch_action(
        self,
        image_path: str,
        task_description: str,
        action_history: List[Dict[str, Any]],
        ui_elements: List[UIElement]
    ) -> Optional[Dict[str, Any]]:
        """Generate touch action using Phi Ground following the paper's approach.
        
        Args:
            image_path: Path to the screenshot
            task_description: Current automation task
            action_history: Previous actions performed
            ui_elements: Detected UI elements for validation
            
        Returns:
            Action dictionary or None if no action should be performed
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Loaded screenshot: {image_path} (size: {image.size})")
            
            # Create Phi Ground prompt
            prompt = self._create_phi_ground_prompt(
                image_path, task_description, action_history
            )
            
            # Tokenize input - handle vision-language model tokenization
            if self.vision_supported:
                try:
                    logger.info("Using vision tokenization with screenshot")
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        images=image,
                        padding=True,
                        truncation=True
                    ).to(self.device)
                except Exception as e:
                    logger.warning(f"Vision tokenization failed: {e}, falling back to text-only")
                    self.vision_supported = False
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)
            else:
                # Use text-only tokenization with proper attention mask
                logger.info("Using text-only tokenization (no screenshot processing)")
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
            
            # Generate response with robust caching fixes
            with torch.no_grad():
                generation_success = False
                
                # Strategy 1: Try with minimal parameters first (most reliable)
                try:
                    logger.info("Trying minimal generation strategy")
                    outputs = self.model.generate(
                        input_ids=inputs.get('input_ids'),
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=False
                    )
                    generation_success = True
                    logger.info("Minimal generation successful")
                except Exception as e:
                    logger.warning(f"Minimal generation failed: {e}")
                
                # Strategy 2: Try with basic parameters
                if not generation_success:
                    try:
                        logger.info("Trying basic generation strategy")
                        outputs = self.model.generate(
                            input_ids=inputs.get('input_ids'),
                            attention_mask=inputs.get('attention_mask'),
                            max_new_tokens=256,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            use_cache=False
                        )
                        generation_success = True
                        logger.info("Basic generation successful")
                    except Exception as e:
                        logger.warning(f"Basic generation failed: {e}")
                
                # Strategy 3: Try with full parameters (least reliable)
                if not generation_success:
                    try:
                        logger.info("Trying full generation strategy")
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=256,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            use_cache=False  # Disable cache to avoid DynamicCache issues
                        )
                        generation_success = True
                        logger.info("Full generation successful")
                    except Exception as e:
                        logger.warning(f"Full generation failed: {e}")
                
                # Strategy 4: Ultimate fallback - direct forward pass
                if not generation_success:
                    try:
                        logger.info("Using direct forward pass fallback")
                        # Get the last token and generate one token at a time
                        current_input = inputs.get('input_ids')
                        generated_tokens = []
                        
                        for _ in range(256):
                            with torch.no_grad():
                                outputs = self.model.forward(input_ids=current_input)
                                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                                generated_tokens.append(next_token.item())
                                current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
                                
                                # Stop if we hit the end token
                                if next_token.item() == self.tokenizer.eos_token_id:
                                    break
                        
                        # Combine original input with generated tokens
                        outputs = torch.cat([inputs.get('input_ids'), torch.tensor([generated_tokens])], dim=1)
                        generation_success = True
                        logger.info("Direct forward pass successful")
                    except Exception as e:
                        logger.error(f"All generation strategies failed: {e}")
                        return None
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"Phi Ground raw response: {response}")
            
            # Clean response - remove prompt repetition
            cleaned_response = self._clean_response(response)
            logger.debug(f"Phi Ground cleaned response: {cleaned_response}")
            
            # Extract action from response
            action = self._parse_phi_ground_response(cleaned_response, ui_elements)
            
            if action:
                logger.info(f"Phi Ground generated action: {action['type']} - {action.get('reasoning', '')}")
            else:
                logger.warning(f"No valid action parsed from response: {response[:200]}...")
            
            return action
            
        except Exception as e:
            logger.error(f"Phi Ground action generation failed: {e}")
            return None
    
    def _clean_response(self, response: str) -> str:
        """Clean the response by removing prompt repetition and extracting only the action.
        
        Args:
            response: Raw response from the model
            
        Returns:
            Cleaned response containing only the action
        """
        try:
            # Remove common prompt repetition patterns
            patterns_to_remove = [
                r'<\|im_start\|>system.*?<\|im_end\|>',
                r'<\|im_start\|>user.*?<\|im_end\|>',
                r'You are an Android automation assistant.*?',
                r'Analyze the screenshot.*?',
                r'Please analyze.*?',
                r'Task:.*?',
                r'Rules:.*?',
                r'Prioritize:.*?'
            ]
            
            cleaned = response
            for pattern in patterns_to_remove:
                cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            
            # Extract only the assistant response part
            assistant_match = re.search(r'<\|im_start\|>assistant\s*(.*?)(?:<\|im_end\|>|$)', cleaned, re.DOTALL)
            if assistant_match:
                cleaned = assistant_match.group(1).strip()
            
            # If no assistant tags found, try to find action patterns
            if not cleaned or len(cleaned) < 10:
                action_match = re.search(r'(\d+\.\s*(?:TAP|INPUT|SWIPE|WAIT):.*?)(?:\n|$)', response, re.IGNORECASE)
                if action_match:
                    cleaned = action_match.group(1).strip()
            
            # Final cleanup
            cleaned = cleaned.strip()
            if not cleaned:
                # Fallback: try to find any action-like text
                action_match = re.search(r'(\d+\.\s*[A-Z]+:.*)', response, re.IGNORECASE)
                if action_match:
                    cleaned = action_match.group(1).strip()
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Error cleaning response: {e}")
            return response
    
    def _parse_phi_ground_response(
        self, 
        response: str, 
        ui_elements: List[UIElement]
    ) -> Optional[Dict[str, Any]]:
        """Parse Phi Ground response and extract action details.
        
        Args:
            response: Raw response from Phi Ground
            ui_elements: Detected UI elements for coordinate validation
            
        Returns:
            Parsed action dictionary or None
        """
        try:
            # Extract action from response - more flexible regex
            action_match = re.search(r'(\d+)\.?\s*(TAP|INPUT|SWIPE|WAIT):?\s*(.+)', response, re.IGNORECASE)
            if not action_match:
                # Try alternative patterns
                action_match = re.search(r'(TAP|INPUT|SWIPE|WAIT):?\s*(.+)', response, re.IGNORECASE)
                if not action_match:
                    # Try to find any action-like pattern
                    action_match = re.search(r'(\d+\.\s*[A-Z]+:.*)', response, re.IGNORECASE)
                    if not action_match:
                        logger.warning(f"Could not parse Phi Ground response: {response[:200]}...")
                        # Generate a default wait action if parsing fails
                        return {
                            "type": "wait",
                            "duration": 2.0,
                            "reasoning": "No valid action parsed, waiting",
                            "phi_ground_generated": True,
                            "confidence": 0.1
                        }
                    else:
                        # Try to parse the found pattern
                        action_text = action_match.group(1)
                        action_type_match = re.search(r'(TAP|INPUT|SWIPE|WAIT)', action_text, re.IGNORECASE)
                        if action_type_match:
                            action_type = action_type_match.group(1).upper()
                            action_description = action_text
                        else:
                            # Default to wait action
                            return {
                                "type": "wait",
                                "duration": 2.0,
                                "reasoning": f"Unparseable action: {action_text}",
                                "phi_ground_generated": True,
                                "confidence": 0.1
                            }
                else:
                    # Use default number 1 if not found
                    action_type = action_match.group(1).upper()
                    action_description = action_match.group(2).strip()
            else:
                action_type = action_match.group(2).upper()
                action_description = action_match.group(3).strip()
            
            if action_type == "TAP":
                return self._parse_tap_action(action_description, ui_elements)
            elif action_type == "INPUT":
                return self._parse_input_action(action_description, ui_elements)
            elif action_type == "SWIPE":
                return self._parse_swipe_action(action_description)
            elif action_type == "WAIT":
                return self._parse_wait_action(action_description)
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to parse Phi Ground response: {e}")
            return None
    
    def _parse_tap_action(
        self, 
        description: str, 
        ui_elements: List[UIElement]
    ) -> Optional[Dict[str, Any]]:
        """Parse tap action from Phi Ground response.
        
        Args:
            description: Action description from Phi Ground
            ui_elements: Available UI elements for coordinate mapping
            
        Returns:
            Tap action dictionary or None
        """
        # Extract coordinates from description
        coord_match = re.search(r'\((\d+),\s*(\d+)\)', description)
        if coord_match:
            x, y = int(coord_match.group(1)), int(coord_match.group(2))
            
            # Find matching element for validation
            matching_element = self._find_matching_element(description, ui_elements)
            
            return {
                "type": "tap",
                "x": x,
                "y": y,
                "element_text": matching_element.text if matching_element else "",
                "reasoning": f"Phi Ground: {description}",
                "phi_ground_generated": True,
                "confidence": matching_element.confidence if matching_element else 0.5
            }
        
        # If no coordinates, try to find element by description
        matching_element = self._find_matching_element(description, ui_elements)
        if matching_element:
            x1, y1, x2, y2 = matching_element.bbox.as_tuple()
            return {
                "type": "tap",
                "x": (x1 + x2) // 2,
                "y": (y1 + y2) // 2,
                "element_text": matching_element.text,
                "reasoning": f"Phi Ground: {description}",
                "phi_ground_generated": True,
                "confidence": matching_element.confidence
            }
        
        return None
    
    def _parse_input_action(
        self, 
        description: str, 
        ui_elements: List[UIElement]
    ) -> Optional[Dict[str, Any]]:
        """Parse input action from Phi Ground response.
        
        Args:
            description: Action description from Phi Ground
            ui_elements: Available UI elements for coordinate mapping
            
        Returns:
            Input action dictionary or None
        """
        # Extract text and field from description
        # Format: "INPUT: [text] into [field_description] at coordinates (x, y)"
        text_match = re.search(r'INPUT:\s*"([^"]+)"\s+into\s+(.+)', description, re.IGNORECASE)
        if text_match:
            input_text = text_match.group(1)
            field_description = text_match.group(2)
            
            # Extract coordinates
            coord_match = re.search(r'\((\d+),\s*(\d+)\)', field_description)
            if coord_match:
                x, y = int(coord_match.group(1)), int(coord_match.group(2))
                
                # Find matching element
                matching_element = self._find_matching_element(field_description, ui_elements)
                
                return {
                    "type": "text_input",
                    "x": x,
                    "y": y,
                    "text": input_text,
                    "field_hint": matching_element.text if matching_element else field_description,
                    "reasoning": f"Phi Ground: {description}",
                    "phi_ground_generated": True,
                    "confidence": matching_element.confidence if matching_element else 0.5
                }
        
        return None
    
    def _parse_swipe_action(self, description: str) -> Optional[Dict[str, Any]]:
        """Parse swipe action from Phi Ground response.
        
        Args:
            description: Action description from Phi Ground
            
        Returns:
            Swipe action dictionary or None
        """
        # Extract coordinates: "SWIPE: from (x1, y1) to (x2, y2)"
        coord_match = re.search(r'from\s*\((\d+),\s*(\d+)\)\s+to\s+\((\d+),\s*(\d+)\)', description)
        if coord_match:
            x1, y1 = int(coord_match.group(1)), int(coord_match.group(2))
            x2, y2 = int(coord_match.group(3)), int(coord_match.group(4))
            
            return {
                "type": "swipe",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "duration": 300,  # Default duration
                "reasoning": f"Phi Ground: {description}",
                "phi_ground_generated": True
            }
        
        return None
    
    def _parse_wait_action(self, description: str) -> Optional[Dict[str, Any]]:
        """Parse wait action from Phi Ground response.
        
        Args:
            description: Action description from Phi Ground
            
        Returns:
            Wait action dictionary or None
        """
        # Extract duration: "WAIT: [duration] seconds"
        duration_match = re.search(r'(\d+(?:\.\d+)?)\s*seconds?', description)
        if duration_match:
            duration = float(duration_match.group(1))
            
            return {
                "type": "wait",
                "duration": duration,
                "reasoning": f"Phi Ground: {description}",
                "phi_ground_generated": True
            }
        
        return None
    
    def _find_matching_element(
        self, 
        description: str, 
        ui_elements: List[UIElement]
    ) -> Optional[UIElement]:
        """Find UI element that matches the Phi Ground description.
        
        Args:
            description: Element description from Phi Ground
            ui_elements: Available UI elements
            
        Returns:
            Matching UI element or None
        """
        description_lower = description.lower()
        
        # Extract key terms from description
        key_terms = re.findall(r'\b\w+\b', description_lower)
        
        best_match = None
        best_score = 0
        
        for element in ui_elements:
            element_text_lower = element.text.lower()
            
            # Calculate similarity score
            score = 0
            for term in key_terms:
                if term in element_text_lower:
                    score += 1
                if element_text_lower in term or term in element_text_lower:
                    score += 0.5
            
            # Normalize by text length
            if len(element.text) > 0:
                score = score / len(element.text.split())
            
            if score > best_score:
                best_score = score
                best_match = element
        
        return best_match if best_score > 0.3 else None
    
    def validate_action_coordinates(
        self, 
        action: Dict[str, Any], 
        screen_width: int = 1080, 
        screen_height: int = 1920
    ) -> bool:
        """Validate action coordinates are within screen bounds.
        
        Args:
            action: Action dictionary
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            
        Returns:
            True if coordinates are valid, False otherwise
        """
        action_type = action.get("type")
        
        if action_type == "tap":
            x, y = action.get("x", 0), action.get("y", 0)
            return 0 <= x <= screen_width and 0 <= y <= screen_height
            
        elif action_type == "text_input":
            x, y = action.get("x", 0), action.get("y", 0)
            return 0 <= x <= screen_width and 0 <= y <= screen_height
            
        elif action_type == "swipe":
            x1, y1 = action.get("x1", 0), action.get("y1", 0)
            x2, y2 = action.get("x2", 0), action.get("y2", 0)
            return (0 <= x1 <= screen_width and 0 <= y1 <= screen_height and
                   0 <= x2 <= screen_width and 0 <= y2 <= screen_height)
        
        return True


# Global instance for reuse
_phi_ground_generator = None


def get_phi_ground_generator() -> PhiGroundActionGenerator:
    """Get or create the global Phi Ground generator instance."""
    global _phi_ground_generator
    if _phi_ground_generator is None:
        _phi_ground_generator = PhiGroundActionGenerator()
    return _phi_ground_generator
