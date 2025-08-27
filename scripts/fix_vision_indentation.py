#!/usr/bin/env python3
"""
Fix Vision Tokenization with Proper Indentation
==============================================

This script fixes the vision tokenization issues in Phi Ground with proper indentation.
"""

import re
from pathlib import Path

def fix_vision_tokenization():
    """Fix vision tokenization with proper indentation."""
    
    phi_ground_path = Path("src/ai/phi_ground.py")
    
    if not phi_ground_path.exists():
        print("❌ Phi Ground file not found")
        return False
    
    # Read the current file
    with open(phi_ground_path, 'r') as f:
        content = f.read()
    
    # Find the vision tokenization section
    pattern = r'(# Tokenize input - handle vision-language model tokenization.*?)(else:)'
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print("❌ Vision tokenization section not found")
        return False
    
    # Create the improved vision tokenization code with proper indentation
    improved_vision_code = '''# Tokenize input - handle vision-language model tokenization
            if self.vision_supported:
                try:
                    logger.info("Using vision tokenization with screenshot")
                    
                    # Try multiple vision tokenization approaches
                    vision_inputs = None
                    
                    # Approach 1: Try with AutoImageProcessor
                    try:
                        from transformers import AutoImageProcessor
                        image_processor = AutoImageProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                        processed_image = image_processor(image, return_tensors="pt")
                        vision_inputs = self.tokenizer(
                            prompt,
                            return_tensors="pt",
                            **processed_image,
                            padding=True,
                            truncation=True
                        )
                        # Move to correct device
                        vision_inputs = {k: v.to(self.device) for k, v in vision_inputs.items()}
                        logger.info("Vision tokenization successful with AutoImageProcessor")
                    except Exception as e1:
                        logger.debug(f"AutoImageProcessor approach failed: {e1}")
                        
                        # Approach 2: Try with model's image processor
                        try:
                            if hasattr(self.tokenizer, 'image_processor'):
                                processed_image = self.tokenizer.image_processor(image, return_tensors="pt")
                                vision_inputs = self.tokenizer(
                                    prompt,
                                    return_tensors="pt",
                                    **processed_image,
                                    padding=True,
                                    truncation=True
                                )
                                # Move to correct device
                                vision_inputs = {k: v.to(self.device) for k, v in vision_inputs.items()}
                                logger.info("Vision tokenization successful with model's image processor")
                        except Exception as e2:
                            logger.debug(f"Model image processor approach failed: {e2}")
                            
                            # Approach 3: Try direct images parameter
                            try:
                                vision_inputs = self.tokenizer(
                                    prompt,
                                    images=image,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True
                                )
                                # Move to correct device
                                vision_inputs = {k: v.to(self.device) for k, v in vision_inputs.items()}
                                logger.info("Vision tokenization successful with direct images parameter")
                            except Exception as e3:
                                logger.debug(f"Direct images parameter approach failed: {e3}")
                                
                                # Approach 4: Manual image processing
                                try:
                                    # Resize image to standard size
                                    image_resized = image.resize((224, 224))
                                    
                                    # Convert to tensor
                                    import torchvision.transforms as transforms
                                    transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
                                    
                                    image_tensor = transform(image_resized).unsqueeze(0)
                                    
                                    # Create inputs with image
                                    vision_inputs = self.tokenizer(
                                        prompt,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True
                                    )
                                    
                                    # Add image tensor
                                    vision_inputs['pixel_values'] = image_tensor.to(self.device)
                                    logger.info("Vision tokenization successful with manual processing")
                                except Exception as e4:
                                    logger.debug(f"Manual processing approach failed: {e4}")
                    
                    if vision_inputs is not None:
                        inputs = vision_inputs
                    else:
                        raise Exception("All vision tokenization approaches failed")
                        
                except Exception as e:
                    logger.warning(f"Vision tokenization failed: {e}, falling back to text-only")
                    self.vision_supported = False
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    # Move to correct device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:'''
    
    # Replace the section
    new_content = re.sub(pattern, improved_vision_code, content, flags=re.DOTALL)
    
    if new_content == content:
        print("⚠️ No changes made - pattern not found")
        return False
    
    # Write the fixed content
    with open(phi_ground_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Successfully fixed vision tokenization with proper indentation")
    return True


def test_syntax():
    """Test if the file has correct syntax."""
    try:
        import ast
        with open("src/ai/phi_ground.py", 'r') as f:
            content = f.read()
        ast.parse(content)
        print("✅ Syntax check passed")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False


def main():
    """Main function to apply the fix."""
    print("🔧 Fixing vision tokenization with proper indentation...")
    
    # Apply the fix
    success = fix_vision_tokenization()
    
    if success:
        # Test syntax
        if test_syntax():
            print("✅ Vision tokenization fix applied successfully!")
            print("🚀 You can now run your scripts normally")
        else:
            print("❌ Syntax error detected - reverting to backup")
            import shutil
            shutil.copy2("src/ai/phi_ground.py.backup", "src/ai/phi_ground.py")
    else:
        print("❌ Failed to apply the fix")


if __name__ == "__main__":
    main()
