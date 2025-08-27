#!/usr/bin/env python3
"""
Simple Vision Tokenization Fix
============================

This script adds missing vision tokenization approaches to the existing structure.
"""

import re
from pathlib import Path

def add_vision_approaches():
    """Add missing vision tokenization approaches."""
    
    phi_ground_path = Path("src/ai/phi_ground.py")
    
    if not phi_ground_path.exists():
        print("❌ Phi Ground file not found")
        return False
    
    # Read the current file
    with open(phi_ground_path, 'r') as f:
        content = f.read()
    
    # Find where to add the additional approaches
    pattern = r'(                        except Exception as e2:\n                        logger\.debug\(f"Model image processor approach failed: \{e2\}"\)\n)'
    
    # Additional approaches to add
    additional_approaches = '''                        except Exception as e2:
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
'''
    
    # Replace the pattern
    new_content = re.sub(pattern, additional_approaches, content)
    
    if new_content == content:
        print("⚠️ No changes made - pattern not found")
        return False
    
    # Write the fixed content
    with open(phi_ground_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Successfully added vision tokenization approaches")
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
    print("🔧 Adding vision tokenization approaches...")
    
    # Apply the fix
    success = add_vision_approaches()
    
    if success:
        # Test syntax
        if test_syntax():
            print("✅ Vision tokenization approaches added successfully!")
            print("🚀 You can now run your scripts normally")
        else:
            print("❌ Syntax error detected - reverting to backup")
            import shutil
            shutil.copy2("src/ai/phi_ground.py.backup", "src/ai/phi_ground.py")
    else:
        print("❌ Failed to apply the fix")


if __name__ == "__main__":
    main()
