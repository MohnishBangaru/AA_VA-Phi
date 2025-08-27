#!/usr/bin/env python3
"""
Simple Device Fix
================

This script only changes the device selection to use GPU without modifying other code.
"""

import re
from pathlib import Path

def fix_device_selection():
    """Fix device selection to use GPU."""
    
    phi_ground_path = Path("src/ai/phi_ground.py")
    
    if not phi_ground_path.exists():
        print("❌ Phi Ground file not found")
        return False
    
    # Read the current file
    with open(phi_ground_path, 'r') as f:
        content = f.read()
    
    # Find the specific line that sets device to CPU and replace it
    # Look for the line in the initialization method
    pattern = r'(\s+)self\.device = "cpu"'
    replacement = r'\1self.device = "cuda" if torch.cuda.is_available() else "cpu"'
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content == content:
        print("⚠️ No device selection found to replace")
        return False
    
    # Write the modified content
    with open(phi_ground_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Device selection fixed to use GPU")
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

def create_backup():
    """Create a backup of the current file."""
    phi_ground_path = Path("src/ai/phi_ground.py")
    backup_path = Path("src/ai/phi_ground.py.backup4")
    
    if phi_ground_path.exists():
        import shutil
        shutil.copy2(phi_ground_path, backup_path)
        print(f"✅ Created backup: {backup_path}")
        return True
    return False

def main():
    """Main function to apply the simple device fix."""
    print("🔧 Applying Simple Device Fix...")
    
    # Create backup
    create_backup()
    
    # Apply the fix
    success = fix_device_selection()
    
    if success:
        # Test syntax
        if test_syntax():
            print("✅ Simple device fix applied successfully!")
            print("💡 Phi Ground will now use GPU if available")
            print("🚀 You can now run your scripts")
        else:
            print("❌ Syntax error detected - reverting to backup")
            import shutil
            shutil.copy2("src/ai/phi_ground.py.backup4", "src/ai/phi_ground.py")
    else:
        print("❌ Failed to apply the fix")

if __name__ == "__main__":
    main()
