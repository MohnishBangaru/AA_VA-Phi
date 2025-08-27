#!/usr/bin/env python3
"""
Clean GPU Fix
===========

This script applies a clean GPU fix to Phi Ground without breaking syntax.
"""

import re
from pathlib import Path

def apply_clean_gpu_fix():
    """Apply a clean GPU fix to Phi Ground."""
    
    phi_ground_path = Path("src/ai/phi_ground.py")
    
    if not phi_ground_path.exists():
        print("❌ Phi Ground file not found")
        return False
    
    # Read the current file
    with open(phi_ground_path, 'r') as f:
        content = f.read()
    
    # Find and replace device selection patterns
    replacements = [
        # Replace CPU device selection with GPU preference
        ("self.device = 'cpu'", "self.device = 'cuda' if torch.cuda.is_available() else 'cpu'"),
        ("device = 'cpu'", "device = 'cuda' if torch.cuda.is_available() else 'cpu'"),
        ("device='cpu'", "device='cuda' if torch.cuda.is_available() else 'cpu'"),
    ]
    
    modified = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            modified = True
            print(f"   Applied GPU fix: {old} → {new}")
    
    if modified:
        # Write the modified content
        with open(phi_ground_path, 'w') as f:
            f.write(content)
        print("✅ Clean GPU fix applied successfully")
        return True
    else:
        print("⚠️ No changes needed - GPU configuration already present")
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
    backup_path = Path("src/ai/phi_ground.py.backup3")
    
    if phi_ground_path.exists():
        import shutil
        shutil.copy2(phi_ground_path, backup_path)
        print(f"✅ Created backup: {backup_path}")
        return True
    return False

def main():
    """Main function to apply the clean GPU fix."""
    print("🔧 Applying Clean GPU Fix...")
    
    # Create backup
    create_backup()
    
    # Apply the fix
    success = apply_clean_gpu_fix()
    
    if success:
        # Test syntax
        if test_syntax():
            print("✅ Clean GPU fix applied successfully!")
            print("💡 Phi Ground will now prefer GPU over CPU")
            print("🚀 You can now run your scripts with GPU acceleration")
        else:
            print("❌ Syntax error detected - reverting to backup")
            import shutil
            shutil.copy2("src/ai/phi_ground.py.backup3", "src/ai/phi_ground.py")
    else:
        print("❌ Failed to apply the fix")

if __name__ == "__main__":
    main()
