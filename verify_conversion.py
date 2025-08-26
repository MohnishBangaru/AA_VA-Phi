#!/usr/bin/env python3
"""
Verification script to ensure all notebooks were created properly.
"""

import json
import os
from pathlib import Path

def verify_notebook_structure(notebook_path: str) -> bool:
    """
    Verify that a notebook has proper structure.
    
    Args:
        notebook_path: Path to the notebook file
        
    Returns:
        True if notebook is valid, False otherwise
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Check required fields
        required_fields = ['cells', 'metadata', 'nbformat', 'nbformat_minor']
        for field in required_fields:
            if field not in notebook:
                print(f"❌ {notebook_path}: Missing required field '{field}'")
                return False
        
        # Check cells
        if not isinstance(notebook['cells'], list):
            print(f"❌ {notebook_path}: 'cells' is not a list")
            return False
        
        # Check metadata
        if not isinstance(notebook['metadata'], dict):
            print(f"❌ {notebook_path}: 'metadata' is not a dict")
            return False
        
        # Check kernel spec
        if 'kernelspec' not in notebook['metadata']:
            print(f"❌ {notebook_path}: Missing kernelspec in metadata")
            return False
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ {notebook_path}: Invalid JSON - {e}")
        return False
    except Exception as e:
        print(f"❌ {notebook_path}: Error reading file - {e}")
        return False

def verify_all_notebooks():
    """
    Verify all notebooks in the repository.
    """
    print("🔍 Verifying notebook conversion...")
    
    # Find all notebooks
    notebooks = []
    for root, dirs, files in os.walk('.'):
        # Skip .git and other hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith('.ipynb'):
                notebooks.append(os.path.join(root, file))
    
    print(f"Found {len(notebooks)} notebooks to verify")
    
    # Verify each notebook
    valid_count = 0
    invalid_count = 0
    
    for notebook in sorted(notebooks):
        if verify_notebook_structure(notebook):
            print(f"✅ {notebook}")
            valid_count += 1
        else:
            invalid_count += 1
    
    print(f"\n📊 Verification Results:")
    print(f"✅ Valid notebooks: {valid_count}")
    print(f"❌ Invalid notebooks: {invalid_count}")
    print(f"📈 Success rate: {(valid_count / len(notebooks)) * 100:.1f}%")
    
    return valid_count == len(notebooks)

def check_notebook_content():
    """
    Check content of key notebooks.
    """
    print("\n🔍 Checking key notebook content...")
    
    key_notebooks = [
        'src/ai/phi_ground.ipynb',
        'scripts/universal_apk_tester.ipynb',
        'src/core/config.ipynb',
        'scripts/phi_ground_example.ipynb'
    ]
    
    for notebook_path in key_notebooks:
        if os.path.exists(notebook_path):
            try:
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook = json.load(f)
                
                cell_count = len(notebook['cells'])
                code_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'code')
                markdown_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
                
                print(f"📓 {notebook_path}:")
                print(f"   Total cells: {cell_count}")
                print(f"   Code cells: {code_cells}")
                print(f"   Markdown cells: {markdown_cells}")
                
            except Exception as e:
                print(f"❌ Error checking {notebook_path}: {e}")
        else:
            print(f"❌ {notebook_path} not found")

if __name__ == "__main__":
    print("🚀 AA_VA-Phi Notebook Conversion Verification")
    print("=" * 50)
    
    # Verify structure
    structure_valid = verify_all_notebooks()
    
    # Check content
    check_notebook_content()
    
    print("\n" + "=" * 50)
    if structure_valid:
        print("🎉 All notebooks verified successfully!")
        print("✅ Conversion completed successfully")
    else:
        print("⚠️  Some notebooks have issues")
        print("❌ Conversion may need review")
    
    print("\n📚 Next steps:")
    print("1. Start Jupyter Lab: jupyter lab")
    print("2. Open src/ai/phi_ground.ipynb to explore Phi-Ground integration")
    print("3. Run scripts/phi_ground_example.ipynb for examples")
    print("4. Check README_NOTEBOOKS.md for detailed usage guide")
