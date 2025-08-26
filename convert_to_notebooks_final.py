#!/usr/bin/env python3
"""
Final script to convert Python files to Jupyter notebooks using nbformat.
This script will convert all .py files in the AA_VA-Phi repository to .ipynb format.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

try:
    import nbformat as nbf
except ImportError:
    print("nbformat not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "nbformat"])
    import nbformat as nbf

def create_notebook_from_python(python_file_path: str, output_dir: str = None) -> str:
    """
    Convert a Python file to a Jupyter notebook using nbformat.
    
    Args:
        python_file_path: Path to the Python file
        output_dir: Output directory for the notebook (defaults to same directory)
    
    Returns:
        Path to the created notebook file
    """
    python_path = Path(python_file_path)
    
    if output_dir is None:
        output_dir = python_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the Python file
    with open(python_file_path, 'r', encoding='utf-8') as f:
        python_content = f.read()
    
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Split content into cells
    cells = split_python_into_cells(python_content)
    nb.cells = cells
    
    # Set metadata
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    }
    
    # Write notebook file
    notebook_path = output_dir / f"{python_path.stem}.ipynb"
    nbf.write(nb, str(notebook_path))
    
    return str(notebook_path)

def split_python_into_cells(python_content: str) -> List[nbf.NotebookNode]:
    """
    Split Python content into notebook cells.
    
    Args:
        python_content: The Python file content
    
    Returns:
        List of notebook cells
    """
    cells = []
    
    # Split by lines
    lines = python_content.split('\n')
    
    current_cell = []
    current_cell_type = "code"
    in_docstring = False
    docstring_content = []
    
    for line in lines:
        stripped = line.strip()
        
        # Handle docstrings
        if '"""' in line or "'''" in line:
            if not in_docstring:
                # Start of docstring
                in_docstring = True
                docstring_content = []
                # Extract content after opening quotes
                if '"""' in line:
                    content = line.split('"""', 1)[1]
                    if '"""' in content:
                        # Single line docstring
                        docstring_content.append(content.split('"""', 1)[0])
                        in_docstring = False
                        # Add markdown cell for docstring
                        if docstring_content:
                            cells.append(nbf.v4.new_markdown_cell('\n'.join(docstring_content)))
                        docstring_content = []
                    else:
                        docstring_content.append(content)
                elif "'''" in line:
                    content = line.split("'''", 1)[1]
                    if "'''" in content:
                        # Single line docstring
                        docstring_content.append(content.split("'''", 1)[0])
                        in_docstring = False
                        # Add markdown cell for docstring
                        if docstring_content:
                            cells.append(nbf.v4.new_markdown_cell('\n'.join(docstring_content)))
                        docstring_content = []
                    else:
                        docstring_content.append(content)
            else:
                # End of docstring
                in_docstring = False
                # Add markdown cell for docstring
                if docstring_content:
                    cells.append(nbf.v4.new_markdown_cell('\n'.join(docstring_content)))
                docstring_content = []
            continue
        
        if in_docstring:
            docstring_content.append(line)
            continue
        
        # Check for markdown cells (comments)
        if is_markdown_line(line):
            # If we have code in current cell, save it
            if current_cell and current_cell_type == "code":
                cells.append(nbf.v4.new_code_cell('\n'.join(current_cell)))
                current_cell = []
            
            # Start or continue markdown cell
            if current_cell_type != "markdown":
                current_cell_type = "markdown"
                current_cell = []
            
            current_cell.append(line)
        else:
            # If we have markdown in current cell, save it
            if current_cell and current_cell_type == "markdown":
                cells.append(nbf.v4.new_markdown_cell('\n'.join(current_cell)))
                current_cell = []
            
            # Start or continue code cell
            if current_cell_type != "code":
                current_cell_type = "code"
                current_cell = []
            
            current_cell.append(line)
    
    # Add the last cell
    if current_cell:
        if current_cell_type == "markdown":
            cells.append(nbf.v4.new_markdown_cell('\n'.join(current_cell)))
        else:
            cells.append(nbf.v4.new_code_cell('\n'.join(current_cell)))
    
    return cells

def is_markdown_line(line: str) -> bool:
    """
    Determine if a line should be treated as markdown.
    
    Args:
        line: The line to check
    
    Returns:
        True if the line should be markdown
    """
    stripped = line.strip()
    
    # Empty lines
    if not stripped:
        return False
    
    # Comments that look like documentation
    if stripped.startswith('#'):
        # Keep documentation comments, skip simple comments
        if len(stripped) > 2 and not stripped.startswith('# '):
            return True
        if any(keyword in stripped.lower() for keyword in ['note:', 'warning:', 'todo:', 'fixme:', 'important:', 'example:', 'usage:']):
            return True
        # Keep section headers
        if stripped.startswith('# ') or stripped.startswith('## ') or stripped.startswith('### '):
            return True
    
    return False

def convert_repository_to_notebooks():
    """
    Convert all Python files in the repository to notebooks.
    """
    # Get all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip .git and other hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files to convert")
    
    # Convert each file
    converted_files = []
    for python_file in python_files:
        try:
            notebook_path = create_notebook_from_python(python_file)
            converted_files.append(notebook_path)
            print(f"✓ Converted: {python_file} -> {notebook_path}")
        except Exception as e:
            print(f"✗ Failed to convert {python_file}: {e}")
    
    print(f"\nConversion complete! Converted {len(converted_files)} files to notebooks.")
    return converted_files

if __name__ == "__main__":
    convert_repository_to_notebooks()
