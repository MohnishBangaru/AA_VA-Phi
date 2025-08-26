#!/usr/bin/env python3
"""
Script to convert Python files to Jupyter notebooks.
This script will convert all .py files in the AA_VA-Phi repository to .ipynb format.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any

def create_notebook_from_python(python_file_path: str, output_dir: str = None) -> str:
    """
    Convert a Python file to a Jupyter notebook.
    
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
    
    # Split content into cells
    cells = split_python_into_cells(python_content)
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
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
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write notebook file
    notebook_path = output_dir / f"{python_path.stem}.ipynb"
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    return str(notebook_path)

def split_python_into_cells(python_content: str) -> List[Dict[str, Any]]:
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
    
    for line in lines:
        # Check for markdown cells (docstrings, comments)
        if is_markdown_line(line):
            # If we have code in current cell, save it
            if current_cell and current_cell_type == "code":
                cells.append(create_code_cell(current_cell))
                current_cell = []
            
            # Start or continue markdown cell
            if current_cell_type != "markdown":
                current_cell_type = "markdown"
                current_cell = []
            
            current_cell.append(line)
        else:
            # If we have markdown in current cell, save it
            if current_cell and current_cell_type == "markdown":
                cells.append(create_markdown_cell(current_cell))
                current_cell = []
            
            # Start or continue code cell
            if current_cell_type != "code":
                current_cell_type = "code"
                current_cell = []
            
            current_cell.append(line)
    
    # Add the last cell
    if current_cell:
        if current_cell_type == "markdown":
            cells.append(create_markdown_cell(current_cell))
        else:
            cells.append(create_code_cell(current_cell))
    
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
    
    # Docstring lines
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return True
    
    # Comments that look like documentation
    if stripped.startswith('#'):
        # Skip simple comments, keep documentation comments
        if len(stripped) > 2 and not stripped.startswith('# '):
            return True
        if any(keyword in stripped.lower() for keyword in ['note:', 'warning:', 'todo:', 'fixme:', 'important:']):
            return True
    
    return False

def create_code_cell(lines: List[str]) -> Dict[str, Any]:
    """
    Create a code cell from lines.
    
    Args:
        lines: The lines of code
    
    Returns:
        Notebook cell dictionary
    """
    # Remove trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()
    
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

def create_markdown_cell(lines: List[str]) -> Dict[str, Any]:
    """
    Create a markdown cell from lines.
    
    Args:
        lines: The lines of markdown
    
    Returns:
        Notebook cell dictionary
    """
    # Convert Python comments to markdown
    markdown_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            # Convert comment to markdown
            markdown_lines.append(stripped[1:].lstrip())
        elif stripped.startswith('"""') or stripped.startswith("'''"):
            # Handle docstrings
            if stripped.endswith('"""') or stripped.endswith("'''"):
                # Single line docstring
                content = stripped[3:-3] if stripped.startswith('"""') else stripped[3:-3]
                markdown_lines.append(content)
            else:
                # Start of multi-line docstring
                content = stripped[3:] if stripped.startswith('"""') else stripped[3:]
                markdown_lines.append(content)
        else:
            markdown_lines.append(line)
    
    # Remove trailing empty lines
    while markdown_lines and not markdown_lines[-1].strip():
        markdown_lines.pop()
    
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": markdown_lines
    }

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
