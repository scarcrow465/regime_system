# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# #!/usr/bin/env python
"""
Setup script to create the regime_system package structure
Run this first to create all directories
"""

import os
import sys

def create_package_structure():
    """Create the regime_system package directory structure"""
    
    base_dir = "regime_system"
    
    # Define the directory structure
    directories = [
        base_dir,
        f"{base_dir}/core",
        f"{base_dir}/optimization",
        f"{base_dir}/backtesting",
        f"{base_dir}/config",
        f"{base_dir}/utils",
        f"{base_dir}/validation",
        f"{base_dir}/cloud",
        f"{base_dir}/future",
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}/")
        
        # Create __init__.py files
        init_file = os.path.join(directory, "__init__.py")
        with open(init_file, "w") as f:
            f.write('"""Package initialization"""\n')
        print(f"Created: {init_file}")
    
    # Create placeholder files
    placeholders = {
        f"{base_dir}/future/README.md": "# Future Enhancements\n\nThis directory is reserved for future features.",
        f"{base_dir}/README.md": "# Regime System\n\nInstitutional-grade regime analysis system.",
    }
    
    for filepath, content in placeholders.items():
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Created: {filepath}")
    
    print("\nPackage structure created successfully!")
    print(f"Next step: Run the code extraction scripts")

if __name__ == "__main__":
    create_package_structure()
