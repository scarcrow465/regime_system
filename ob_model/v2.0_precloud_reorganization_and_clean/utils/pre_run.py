#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Pre-Run Checks for Fluid Development
Validates imports/paths before runs—no surprises like missing dirs or imports.
Why: Like a pre-flight checklist—ensures smooth takeoff for fingerprint tests.
Ties to vision: Saves time for edge extraction (OB's "why"), not fixing errors.
"""

import importlib
import os
from utils.logger import get_logger

logger = get_logger('pre_run')

def check_prerequisites():
    """
    Checks imports, paths—logs issues, returns True if OK.
    """
    # Check key modules
    modules = ['pandas', 'numpy', 'scipy', 'matplotlib', 'tqdm']
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            logger.error(f"Missing module {mod}—pip install {mod}, update environment.md")
            return False
    
    # Check paths
    paths = ['docs/plots', 'logs/fingerprint_test']
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Ensured path exists: {path}")
    
    return True

# Run before tests: python -c "from utils.pre_run import check_prerequisites; check_prerequisites()"

