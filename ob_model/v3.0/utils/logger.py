#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# utils/logger.py
from loguru import logger
import os
import sys

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "exports", "logs")

def setup_logging(level="INFO", advanced=False):
    logger.remove()  # Clear defaults
    log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}" if not advanced else "{time} | {level} | {file}:{line} | {message}"
    logger.add(sys.stdout, format=log_format, level=level)
    logger.add(os.path.join(LOG_DIR, "{time:YYYYMMDD}_regime.log"), rotation="10 MB", format=log_format, level=level)
    logger.info(f"Logging setup: level={level}, advanced={advanced}")

# Usage: from utils.logger import setup_logging; setup_logging("DEBUG", True)

