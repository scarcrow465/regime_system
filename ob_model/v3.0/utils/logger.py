#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from datetime import datetime
import loguru
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import pretty_errors
from tqdm import tqdm
from config.settings import DEBUG_LEVEL, LOGS_DIR

console = Console()

# Configure pretty_errors for verbose traces
pretty_errors.configure(
    filename_display=pretty_errors.FILENAME_EXTENDED,
    line_number_first=True,
    display_locals=True
)

# Loguru setup
logger = loguru.logger
logger.remove()  # Clear defaults
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOGS_DIR, f"{timestamp}.log")
logger.add(log_file, level="DEBUG")  # Always log debug to file

def log_message(msg, level='info'):
    if DEBUG_LEVEL == 'none' and level != 'error':
        return
    if level == 'error':
        console.print(Panel(msg, title="ERROR", style="bold red"))
        logger.error(msg)
    elif DEBUG_LEVEL in ['summary', 'debug', 'verbose']:
        if level == 'info' and DEBUG_LEVEL != 'none':
            console.print(msg, style="green")
            logger.info(msg)
        # Add more for debug/verbose

def progress_bar(iterable, desc, total=None):
    if DEBUG_LEVEL != 'none':
        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, mininterval=1)
    return iterable  # Silent if none

# Adapt old code prints to this (e.g., wrap in log_message)

