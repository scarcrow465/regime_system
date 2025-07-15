#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# setup_project.py
import os
from config.settings import BASE_DIR
from utils.logger import log_message

BASE_DIR = os.path.join(os.path.expanduser("~"), "rs", "GitProjects", "regime_system", "v3.0")  # Align with your GitHub repo path

DIRS = [
    os.path.join(BASE_DIR, "core"),  # For classifiers, indicators, data_loader
    os.path.join(BASE_DIR, "optimization"),  # For GMM/Optuna
    os.path.join(BASE_DIR, "backtesting"),  # For OB integration probes
    os.path.join(BASE_DIR, "validation"),  # For persistence/distribution checks
    os.path.join(BASE_DIR, "config"),  # Settings/yaml
    os.path.join(BASE_DIR, "utils"),  # Logging, checkpoint
    os.path.join(BASE_DIR, "exports", "plots"),  # PNGs
    os.path.join(BASE_DIR, "exports", "logs"),  # TXT/JSON logs
    os.path.join(BASE_DIR, "exports", "csv"),  # CSVs like probed_trades
    os.path.join(BASE_DIR, "cloud"),  # AWS deploy scripts
    os.path.join(BASE_DIR, "future")  # For expansions like fingerprint_*
]

def create_dirs():
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
        log_message(f"Created/verified dir: {d}", 'info')

def init_git():
    os.chdir(BASE_DIR)
    subprocess.run(["git", "init"])
    with open(os.path.join(BASE_DIR, ".gitignore"), "w") as f:
        f.write("*.pyc\n__pycache__/\nexports/*\n")
    print("Git initialized with .gitignore.")

if __name__ == "__main__":
    create_dirs()
    log_message("Phase 0 setup complete!", 'info')

