#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# config/settings.py
import yaml

DEFAULTS = {
    "symbols": ["NQ"],
    "timeframe": "15min",
    "gmm_components": [2, 5],
    "persistence_target": 0.75,
    "oos_drop_max": 0.10,
    "cloud": {"instance": "t3.medium", "budget": 50},
    # Add OB-specific: "ob_csv_path": "path/to/backtest.csv"
}

def load_config(file="config.yaml"):
    if os.path.exists(file):
        with open(file, "r") as f:
            return yaml.safe_load(f)
    return DEFAULTS

# Usage: config = load_config()

