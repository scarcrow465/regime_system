#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Fingerprint Classifier for Edge Taxonomy
Tags edges with primary/sub descriptions, scopes, analytical metrics.
Why: Classifies "what" the edge is (e.g., behavioral reversion in day trading scope)—extracts "why" OB wins from latent asymmetries.
Use: Input edge_map, output tagged_map with taxonomy.
"""

import pandas as pd
from utils.logger import get_logger, log_execution_time, log_errors
from config.edge_taxonomy import PRIMARY_CATEGORIES, SUB_CLASSIFIERS, THRESHOLDS, SCOPES  # Add SCOPES

logger = get_logger('fingerprint_classifier')

@log_execution_time
@log_errors()
def classify_edges(edge_map: dict, timeframe: str = 'daily') -> dict:
    """
    Classify edges with taxonomy, adjust scopes for timeframe.
    - Input: edge_map, timeframe ('1h', 'daily', 'weekly').
    - Output: tagged_map with desc, scores, scopes (multiplied for timeframe).
    """
    tagged_map = {}
    scope_multipliers = SCOPES.get(timeframe, SCOPES['daily'])  # Default daily
    for category, data in edge_map.items():
        logger.info(f"Classifying {category}")
        primary_desc = PRIMARY_CATEGORIES.get(category, 'Unknown')
        all_scopes = {scope: data['scopes'].get(scope, 0) * mult for scope, mult in scope_multipliers.items()}
        best_scope = max(all_scopes, key=all_scopes.get) if all_scopes else 'unknown'
        sub = {'best_scope': best_scope, 'all_scopes': all_scopes}
        analytical = {'robustness': 0}  # Placeholder (e.g., bootstrap tests later)
        final_score = data['broad_score'] * 0.5 + data['conditional_score'] * 0.5
        tagged_map[category] = {
            'primary_desc': primary_desc,
            'sub': sub,
            'analytical': analytical,
            'final_score': final_score
        }
    logger.info(f"Classification complete: {len(tagged_map)} tagged edges")
    return tagged_map

