#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Fingerprint Classifier for Edge Tagging
Tags/scores scan results with taxonomy (primary like behavioral, sub like scopes, analytical like evolution).
Why: Adds context to potentials—e.g., RSI2 as 'behavioral reversion, scope=day trading, conditional on low-vol'.
How it ties to vision: Scores prove OB edge is market behavior (e.g., trending robust >5 days), not luck—multiples OK.
Use: Input edge_map from scanner, output tagged dict.
"""

from utils.logger import get_logger, log_execution_time, log_errors
from utils.debug_utils import log_var_state
from config.edge_taxonomy import PRIMARY_CATEGORIES, SUB_CLASSIFIERS, ANALYTICAL_METRICS, THRESHOLDS
import logging  # For logging.WARNING
from config.settings import VERBOSE

logger = get_logger('fingerprint_classifier')  # Define logger first

if not VERBOSE:
    logger.setLevel(logging.WARNING)  # Suppress INFO if not VERBOSE

@log_execution_time(logger)
@log_errors(logger)
def classify_edges(edge_map: dict) -> dict:
    """
    Classify/score edges—tag with taxonomy, compute analytical (e.g., robustness p-value).
    - Input: edge_map from scanner.
    - Output: tagged_map with scores—e.g., {'behavioral': {'primary': 'reversion', 'sub': {'scope': 'day_trading'}, 'analytical': {'robustness': 0.8}}}.
    - Why visual: Dict structure easy for heatmaps (rows=primary, columns=scopes).
    """
    tagged_map = {}
    
    for category, data in edge_map.items():
        logger.info(f"Classifying {category}")
        
        # Primary tag: Already from scan—refine if needed
        tagged_map[category] = {'primary_desc': PRIMARY_CATEGORIES[category]}
        
        # Sub-tags: e.g., best scope from results
        if 'scopes' in data:
            best_scope = max(data['scopes'], key=data['scopes'].get)  # Highest score
            tagged_map[category]['sub'] = {'best_scope': best_scope, 'all_scopes': data['scopes']}
            log_var_state('scope_results', data['scopes'], logger)
        
        # Analytical: Score viability/evolution
        analytical = {}
        for metric, desc in ANALYTICAL_METRICS.items():
            if metric == 'robustness':
                # Example: p-value <0.05 = robust
                analytical[metric] = 1 if data.get('broad_p', 1) < 0.05 else 0
            # Add for others (e.g., evolution: Slope on scores)
        tagged_map[category]['analytical'] = analytical
        
        # Overall score: Average broad + conditional, boosted if robust
        final_score = (data['broad_score'] + data.get('conditional_score', 0)) / 2
        if analytical.get('robustness', 0) > 0:
            final_score += THRESHOLDS['conditional_boost']
        tagged_map[category]['final_score'] = min(final_score, 1.0)  # Cap at 1
        
    logger.info(f"Classification complete: {len(tagged_map)} tagged edges")
    return tagged_map

# Example Test
if __name__ == "__main__":
    fake_map = {'behavioral': {'broad_score': 0.45, 'conditional_score': 0.7, 'broad_p': 0.03, 'scopes': {'day_trading': 0.8}}}
    tagged = classify_edges(fake_map)
    print(tagged)  # See in terminal

