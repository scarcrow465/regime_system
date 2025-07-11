#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Pattern Classifier for Easy Tags
Tags patterns with simple names, best hold times.
Why: Turns numbers into "what it is" (e.g., "Upward Pull: Better for medium holds")—easy "why" for trades.
Use: Input edge_map/timeframe, output tagged_map with names/strengths.
"""

from utils.logger import get_logger, log_execution_time, log_errors
from config.edge_taxonomy import PRIMARY_CATEGORIES, SCOPES, THRESHOLDS

logger = get_logger('fingerprint_classifier')

@log_execution_time
@log_errors()
def classify_edges(edge_map: dict, timeframe: str = 'daily') -> dict:
    tagged_map = {}
    hold_multipliers = SCOPES.get(timeframe, SCOPES['daily'])  # Adjust for timeframe
    for category, data in edge_map.items():
        logger.info(f"Tagging {category}")
        name = PRIMARY_CATEGORIES.get(category, 'Unknown Pattern')
        all_holds = {hold: data['scopes'].get(hold, 0) * mult for hold, mult in hold_multipliers.items()}
        best_hold = max(all_holds, key=all_holds.get) if all_holds else 'unknown'
        strength = data['broad_strength'] * 0.5 + data['conditional_strength'] * 0.5  # Average overall/better conditions
        tagged_map[category] = {
            'name': name,
            'strength': strength,
            'best_hold': best_hold,
            'all_holds': all_holds
        }
    logger.info(f"Tagging complete: {len(tagged_map)} patterns named")
    return tagged_map

