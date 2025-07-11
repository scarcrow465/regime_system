#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Edge Taxonomy for Fingerprint Detection
Defines primary categories (e.g., behavioral for trend/reversion), sub-classifiers (scopes like scalping to position), and analytical metrics (e.g., robustness for evolutions). 
Why: Broad map to scan without missing latent edges (e.g., RSI2 weak globally but strong conditional post-1983/low-vol).
How it ties to vision: Captures multiple edges per market (e.g., NQ trending + reversion pullbacks), testing all scopes to prove OB's edge is market truth, not luck.
Thresholds low (0.1) to flag potentials—regimes/weekly unlock them.
"""

# Primary Edge Categories: What type of asymmetry? (Scan for all)
PRIMARY_CATEGORIES = {
    'temporal': 'Calendar patterns (e.g., day-of-week returns diffs)',
    'directional': 'Long/short bias (e.g., positive drift in sims)',
    'behavioral': 'Trend/reversion/chop (e.g., autocorrelation tests)',
    'conditional': 'Regime-dependent (e.g., subset tests on vol)',
    'event_driven': 'News reactions (e.g., isolate Fed days)',
    'intermarket': 'Cross-asset links (e.g., correlation tests)',
    'microstructural': 'Volume/VWAP patterns',
    'sentiment_flow': 'Positioning extremes (e.g., COT data if available)'
}

# Sub-Classifiers: Context like your scopes—test holds matching each
SUB_CLASSIFIERS = {
    'scopes': {  # Your definitions—test sims for each hold range
        'scalping': '<2 hours (small portion of daily range)',
        'intraday_swing': 'Few hours to session (majority of daily range)',
        'intraday_full': 'Session to full day (full daily range)',
        'day_trading': '1 day hold only',
        'short_term': 'More than 1 day to full week',
        'swing': '1 to 3 weeks',
        'position': '3+ weeks to month or longer'
    },
    'time_horizon': ['intraday', 'daily', 'weekly'],  # TF fit
    'execution_context': ['open_to_close', 'close_to_open', 'overnight'],  # Timing
    'instrument_sensitivity': ['futures_like_nq', 'commodities_like_gc'],  # Asset fit
    'condition_filters': ['low_vol', 'high_vol', 'expansionary_economy'],  # Activators like post-1983
    'path_dependence': ['post_vol_spike', 'after_gap'],  # Sequence
    'tail_risk_bias': ['small_wins_big_losses', 'lumpy_winners'],  # Win/loss shape
    'signal_density': ['frequent', 'rare']  # How often
}

# Analytical Considerations: Viability/evolution—score edges here
ANALYTICAL_METRICS = {
    'robustness': 'Holds across samples (e.g., t-test p<0.05)',
    'capital_efficiency': 'Sharpe per turnover (high for frequent scopes)',
    'behavioral_signature': 'Links to biases (e.g., overreaction in reversion)',
    'market_archetype_alignment': 'Fits fingerprint (e.g., trending in NQ)',
    'macro_sensitivity': 'Changes with regimes (parked for later)',
    'survivability_threshold': 'Fade if Sharpe <0.5 in recent 1y',
    'rotation_score': 'Vs. other edges (e.g., reversion rotates out in trends)',
    'lead_lag_behavior': 'Predictive (e.g., LTF reversion leads HTF trend)'
}

# Thresholds: Low to catch potentials—regimes unlock (e.g., 0.1 global, >0.5 conditional)
THRESHOLDS = {
    'min_edge_score': 0.1,  # Flag potential (any asymmetry)
    'conditional_boost': 0.3,  # Add if regime/weekly aligns (e.g., RSI2 + low-vol)
    'evolution_significance': 0.05,  # p-value for changes (e.g., post-1983 shift)
    'persistence_min_days': 5  # Edge reliable if lasts this long
}

# Example Usage (in edge_scanner.py): for category in PRIMARY_CATEGORIES: test_asymmetry(category)
# For scopes: for scope in SUB_CLASSIFIERS['scopes']: simulate_holds(scope_range)

