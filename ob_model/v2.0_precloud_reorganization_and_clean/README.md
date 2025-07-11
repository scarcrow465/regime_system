# Regime System Project

## Vision Overview
Build an institutional-grade system to extract market edges via fingerprint (primary: broad scans for asymmetries, multiples per market, across scopes from scalping <2h to position 3+wks), apply dynamically with regimes (secondary: timing/sizing), and refine with weekly context. Starts with NQ/OB model (515% returns, 76% win rate)—scales to multi-asset meta-strategies for independence. Fingerprint gates everything: No edge? Monitor. Latent (e.g., RSI2 conditional post-1983)? Unlock with regimes.

## Quick-Start
1. Setup: pip install -r requirements.txt (see docs/environment.md for versions/hardware).
2. Run: python main.py --data=nq.csv --mode=fingerprint (see guides/fingerprint_tutorial.md).
3. Debug: Set config logging_level='DEBUG'—check logs/module/yyyy-mm-dd.txt.
4. Personal Guide: See docs/knowledge_base/personal_profile.md for style (visuals first, simple explanations).

## Structure Map
- core/: Edge scanning/classification (fingerprint primary).
- config/: Thresholds/scopes (low for potentials).
- utils/: Logging/debug (multi-level for easy fixes).
- [Full map in docs/structure_diagram.png—visual flow from fingerprint to regimes.]

Next Steps: [See DEVELOPMENT_LOG.md—e.g., "Finish fingerprint, then weekly."]


## 7/10/2025 - 09:58

Phase 1 done—your foundation is set! Test: In a script (e.g., main.py), add logger = get_logger('test'); logger.info("Hello"); logger.debug("Deep hello"); for i in progress_wrapper(range(10), desc="Test Bar"): pass. See terminal bar/color, file in logs/test/yyyy-mm-dd.log with details. 
Big Picture: This is your "safety net"—catches falls (bugs) early, so you soar toward independence with OB scaling. Phase 1 ties to vision by making edges "visible" (bars/logs) without frustration.



## 7/11/2025 - 16:32

# Regime System OB Model v2.0
Project for Cameron's trading breakthrough at 26—fingerprint market edges to explain "why" OB wins, scaling to multi-asset (NQ/ES/GC), multi-timeframe (1h/daily/weekly).

## Overview
- **Goal**: Extract asymmetries (edges) from data, classify/taxonomy (e.g., behavioral reversion in day_trading scope), evolve/track persistence (rolling scores/slope/breaks).
- **Why**: "Treasure hunter" for OB's why (e.g., RSI2 in low-vol = conditional edge), persistence "glue" for sizing (weekly hold = position trades).
- **Tech**: Python 3.12, pandas/numpy/scipy for data, Loguru/pretty_errors for logs, Rich for pretty terminal, matplotlib for plots (line/bar in docs/plots/date/time).
- **Structure**:
  - config/: settings.py (VERBOSE, PLOT_ENABLED), edge_taxonomy.py (categories/scopes).
  - core/: edge_scanner.py (scan asymmetries), fingerprint_classifier.py (tag scopes), fingerprint_evolver.py (track evolution), weekly_resampler.py (if needed).
  - utils/: logger.py (Loguru decorators), debug_utils.py (sanity/safe_save).
  - fingerprint_test.py: Test chain on timeframe CSV (set TIMEFRAME='daily'), outputs tables/plots.
- **Progress**:
  - Phase 1: Scan/classify basic edges (directional/behavioral, etc.) on fake data.
  - Phase 2: Evolve/track (rolling/slope/persistence), pretty terminal (Rich panels/tables), organized exports (date/time subs).
  - Phase 3: Weekly/daily/1h context for persistence, timeframe variable in test.
  - Next: Integrate with OB model (if edge >0.1 weekly, size up), multi-asset (loop CSVs), add RSI/VIX for stronger scores.
- **Key Discussions**: VERBOSE toggle for clean/detailed logs, Rich for dashboard output, Loguru migration for prettier errors, no resampling (use pre-gen CSVs), scopes timeframe-specific (scalping=0.04 days for 1h).
- **Run**: python fingerprint_test.py (set TIMEFRAME), check tables/plots/logs.
- **Future**: Cloud optimization (checkpoint/cost monitor), real-time (databento).

Updated: July 11, 2025.