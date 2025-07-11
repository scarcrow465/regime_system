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


## 7/11/2025 - 09:58
Phase 1 done—your foundation is set! Test: In a script (e.g., main.py), add logger = get_logger('test'); logger.info("Hello"); logger.debug("Deep hello"); for i in progress_wrapper(range(10), desc="Test Bar"): pass. See terminal bar/color, file in logs/test/yyyy-mm-dd.log with details. 
Big Picture: This is your "safety net"—catches falls (bugs) early, so you soar toward independence with OB scaling. Phase 1 ties to vision by making edges "visible" (bars/logs) without frustration.

Updated: July 11, 2025.