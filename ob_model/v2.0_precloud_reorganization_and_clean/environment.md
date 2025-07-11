# Environment Snapshot for Regime System Project

This file captures all technical specs for reproducibility—use it to recreate the setup on any machine or for debugging version issues. If problems arise (e.g., "code worked before but not now"), start here. Update after changes (e.g., pip install).

## Hardware & System Details
- **Computer Model**: Asus G513QY (gaming laptop—good for data processing but monitor heat during long runs like edge scans across scopes).
- **RAM**: 64GB (handles ~400k rows comfortably; for larger datasets like multi-asset or long-term fingerprint evolution, use chunk processing in data_loader.py to avoid overload).
- **Processor**: [Add your CPU details if known, e.g., AMD Ryzen 9—fast for optimizations but test memory usage during brute-alternative scans].
- **Storage**: [Add if relevant, e.g., SSD for quick CSV loads during weekly resampling].
- **Operating System**: Windows [version, e.g., 11—note: Some libs like ta-lib need Windows wheels; if issues, see notes below].
- **Why Important**: Hardware limits guide optimizations (e.g., your 64GB is great for pandas-based edge tests but watch during intensive rolling windows for evolutions like SP500 post-1983 shifts).

## Software & Python Environment
- **Python Version**: [Your exact, e.g., 3.12.3—check with `python --version`; stick to this to avoid breaking changes in libs like pandas for rolling scans].
- **Package Manager**: pip (use `pip install -r requirements.txt` to setup—generated from `pip freeze`).
- **Key Packages & Versions** (Update by running `pip freeze > temp.txt` and pasting relevant ones here):
  - numpy==[version, e.g., 1.26.4]  # For arrays in indicators.py—essential for edge scans like autocorrelation tests.
  - pandas==[version, e.g., 2.2.2]  # For data handling in data_loader.py—your CSVs load here; critical for resampling to weekly.
  - ta==[version, e.g., 0.11.0]  # For technical indicators like RSI—watch for version bugs; if issues, downgrade to 0.10.x and note here.
  - matplotlib==[version, e.g., 3.8.4]  # For visuals (e.g., heatmaps of edges by scope)—your visual style needs this for plots like "edge evolution line chart showing strengthening like a snowball".
  - scipy==[version, e.g., 1.13.1]  # For stats in edge_validator.py (t-tests for edge significance, e.g., RSI2 conditional p-values).
  - tqdm==[version, e.g., 4.66.4]  # For progress bars in terminal logs—keeps scans visual and fun, like a loading bar in a video game.
  - [Add all relevant from pip freeze, e.g., logging, json for structured outputs].
- **Other Tools**: PostgreSQL for database (schema: [describe if custom, e.g., tables for CSVs]), BarChart Excel plugin for data ($90/month—automate hourly updates via macros).
- **Installation Notes**: 
  - For ta-lib on Windows: Download wheel from unofficial site (lfd.uci.edu/~gohlke/pythonlibs/), pip install wheel_file.whl—avoids compile errors.
  - No internet in code_execution tools—use pre-installed libs; test offline.
  - Environment Setup Command: pip install -r requirements.txt; if errors, check Windows wheels.
- **Why Important**: Prevents "it worked on my old setup" frustrations—e.g., ta 0.11 broke before; this ensures consistency for fingerprint scans across scopes (scalping to position) and evolutions (like RSI2 post-1983).

## Development Preferences & Troubleshooting Tips
- See knowledge_base/personal_profile.md for my (Cameron's) style—visuals first (e.g., plots for edge maps), simple explanations with analogies, structure with sections/bullets.
- Common Issues & Fixes:
  - NaNs in Data: From data_loader.py—use fillna(0) as in past fixes; log.warning if detected.
  - Version Mismatch: pip uninstall [lib]; pip install [version from here]—e.g., for ta-lib issues.
  - Memory Overload: Your 64GB—chunk data in scans (e.g., process 100k rows at a time); monitor with task manager.
  - Logging Mess: Set level='INFO' for clean terminal; DEBUG to files only.
- Test Command: python -m unittest (after adding tests)—verifies env and catches early bugs.
- Big Picture Note: This snapshot ties to the vision—stable env means more time on edges (fingerprint extracting OB's "why") than fixing, scaling to multi-asset meta-strategies.

Last updated: July 11, 2025. Run `pip freeze > temp.txt` and update packages section after changes.