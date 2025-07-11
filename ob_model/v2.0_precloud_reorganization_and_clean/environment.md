# Environment Snapshot for Regime System Project

This file captures all technical specs for reproducibility—use it to recreate the setup on any machine or for debugging version issues. If problems arise (e.g., "code worked before but not now"), start here. Update after changes (e.g., pip install).

## Hardware & System Details
- **Computer Model**: Asus G513QY (gaming laptop—good for data processing but monitor heat during long runs).
- **RAM**: 64GB (handles ~400k rows comfortably; for larger datasets like multi-asset, chunk processing in data_loader.py).
- **Processor**: [Add your CPU details if known, e.g., AMD Ryzen 9—fast for optimizations but test memory usage].
- **Storage**: [Add if relevant, e.g., SSD for quick CSV loads].
- **Operating System**: Windows [version, e.g., 11—note: Some libs like ta-lib need Windows wheels].
- **Why Important**: Hardware limits guide optimizations (e.g., no massive ML models yet—your 64GB is great for pandas but watch during brute-force edge scans).

## Software & Python Environment
- **Python Version**: 3.12.3 (or your exact—check with `python --version`; stick to this to avoid breaking changes in libs like pandas).
- **Package Manager**: pip (use `pip install -r requirements.txt` to setup).
- **Key Packages & Versions** (Generated from `pip freeze`—run it and paste/update here):
  - numpy==1.26.4  # For arrays in indicators.py—essential for edge scans.
  - pandas==2.2.2  # For data handling in data_loader.py—your CSVs load here.
  - ta==0.11.0  # For technical indicators—watch for version bugs; if issues, downgrade to 0.10.x.
  - matplotlib==3.8.4  # For visuals (e.g., edge heatmaps)—your visual style needs this for plots.
  - scipy==1.13.1  # For stats in edge_validator.py (t-tests for edge significance).
  - tqdm==4.66.4  # For progress bars in terminal logs—keeps things clean during scans.
  - [Add all from pip freeze, e.g., logging if custom].
- **Other Tools**: PostgreSQL for database (schema: [describe if custom]), BarChart Excel plugin for data ($90/month—automate hourly updates).
- **Installation Notes**: For ta-lib on Windows: Download wheel from unofficial site, pip install wheel_file.whl. No internet in code_execution tool—use pre-installed libs.
- **Why Important**: Prevents "it worked on my old setup" frustrations—e.g., ta 0.11 broke before; this ensures consistency for fingerprint scans across scopes.

## Development Preferences & Troubleshooting Tips
- See knowledge_base/personal_profile.md for my (Cameron's) style—visuals first, simple explanations.
- Common Issues & Fixes:
  - NaNs in data: Check data_loader.py—use fillna(0) as in past fixes.
  - Version Mismatch: pip uninstall [lib]; pip install [version from here].
  - Memory Overload: Your 64GB—chunk data in scans (e.g., process 100k rows at a time).
- Test Command: python -m unittest (after adding tests)—verifies env.

Last updated: July 11, 2025. Run `pip freeze > temp.txt` and update packages section after changes.