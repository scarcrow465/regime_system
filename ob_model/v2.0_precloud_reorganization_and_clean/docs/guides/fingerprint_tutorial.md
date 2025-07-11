# Fingerprint Tutorial: Your Edge Detection Guide

This guide walks you through running the fingerprint—think of it as a treasure hunt: Scanner sweeps for clues (asymmetries), classifier labels them (e.g., "reversion gold"), evolver watches growth (like a plant over time). Why? Reveals "why" OB works (market behavior, not luck)—broad scans catch latent edges (e.g., RSI2 conditional), testing all scopes (scalping <2h to position 3+wks).

## Step 1: Setup
- Load data in main.py: df = load_csv('nq.csv')—see data_loader.py.
- Config: In edge_taxonomy.py, thresholds low (0.1) to flag potentials—regimes unlock.

## Step 2: Run Scan (Broad Search)
- In edge_scanner.py: edge_map = scan_for_edges(df)
- What happens: Tests categories (e.g., behavioral autocorr for reversion)—no filters, then conditionals (e.g., low-vol subset for RSI2-like).
- Visual: Add plot: import matplotlib.pyplot as plt; plt.bar(edge_map.keys(), [d['broad_score'] for d in edge_map.values()]); plt.show()—bar chart "painting" edge strengths like a skyline.

## Step 3: Classify & Tag
- fingerprint_classifier.py: tagged_map = classify_edges(edge_map)
- What: Tags primary (e.g., behavioral), sub (scopes: Test 1-day for day trading), analytical (robustness p-value).
- Visual: Heatmap of scopes: plt.imshow(scope_matrix, cmap='hot')—red hot for strong edges like a heat map treasure spot.

## Step 4: Evolve & Track
- fingerprint_evolver.py: evolved_map = evolve_edges(tagged_map, df)
- What: Rolling slopes for intensity (strengthening?), breaks (post-1983 shift).
- Visual: Line plot: plt.plot(rolling_scores); plt.title("Edge Evolution—rising like a rocket launch")—see if RSI2 "activated."

## Debugging Tips
- Set logging_level='DEBUG' in config—files in logs/fingerprint/ show details (e.g., "p=0.03 for reversion").
- If missed edge: Check conditional_subset_test()—logs "Low-vol boosted 0.2."
- Test: Run if __name__ in files—see outputs/logs.

## Ties to Vision
Fingerprint proves OB's edge is market DNA (e.g., trending reliable in NQ)—regimes apply it, weekly checks persistence. Finish this, then expand—no sidetracks.

Next Steps: Integrate with regimes—see DEVELOPMENT_LOG.md.