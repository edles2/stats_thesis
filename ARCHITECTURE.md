# Architecture

## Components
- `analysis/data.py`: parses workbook sectors into tidy rows.
- `analysis/analyse.py`: computes method comparison metrics.
- `analysis/streamlit.py`: interactive result exploration.
- `analysis/smoke.py`: synthetic, privacy-safe smoke check.

## Flow
1. Build tidy dataframe from workbook.
2. Create `(manual, method)` tooth-level pairs.
3. Aggregate by method and zone into reviewer-friendly metrics.
