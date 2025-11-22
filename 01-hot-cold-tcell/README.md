# 01-hot-cold-tcell

Scripts for hot/cold profiling using T-cell panels.

## Files
- `scripts/01_make_heatmap.py` – builds clustered, scaled heatmaps.
- `scripts/02_compute_hot_score.py` – computes T/Activation/PEX/TEX indices, HotScore (Z), p_consistency, labels, and per-cancer summaries.

## Data (not included)
Place CSVs here: `01-hot-cold-tcell/data/`  
Format (tidy): `SAMPLE_ID, cyt, zsc`

## Quick start
```bash
pip install -r ../../requirements.txt
python scripts/01_make_heatmap.py
python scripts/02_compute_hot_score.py

