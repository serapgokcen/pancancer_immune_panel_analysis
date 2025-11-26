# 01-hot-cold-tcell

Hot / Intermediate / Cold T-cell classification using T-cell panels.

## Scripts

- `scripts/01_make_heatmap.py`  
  Builds clustered, scaled heatmaps from tidy expression tables.

- `scripts/02_compute_hot_score.py`  
  Computes T-cell abundance, activation, PEX and TEX indices, HotScore (Z),
  p_consistency, per-sample labels (hot / intermediate-hot / intermediate-cold / cold),
  and per-cancer summary metrics (fractions, CIs, medians, WHC, HI, HSS).

## Data

Processed TCGA PanCancer Atlas z-score tables (323 immune-related genes) for
seven cancer types are stored in the root `data/` directory:

- `data/melanoma_323_transformed_fixed.csv`
- `data/breast_323_transformed_fixed.csv`
- `data/colorectalcorr_323_transformed_fixed.csv`
- `data/lung_323_transformed_fixed.csv`
- `data/ovariancorr_323_transformed_fixed.csv`
- `data/pancreatic_323_transformed_fixed.csv`
- `data/prostatecorr_323_transformed_fixed.csv`

Each CSV is tidy, with columns:

```text
SAMPLE_ID, cyt, zsc```

## Quick start

From the repository root:

pip install -r requirements.txt
cd 01-hot-cold-tcell

Then, from the 01-hot-cold-tcell folder:

# Heatmap generation (optional)
python scripts/01_make_heatmap.py

# Hot / intermediate / cold classification
python scripts/02_compute_hot_score.py --outdir results


## Outputs

From scripts/01_make_heatmap.py:

One or more clustered, scaled heatmap figure files (e.g. PNG/PDF), saved in the working directory or in the output location configured inside the script.

From scripts/02_compute_hot_score.py (written to the directory given by --outdir, e.g. results/):

activation_labels_per_sample.csv – per-sample table (SAMPLE_ID, cancer_type, HotScore, Z, p_consistency, panel indices, group).

activation_summary_by_cancer.csv – per-cancer summary (N, class fractions with Wilson CIs, median Zs for hot/cold sides, WHC, HI, HSS).
