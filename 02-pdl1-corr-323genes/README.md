# 02-pdl1-corr-323genes

Analysis of genes positively correlated with PDL1 (CD274) across multiple
TCGA cancer types.

## Scripts

- `scripts/01_pdl1_corr_heatmap.py`  
  For one cancer type, builds a CD274-anchored expression heatmap and a
  ranked correlation heatmap (genes vs CD274).

- `scripts/02_pdl1_corr_upsetplot.py`  
  Uses per-cancer lists of strongly PDL1-correlated genes (r ≥ 0.5) to
  generate an UpSet plot and an Excel table of gene-set intersections.

## Data

Processed inputs live in the root `data/` directory.

### 1. 323-gene expression tables (tidy z-scores)

- `data/melanoma_323_transformed_fixed.csv`
- `data/breast_323_transformed_fixed.csv`
- `data/colorectalcorr_323_transformed_fixed.csv`
- `data/lung_323_transformed_fixed.csv`
- `data/ovariancorr_323_transformed_fixed.csv`
- `data/pancreatic_323_transformed_fixed.csv`
- `data/prostatecorr_323_transformed_fixed.csv`

Each has columns:
SAMPLE_ID, cyt, zsc

### 2. PDL1-correlated gene lists (r ≥ 0.5)

One gene symbol per row. Examples:

- `data/MEL-T-COM_genes_pos_correlated_rge0.5.csv`
- `data/bladder_genes_pos_correlated_rge0.5.csv`
- `data/brainlower_genes_pos_correlated_rge0.5.csv`
- `data/breast_genes_pos_correlated_rge0.5.csv`
- `data/cervical_genes_pos_correlated_rge0.5.csv`
- `data/colorectalcorrgreen_genes_pos_correlated_rge0.5.csv`
- `data/esophagealcorr_genes_pos_correlated_rge0.5.csv`
- `data/glioblastoma_genes_pos_correlated_rge0.5.csv`
- `data/headandneck_genes_pos_correlated_rge0.5.csv`
- `data/kidneyrenalclear_genes_pos_correlated_rge0.5.csv`
- `data/kidneyrenalpapil_genes_pos_correlated_rge0.5.csv`
- `data/liverhepato_genes_pos_correlated_rge0.5.csv`
- `data/lung_genes_pos_correlated_rge0.5.csv`
- `data/lungsquamous_genes_pos_correlated_rge0.5.csv`
- `data/ovariancorr_genes_pos_correlated_rge0.5.csv`
- `data/pancreatic_genes_pos_correlated_rge0.5.csv`
- `data/pheochromocytoma_genes_pos_correlated_rge0.5.csv`
- `data/prostatecorr_genes_pos_correlated_rge0.5.csv`
- `data/sarcoma_genes_pos_correlated_rge0.5.csv`
- `data/stomachcorr_genes_pos_correlated_rge0.5.csv`
- `data/thyroid_genes_pos_correlated_rge0.5.csv`

The scripts first try the original Windows paths; if those are not present,
they fall back to these filenames in ./data/.
Raw TCGA matrices are not included.

## Quick Start

pip install -r requirements.txt
cd 02-pdl1-corr-323genes

# Heatmap + per-gene correlations for one cancer type
python scripts/01_pdl1_corr_heatmap.py --outdir results_pdl1_corr

# UpSet of PDL1-correlated gene overlaps across cancers
python scripts/02_pdl1_corr_upsetplot.py --outdir results_pdl1_upset


## Outputs

From scripts/01_pdl1_corr_heatmap.py (in results_pdl1_corr/):
pdl1_corr_heatmap.png – CD274-anchored expression heatmap.
pdl1_corr_ranked_genes.png – ranked gene–CD274 correlation heatmap.

From scripts/02_pdl1_corr_upsetplot.py (in results_pdl1_upset/):
pdl1_corr_323genes_upsetplot.png – UpSet plot of gene-set intersections.
Gene_intersections.xlsx – table listing each intersection, its gene count,
and the gene names.

