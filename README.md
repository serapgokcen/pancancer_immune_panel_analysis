# Tcellmarker_Heatmap_analysis

This repository contains two related workflows built around TCGA PanCancer
Atlas z-score tables (323 immune-related genes per cancer type).

## Workflows

1. **01-hot-cold-tcell/**  
   Classifies samples as hot / intermediate-hot / intermediate-cold / cold  
   using T-cell abundance, activation, PEX and TEX panels.

2. **02-pdl1-corr-323genes/**  
   Computes Pearson correlations between CD274 (PDL1) and 323 immune-related
   genes across multiple cancer types, and visualises shared genes with an
   UpSet plot.

## Data

Processed z-score tables (323 immune-related genes) for seven cancer types
are stored in the root `data/` directory and are used by both workflows.

Each CSV is tidy, with columns:
`SAMPLE_ID`, `cyt`, `zsc`

## Quick start

From the repository root:
`pip install -r requirements.txt`

Then see the workflow-specific README files for details:
-01-hot-cold-tcell/README.md
-02-pdl1-corr-323genes/README.md

