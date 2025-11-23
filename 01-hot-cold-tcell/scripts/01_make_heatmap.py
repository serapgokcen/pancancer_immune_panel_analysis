#!/usr/bin/env python
# coding: utf-8
"""
Heatmap-only (no HotScore, no labels, no CSV outputs).

Usage (GitHub Codespaces):
  pip install -r requirements.txt
  python 01-hot-cold-tcell/scripts/01_make_heatmap.py
  # or to point to a specific file:
  python 01-hot-cold-tcell/scripts/01_make_heatmap.py --source data/pancreatic_323_transformed_fixed.csv
"""

import os, warnings, argparse
from typing import List, Dict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------
# CONFIG — defaults for Codespaces
# ----------------
SOURCE = "data/pancreatic_323_transformed_fixed.csv"
VALUE_CANDIDATES = ("zsc", "value")

# Where we look for data inside the repo
DATA_SEARCH_PATHS = ["data", "01-hot-cold-tcell/data", "02-pdl1-corr-323genes/data"]

def prefer_any_file(p: str) -> str:
    fn = os.path.basename(p)
    for root in DATA_SEARCH_PATHS:
        cand = os.path.join(root, fn)
        if os.path.exists(cand):
            return cand
    return p

# Panels
T_ABUNDANCE = ['CD3D','CD3E','CD8A','PTPRC']
PEX         = ['SLAMF6','CCR7','TCF7']
TEX         = ['TOX','CXCL13','HAVCR2','TIGIT','LAG3','PDCD1','CTLA4']
ACTIVATION  = ['IFNG','ICOS','TNFRSF9','CD69','CD40LG','CD274']

# Which blocks to plot (top→bottom)
PANEL_BLOCKS: Dict[str, List[str]] = {
    "Exhaustion (TEX)": TEX,
    "Progenitor (PEX)": PEX,
    "T-cell Abundance": T_ABUNDANCE,
    "Activation": ACTIVATION,
}

# Scaling for colours
DISPLAY_SCALE = "rankz"   # "rankz" | "zsc" | "robustz"

# Clustering controls
CLUSTER_MODE = "global_samples"   # "none" | "global_samples" | "per_panel"
ROW_CLUSTER_PER_PANEL = True

# Colour scale limits
VMIN, VMAX = -4, 4

# Parse optional --source for convenience
ap = argparse.ArgumentParser()
ap.add_argument("--source", help="Path to CSV (repo-relative or absolute).")
args, _ = ap.parse_known_args()
if args.source:
    SOURCE = args.source
SOURCE = prefer_any_file(SOURCE)

# -------------
# HELPERS
# -------------
def inv_norm_cdf(p):
    p = np.asarray(p, dtype=float)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]
    plow, phigh = 0.02425, 0.97575
    q = np.empty_like(p)
    m = p < plow
    if m.any():
        pp = p[m]; t = np.sqrt(-2*np.log(pp))
        q[m] = (((((c[0]*t + c[1])*t + c[2])*t + c[3])*t + c[4])*t + c[5]) / \
               ((((d[0]*t + d[1])*t + d[2])*t + d[3])*t + 1)
    m = (p >= plow) & (p <= phigh)
    if m.any():
        pp = p[m] - 0.5; t2 = pp*pp
        q[m] = (((((a[0]*t2 + a[1])*t2 + a[2])*t2 + a[3])*t2 + a[4])*t2 + a[5]) * pp / \
               (((((b[0]*t2 + b[1])*t2 + b[2])*t2 + b[3])*t2 + b[4])*t2 + 1)
    m = p > phigh
    if m.any():
        pp = 1 - p[m]; t = np.sqrt(-2*np.log(pp))
        q[m] = -(((((c[0]*t + c[1])*t + c[2])*t + c[3])*t + c[4])*t + c[5]) / \
                 ((((d[0]*t + d[1])*t + d[2])*t + d[3])*t + 1)
    return q

def pooled_rank_to_norm(series: pd.Series) -> pd.Series:
    r = (series.rank(method="average") - 0.375) / (len(series) + 0.25)
    r = r.clip(1e-12, 1-1e-12)
    return pd.Series(inv_norm_cdf(r), index=series.index)

def mad(x: np.ndarray) -> float:
    return np.median(np.abs(x - np.median(x))) * 1.4826 + 1e-12

def add_gene_scaled(df_long: pd.DataFrame, scale: str, value_col: str = "value") -> pd.DataFrame:
    D = df_long.copy()
    if scale == "rankz":
        D['z'] = D.groupby('cyt')[value_col].transform(pooled_rank_to_norm)
    elif scale == "zsc":
        D['z'] = D[value_col]
    elif scale == "robustz":
        D['z'] = D.groupby('cyt')[value_col].transform(lambda v: (v - v.median()) / mad(v.values))
    else:
        raise ValueError("scale must be one of: 'rankz', 'zsc', 'robustz'")
    return D

def load_panel(source: str,
               value_candidates=("zsc","value"),
               required=("SAMPLE_ID","cyt")) -> pd.DataFrame:
    """Read CSV. If not found in repo and not a seaborn demo name, raise a clear error."""
    if os.path.exists(source):
        df = pd.read_csv(source)
    else:
        looks_like_name = (("/" not in source) and ("\\" not in source) and (":" not in source))
        if looks_like_name:
            df = sns.load_dataset(source)
        else:
            raise FileNotFoundError(f"CSV not found: {source}. "
                                    f"Put your file under one of: {DATA_SEARCH_PATHS}")
    df.columns = [c.strip() for c in df.columns]
    if not set(required).issubset(df.columns):
        raise ValueError(f"{source}: need columns {required}, found {list(df.columns)}")
    vcol = next((c for c in value_candidates if c in df.columns), None)
    if vcol is None:
        raise ValueError(f"{source}: expected one of {value_candidates}, found {list(df.columns)}")
    return df[list(required)+[vcol]].rename(columns={vcol:"value"})

def intersect_panel(all_genes: List[str], wanted: List[str]) -> List[str]:
    genes = [g for g in wanted if g in all_genes]
    missing = [g for g in wanted if g not in all_genes]
    if missing:
        print(f"  ! Missing genes (ignored): {missing}")
    return genes

def cluster_columns(M: pd.DataFrame, method="average", metric="euclidean") -> List[str]:
    try:
        g = sns.clustermap(M, row_cluster=False, col_cluster=True, method=method,
                           metric=metric, cmap="RdBu_r", xticklabels=False, yticklabels=False, cbar=False)
        order = M.columns[g.dendrogram_col.reordered_ind].tolist()
        plt.close(g.fig)
        return order
    except Exception as e:
        warnings.warn(f"clustermap fallback (no SciPy?): {e}. Using SVD order.")
        X = M.fillna(0).values
        X = X - X.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        idx = np.argsort(vt[0])
        return M.columns[idx].tolist()

def cluster_rows(M: pd.DataFrame, method="average", metric="euclidean") -> List[str]:
    try:
        g = sns.clustermap(M, row_cluster=True, col_cluster=False, method=method,
                           metric=metric, cmap="RdBu_r", xticklabels=False, yticklabels=False, cbar=False)
        order = M.index[g.dendrogram_row.reordered_ind].tolist()
        plt.close(g.fig)
        return order
    except Exception as e:
        warnings.warn(f"Row clustering fallback: {e}. Keeping given row order.")
        return M.index.tolist()

def orient_left_to_right(M: pd.DataFrame, cols: List[str]) -> List[str]:
    """Flip order if needed so left→right ≈ hot→cold (by mean z across genes)."""
    if not cols:
        return cols
    s = M.mean(axis=0).reindex(cols)
    if s.iloc[0] < s.iloc[-1]:
        cols = cols[::-1]
    return cols

# -----------------
# LOAD & PREP
# -----------------
print("Using SOURCE:", SOURCE)
panel = load_panel(SOURCE, VALUE_CANDIDATES)
all_genes = sorted(panel['cyt'].unique().tolist())
print(f"Loaded {os.path.basename(SOURCE)}: "
      f"{panel['SAMPLE_ID'].nunique()} samples, {len(all_genes)} genes.")

# Keep TEX names for display flipping
neg_genes = set(TEX)

# Intersect with available genes
T_ABUNDANCE = intersect_panel(all_genes, T_ABUNDANCE)
PEX         = intersect_panel(all_genes, PEX)
TEX         = intersect_panel(all_genes, TEX)
ACTIVATION  = intersect_panel(all_genes, ACTIVATION)

def oriented_matrix(df_long: pd.DataFrame, genes: List[str], scale: str) -> pd.DataFrame:
    if not genes:
        return pd.DataFrame()
    D = df_long[df_long['cyt'].isin(genes)].copy()
    D = add_gene_scaled(D, scale, "value")
    # Flip TEX so red = hot visually
    D.loc[D['cyt'].isin(neg_genes), 'z'] *= -1
    return D.pivot(index='cyt', columns='SAMPLE_ID', values='z')

# Column order
if CLUSTER_MODE == "none":
    all_block_genes = sorted(set().union(*[set(g) for g in PANEL_BLOCKS.values()]))
    M_all = oriented_matrix(panel, all_block_genes, DISPLAY_SCALE).fillna(0)
    ordered_samples = orient_left_to_right(M_all, cluster_columns(M_all))
elif CLUSTER_MODE == "global_samples":
    all_block_genes = sorted(set().union(*[set(g) for g in PANEL_BLOCKS.values()]))
    M_all = oriented_matrix(panel, all_block_genes, DISPLAY_SCALE).fillna(0)
    ordered_samples = orient_left_to_right(M_all, cluster_columns(M_all))
elif CLUSTER_MODE == "per_panel":
    ordered_samples = None
else:
    raise ValueError("CLUSTER_MODE must be one of: 'none', 'global_samples', 'per_panel'")

# Blocks to draw
panel_items = []
for title, genes in PANEL_BLOCKS.items():
    g_avail = [g for g in genes if g in all_genes]
    if not g_avail:
        print(f"Skipping '{title}' (no genes present).")
        continue
    panel_items.append((title, g_avail))

if not panel_items:
    raise ValueError("No panels with available genes to plot.")

# Figure size proportional to rows
ROW_HEIGHT_INCH = 0.35
MIN_BLOCK_ROWS = 2
height_ratios = [max(MIN_BLOCK_ROWS, len(g)) for _, g in panel_items]
fig_height = sum(r * ROW_HEIGHT_INCH for r in height_ratios) + 1.5

fig, axes = plt.subplots(
    len(panel_items), 1,
    figsize=(12, fig_height),
    gridspec_kw={'height_ratios': height_ratios},
    constrained_layout=True
)
if len(panel_items) == 1:
    axes = [axes]

def plot_block(ax, df_long, genes, title, col_order=None):
    D = df_long[df_long['cyt'].isin(genes)].copy()
    D = add_gene_scaled(D, DISPLAY_SCALE, "value")
    D.loc[D['cyt'].isin(neg_genes), 'z'] *= -1
    Mz = D.pivot(index='cyt', columns='SAMPLE_ID', values='z').reindex(index=genes)

    if col_order is not None:
        Mz = Mz.reindex(columns=col_order)
    elif CLUSTER_MODE == "per_panel":
        Mz = Mz.fillna(0)
        col_order = orient_left_to_right(Mz, cluster_columns(Mz))
        Mz = Mz.reindex(columns=col_order)

    if ROW_CLUSTER_PER_PANEL and len(genes) > 1:
        Mz = Mz.reindex(index=cluster_rows(Mz.fillna(0)))

    sns.heatmap(Mz, ax=ax, cmap="RdBu_r", center=0, vmin=VMIN, vmax=VMAX,
                cbar=False, xticklabels=False, yticklabels=True)

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    for lbl in ax.get_yticklabels():
        lbl.set_rotation(0); lbl.set_ha('right')

# Draw all blocks
for ax, (title, genes) in zip(axes, panel_items):
    col_order = None if CLUSTER_MODE == "per_panel" else ordered_samples
    plot_block(ax, panel, genes, title, col_order=col_order)

# Colourbar
sm = mpl.cm.ScalarMappable(cmap="RdBu_r", norm=mpl.colors.Normalize(vmin=VMIN, vmax=VMAX))
sm.set_array([])
cax = fig.add_axes([1.1, 0.12, 0.02, 0.76])
cb = plt.colorbar(sm, cax=cax)
label_map = {"rankz": "gene-wise rank-normalised score (z)",
             "zsc": "expression z-score",
             "robustz": "gene-wise robust z-score"}
cb.set_label(label_map.get(DISPLAY_SCALE, "z-score"), fontsize=18)
cb.ax.tick_params(labelsize=16)

header = {"none":"(no sample clustering)",
          "global_samples":"(samples clustered globally)",
          "per_panel":"(each panel clustered independently)"}[CLUSTER_MODE]
axes[0].set_title(list(dict(panel_items).keys())[0] + "  " + header, fontsize=18)

# Save PNG inside repo
base = os.path.splitext(os.path.basename(SOURCE))[0].lower().replace(" ", "_")
FIGDIR = "01-hot-cold-tcell/figures"
os.makedirs(FIGDIR, exist_ok=True)
FIGPATH = f"{FIGDIR}/{base}_heatmap_example.png"
plt.savefig(FIGPATH, dpi=300, bbox_inches="tight")
print(f"Saved figure: {FIGPATH}")

plt.show()
