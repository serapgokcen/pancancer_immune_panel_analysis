#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ====================================
# HOT/COLD PROFILING with CLUSTERED, SCALED HEATMAP BLOCKS
# (CSV path or seaborn dataset name)
# ======================================
import os, warnings
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Dict

# ----------------
# CONFIG — EDIT ME
# ----------------
SOURCE = r"C:\TRANSFORMED_CYTO\323\colorectalcorr_323_transformed_fixed.csv"   # or "BREAST_ALL_transformed_fixed"
VALUE_CANDIDATES = ("zsc", "value")

# Panels (tweak to your dataset)
T_ABUNDANCE = ['CD3D','CD3E','CD8A','PTPRC']             # quantity
PEX         = ['SLAMF6','CCR7','TCF7']                   # progenitor-exhausted
TEX         = ['TOX','CXCL13','HAVCR2','TIGIT','LAG3','PDCD1','CTLA4']   # terminal-exhausted
ACTIVATION  = ['IFNG','ICOS','TNFRSF9','CD69','CD40LG','CD274']          # activation/IFNγ axis

# Which blocks to plot (top→bottom)
PANEL_BLOCKS: Dict[str, List[str]] = {
    "Exhaustion (TEX)": TEX,
    "Progenitor (PEX)": PEX,  
    "T-cell Abundance": T_ABUNDANCE,
    "Activation": ACTIVATION,
}

# === SCALING CHOICES ===
# "rankz"  = pooled rank → normal z (robust, cross-panel comparable)
# "zsc"    = use dataset-provided per-gene z-scores directly (easy to explain)
# "robustz"= per-gene median/MAD z
SCORE_SCALE   = "rankz"   # used for indices, HotScore, consistency, clustering
DISPLAY_SCALE = "rankz"   # used only for heatmap colouring

# Label thresholds
THRESHOLDS = dict(z_hot=0.7, z_ihot_lo=0.2, z_icold_hi=-0.2, z_cold=-0.7,
                  p_hot=0.60, p_ihot=0.50, p_icold=0.35, p_cold=0.30)

# Clustering controls
CLUSTER_MODE = "global_samples"       # "none" | "global_samples" | "per_panel"
ROW_CLUSTER_PER_PANEL = True          # cluster genes (rows) inside each block?

# Fixed display limits for all heatmaps & colourbar
VMIN, VMAX = -4, 4

# Output names
base = os.path.splitext(os.path.basename(SOURCE))[0].lower().replace(" ", "_")
LABELS_OUT  = f"{base}_hot_labels.csv"
SUMMARY_OUT = f"{base}_hot_summary.csv"

# -------------
# HELPERS
# -------------
def inv_norm_cdf(p):
    # Acklam probit approximation
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
        q[m] = (((((c[0]*t + c[1])*t + c[2])*t + c[3])*t + c[4])*t + c[5]) /                ((((d[0]*t + d[1])*t + d[2])*t + d[3])*t + 1)
    m = (p >= plow) & (p <= phigh)
    if m.any():
        pp = p[m] - 0.5; t2 = pp*pp
        q[m] = (((((a[0]*t2 + a[1])*t2 + a[2])*t2 + a[3])*t2 + a[4])*t2 + a[5]) * pp /                (((((b[0]*t2 + b[1])*t2 + b[2])*t2 + b[3])*t2 + b[4])*t2 + 1)
    m = p > phigh
    if m.any():
        pp = 1 - p[m]; t = np.sqrt(-2*np.log(pp))
        q[m] = -(((((c[0]*t + c[1])*t + c[2])*t + c[3])*t + c[4])*t + c[5]) /                  ((((d[0]*t + d[1])*t + d[2])*t + d[3])*t + 1)
    return q

def pooled_rank_to_norm(series: pd.Series) -> pd.Series:
    r = (series.rank(method="average") - 0.375) / (len(series) + 0.25)
    r = r.clip(1e-12, 1-1e-12)
    return pd.Series(inv_norm_cdf(r), index=series.index)

def mad(x: np.ndarray) -> float:
    return np.median(np.abs(x - np.median(x))) * 1.4826 + 1e-12

def standardise(x: pd.Series) -> pd.Series:
    return (x - x.median()) / mad(x.values)

def load_panel(source: str,
               value_candidates=("zsc","value"),
               required=("SAMPLE_ID","cyt")) -> pd.DataFrame:
    if os.path.exists(source):
        df = pd.read_csv(source)
    else:
        df = sns.load_dataset(source)
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
    if missing: print(f"  ! Missing genes (ignored): {missing}")
    return genes

def add_gene_scaled(df_long: pd.DataFrame, scale: str, value_col: str = "value") -> pd.DataFrame:
    D = df_long.copy()
    if scale == "rankz":
        D['z'] = D.groupby('cyt')[value_col].transform(pooled_rank_to_norm)
    elif scale == "zsc":
        D['z'] = D[value_col]   # dataset z-scores already
    elif scale == "robustz":
        D['z'] = D.groupby('cyt')[value_col].transform(lambda v: (v - v.median()) / mad(v.values))
    else:
        raise ValueError("scale must be one of: 'rankz', 'zsc', 'robustz'")
    return D

def compute_index(df_long: pd.DataFrame, genes: List[str], scale: str = "rankz") -> pd.Series:
    D = df_long[df_long['cyt'].isin(genes)].copy()
    if D.empty:
        return pd.Series(dtype=float)
    D = add_gene_scaled(D, scale, "value")
    return D.groupby('SAMPLE_ID')['z'].mean()

def cluster_columns(M: pd.DataFrame, method="average", metric="euclidean") -> List[str]:
    """Return column order from hierarchical clustering. Falls back to SVD order if SciPy missing."""
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

# --- NEW: ensure left→right = red→blue orientation without changing clusters ---
def orient_left_to_right(M: pd.DataFrame, cols: List[str]) -> List[str]:
    """
    Given an oriented matrix M (where higher values = 'hot'), and a column order,
    flip order if needed so that leftmost has the higher overall heat (red) and
    rightmost is cooler (blue).
    """
    if not cols:
        return cols
    s = M.mean(axis=0).reindex(cols)
    # If left column is cooler than right column, reverse the order
    if s.iloc[0] < s.iloc[-1]:
        cols = cols[::-1]
    return cols

# -----------------
# LOAD & PREP
# -----------------
panel = load_panel(SOURCE, VALUE_CANDIDATES)
all_genes = sorted(panel['cyt'].unique().tolist())
print(f"Loaded {os.path.basename(SOURCE)}: {panel['SAMPLE_ID'].nunique()} samples, {len(all_genes)} genes.")

# Intersect requested lists
T_ABUNDANCE = intersect_panel(all_genes, T_ABUNDANCE)
PEX         = intersect_panel(all_genes, PEX)
TEX         = intersect_panel(all_genes, TEX)
ACTIVATION  = intersect_panel(all_genes, ACTIVATION)

# --------------
# SCORES (indices, HotScore, consistency)
# --------------
T_idx   = compute_index(panel, T_ABUNDANCE, scale=SCORE_SCALE)
PEX_idx = compute_index(panel, PEX,         scale=SCORE_SCALE)
TEX_idx = compute_index(panel, TEX,         scale=SCORE_SCALE)
A_idx   = compute_index(panel, ACTIVATION,  scale=SCORE_SCALE)

parts = []
for name, s, sign in [('T',T_idx,+1), ('A',A_idx,+1), ('PEX',PEX_idx,+1), ('TEX',TEX_idx,-1)]:
    if len(s): parts.append(sign * standardise(s))
if not parts:
    raise ValueError("No valid panels—check gene lists/dataset.")

HotScore   = pd.concat(parts, axis=1).mean(axis=1).rename('HotScore')
HotScore_Z = standardise(HotScore).rename('HotScore_Z')

# Consistency (signed): pos panels should be >0, TEX should be <0 in the chosen SCORE_SCALE
pos_genes = set(T_ABUNDANCE) | set(ACTIVATION) | set(PEX)
neg_genes = set(TEX)
Pz = add_gene_scaled(panel, SCORE_SCALE, "value")

# --- Make 'aligns_hot' numeric (True→1, False→0) to avoid aggregation errors
tmp = np.where(Pz['cyt'].isin(pos_genes), Pz['z'] > 0,
               np.where(Pz['cyt'].isin(neg_genes), Pz['z'] < 0, np.nan))
Pz['aligns_hot'] = pd.Series(tmp, index=Pz.index).map({True:1.0, False:0.0}).astype(float)

p_consistency = (Pz.dropna(subset=['aligns_hot'])
                   .groupby('SAMPLE_ID')['aligns_hot'].mean()
                   .rename('p_consistency'))

def label_by_hotness(Z_hot, p_consistency, thr: Dict[str, float]) -> pd.Series:
    labs = []
    for z, p in zip(Z_hot, p_consistency):
        if (z >= thr['z_hot']) and (p >= thr['p_hot']):
            labs.append('hot')
        elif (z >= thr['z_ihot_lo']) or (p >= thr['p_ihot']):
            labs.append('intermediate-hot')
        elif (z <= thr['z_cold']) and (p <= thr['p_cold']):
            labs.append('cold')
        elif (z <= thr['z_icold_hi']) or (p <= thr['p_icold']):
            labs.append('intermediate-cold')
        else:
            labs.append('intermediate-hot' if z >= 0 else 'intermediate-cold')
    return pd.Series(labs, index=Z_hot.index)

labels = label_by_hotness(HotScore_Z, p_consistency, THRESHOLDS).rename('label')

labels_df = (
    pd.concat([HotScore, HotScore_Z, p_consistency,
               T_idx.rename('T_idx'), A_idx.rename('A_idx'),
               PEX_idx.rename('PEX_idx'), TEX_idx.rename('TEX_idx'),
               labels], axis=1)
      .sort_values('HotScore_Z', ascending=False)
)
summary_df = (labels_df['label'].value_counts(normalize=True)
              .rename('fraction').to_frame()
              .assign(n=lambda d: (d['fraction']*len(labels_df)).round().astype(int)))

labels_df.to_csv(LABELS_OUT, index=True)
summary_df.to_csv(SUMMARY_OUT, index=True)
print(f"Saved:\n - {LABELS_OUT}\n - {SUMMARY_OUT}")

# ------------------------
# HEATMAPS with clustering (height ∝ #genes)
# ------------------------
def oriented_matrix(df_long: pd.DataFrame, genes: List[str], scale: str) -> pd.DataFrame:
    if not genes:
        return pd.DataFrame()
    D = df_long[df_long['cyt'].isin(genes)].copy()
    D = add_gene_scaled(D, scale, "value")
    # flip TEX to align with "hot" across blocks
    D.loc[D['cyt'].isin(neg_genes), 'z'] *= -1
    M = D.pivot(index='cyt', columns='SAMPLE_ID', values='z')
    return M

# Decide column order
if CLUSTER_MODE == "none":
    ordered_samples = labels_df.index.tolist()  # HotScore order (desc)
elif CLUSTER_MODE == "global_samples":
    all_block_genes = sorted(set().union(*[set(glist) for glist in PANEL_BLOCKS.values()]))
    M_all = oriented_matrix(panel, all_block_genes, scale=SCORE_SCALE).fillna(0)
    ordered_samples = cluster_columns(M_all)
    # --- NEW: enforce left→right = hot→cold orientation
    ordered_samples = orient_left_to_right(M_all, ordered_samples)
elif CLUSTER_MODE == "per_panel":
    ordered_samples = None  # handled inside each block
else:
    raise ValueError("CLUSTER_MODE must be one of: 'none', 'global_samples', 'per_panel'")

# ---- compute height ratios from available genes per block ----
panel_items = []
for title, genes in PANEL_BLOCKS.items():
    g_avail = [g for g in genes if g in all_genes]
    if len(g_avail) == 0:
        print(f"Skipping '{title}' (no genes present).")
        continue
    panel_items.append((title, g_avail))

if not panel_items:
    raise ValueError("No panels with available genes to plot.")

ROW_HEIGHT_INCH = 0.35
MIN_BLOCK_ROWS = 2
height_ratios = [max(MIN_BLOCK_ROWS, len(glist)) for _, glist in panel_items]
fig_height = sum(r * ROW_HEIGHT_INCH for r in height_ratios) + 1.5  # +margin for titles/colourbar

fig, axes = plt.subplots(
    len(panel_items), 1,
    figsize=(12, fig_height),
    gridspec_kw={'height_ratios': height_ratios},
    constrained_layout=True
)
if len(panel_items) == 1:
    axes = [axes]

def plot_block(ax, df_long, genes, title, col_order=None):
    # scale for display
    D = df_long[df_long['cyt'].isin(genes)].copy()
    D = add_gene_scaled(D, DISPLAY_SCALE, "value")
    # --- NEW: flip TEX rows for display so red = hot in every block
    D.loc[D['cyt'].isin(neg_genes), 'z'] *= -1
    Mz = D.pivot(index='cyt', columns='SAMPLE_ID', values='z').reindex(index=genes)

    # column order
    if col_order is not None:
        Mz = Mz.reindex(columns=col_order)
    elif CLUSTER_MODE == "per_panel":
        col_order = cluster_columns(Mz.fillna(0))
        # --- NEW: enforce left→right = hot→cold per panel
        col_order = orient_left_to_right(Mz.fillna(0), col_order)
        Mz = Mz.reindex(columns=col_order)

    # optional row clustering
    if ROW_CLUSTER_PER_PANEL and len(genes) > 1:
        row_order = cluster_rows(Mz.fillna(0))
        Mz = Mz.reindex(index=row_order)

    sns.heatmap(Mz, ax=ax, cmap="RdBu_r", center=0, vmin=VMIN, vmax=VMAX,
                cbar=False, xticklabels=False, yticklabels=True)

    # styling: horizontal gene labels, size 18
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    for lbl in ax.get_yticklabels():
        lbl.set_rotation(0)
        lbl.set_ha('right')

# draw blocks
for ax, (title, genes) in zip(axes, panel_items):
    col_order = None if CLUSTER_MODE == "per_panel" else ordered_samples
    plot_block(ax, panel, genes, title, col_order=col_order)

# colourbar (fixed relative position) — matches heatmap limits
sm = mpl.cm.ScalarMappable(cmap="RdBu_r", norm=mpl.colors.Normalize(vmin=VMIN, vmax=VMAX))
sm.set_array([])
cax = fig.add_axes([1.1, 0.12, 0.02, 0.76])
cb = plt.colorbar(sm, cax=cax)

label_map = {
    "rankz":  "gene-wise rank-normalised score (z)",
    "zsc":    "expression z-score",
    "robustz":"gene-wise robust z-score"
}
cb.set_label(label_map.get(DISPLAY_SCALE, "z-score"), fontsize=18)
cb.ax.tick_params(labelsize=16)

header = {"none":"(samples sorted by HotScore)",
          "global_samples":"(samples clustered globally)",
          "per_panel":"(each panel clustered independently)"}[CLUSTER_MODE]
axes[0].set_title(list(dict(panel_items).keys())[0] + "  " + header, fontsize=18)

plt.show()


# In[ ]:




