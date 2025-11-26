#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

"""
01_pdl1_corr_323genes.py

Part 1: build PDL1 (CD274)-anchored heatmap for one cancer type.
Part 2: build a 1D correlation heatmap (genes sorted by correlation with CD274).

- Loads tidy CSV [SAMPLE_ID, cyt, zsc]
- Reorders genes by a fixed gene list (desired_order)
- Sorts samples by CD274 expression (high -> low)
- Clusters other genes by expression pattern (rows)
- Colours rows by correlation with CD274
- Saves:
    - pdl1_corr_heatmap.png           (expression heatmap with row colours)
    - pdl1_corr_ranked_genes.png      (1-column correlation heatmap)
"""

import argparse
from pathlib import Path, PureWindowsPath

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# -------------------------------------------------------------------
# Repo-safe path resolver
# -------------------------------------------------------------------

SEARCH_DIRS = [
    Path("./data"),                        # repo root /data
    Path("./02-pdl1-corr-323genes/data"), # local subfolder
    Path("../data"),                       # if run from inside subfolder
]

def resolve_repo_path(original_path: str) -> Path:
    """
    Keep your Windows path for local runs.
    If that file does not exist, try to find the same filename
    inside SEARCH_DIRS.
    """
    p = Path(original_path)

    # 1) Windows / local machine
    if p.exists():
        return p

    # 2) Look for filename inside repo search dirs
    fname = PureWindowsPath(original_path).name  # e.g. "pancreatic_323_transformed_fixed.csv"
    for d in SEARCH_DIRS:
        candidate = d / fname
        if candidate.exists():
            return candidate

    # 3) Not found anywhere
    return p


def parse_args():
    ap = argparse.ArgumentParser(description="PDL1-anchored heatmaps for one cancer type.")
    ap.add_argument(
        "--file",
        default=r"C:\TRANSFORMED_CYTO\323\pancreatic_323_transformed_fixed.csv",
        help="Path to tidy CSV with columns SAMPLE_ID, cyt, zsc "
             "(default: pancreatic_323_transformed_fixed.csv on Windows)."
    )
    ap.add_argument(
        "--outdir",
        default="results_pdl1_corr",
        help="Output directory for the heatmap figures (default: results_pdl1_corr)."
    )
    # Notebook/IPython-safe: ignore unknown args like "-f kernel-xxxx.json"
    args, _ = ap.parse_known_args()
    return args


def main(args):
    # Resolve paths and create output folder
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    file_path = resolve_repo_path(args.file)
    print(f"[INFO] Using file: {file_path} (exists? {file_path.exists()})")

    # ----------------------------
    # BLOCK 1: PDL1-ANCHORED HEATMAP
    # ----------------------------
    df = pd.read_csv(file_path)
    df_pivot = df.pivot(index="cyt", columns="SAMPLE_ID", values="zsc")

    # Define genes of interest (CD274 in the list)
    desired_order = ['ADAM8','AHR','AIF1','JAML','ANXA1','ANXA2','APBB1','AQP3','ARG1','ATM','BACH2','BATF','BATF3','BCL2','BCL6','BHLHE40',
                     'BCL2L11','BNC2','BST2','BTG1','BTG2','CAPN3','CCL18','CCL20','CCL3','CCL4','CCL5','CCND3','CCR10','CCR2','CCR3','CCR4',
                     'CCR5','CCR6','CCR7','CD109','CSF1R','ITGAM','ITGAX','TNFRSF9','CD163','CD2','MS4A1','CD200',
                     'CD226','CD248','DPP4','CD27','CD274','CD276','BTN3A1','CD28','CD38','ENTPD1','CD3D','CD3E','CD4','CD40',
                     'CD40LG','CD44','CD47','CD48','NCAM1','CD58','CD59','CEACAM8','CD68','CD69','CD7','CD70','CD83',
                     'CD86','CD8A','CD8B','KLRD1','CDK5R1','CDKN1A','CENPV','CIITA','CLEC9A','CMKLR1','COL10A1','CORO1A','PTGDR2','CSF1','CSF2',
                     'CTLA4','CTSW','CXCL10','CXCL11','CXCL13','CXCL9','CXCR3','CXCR4','CXCR5','CXCR6','CYTOR','DACT1','DUSP1',
                     'EGR2','EOMES','EZR','FABP5','FAS','FASLG','FCGR1A','FGFBP2','FHL1','FOS','FOXO4','FOXP3','G0S2',
                     'GATA3','GCK','GIMAP7','GNLY','GZMA','GZMB','GZMH','GZMK','GZMM','HAVCR2','HBP1','IKZF2','HES1','HIF1A',
                     'HLA-DQA1','HLA-DRA','HLA-DRB1','HLA-E','HMOX1','HOPX','ICAM3','ICOS','ID2','ID3','IDO1','IFI16','IFI35','IFI44L','IFI6',
                     'IFIT1','IFIT2','IFIT3','IFITM1','IFITM3','IFNAR1','IFNG','IFNGR1','IKZF1','IL10','IL10RA','IL12A','IL12B','IL12RB1','IL13',
                     'IL17A','IL17F','IL17RB','IL18','IL18R1','IL1B','IL1R1','IL1R2','IL2','IL21','IL21R','IL22','IL23R','IL26','IL2RA','IL2RB',
                     'IL2RG','IL32','IL1RL1','IL4','IL4R','IL5','IL6','IL6R','IL6ST','IL7R','IL9','IRF1','IRF4','IRF7','IRF8','ISG15','ITGA1',
                     'ITGA4','ITGAE','ITGAL','ITGB7','ITM2C','JAK2','JUN','KLF2','KLF6','KLRB1','KLRG1','LAG3','LEF1','LGALS1','LGALS3','LIF',
                     'LIMS1','LTA','LTB','LY9','LYAR','LYZ','MAF','MDM4','MIR155HG','MIR4435-2HG','MKI67','MX1','MX2','MYB','NCR3','NFATC1',
                     'NFKBID','NFKBIZ','NKG7','NLRP3','NME1','NOS2','NOSIP','NR4A1','NR4A3','NSG2','NT5E','OAS1','OAS2','OAS3','OASL',
                     'ORC6','P2RX7','PACSIN1','PDCD1','PDCD4','POU6F1','PPARG','PRDM1','PRF1','PRKAB2','PSMB10','PTGER2','PTPRC',
                     'PYCARD','RAI14','RGS10','RORA','RORC','RPS6KA2','RSAD2','RUNX1','RUNX2','S100A11','S100A4','S100A7','SELL','SELPLG',
                     'SERPINE1','SH2B3','SH2D1A','SIGLEC10','SIGLECL1','SIRPA','SLAMF6','SLAMF7','SLC4A10','SLPI','SMAD2','SMAD7','SOCS1',
                     'SPI1','SPINK1','SPN','STAT1','STAT3','STAT4','STAT5A','STAT6','SYNPO','TAGAP','TBX21','TCF12','TCF7','TGFB1','TGFB3',
                     'TIGIT','TIMP1','TMEM163','TNF','TNFRSF14','TNFRSF18','TNFRSF1B','TNFRSF4','TOX','TOX2','TRADD',
                     'TRIM21','TSPYL2','TUBA1A','TUBA1B','TUBB','TUBB4B','USP10','VDR','VEGFA','VIM','WARS1','WHAMMP3','WNT1','XAF1','XCL1','XCL2','ZBTB16','ZBTB32','ZBTB49','ZEB2','ZNF683']

    # Reorder genes by desired_order
    reordered_df = df_pivot.loc[desired_order]

    # Fix sample order based on CD274 expression
    sample_order = reordered_df.loc['CD274'].sort_values(ascending=False).index.tolist()
    reordered_df = reordered_df[sample_order]

    # Split CD274 from other genes
    cd274_row = reordered_df.loc[['CD274']]
    other_rows = reordered_df.drop(index='CD274')

    # Compute correlation of each gene with CD274
    correlations = other_rows.apply(lambda row: row.corr(cd274_row.loc['CD274']), axis=1)

    # Cluster other genes (rows only), based on fixed sample order
    linkage_matrix = linkage(other_rows.values, method='average', metric='euclidean')
    clustered_indices = leaves_list(linkage_matrix)
    clustered_gene_order = other_rows.index[clustered_indices]

    # Final data for plotting: CD274 on top, others clustered
    final_df = pd.concat([cd274_row, other_rows.loc[clustered_gene_order]])
    final_df.index.name = ''
    final_df.columns.name = ''

    # Use PRGn colormap for correlation bar (green to purple)
    norm = Normalize(vmin=correlations.min(), vmax=correlations.max())
    corr_cmap = cm.get_cmap("PRGn")
    row_color_list = [(0.5, 0.5, 0.5)]  # CD274 in gray
    row_color_list += [corr_cmap(norm(corr))[:3] for corr in correlations.loc[clustered_gene_order]]

    g = sns.clustermap(
        final_df,
        cmap='RdBu_r',              # Heatmap colormap: red–blue
        annot=False,
        xticklabels=False,
        yticklabels=True,
        center=0,
        vmax=4,
        vmin=-4,
        row_cluster=False,
        col_cluster=False,
        figsize=(60,40),
        cbar_pos=(0.1, 0.6, 0.03, 0.18),
        row_colors=row_color_list
    )

    for ax in [g.ax_row_dendrogram, g.ax_col_dendrogram]:
        ax.set_visible(False)

    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.tick_params(axis='y', labelsize=12)

    # --- HEATMAP COLORBAR LABEL & FONTSIZE ---
    heatmap_cbar = g.cax
    heatmap_cbar.set_ylabel("Heatmap", rotation=270, labelpad=25, fontsize=60)
    heatmap_cbar.tick_params(labelsize=40)

    # --- CORRELATION COLORBAR: Green–Purple ---
    sm = ScalarMappable(cmap=corr_cmap, norm=norm)
    sm.set_array([])

    cbar_ax = g.fig.add_axes([0.1, 0.029, 0.02, 0.45])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Correlation", rotation=270, labelpad=40, fontsize=60)
    cbar.ax.tick_params(labelsize=40)

    # Save big heatmap
    fig_path = outdir / "pdl1_corr_heatmap.png"
    g.fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)
    print(f"[INFO] Saved heatmap to {fig_path}")

    # ----------------------------
    # BLOCK 2: 1-COLUMN CORRELATION HEATMAP
    # ----------------------------

    # 1) Sort genes by correlation descending
    sorted_genes = correlations.sort_values(ascending=False).index

    # 2) Build a one-column DataFrame in that order
    corr_df = pd.DataFrame(
        correlations.loc[sorted_genes],
        columns=['Correlation']
    )

    # 3) Plot into a new fig/ax so we can tweak fonts correctly
    fig2, ax = plt.subplots(figsize=(4, len(sorted_genes) * 0.4))

    sns.heatmap(
        corr_df,
        ax=ax,
        cmap='PRGn',           # green–purple map
        cbar=True,
        cbar_kws={
            'fraction': 0.3,   # make the bar narrower
            'aspect': 60,      # tall colour bar
        },
        yticklabels=True,
        xticklabels=['']
    )

    # 4) Increase label size and make bold
    ax.tick_params(axis='y', labelsize=16)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # 5) Clean up axes
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')

    plt.tight_layout()

    # 6) Save as high-quality PNG
    corr_fig_path = outdir / "pdl1_corr_ranked_genes.png"
    plt.savefig(corr_fig_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved ranked-gene correlation heatmap to {corr_fig_path}")

    plt.show()
    plt.close(fig2)


if __name__ == "__main__":
    args = parse_args()
    main(args)


# In[ ]:




