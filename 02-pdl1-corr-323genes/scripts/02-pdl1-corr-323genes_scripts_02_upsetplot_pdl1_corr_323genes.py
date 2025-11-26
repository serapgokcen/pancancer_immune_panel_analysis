#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

"""
PDL1-correlated genes – UpSet plot across cancers
-------------------------------------------------
1) Loads one CSV per cancer type, each containing the list of genes
   positively correlated with CD274 (r >= 0.5).
2) Builds a presence/absence matrix (genes x cancer types).
3) Generates an UpSet plot of gene-set intersections.
4) Exports the genes in each intersection to an Excel file.
"""

import argparse
from pathlib import Path, PureWindowsPath
from collections import defaultdict

import pandas as pd
from upsetplot import UpSet
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Repo-safe path resolver
#   - Keeps your original Windows paths untouched.
#   - If a Windows path does not exist, it searches the repo for a file
#     with the same filename in these directories.
# -------------------------------------------------------------------------
SEARCH_DIRS = [
    Path("./data"),
    Path("./02-pdl1-corr-323genes/data"),
    Path("./01-hot-cold-tcell/data"),
]


def resolve_repo_path(original_path: str) -> Path:
    p = Path(original_path)

    # 1) On your Windows machine, use the original path if it exists
    if p.exists():
        return p

    # 2) Otherwise, try to find a file with the same basename in SEARCH_DIRS
    fname = PureWindowsPath(original_path).name
    for d in SEARCH_DIRS:
        candidate = d / fname
        if candidate.exists():
            return candidate

    # 3) Not found anywhere
    return p


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build an UpSet plot of PDL1-correlated gene overlaps across cancers."
    )
    ap.add_argument(
        "--outdir",
        default="results_pdl1_upset",
        help="Output directory for the UpSet plot and Excel file.",
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="If set, skip cancers whose CSVs are missing instead of failing.",
    )

    # Notebook/IPython-safe: ignore unknown args like "-f kernel-xxxx.json"
    args, _ = ap.parse_known_args()
    return args


# =========================
# CONFIG – KEEP WINDOWS PATHS
# =========================

WINDOWS_FILES = [
    r"C:\extracted_strong_correlations\bladder_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\brainlower_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\colorectalcorrgreen_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\esophagealcorr_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\MEL-T-COM_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\ovariancorr_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\prostatecorr_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\stomachcorr_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\breast_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\cervical_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\glioblastoma_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\headandneck_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\kidneyrenalclear_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\kidneyrenalpapil_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\liverhepato_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\lung_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\lungsquamous_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\pancreatic_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\pheochromocytoma_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\sarcoma_genes_pos_correlated_rge0.5.csv",
    r"C:\extracted_strong_correlations\thyroid_genes_pos_correlated_rge0.5.csv",
]

CANCER_TYPES = [
    "Bladder",
    "Brain Lower",
    "Colorectal",
    "Esophageal",
    "Melanoma",
    "Ovarian",
    "Prostate",
    "Stomach",
    "Breast",
    "Cervical",
    "Glioblastoma",
    "Head and Neck",
    "Kidney Renal Clear",
    "Kidney Renal Papillary",
    "Liver Hepatocarcinoma",
    "Lung Adenocarcinoma",
    "Lung Squamous Cell",
    "Pancreatic",
    "Pheochromocytoma",
    "Sarcoma",
    "Thyroid",
]


# =========================
# MAIN
# =========================
def main(args):
    if len(WINDOWS_FILES) != len(CANCER_TYPES):
        raise SystemExit(
            f"CONFIG ERROR: WINDOWS_FILES ({len(WINDOWS_FILES)}) and "
            f"CANCER_TYPES ({len(CANCER_TYPES)}) have different lengths."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve paths and optionally skip missing
    pairs = []  # list of (cancer_label, resolved_path)
    missing = []
    for label, win_path in zip(CANCER_TYPES, WINDOWS_FILES):
        rp = resolve_repo_path(win_path)
        if rp.exists():
            pairs.append((label, rp))
        else:
            missing.append((label, rp))

    if missing and not args.allow_missing:
        msg = "\n".join([f"  - {lab}: {p}" for lab, p in missing])
        raise SystemExit(
            "Missing required CSVs for one or more cancers.\n"
            "Either add the files to the repo data folders (matching filenames), "
            "or rerun with --allow-missing to skip them.\n"
            f"Missing:\n{msg}"
        )

    if missing and args.allow_missing:
        msg = "\n".join([f"  - {lab}: {p}" for lab, p in missing])
        print("[WARN] Running in demo mode, skipping missing cancers:\n" + msg)

    if not pairs:
        raise SystemExit("No valid cancer CSVs found; nothing to plot.")

    # Read all genes into sets
    gene_sets = {}
    for label, path in pairs:
        df = pd.read_csv(path)
        # first column assumed to contain gene symbols
        genes = set(df.iloc[:, 0].astype(str).str.strip())
        gene_sets[label] = genes

    # Build presence/absence matrix
    all_genes = set().union(*gene_sets.values())
    gene_matrix = pd.DataFrame(
        {
            label: [gene in gene_sets[label] for gene in all_genes]
            for label in gene_sets.keys()
        },
        index=sorted(all_genes),
    ).astype(int)

    # UpSet expects groupby-style data (MultiIndex)
    labels_in_use = list(gene_sets.keys())
    upset_data = gene_matrix.groupby(labels_in_use).size()

    # Plot UpSet
    plt.figure(figsize=(22, 8))
    upset = UpSet(
        upset_data,
        show_counts="%d",
        intersection_plot_elements=35,
    )
    upset.plot()
    plt.suptitle(
        "Gene Set Intersections Across Cancer Types",
        fontsize=20,
        fontweight="bold",
    )

    # Save plot
    fig = plt.gcf()
    png_path = outdir / "pdl1_corr_323genes_upsetplot.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved UpSet plot to: {png_path}")

    # Show (useful when running locally)
    plt.show()

    # Export genes in each intersection to Excel
    intersection_dict = defaultdict(list)
    for gene, row in gene_matrix.iterrows():
        present_types = tuple(
            [lab for lab, val in zip(labels_in_use, row.values) if val]
        )
        if present_types:  # skip genes with no presence at all
            intersection_dict[present_types].append(gene)

    intersection_df = pd.DataFrame(
        [
            {
                "Intersection": "&".join(inter),
                "GeneCount": len(genes),
                "Genes": ", ".join(sorted(genes)),
            }
            for inter, genes in intersection_dict.items()
        ]
    )

    excel_path = outdir / "Gene_intersections.xlsx"
    intersection_df.to_excel(excel_path, index=False)
    print(f"[INFO] Saved intersection table to: {excel_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)


# In[ ]:




