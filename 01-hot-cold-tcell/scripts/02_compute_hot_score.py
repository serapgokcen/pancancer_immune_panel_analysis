#!/usr/bin/env python
# coding: utf-8

"""
PD1/PDL1 project — Hot/Intermediate/Cold via Abundance + Activation vs Exhaustion (zsc)
---------------------------------------------------------------------------------------
1) Loads one CSV per cancer type (tidy: [SAMPLE_ID, cyt, zsc]).
2) Uses dataset-provided z-scores (zsc) for per-gene normalization.
3) Builds panel indices:
     - T-cell Abundance (pro-hot)   -> + sign
     - Activation (pro-hot)         -> + sign
     - Progenitor-exhausted (pro-hot, optional) -> + sign
     - Terminal Exhaustion (anti-hot or gated) -> sign depends on mode
   Each panel index is median/MAD-standardized across all samples, then combined.
4) Computes HotScore → Z, and a signed “p_consistency” (% genes aligning with hot).
5) Classifies samples: cold / intermediate-cold / intermediate-hot / hot.
6) Outputs:
   - activation_labels_per_sample.csv (per-sample HotScore, Z, p, panel indices, label)
   - activation_summary_by_cancer.csv (fractions, CIs, medians, WHC, HI, HSS)
"""

import argparse
from pathlib import Path, PureWindowsPath
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

# -----------------------------------------------------------------------------
# Repo-safe path resolver:
# - Keeps your original Windows FILES untouched.
# - If the Windows path doesn't exist (Codespaces/Linux),
#   it searches the repo for the same filename.
# -----------------------------------------------------------------------------

SEARCH_DIRS = [
    Path("./data"),
    Path("./01-hot-cold-tcell/data"),
    Path("./02-pdl1-corr-323genes/data"),
]

def resolve_repo_path(original_path: str) -> Path:
    p = Path(original_path)

    # 1) If original path exists (your Windows machine), use it
    if p.exists():
        return p

    # 2) Otherwise try to find by filename inside repo search dirs
    fname = PureWindowsPath(original_path).name  # e.g., "BRCA.csv"
    for d in SEARCH_DIRS:
        candidate = d / fname
        if candidate.exists():
            return candidate

    # 3) Not found anywhere
    return p  # return original, caller decides what to do

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only selected cancers, e.g. --only melanoma lung"
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="If set, missing files are skipped with a warning (demo/Codespaces mode)."
    )
    ap.add_argument(
        "--outdir",
        default="results",
        help="Output directory for tables/figures."
    )
    # Notebook/IPython-safe: ignore unknown args like "-f kernel-xxxx.json"
    args, _ = ap.parse_known_args()
    return args


# =========================
# CONFIG — KEEP WINDOWS PATHS
# =========================
FILES = {
    "melanoma":   r"C:\TRANSFORMED_CYTO\323\melanoma_323_transformed_fixed.csv",   # should include abundance/exhaustion genes too
    "breast":     r"C:\TRANSFORMED_CYTO\323\breast_323_transformed_fixed.csv",
    "colorectal": r"C:\TRANSFORMED_CYTO\323\colorectalcorr_323_transformed_fixed.csv",
    "lung":       r"C:\TRANSFORMED_CYTO\323\lung_323_transformed_fixed.csv",
    "ovarian":    r"C:\TRANSFORMED_CYTO\323\ovariancorr_323_transformed_fixed.csv",
    "pancreatic": r"C:\TRANSFORMED_CYTO\323\pancreatic_323_transformed_fixed.csv",
    "prostate":   r"C:\TRANSFORMED_CYTO\323\prostatecorr_323_transformed_fixed.csv",
}

# =========================
# ARGUMENTS + FILE RESOLUTION
# =========================
args = parse_args()

# output directory (Path)
OUTDIR = Path(args.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)

# decide which cancers to run
selected = args.only if args.only else list(FILES.keys())

# filter to valid keys (avoid typos)
valid_selected = []
invalid = []
for k in selected:
    if k in FILES:
        valid_selected.append(k)
    else:
        invalid.append(k)

if invalid:
    print(f"[WARN] These --only keys are not in FILES and will be ignored: {invalid}")

# resolve paths
RESOLVED_FILES: Dict[str, str] = {}
missing = []
for cancer in valid_selected:
    rp = resolve_repo_path(FILES[cancer])
    if rp.exists():
        RESOLVED_FILES[cancer] = str(rp)  # keep downstream expecting str paths
    else:
        missing.append((cancer, str(rp)))

# enforce "require all" unless --allow-missing
if missing and not args.allow_missing:
    msg = "\n".join([f"  - {c}: {p}" for c, p in missing])
    raise SystemExit(
        "Missing required cancer CSVs (full 7-cancer run needs all files).\n"
        "Either add the missing files to the repo data folders, or run with --allow-missing for demo.\n"
        f"Missing:\n{msg}"
    )

# demo mode warning if skipping
if missing and args.allow_missing:
    msg = "\n".join([f"  - {c}: {p}" for c, p in missing])
    print("[WARN] Running in demo mode, skipping missing cancers:\n" + msg)

# overwrite FILES so the rest of your script uses resolved paths automatically
FILES = RESOLVED_FILES


# =========================
# PROJECT SETTINGS (UNCHANGED)
# =========================
INPUT_SHAPE = "tidy"        # our exports are tidy: [SAMPLE_ID, cyt, zsc]
VALUE_COL   = "zsc"         # column name in the CSVs
USE_RANK_FALLBACK = False   # Set True to use pooled rank→N(0,1) instead of supplied z-scores.

# Panels (edit if your dataset uses different symbols)
T_ABUNDANCE = ['CD3D','CD3E','CD8A','PTPRC']                 # quantity (pro-hot)
ACTIVATION  = ['IFNG','ICOS','TNFRSF9','CD69','CD40LG','CD274']  # pro-hot (PD-L1 axis included)
PEX         = ['SLAMF6','TCF7','CCR7']                       # progenitor exhausted (pro-hot, optional)
EXHAUSTION  = ['PDCD1','CTLA4','LAG3','TIGIT','HAVCR2','TOX','CXCL13']  # exhaustion panel

# Outputs
LABELS_OUT  = "activation_labels_per_sample.csv"
SUMMARY_OUT = "activation_summary_by_cancer.csv"

# Thresholds (amplitude Z + consistency p)
THRESHOLDS = dict(
    z_hot=1.0, z_ihot_lo=0.3, z_icold_hi=-0.3, z_cold=-1.0,
    p_hot=0.70, p_ihot=0.55, p_icold=0.30, p_cold=0.30
)

# PEX toggle (include progenitor-exhausted panel or not)
USE_PEX = True   # set to False to exclude PEX completely

# --- TEX gating by T-cell abundance ---
# How to combine Terminal Exhaustion (TEX):
#   "static_neg": old behaviour (always subtract TEX)
#   "gated_by_t": +TEX when T-cell abundance is high (within cancer), −TEX when low
TEX_MODE   = "gated_by_t"   # or "static_neg"

# Within-cancer threshold to decide "high" vs "low" abundance for gating
# Use "mean" to match your proposal; "median" is a robust alternative.
GATE_STAT  = "mean"         # "mean" or "median"

# Optional: use a smooth weight instead of a hard flip (helps near the threshold)
GATE_STYLE = "binary"       # "binary" or "tanh"
GATE_K     = 0.75           # slope for tanh; used only if GATE_STYLE=="tanh"

rng = np.random.default_rng(42)

def mad(x: np.ndarray) -> float:
    """Median absolute deviation scaled to ~SD for normal data."""
    return np.median(np.abs(x - np.median(x))) * 1.4826 + 1e-12

def standardise(series: pd.Series) -> pd.Series:
    """Median/MAD standardisation."""
    return (series - series.median()) / mad(series.values)

def binom_ci_wilson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval without SciPy."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

def bootstrap_ci(x: np.ndarray, fn, n_boot: int = 5000, alpha: float = 0.05) -> Tuple[float,float]:
    """Bootstrap CI for a statistic (e.g., np.median)."""
    x = np.asarray(x)
    if x.size == 0:
        return (np.nan, np.nan)
    stats = []
    for _ in range(n_boot):
        resample = rng.choice(x, size=x.size, replace=True)
        stats.append(fn(resample))
    lo, hi = np.quantile(stats, [alpha/2, 1 - alpha/2])
    return float(lo), float(hi)

# Accurate inverse normal CDF (probit) — Acklam approximation (only used if USE_RANK_FALLBACK=True)
def inv_norm_cdf(p):
    p = np.asarray(p, dtype=float)
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
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

# --------- Loading ----------
def load_tidy_csv(path: str, cancer_label: str, value_col: str) -> pd.DataFrame:
    """Load tidy CSV (SAMPLE_ID, cyt, value_col) and tag cancer_type. Auto-accepts 'value' or 'zsc'."""
    x = pd.read_csv(path)
    x.columns = [c.strip() for c in x.columns]
    candidates = [value_col, "value", "zsc", "expr", "expression"]
    vcol = next((c for c in candidates if c in x.columns), None)
    if not {"SAMPLE_ID", "cyt"}.issubset(x.columns) or vcol is None:
        raise ValueError(f"{path}: need columns SAMPLE_ID, cyt, and one of {candidates}. "
                         f"Found: {list(x.columns)}")
    out = x[["SAMPLE_ID", "cyt", vcol]].rename(columns={vcol: "value"}).copy()
    out["cancer_type"] = cancer_label
    return out[["cancer_type","SAMPLE_ID","cyt","value"]]

def build_pooled_dataframe(files: Dict[str, str],
                           input_shape: str,
                           value_col: str) -> pd.DataFrame:
    """Load all panels, return pooled tidy DataFrame with columns: [cancer_type, SAMPLE_ID, cyt, value]."""
    frames = []
    for ct, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file for {ct}: {path}")
        df = load_tidy_csv(path, ct, value_col=value_col)
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    # Drop missing and coerce numeric
    pooled = pooled.dropna(subset=["value"]).copy()
    pooled["value"] = pd.to_numeric(pooled["value"], errors="coerce")
    pooled = pooled.dropna(subset=["value"])
    # clean types
    pooled["SAMPLE_ID"] = pooled["SAMPLE_ID"].astype(str)
    pooled["cyt"] = pooled["cyt"].astype(str)
    pooled["cancer_type"] = pooled["cancer_type"].astype(str)
    return pooled

def add_gene_scaled(df_long: pd.DataFrame, use_rank_fallback: bool, value_col: str = "value") -> pd.DataFrame:
    """Return df with a 'z' column per gene: zsc if available (preferred), else pooled rank→z."""
    D = df_long.copy()
    if use_rank_fallback:
        D["z"] = D.groupby("cyt")[value_col].transform(pooled_rank_to_norm)
    else:
        D["z"] = D[value_col]  # use dataset z-scores
    return D

def compute_panel_index(df_long: pd.DataFrame, genes: List[str]) -> pd.Series:
    """Mean z across given genes per sample (returns empty Series if none present)."""
    D = df_long[df_long["cyt"].isin(genes)].copy()
    if D.empty:
        return pd.Series(dtype=float)
    D = add_gene_scaled(D, USE_RANK_FALLBACK, "value")
    return D.groupby("SAMPLE_ID")["z"].mean()

def preflight_gene_coverage(df_all: pd.DataFrame,
                            panels: Dict[str, List[str]]) -> None:
    """Print how many panel genes are present in the pooled table (across all cancers)."""
    present = set(df_all["cyt"].unique())
    print("\nGene coverage across ALL files:")
    for name, glist in panels.items():
        have = sorted(set(glist) & present)
        missing = sorted(set(glist) - present)
        print(f" - {name:18}: {len(have)}/{len(glist)} present" +
              ("" if not missing else f"  | missing: {missing}"))

# --------- Classification ----------
def classify_hot_cold(df_all: pd.DataFrame,
                      t_abund: List[str],
                      activation: List[str],
                      exhaustion: List[str],
                      pex: List[str],
                      thresholds: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # panel indices on zsc (or rank fallback)
    T_idx_raw   = compute_panel_index(df_all, t_abund)
    A_idx_raw   = compute_panel_index(df_all, activation)
    TEX_idx_raw = compute_panel_index(df_all, exhaustion)
    PEX_idx_raw = compute_panel_index(df_all, pex)

    if all(len(x)==0 for x in [T_idx_raw, A_idx_raw, TEX_idx_raw, PEX_idx_raw]):
        raise ValueError("No panel indices could be computed — check that your files contain the required genes.")

    # standardise each panel index across ALL samples to equalise scales
    parts = []
    T_idx = A_idx = PEX_idx = None
    if len(T_idx_raw):
        T_idx = standardise(T_idx_raw).rename("T_idx"); parts.append(T_idx)
    if len(A_idx_raw):
        A_idx = standardise(A_idx_raw).rename("A_idx"); parts.append(A_idx)
    if len(PEX_idx_raw):
        PEX_idx = standardise(PEX_idx_raw).rename("PEX_idx"); parts.append(PEX_idx)

    # ---- TEX contribution (static vs gated) ----
    TEX_contrib = None
    full_gate = None  # for later use in p_consistency gating
    if len(TEX_idx_raw):
        TEX_std = standardise(TEX_idx_raw)

        if TEX_MODE == "static_neg":
            TEX_contrib = (-TEX_std).rename("TEX_contrib")

        elif TEX_MODE == "gated_by_t":
            # Build per-sample gate based on within-cancer T abundance
            any_lookup = (df_all.drop_duplicates("SAMPLE_ID")[["SAMPLE_ID","cancer_type"]]
                              .set_index("SAMPLE_ID"))
            gate_df = pd.DataFrame({
                "cancer_type": any_lookup["cancer_type"],
                "T_idx_raw":   T_idx_raw
            }).dropna()

            # Compute within-cancer statistic for thresholding
            if GATE_STAT == "mean":
                thr = gate_df.groupby("cancer_type")["T_idx_raw"].transform("mean")
            else:
                thr = gate_df.groupby("cancer_type")["T_idx_raw"].transform("median")

            # Abundance deviation (centre within cancer)
            dev = gate_df["T_idx_raw"] - thr

            if GATE_STYLE == "binary":
                gate_weight = np.where(dev >= 0, +1.0, -1.0)  # +TEX if high, −TEX if low
            else:
                # Smooth gate: within-cancer robust z-score -> tanh
                def within_cancer_mad_z(s: pd.Series) -> pd.Series:
                    return (s - s.median()) / (mad(s.values) or 1.0)
                z_within = gate_df.groupby("cancer_type")["T_idx_raw"].transform(within_cancer_mad_z)
                gate_weight = np.tanh(z_within / GATE_K)

            gate_weight = pd.Series(gate_weight, index=gate_df.index)

            # Align with TEX_std index
            # If a sample lacks T_idx_raw, give weight=0 (drop TEX effect conservatively)
            full_gate = pd.Series(0.0, index=TEX_std.index)
            full_gate.loc[gate_weight.index] = gate_weight.values

            TEX_contrib = (full_gate * TEX_std).rename("TEX_contrib")
        else:
            raise ValueError(f"Unknown TEX_MODE: {TEX_MODE}")

    if TEX_contrib is not None:
        parts.append(TEX_contrib)

    panel_mat = pd.concat(parts, axis=1)

    # HotScore = mean of available parts
    HotScore = panel_mat.mean(axis=1).rename("HotScore")

    # Global Z of the HotScore (amplitude)
    HotScore_Z = standardise(HotScore).rename("Z")

    # Consistency p: gene-level alignment with “hot” direction
    pos_genes = set(t_abund) | set(activation) | set(pex)
    neg_genes = set(exhaustion)

    Dz = add_gene_scaled(df_all, USE_RANK_FALLBACK, "value")[["SAMPLE_ID","cyt","z"]].copy()

    # Map gate to samples for exhaustion alignment
    ex_gate = None
    if TEX_MODE == "gated_by_t" and len(TEX_idx_raw) and full_gate is not None:
        # keep as Series; replace zeros with NaN
        ex_gate = full_gate.apply(np.sign).replace(0.0, np.nan)

    # Define alignment per row → True/False/NaN
    def _align_row(row):
        gene = row["cyt"]; z = row["z"]; sid = row["SAMPLE_ID"]
        if gene in pos_genes:
            return True if z > 0 else False
        if gene in neg_genes:
            if TEX_MODE == "gated_by_t" and ex_gate is not None:
                s = ex_gate.get(sid, np.nan)
                if pd.isna(s):   # no gate -> ignore
                    return np.nan
                # if gate is +1, +z aligns with hot; if gate is −1, −z aligns with hot
                return True if (z * s) > 0 else False
            else:
                # static: exhaustion aligns with hot if negative
                return True if z < 0 else False
        return np.nan

    tmp = Dz.apply(_align_row, axis=1)

    # Map to numeric: True→1.0, False→0.0, NaN stays NaN
    Dz["aligns_hot"] = pd.Series(tmp, index=Dz.index).map({True: 1.0, False: 0.0}).astype(float)

    p_consistency = (Dz.dropna(subset=["aligns_hot"])
                       .groupby("SAMPLE_ID")["aligns_hot"].mean()
                       .rename("p_consistency"))

    # labels (keep original zip behavior to preserve results)
    def label_row(z, p, t=THRESHOLDS):
        if (z >= t["z_hot"]) and (p >= t["p_hot"]):
            return "hot"
        if (t["z_ihot_lo"] <= z < t["z_hot"]) or (t["p_ihot"] <= p < t["p_hot"]):
            return "intermediate-hot"
        if (t["z_cold"] < z < t["z_icold_hi"]) or (t["p_icold"] < p <= t["p_cold"]):
            return "intermediate-cold"
        if (z <= t["z_cold"]) and (p <= t["p_cold"]):
            return "cold"
        return "intermediate-hot" if z >= 0 else "intermediate-cold"

    labels = pd.Series([label_row(z, p) for z, p in zip(HotScore_Z, p_consistency)],
                       index=HotScore_Z.index, name="group")

    # attach cancer_type back
    any_lookup = df_all.drop_duplicates("SAMPLE_ID")[["SAMPLE_ID","cancer_type"]].set_index("SAMPLE_ID")
    L = pd.concat([HotScore, HotScore_Z, p_consistency, panel_mat, labels], axis=1)
    L = any_lookup.join(L, how="right").reset_index().rename(columns={"index":"SAMPLE_ID"})

    # per-cancer summaries
    summaries = []
    for ct, G in L.groupby("cancer_type"):
        N = len(G)
        n_hot   = int((G["group"]=="hot").sum())
        n_ihot  = int((G["group"]=="intermediate-hot").sum())
        n_icold = int((G["group"]=="intermediate-cold").sum())
        n_cold  = int((G["group"]=="cold").sum())

        def frac_ci(k):
            p = k/N if N>0 else np.nan
            lo, hi = binom_ci_wilson(k, N)
            return p, lo, hi

        hot_cov, hot_lo, hot_hi       = frac_ci(n_hot)
        ihot_cov, ihot_lo, ihot_hi    = frac_ci(n_ihot)
        icold_cov, icold_lo, icold_hi = frac_ci(n_icold)
        cold_cov, cold_lo, cold_hi    = frac_ci(n_cold)

        hot_side  = G.loc[G["group"].isin(["hot","intermediate-hot"]),"Z"].values
        cold_side = G.loc[G["group"].isin(["cold","intermediate-cold"]),"Z"].values

        medZ_hot   = float(np.median(hot_side)) if hot_side.size else np.nan
        medZ_cold  = float(np.median(cold_side)) if cold_side.size else np.nan
        lo_hot, hi_hot   = bootstrap_ci(hot_side,  np.median) if hot_side.size else (np.nan, np.nan)
        lo_cold, hi_cold = bootstrap_ci(cold_side, np.median) if cold_side.size else (np.nan, np.nan)

        WHC = (n_hot + 0.5*n_ihot) / N if N>0 else np.nan            # Width of Hot Coverage
        HI  = medZ_hot                                               # Hot Intensity (median Z of hot-side)
        HSS = min(1.0, WHC * (HI/1.5)) if np.isfinite(WHC) and np.isfinite(HI) else np.nan  # composite

        summaries.append(dict(
            cancer_type=ct, N=N,
            n_hot=n_hot, n_intermediate_hot=n_ihot, n_intermediate_cold=n_icold, n_cold=n_cold,
            hot_frac=hot_cov, hot_lo=hot_lo, hot_hi=hot_hi,
            i_hot_frac=ihot_cov, i_hot_lo=ihot_lo, i_hot_hi=ihot_hi,
            i_cold_frac=icold_cov, i_cold_lo=icold_lo, i_cold_hi=icold_hi,
            cold_frac=cold_cov, cold_lo=cold_lo, cold_hi=cold_hi,
            hot_medianZ=medZ_hot, hot_medianZ_lo=lo_hot, hot_medianZ_hi=hi_hot,
            cold_medianZ=medZ_cold, cold_medianZ_lo=lo_cold, cold_medianZ_hi=hi_cold,
            WHC=WHC, HI=HI, HSS=HSS
        ))

    labels_df  = L.sort_values(["cancer_type","Z"], ascending=[True, False]).reset_index(drop=True)
    summary_df = pd.DataFrame(summaries).sort_values("HSS", ascending=False).reset_index(drop=True)
    return labels_df, summary_df

# =========================
# RUN
# =========================
if __name__ == "__main__":
    for ct, path in FILES.items():
        print(f"[{ct:10}] exists? {os.path.exists(path)}  -> {path}")

    # load
    df_all = build_pooled_dataframe(FILES, INPUT_SHAPE, VALUE_COL)

    # preflight: gene coverage
    PANELS = {
        "T_abundance": T_ABUNDANCE,
        "Activation":  ACTIVATION,
        "PEX":         PEX if USE_PEX else [],
        "Exhaustion":  EXHAUSTION
    }
    preflight_gene_coverage(df_all, PANELS)

    # classify
    labels_df, summary_df = classify_hot_cold(
        df_all,
        t_abund=T_ABUNDANCE,
        activation=ACTIVATION,
        exhaustion=EXHAUSTION,
        pex=PEX if USE_PEX else [],
        thresholds=THRESHOLDS
    )

    # save (keep same filenames; location controlled by OUTDIR)
    labels_df.to_csv(OUTDIR / LABELS_OUT, index=False)
    summary_df.to_csv(OUTDIR / SUMMARY_OUT, index=False)

    print("\nSaved:")
    print(" -", str(OUTDIR / LABELS_OUT))
    print(" -", str(OUTDIR / SUMMARY_OUT))
    print("\nPreview (labels):\n", labels_df.head(10).to_string(index=False))
    print("\nPreview (summary):\n", summary_df.head(10).to_string(index=False))
