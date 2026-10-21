"""Gate 1 / Tier 1 (feature-based): for every electrode and band, how well does each model layer
predict the ECoG signal? Ridge regression from lagged layer activity -> electrode, scored by
held-out Pearson r (contiguous time-block cross-validation). Then: does the best-matching layer
get deeper as electrodes get farther from A1?

New in this version
  * wider alpha grid (1e3 ... 1e7); the chosen alpha is printed per layer
  * shuffled-time null (--n_null): ECoG is circularly shifted by a random offset, the whole
    layer-selection procedure is rerun, and the max-over-layers r from those shifts gives a null
    for "best layer r". Electrodes are called significant if best r beats it (pooled per band).
  * subject-level bootstrap CI for the distance-vs-best-layer Spearman rho (electrodes within a
    subject are not independent, so the plain p-value is not trustworthy)
  * box plots of distance per best layer drawn behind the scatter

Inputs
  --stitched   folder from stitch_layers.py (mems_layer{N}.npz)
  --ecog       ecog_final_bands.h5  (sub-XX/{theta,beta,low_gamma,mid_gamma,high_gamma}, electrode_names)
  --electrodes electrodes_good_labeled.csv (needs: sub, name, dist_to_A1_mm)
Outputs go to figures/<short auto name>/, e.g. gate1_L0_7_14_hg_partial or gate1_all15L_allbands. The name
has the layers, bands, a 'partial' flag, and only settings that differ from defaults (full settings are in
run_info.json and the figure titles):
  results_long.csv, electrodes_best_layer.csv, gate1_summary.csv, gate1_dist_vs_layer.png,
  layer_curves.png, run_info.json

Example (from the topographic folder):
  python gate1_tier1.py --stitched data/stitched --ecog data/pavlos_handoff/ecog_final_bands.h5 \
      --electrodes electrodes_good_labeled.csv --layers 0 7 14 --bands high_gamma --n_null 20
"""
import argparse
import datetime
import glob
import json
import os
import re

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import spearmanr

ALL_BANDS = ["theta", "beta", "low_gamma", "mid_gamma", "high_gamma"]


def load_ecog(path, bands):
    cols, blocks = [], []
    with h5py.File(path, "r") as f:
        for sub in sorted(k for k in f.keys() if k.startswith("sub-")):
            names = [n.decode() for n in f[sub]["electrode_names"][:]]
            for b in bands:
                blocks.append(f[sub][b][:].astype(np.float32))
                cols += [(sub, n, b) for n in names]
    Y = np.concatenate(blocks, axis=0).T  # (T, n_targets)
    meta = pd.DataFrame(cols, columns=["sub", "name", "band"])
    mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0)
    Y = (Y - mu) / np.where(sd > 0, sd, 1)
    if np.isnan(Y).any():
        print("warning: NaNs in ECoG, filling with 0")
        Y = np.nan_to_num(Y)
    return Y.astype(np.float32), meta


def audio_envelope(wav_path, n_frames, dt=0.08):
    """Same envelope as the dry run: mono mean -> 100 Hz..min(8 kHz, 0.45 sr) band-pass -> Hilbert -> 80 ms mean bins."""
    sr, x = wavfile.read(wav_path)
    x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(1)
    sos = butter(4, [100, min(8000, 0.45 * sr)], "band", fs=sr, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x)))
    idx = np.floor(np.arange(len(env)) / (sr * dt)).astype(int)
    keep = idx < n_frames
    tot = np.bincount(idx[keep], weights=env[keep], minlength=n_frames)
    cnt = np.bincount(idx[keep], minlength=n_frames)
    return tot / np.maximum(cnt, 1)


def regress_out_envelope(Y, env, n_lags, chunk=1000):
    """Remove from every column of Y (T, targets) whatever a lagged copy of the audio envelope explains
    (lags 0..n_lags-1 frames, env at t-l, plus intercept). Returns the residuals (re-z-scored) and the fraction
    of variance the envelope explained per column (env_r2)."""
    T = Y.shape[0]
    e = (env - env.mean()) / env.std()
    D = np.ones((T, n_lags + 1))
    for l in range(n_lags):
        D[l:, l + 1] = e[: T - l]
        D[:l, l + 1] = 0.0
    G = np.linalg.inv(D.T @ D)
    out = np.empty_like(Y)
    r2 = np.empty(Y.shape[1])
    for a in range(0, Y.shape[1], chunk):
        y = Y[:, a:a + chunk].astype(np.float64)
        res = y - D @ (G @ (D.T @ y))
        r2[a:a + chunk] = 1 - res.var(0) / np.maximum(y.var(0), 1e-12)
        out[:, a:a + chunk] = ((res - res.mean(0)) / np.maximum(res.std(0), 1e-12)).astype(Y.dtype)
    return out, r2


def layer_features(npz_path, n_pcs, n_lags):
    d = np.load(npz_path)
    x, covered = d["x"], d["covered"]
    x = np.nan_to_num(x)
    xc = x[covered] - x[covered].mean(0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(n_pcs, vt.shape[0])
    P = (x - x[covered].mean(0)) @ vt[:k].T
    P[~covered] = 0
    P = P / (P[covered].std(0) + 1e-8)
    T = P.shape[0]
    X = np.zeros((T, k * n_lags), np.float32)
    ok = np.ones(T, bool)
    for l in range(n_lags):  # frame t uses model frames t, t-1, ..., t-(n_lags-1)
        X[l:, l * k:(l + 1) * k] = P[: T - l]
        ok[l:] &= covered[: T - l]
        ok[:l] = False
    return X, ok


def prepare_folds(X, ok, folds, buffer):
    """Everything that depends only on X (reused for the real data and every null shift)."""
    idx = np.flatnonzero(ok)
    edges = np.linspace(0, len(idx), folds + 1).astype(int)
    prep = []
    for f in range(folds):
        te = idx[edges[f]:edges[f + 1]]
        tr = idx[(idx < te[0] - buffer) | (idx > te[-1] + buffer)]
        mx = X[tr].mean(0)
        Xtr = X[tr] - mx
        w, V = np.linalg.eigh((Xtr.T @ Xtr).astype(np.float64))
        prep.append({"tr": tr, "te": te, "Xtr": Xtr, "w": w, "V": V,
                     "Xte": ((X[te] - mx) @ V).astype(np.float64)})
    return prep


def score(prep, Y, alphas):
    rs = np.zeros((len(alphas), Y.shape[1]))
    for fp in prep:
        Ytr = Y[fp["tr"]]
        Ytr = Ytr - Ytr.mean(0)
        VB = fp["V"].T @ (fp["Xtr"].T @ Ytr).astype(np.float64)
        Yte = Y[fp["te"]]
        Yte = Yte - Yte.mean(0)
        for ai, a in enumerate(alphas):
            Pte = fp["Xte"] @ (VB / (fp["w"] + a)[:, None])
            Pte -= Pte.mean(0)
            num = (Pte * Yte).sum(0)
            den = np.sqrt((Pte ** 2).sum(0) * (Yte ** 2).sum(0)) + 1e-12
            rs[ai] += num / den
    return rs / len(prep)


def subject_bootstrap_rho(df, n_boot, rng):
    groups = {s: (g["dist_to_A1_mm"].to_numpy(), g["best_layer"].to_numpy()) for s, g in df.groupby("sub")}
    subs = list(groups)
    if len(subs) < 3:
        return np.nan, np.nan
    rhos = []
    for _ in range(n_boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        d = np.concatenate([groups[s][0] for s in pick])
        l = np.concatenate([groups[s][1] for s in pick])
        if len(np.unique(l)) > 1 and len(np.unique(d)) > 1:
            rhos.append(spearmanr(d, l)[0])
    if len(rhos) < 50:
        return np.nan, np.nan
    return tuple(np.percentile(rhos, [2.5, 97.5]))


BAND_ABBR = {"theta": "th", "beta": "be", "low_gamma": "lg", "mid_gamma": "mg", "high_gamma": "hg"}
DEFAULTS = {"array": "mems", "pcs": 64, "lags": 8, "n_null": 20, "alphas": [1e3, 1e4, 1e5, 1e6, 1e7]}


def layers_label(layers, total):
    if list(layers) == list(range(total)):
        return f"all{total}L"
    if len(layers) >= 3 and list(layers) == list(range(layers[0], layers[-1] + 1)):
        return f"L{layers[0]}to{layers[-1]}"
    return "L" + "_".join(map(str, layers))


def make_run_name(args, layers, partial):
    """Short name: layers, bands, 'partial' flag, plus only the settings that differ from defaults.
    Everything (including defaults) is saved in run_info.json and in the figure titles."""
    bands = "allbands" if set(args.bands) == set(ALL_BANDS) else "_".join(BAND_ABBR[b] for b in args.bands)
    parts = ["gate1", layers_label(layers, args.n_layers_total), bands]
    if args.model_name:
        parts.insert(0, args.model_name)
    if partial:
        parts.append("partial")
    if args.array != DEFAULTS["array"]:
        parts.append(args.array)
    if args.pcs != DEFAULTS["pcs"]:
        parts.append(f"pc{args.pcs}")
    if args.lags != DEFAULTS["lags"]:
        parts.append(f"lag{args.lags}")
    if args.n_null != DEFAULTS["n_null"]:
        parts.append("nonull" if args.n_null == 0 else f"null{args.n_null}")
    if sorted(args.alphas) != DEFAULTS["alphas"]:
        parts.append(f"a{int(round(np.log10(min(args.alphas))))}-{int(round(np.log10(max(args.alphas))))}")
    if getattr(args, "regress_envelope", False):
        parts.append("envreg")
    return "_".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitched", required=True)
    ap.add_argument("--ecog", required=True)
    ap.add_argument("--electrodes", required=True)
    ap.add_argument("--out_root", default="figures", help="parent folder; a run-specific subfolder is created inside")
    ap.add_argument("--out", default=None, help="override the auto-named run folder")
    ap.add_argument("--n_layers_total", type=int, default=15, help="layers in the full model (to flag partial runs)")
    ap.add_argument("--model_name", default=None, help="label for the model whose layers are in --stitched (e.g. stgru); prefixes the run name")
    ap.add_argument("--dataset", default=None, help="label for the recording dataset (e.g. podcast_ecog); saved in run_info.json")
    ap.add_argument("--array", default="mems")
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--bands", nargs="+", default=ALL_BANDS, choices=ALL_BANDS)
    ap.add_argument("--pcs", type=int, default=64, help="PCA components per layer")
    ap.add_argument("--lags", type=int, default=8, help="model frames back (8 = 0-560ms)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--buffer", type=int, default=25, help="frames dropped around test block")
    ap.add_argument("--alphas", nargs="+", type=float, default=[1e3, 1e4, 1e5, 1e6, 1e7])
    ap.add_argument("--n_null", type=int, default=20, help="shuffled-time repeats per layer (0 = skip null)")
    ap.add_argument("--null_min_shift_s", type=float, default=60.0, help="min circular shift in seconds")
    ap.add_argument("--sig", type=float, default=0.01, help="null p-value cutoff for 'significant' electrodes")
    ap.add_argument("--min_r", type=float, default=0.05, help="only used when --n_null 0")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regress_envelope", action="store_true",
                    help="acoustic control: remove what the lagged audio envelope explains from each electrode before fitting")
    ap.add_argument("--wav", default=None, help="podcast.wav (needed with --regress_envelope)")
    ap.add_argument("--env_lags", type=int, default=13, help="envelope lags removed, in 80 ms frames (13 = 0-960 ms)")
    args = ap.parse_args()
    if args.regress_envelope and not args.wav:
        ap.error("--regress_envelope needs --wav")
    rng = np.random.default_rng(args.seed)

    found = sorted(int(re.search(rf"{args.array}_layer(\d+)\.npz", p).group(1))
                   for p in glob.glob(os.path.join(args.stitched, f"{args.array}_layer*.npz")))
    layers = args.layers if args.layers is not None else found
    partial = len(layers) < args.n_layers_total
    run_name = make_run_name(args, layers, partial)
    args.out = args.out or os.path.join(args.out_root, run_name)
    os.makedirs(args.out, exist_ok=True)
    run_label = (f"{args.array} | layers {','.join(map(str, layers))} ({len(layers)} of {args.n_layers_total})"
                 f"{' - PARTIAL RUN' if partial else ''} | pcs {args.pcs}, lags {args.lags}, null {args.n_null}"
                 f"{' | ENVELOPE REGRESSED OUT' if args.regress_envelope else ''}")
    print("run folder:", args.out)
    if partial:
        print(f"NOTE: partial run, {len(layers)} of {args.n_layers_total} layers. Figures are labeled accordingly.")

    Y, meta = load_ecog(args.ecog, args.bands)
    if args.regress_envelope:
        env = audio_envelope(args.wav, Y.shape[0])
        Y, meta["env_r2"] = regress_out_envelope(Y, env, args.env_lags)
        for b in args.bands:
            m = (meta["band"] == b).to_numpy()
            print(f"envelope regressed out ({args.env_lags} lags): {b} median variance explained {np.median(meta.loc[m, 'env_r2']):.3f}, "
                  f"max {meta.loc[m, 'env_r2'].max():.3f}")
    el = pd.read_csv(args.electrodes)[["sub", "name", "dist_to_A1_mm"]].drop_duplicates(["sub", "name"])
    meta = meta.merge(el, on=["sub", "name"], how="left")
    n_t, n_targets = Y.shape
    print(f"{n_targets} targets ({meta[['sub', 'name']].drop_duplicates().shape[0]} electrodes x {len(args.bands)} bands), {n_t} frames")
    print(f"{meta[['sub', 'name', 'dist_to_A1_mm']].drop_duplicates()['dist_to_A1_mm'].isna().sum()} electrodes had no distance in the electrodes csv")

    R = np.zeros((len(layers), n_targets))
    null_best = np.full((args.n_null, n_targets), -np.inf) if args.n_null > 0 else None
    chosen_alpha = []
    min_shift = int(args.null_min_shift_s / 0.08)
    for li, layer in enumerate(layers):
        X, ok = layer_features(os.path.join(args.stitched, f"{args.array}_layer{layer}.npz"), args.pcs, args.lags)
        prep = prepare_folds(X, ok, args.folds, args.buffer)
        rs = score(prep, Y, args.alphas)
        best = int(np.argmax(rs.mean(1)))
        R[li] = rs[best]
        chosen_alpha.append(args.alphas[best])
        edge = " (EDGE of alpha grid, consider widening)" if best in (0, len(args.alphas) - 1) else ""
        print(f"layer {layer:2d}: rows used {ok.sum()}, alpha {args.alphas[best]:g}{edge}, mean r {R[li].mean():.4f}, max r {R[li].max():.3f}")
        for s in range(args.n_null):
            shift = int(rng.integers(min_shift, n_t - min_shift))
            rn = score(prep, np.roll(Y, shift, axis=0), [args.alphas[best]])[0]
            null_best[s] = np.maximum(null_best[s], rn)
        if args.n_null:
            print(f"          null done ({args.n_null} shifts)")

    long = pd.DataFrame({"layer": np.repeat(layers, n_targets),
                         "sub": np.tile(meta["sub"], len(layers)),
                         "name": np.tile(meta["name"], len(layers)),
                         "band": np.tile(meta["band"], len(layers)),
                         "dist_to_A1_mm": np.tile(meta["dist_to_A1_mm"], len(layers)),
                         "r": R.ravel()})
    long.to_csv(os.path.join(args.out, "results_long.csv"), index=False)

    best_tables, summary, fig_rows = [], [], []
    for band in args.bands:
        idx = np.flatnonzero((meta["band"] == band).to_numpy())
        rmat = R[:, idx].T
        t = meta.iloc[idx][["sub", "name", "band", "dist_to_A1_mm"] + (["env_r2"] if "env_r2" in meta else [])].reset_index(drop=True)
        t["best_layer"] = np.array(layers)[rmat.argmax(1)]
        t["best_r"] = rmat.max(1)
        if args.n_null > 0:
            pooled = np.sort(null_best[:, idx].ravel())
            thr = float(np.quantile(pooled, 1 - args.sig))
            t["p_null"] = (1 + (len(pooled) - np.searchsorted(pooled, t["best_r"].to_numpy(), side="left"))) / (1 + len(pooled))
            t["significant"] = t["p_null"] < args.sig
        else:
            thr = args.min_r
            t["p_null"] = np.nan
            t["significant"] = t["best_r"] >= args.min_r
        best_tables.append(t)
        has_d = t.dropna(subset=["dist_to_A1_mm"])
        keep = has_d[has_d["significant"]]
        rho_all = spearmanr(has_d["dist_to_A1_mm"], has_d["best_layer"])[0] if len(has_d) > 3 else np.nan
        if len(keep) > 3 and keep["best_layer"].nunique() > 1:
            rho, p = spearmanr(keep["dist_to_A1_mm"], keep["best_layer"])
            lo, hi = subject_bootstrap_rho(keep, args.n_boot, rng)
        else:
            rho = p = lo = hi = np.nan
        summary.append({"band": band, "n_total": len(t), "n_significant": len(keep),
                        "null_r_threshold": thr, "rho_significant": rho, "rho_ci_lo_subject_boot": lo,
                        "rho_ci_hi_subject_boot": hi, "p_naive": p, "rho_all_electrodes": rho_all})
        fig_rows.append((band, keep, rho, lo, hi))
    pd.concat(best_tables).to_csv(os.path.join(args.out, "electrodes_best_layer.csv"), index=False)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(args.out, "gate1_summary.csv"), index=False)
    print(sdf.round(4).to_string(index=False))
    print("(p_naive treats electrodes as independent; use the subject-bootstrap CI instead)")

    fig, axes = plt.subplots(1, len(fig_rows), figsize=(4.2 * len(fig_rows), 3.9), squeeze=False)
    for ax, (band, keep, rho, lo, hi) in zip(axes[0], fig_rows):
        data = [keep.loc[keep["best_layer"] == L, "dist_to_A1_mm"].to_numpy() for L in layers]
        pos = [L for L, d in zip(layers, data) if len(d)]
        data = [d for d in data if len(d)]
        if data:
            kw = dict(positions=pos, widths=0.6, manage_ticks=False, showfliers=False, medianprops={"color": "k"})
            try:
                ax.boxplot(data, orientation="horizontal", **kw)
            except TypeError:  # older matplotlib
                ax.boxplot(data, vert=False, **kw)
        jit = np.random.default_rng(0).uniform(-0.2, 0.2, len(keep))
        ax.scatter(keep["dist_to_A1_mm"], keep["best_layer"] + jit, s=8, alpha=0.4)
        ax.set_yticks(layers)
        ci = f", 95% CI [{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) else ""
        ax.set_title(f"{band}\nrho={rho:.2f}{ci}, n={len(keep)}", fontsize=9)
        ax.set_xlabel("distance from A1 (mm)")
        ax.set_ylabel("best-matching layer")
    fig.suptitle(run_label, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "gate1_dist_vs_layer.png"), dpi=150, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    for band in args.bands:
        m = long[long["band"] == band].groupby("layer")["r"].mean()
        ax2.plot(m.index, m.values, marker="o", label=band)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("mean held-out r (all electrodes)")
    ax2.legend(fontsize=7)
    ax2.set_title(run_label, fontsize=6)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.out, "layer_curves.png"), dpi=150, bbox_inches="tight")

    with open(os.path.join(args.out, "run_info.json"), "w") as fh:
        json.dump({**vars(args), "layers_used": layers, "partial_run": partial, "chosen_alphas": chosen_alpha,
                   "run_name": run_name, "finished": datetime.datetime.now().isoformat(timespec="seconds")},
                  fh, indent=2)


if __name__ == "__main__":
    main()
