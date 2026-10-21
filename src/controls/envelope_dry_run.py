"""Dry run for Gate 1 using the raw audio envelope instead of model activations.
Question: do electrodes closer to primary auditory cortex (A1) track the speech envelope more strongly?

Per electrode: correlate the band-filtered ECoG with the audio envelope (80ms frames, the same clock as the
ECoG export) over a window of lags (default -400..+1000ms), keep the peak r and its lag. The lag histogram of
the top 10% electrodes is an alignment check (peaks should sit at small positive lags, brain after sound).
Then Spearman(distance from A1, peak envelope r), shown on the figure with:
  * the plain Spearman p (treats electrodes as independent; this is the "p < 1e-32" style number)
  * a subject-level bootstrap CI (resample subjects)
  * a within-subject permutation p (shuffle distances among each subject's own electrodes)

Usage (from the topographic folder):
  python envelope_dry_run.py --wav data/ds005574/stimuli/podcast.wav \
      --ecog data/pavlos_handoff/ecog_final_bands.h5 --electrodes electrodes_good_labeled.csv --band high_gamma
Output: figures/envelope_dry_run_<band>/ (two-panel figure like the notebook one: r vs distance colored by
anatomical stage + alignment check, with stats; per-electrode csv; summary json)
"""
import argparse
import json
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import spearmanr

DT = 0.08
BAND_ABBR = {"theta": "th", "beta": "be", "low_gamma": "lg", "mid_gamma": "mg", "high_gamma": "hg"}


def to_frames(x, sfreq, n_frames):
    """Mean-bin (channels, samples) into 80 ms frames; same binning as the notebook."""
    spf = sfreq * DT
    idx = np.floor(np.arange(x.shape[-1]) / spf).astype(int)
    keep = idx < n_frames
    out = np.zeros((x.shape[0], n_frames))
    np.add.at(out.T, idx[keep], x[:, keep].T)
    counts = np.bincount(idx[keep], minlength=n_frames)
    return out / np.maximum(counts, 1)


def envelope_hilbert(wav_path, n_frames):
    """Notebook envelope: mono mean -> 4th-order Butterworth 100 Hz..min(8 kHz, 0.45*sr) -> Hilbert amplitude -> 80 ms mean bins."""
    sr, x = wavfile.read(wav_path)
    x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(1)
    sos = butter(4, [100, min(8000, 0.45 * sr)], "band", fs=sr, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x)))
    return to_frames(env[None], sr, n_frames)[0]


def load_fif_highgamma(root, sub, keep_names, n_frames):
    """Dataset's own derivative high gamma (*highgamma*.fif), 80 ms mean bins, z-scored per channel (ddof=1)."""
    import glob
    import mne
    fn = glob.glob(os.path.join(root, sub, "ieeg", "*highgamma*.fif"))
    if not fn:
        return None, None
    raw = mne.io.read_raw_fif(fn[0], preload=True, verbose=False)
    keep = [c for c in raw.ch_names if c in keep_names]
    if not keep:
        return None, None
    Y = to_frames(raw.get_data(picks=keep), raw.info["sfreq"], n_frames)
    Y = (Y - Y.mean(1, keepdims=True)) / Y.std(1, ddof=1, keepdims=True)
    return keep, Y.T


def envelope_frames(wav_path, n_frames, use_log):
    sr, x = wavfile.read(wav_path, mmap=True)
    spf = int(round(sr * DT))
    env = np.zeros(n_frames)
    for k in range(n_frames):
        a, b = k * spf, min((k + 1) * spf, x.shape[0])
        if a >= x.shape[0]:
            break
        seg = np.asarray(x[a:b], dtype=np.float64)
        if seg.ndim == 2:
            seg = seg.mean(1)
        env[k] = np.sqrt((seg ** 2).mean())
    return np.log(env + 1e-6 * env.max()) if use_log else env


def lagged_best_r(Y, e, lag_min, lag_max):
    """Y (T,E), e (T,). Lag L>0: ECoG at frame t vs envelope at t-L (brain after sound). Returns peak r and lag (frames)."""
    T = len(e)
    best = np.full(Y.shape[1], -np.inf)
    best_lag = np.zeros(Y.shape[1], int)
    for lag in range(lag_min, lag_max + 1):
        if lag >= 0:
            ys, es = Y[lag:], e[: T - lag]
        else:
            ys, es = Y[: T + lag], e[-lag:]
        ysc, esc = ys - ys.mean(0), es - es.mean()
        r = (ysc * esc[:, None]).sum(0) / np.sqrt((ysc ** 2).sum(0) * (esc ** 2).sum() + 1e-20)
        better = r > best
        best[better], best_lag[better] = r[better], lag
    return best, best_lag


def subject_bootstrap_ci(df, n_boot, rng):
    groups = {s: (g["dist"].to_numpy(), g["r"].to_numpy()) for s, g in df.groupby("sub")}
    subs = list(groups)
    rhos = []
    for _ in range(n_boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        d = np.concatenate([groups[s][0] for s in pick])
        r = np.concatenate([groups[s][1] for s in pick])
        rhos.append(spearmanr(d, r)[0])
    return tuple(np.percentile(rhos, [2.5, 97.5]))


def within_subject_perm_p(df, rho_obs, n_perm, rng):
    groups = [(g["dist"].to_numpy(), g["r"].to_numpy()) for _, g in df.groupby("sub")]
    r_all = np.concatenate([g[1] for g in groups])
    count = 0
    for _ in range(n_perm):
        d_all = np.concatenate([rng.permutation(g[0]) for g in groups])
        if abs(spearmanr(d_all, r_all)[0]) >= abs(rho_obs):
            count += 1
    return (1 + count) / (1 + n_perm)


def trend_stats(d, n_boot, n_perm, rng):
    """Spearman(distance, r) with naive p, subject-bootstrap CI, within-subject permutation p."""
    rho, p_naive = spearmanr(d["dist"], d["r"])
    lo, hi = subject_bootstrap_ci(d, n_boot, rng)
    p_perm = within_subject_perm_p(d, rho, n_perm, rng)
    return {"n_electrodes": int(len(d)), "n_subjects": int(d["sub"].nunique()), "spearman_rho": float(rho),
            "p_naive": float(p_naive), "ci_lo": float(lo), "ci_hi": float(hi), "p_perm": float(p_perm)}


def fmt_stats(st, n_perm, label):
    p_txt = f"p_perm={st['p_perm']:.4f}" if st["p_perm"] > 1 / (1 + n_perm) else f"p_perm<{1 / (1 + n_perm):.4f}"
    return (f"{label} (n={st['n_electrodes']}, {st['n_subjects']} subj)\n"
            f"rho = {st['spearman_rho']:.3f}, naive p = {st['p_naive']:.1e}\n"
            f"subj-bootstrap CI [{st['ci_lo']:.2f}, {st['ci_hi']:.2f}], {p_txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--ecog", default=None, help="our ecog_final_bands.h5 (omit if using --fif_root)")
    ap.add_argument("--electrodes", required=True)
    ap.add_argument("--band", default="high_gamma", choices=list(BAND_ABBR))
    ap.add_argument("--lag_min", type=int, default=-5, help="min lag in 80ms frames (-5 = -400ms)")
    ap.add_argument("--lag_max", type=int, default=12, help="max lag in 80ms frames (12 = ~1000ms)")
    ap.add_argument("--log", action="store_true", help="use log envelope instead of linear RMS (only with --env rms)")
    ap.add_argument("--env", default="hilbert", choices=["hilbert", "rms"],
                    help="hilbert = notebook envelope (100 Hz-8 kHz bandpass + Hilbert); rms = RMS of raw waveform")
    ap.add_argument("--fif_root", default=None,
                    help="e.g. data/ds005574/derivatives/ecogprep : use the dataset's own *highgamma*.fif instead of --ecog (needs mne)")
    ap.add_argument("--clean", action="store_true", help="figure shows only rho and n (no CI / p values); saved as envelope_vs_distance_clean.png")
    ap.add_argument("--out_root", default="figures")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    name = "envelope_dry_run_" + ("dataset_hg" if args.fif_root else BAND_ABBR[args.band])
    if args.env == "rms":
        name += "_rms"
    if (args.lag_min, args.lag_max) != (-5, 12):
        name += f"_lag{args.lag_min}to{args.lag_max}"
    if args.log:
        name += "_log"
    out = os.path.join(args.out_root, name)
    os.makedirs(out, exist_ok=True)

    rows = []
    if args.fif_root:
        el0 = pd.read_csv(args.electrodes)
        subs = sorted(el0["sub"].unique())
        n_frames = 22500
        env = envelope_hilbert(args.wav, n_frames) if args.env == "hilbert" else envelope_frames(args.wav, n_frames, args.log)
        e = (env - env.mean()) / env.std()
        for sub in subs:
            keep_names = set(el0.loc[el0["sub"] == sub, "name"])
            names, Y = load_fif_highgamma(args.fif_root, sub, keep_names, n_frames)
            if names is None:
                print("no fif for", sub)
                continue
            r, lag = lagged_best_r(np.nan_to_num(Y), e, args.lag_min, args.lag_max)
            rows += [{"sub": sub, "name": n, "r": rv, "best_lag_frames": int(lg)} for n, rv, lg in zip(names, r, lag)]
    else:
        with h5py.File(args.ecog, "r") as f:
            subs = sorted(k for k in f.keys() if k.startswith("sub-"))
            n_frames = f[subs[0]][args.band].shape[1]
            env = envelope_hilbert(args.wav, n_frames) if args.env == "hilbert" else envelope_frames(args.wav, n_frames, args.log)
            e = (env - env.mean()) / env.std()
            for sub in subs:
                names = [n.decode() for n in f[sub]["electrode_names"][:]]
                Y = f[sub][args.band][:].astype(np.float64).T  # (T, E)
                Y = (Y - np.nanmean(Y, 0)) / np.where(np.nanstd(Y, 0) > 0, np.nanstd(Y, 0), 1)
                r, lag = lagged_best_r(np.nan_to_num(Y), e, args.lag_min, args.lag_max)
                rows += [{"sub": sub, "name": n, "r": rv, "best_lag_frames": int(lg)} for n, rv, lg in zip(names, r, lag)]
    df = pd.DataFrame(rows)
    el_all = pd.read_csv(args.electrodes)
    el = el_all[["sub", "name", "dist_to_A1_mm"] + (["stage"] if "stage" in el_all.columns else [])].drop_duplicates(["sub", "name"])
    df = df.merge(el, on=["sub", "name"], how="left").rename(columns={"dist_to_A1_mm": "dist"})
    df = df.dropna(subset=["dist"])
    df.to_csv(os.path.join(out, "electrode_envelope_r.csv"), index=False)

    has_stage = "stage" in df.columns
    df["in_model_region"] = df["stage"].astype(str).str.strip().str[:1].str.isdigit() if has_stage else True
    st_all = trend_stats(df, args.n_boot, args.n_perm, rng)
    st_in = trend_stats(df[df["in_model_region"]], args.n_boot, args.n_perm, rng) if has_stage and df["in_model_region"].sum() > 10 else None
    stats = {"band": args.band, "all_electrodes": st_all, "in_model_regions_stages_1_to_5": st_in,
             "median_best_lag_frames": float(df["best_lag_frames"].median()), "args": vars(args)}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"median best lag {df['best_lag_frames'].median():.0f} frames")
    print(fmt_stats(st_all, args.n_perm, "ALL electrodes").replace("\n", "\n  "))
    if st_in:
        print(fmt_stats(st_in, args.n_perm, "Only electrodes in the model's regions (stages 1-5)").replace("\n", "\n  "))

    top = df.nlargest(max(1, int(round(0.1 * len(df)))), "r")
    frac_ok = float(((top["best_lag_frames"] >= 0) & (top["best_lag_frames"] <= 8)).mean())
    stats["top10pct_peak_lag_ms_median"] = float(top["best_lag_frames"].median() * 80)
    stats["top10pct_fraction_lag_0_to_640ms"] = frac_ok
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"top 10% electrodes by r: median peak lag {stats['top10pct_peak_lag_ms_median']:.0f} ms, {frac_ok:.0%} within 0-640 ms")

    fig, (ax, axh) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax.scatter(df["dist"], df["r"], s=7, c="0.6", label="unlabeled / other", zorder=1)
    if "stage" in df.columns:
        stages = sorted(s_ for s_ in df["stage"].dropna().unique())
        cmap = plt.get_cmap("tab10")
        for st in stages:
            g = df[df["stage"] == st]
            if str(st).strip()[:1].isdigit():
                ax.scatter(g["dist"], g["r"], s=12, color=cmap(int(str(st).strip()[0]) - 1), label=str(st))
            else:
                ax.scatter(g["dist"], g["r"], s=7, c="0.6", label=str(st))
    edges = np.quantile(df["dist"], np.linspace(0, 1, 9))
    mids, meds = [], []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        m = (df["dist"] >= a_) & (df["dist"] <= b_)
        if m.sum():
            mids.append(df.loc[m, "dist"].median())
            meds.append(df.loc[m, "r"].median())
    ax.plot(mids, meds, "k-o", lw=2, ms=4, label="binned median")
    ax.set_xlabel("distance to A1 centroid (mm)")
    sig = "dataset high gamma (70-200 Hz)" if args.fif_root else args.band
    ax.set_ylabel(f"peak envelope r ({sig})")
    ax.set_title("Acoustic tracking falls off with distance from A1", fontsize=10)
    if args.clean:
        txt = f"Spearman rho = {st_all['spearman_rho']:.2f}  (n = {st_all['n_electrodes']})"
    else:
        txt = fmt_stats(st_all, args.n_perm, "ALL electrodes")
        if st_in:
            txt += "\n\n" + fmt_stats(st_in, args.n_perm, "Stages 1-5 only")
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9 if args.clean else 6.5,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})
    ax.legend(fontsize=5.5, loc="center right", bbox_to_anchor=(1.0, 0.42))
    bins = np.arange(args.lag_min - 0.5, args.lag_max + 1.5) * 80
    axh.hist(top["best_lag_frames"] * 80, bins=bins)
    axh.axvline(0, color="k", lw=0.8, ls="--")
    axh.set_xlabel("peak lag (ms)  [>0 = brain after sound]")
    axh.set_ylabel("electrodes (top 10% by r)")
    axh.set_title("Alignment check", fontsize=10)
    fig.suptitle(f"Dry run: audio envelope in place of model activity | {sig} | {args.env} env | lags {args.lag_min * 80} to {args.lag_max * 80} ms"
                 f"{' | log envelope' if args.log else ''}", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "envelope_vs_distance_clean.png" if args.clean else "envelope_vs_distance.png"), dpi=150, bbox_inches="tight")
    print("saved to", out)


if __name__ == "__main__":
    main()
