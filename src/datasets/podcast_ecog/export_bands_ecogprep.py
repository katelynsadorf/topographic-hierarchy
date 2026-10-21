"""Re-export our five ECoG bands from the dataset's preprocessed (unfiltered) ecogprep files.

Why: the dataset's ecogprep derivatives already have spike removal, common average reference and notch filters
(60/120/180/240 Hz) applied. Our first export filtered the raw EDF, which has none of that. Here we start from the
preprocessed signal and apply OUR band definitions, so band choice is separated from preprocessing.

Per subject and band: Butterworth band-pass (order 4, zero-phase) -> Hilbert envelope -> mean-binned into 80 ms frames
-> z-scored per electrode (ddof=1). Same steps as the notebook's bandpass_frames. Output has the same layout as
ecog_final_bands.h5 (sub-XX/{band} float64 (n_elec, 22500) and sub-XX/electrode_names), so gate1_tier1.py and
envelope_dry_run.py take it with --ecog unchanged.

Usage (from the topographic folder):
  python export_bands_ecogprep.py --root data/ds005574/derivatives/ecogprep --electrodes electrodes_good_labeled.csv
  (add --list to only show which .fif files it finds per subject, without processing)
Default output: data/pavlos_handoff/ecog_bands_ecogprep.h5 (the old file is not touched).
Needs: pip install mne h5py scipy
"""
import argparse
import glob
import os

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

DT = 0.08
BANDS = {"theta": (4, 8), "beta": (13, 30), "low_gamma": (30, 60), "mid_gamma": (60, 90), "high_gamma": (90, 150)}


def to_frames(x, sfreq, n_frames):
    spf = sfreq * DT
    idx = np.floor(np.arange(x.shape[-1]) / spf).astype(int)
    keep = idx < n_frames
    out = np.zeros((x.shape[0], n_frames))
    np.add.at(out.T, idx[keep], x[:, keep].T)
    counts = np.bincount(idx[keep], minlength=n_frames)
    return out / np.maximum(counts, 1)


def band_frames(x, sfreq, lo, hi, n_frames, chunk=32):
    """x (channels, samples) -> z-scored 80 ms Hilbert-envelope frames (channels, n_frames)."""
    hi = min(hi, 0.45 * sfreq)
    sos = butter(4, [lo, hi], btype="band", fs=sfreq, output="sos")
    out = np.zeros((x.shape[0], n_frames))
    for a in range(0, x.shape[0], chunk):
        env = np.abs(hilbert(sosfiltfilt(sos, x[a:a + chunk].astype(np.float64), axis=-1), axis=-1))
        f = to_frames(env, sfreq, n_frames)
        out[a:a + chunk] = (f - f.mean(1, keepdims=True)) / f.std(1, ddof=1, keepdims=True)
    return out


def find_fif(root, sub, pattern):
    fs = sorted(glob.glob(os.path.join(root, sub, "ieeg", pattern)))
    return [f for f in fs if "highgamma" not in os.path.basename(f).lower()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="derivatives/ecogprep folder")
    ap.add_argument("--electrodes", required=True, help="electrodes_good_labeled.csv (sub, name)")
    ap.add_argument("--out", default="data/pavlos_handoff/ecog_bands_ecogprep.h5")
    ap.add_argument("--pattern", default="*.fif", help="file pattern inside <root>/<sub>/ieeg (highgamma files are always skipped)")
    ap.add_argument("--n_frames", type=int, default=22500)
    ap.add_argument("--list", action="store_true", help="only list the files found, then exit")
    args = ap.parse_args()

    el = pd.read_csv(args.electrodes)[["sub", "name"]].drop_duplicates()
    subs = sorted(el["sub"].unique())
    if args.list:
        for sub in subs:
            print(sub, [os.path.basename(f) for f in find_fif(args.root, sub, args.pattern)])
        return

    import mne

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with h5py.File(args.out, "w") as h:
        for sub in subs:
            fs = find_fif(args.root, sub, args.pattern)
            if len(fs) != 1:
                print(f"{sub}: found {len(fs)} candidate files {[os.path.basename(f) for f in fs]}; "
                      f"use --pattern to pick exactly one, skipping")
                continue
            raw = mne.io.read_raw_fif(fs[0], preload=True, verbose=False)
            want = set(el.loc[el["sub"] == sub, "name"])
            keep = [c for c in raw.ch_names if c in want]
            x, sfreq = raw.get_data(picks=keep), raw.info["sfreq"]
            print(f"{sub}: {os.path.basename(fs[0])}, {len(keep)}/{len(want)} electrodes, {sfreq:g} Hz, {x.shape[1] / sfreq:.0f} s")
            g = h.create_group(sub)
            g.create_dataset("electrode_names", data=np.array(keep, dtype="S"))
            for b, (lo, hi) in BANDS.items():
                g.create_dataset(b, data=band_frames(x, sfreq, lo, hi, args.n_frames))
            h.flush()
    print("saved", args.out)


if __name__ == "__main__":
    main()
