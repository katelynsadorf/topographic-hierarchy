"""Gate 1 by brain area: which layers do electrodes in each anatomical area pick?
Costas's request: same idea as the distance-vs-layer plot, but with brain area on the y axis, ordered by distance
from A1 (A1 / Heschl at the bottom), one box per area since each area has several electrodes.

Reads electrodes_best_layer.csv from a gate1 run folder (significant electrodes only) and the area labels
(parcel, stage) from electrodes_good_labeled.csv. Nothing is refit.

Usage (from the topographic folder):
  python gate1_areas.py --run figures/gate1_all15L_hg --electrodes electrodes_good_labeled.csv
  python gate1_areas.py --run figures/gate1_all15L_lg_mg --electrodes electrodes_good_labeled.csv --min_n 4
Writes into the run folder: gate1_areas_<band>.png and gate1_areas_<band>.csv (per-area counts and medians).
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def box(ax, data, pos, **kw):
    try:
        return ax.boxplot(data, positions=pos, orientation="horizontal", **kw)
    except TypeError:  # older matplotlib
        return ax.boxplot(data, positions=pos, vert=False, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--electrodes", required=True)
    ap.add_argument("--area_col", default="parcel", help="'parcel' (fine) or 'stage' (coarse anatomical stage)")
    ap.add_argument("--min_n", type=int, default=3, help="only show areas with at least this many significant electrodes")
    args = ap.parse_args()

    t = pd.read_csv(os.path.join(args.run, "electrodes_best_layer.csv"))
    lab = pd.read_csv(args.electrodes)[["sub", "name", args.area_col]].drop_duplicates(["sub", "name"])
    t = t.merge(lab, on=["sub", "name"], how="left")
    t = t[t["significant"] & t["dist_to_A1_mm"].notna()]
    t[args.area_col] = t[args.area_col].fillna("unlabeled").astype(str)

    for band in dict.fromkeys(t["band"]):
        k = t[t["band"] == band]
        g = k.groupby(args.area_col)
        summ = g.agg(n=("best_layer", "size"), n_subjects=("sub", "nunique"),
                     median_dist_mm=("dist_to_A1_mm", "median"), median_layer=("best_layer", "median"),
                     mean_layer=("best_layer", "mean")).reset_index().sort_values("median_dist_mm")
        summ.to_csv(os.path.join(args.run, f"gate1_areas_{band}.csv"), index=False)
        shown = summ[summ["n"] >= args.min_n]
        dropped = int(summ.loc[summ["n"] < args.min_n, "n"].sum())
        print(f"\n{band}: {len(summ)} areas, {len(shown)} with >= {args.min_n} electrodes "
              f"({dropped} electrodes in smaller areas not shown)")
        print(shown.round(2).to_string(index=False))
        if shown.empty:
            continue

        h = max(3.2, 0.42 * len(shown) + 1.5)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, h), sharey=True)
        pos = np.arange(len(shown))
        kw = dict(widths=0.6, manage_ticks=False, showfliers=False, medianprops={"color": "k"})
        for ax, col, xl in ((a1, "dist_to_A1_mm", "distance from A1 (mm)"), (a2, "best_layer", "best-matching layer")):
            data = [k.loc[k[args.area_col] == a, col].to_numpy() for a in shown[args.area_col]]
            box(ax, data, pos, **kw)
            for p_, d in zip(pos, data):
                ax.scatter(d, p_ + np.random.default_rng(0).uniform(-0.18, 0.18, len(d)), s=8, alpha=0.4, color="C0", zorder=3)
            ax.set_xlabel(xl)
            for p_ in pos[::2]:
                ax.axhspan(p_ - 0.5, p_ + 0.5, color="0.95", zorder=0)
        a2.set_xlim(-0.7, k["best_layer"].max() + 0.7)
        a1.set_yticks(pos)
        a1.set_yticklabels([f"{a} (n={n})" for a, n in zip(shown[args.area_col], shown["n"])], fontsize=7)
        a1.set_ylabel(f"{args.area_col}, ordered by median distance from A1 (closest at bottom)")
        fig.suptitle(f"{os.path.basename(os.path.normpath(args.run))} | {band} | significant electrodes only, areas with n >= {args.min_n}",
                     fontsize=8)
        fig.tight_layout()
        out = os.path.join(args.run, f"gate1_areas_{band}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("saved", out)


if __name__ == "__main__":
    main()
