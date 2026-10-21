"""Permutation test for the Gate 1 trend: is Spearman(distance from A1, best-matching layer) bigger than a lucky shuffle?

It reads electrodes_best_layer.csv from an existing gate1 run folder, so nothing is refit and the electrode
selection (shuffled-time null, same filter) stays exactly as in that run.

Among the significant electrodes of each band it shuffles the best-layer labels many times and recomputes rho:
  * within-subject shuffle (primary): labels are shuffled only among electrodes of the same subject, so each
    subject keeps its own layer mix and distance spread. This respects that electrodes from one subject are not
    independent, and is the one to report.
  * global shuffle (secondary): labels shuffled across all electrodes, ignoring subjects. Less conservative.
p = (1 + number of shuffles with |rho| >= observed) / (1 + n_perm), two-sided; a one-sided p (rho >= observed) is
also saved because the hypothesis has a direction (farther from A1 = deeper layer).

Usage (from the topographic folder):
  python gate1_permutation.py --run figures/gate1_all15L_hg
  python gate1_permutation.py --run figures/gate1_all15L_hg figures/gate1_all15L_lg_mg --n_perm 10000
Writes into each run folder: gate1_permutation.csv and gate1_permutation.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))


def perm_rhos(dist_rank, layer_rank, subs, n_perm, rng, within):
    """Spearman = Pearson on ranks. Shuffle the layer ranks (within subject or globally) and recompute."""
    groups = [np.flatnonzero(subs == s) for s in np.unique(subs)] if within else [np.arange(len(subs))]
    out = np.empty(n_perm)
    d = dist_rank - dist_rank.mean()
    dn = np.sqrt((d ** 2).sum())
    for i in range(n_perm):
        sh = layer_rank.copy()
        for g in groups:
            sh[g] = layer_rank[rng.permutation(g)]
        sh = sh - sh.mean()
        out[i] = (d * sh).sum() / (dn * np.sqrt((sh ** 2).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="gate1 run folder(s) containing electrodes_best_layer.csv")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    for run in args.run:
        t = pd.read_csv(os.path.join(run, "electrodes_best_layer.csv"))
        bands = list(dict.fromkeys(t["band"]))
        rows, fig_data = [], []
        for band in bands:
            k = t[(t["band"] == band) & t["significant"] & t["dist_to_A1_mm"].notna()]
            if len(k) < 10 or k["best_layer"].nunique() < 2:
                print(f"{run} {band}: too few significant electrodes ({len(k)})")
                continue
            dr = rankdata(k["dist_to_A1_mm"].to_numpy())
            lr = rankdata(k["best_layer"].to_numpy())
            subs = k["sub"].to_numpy()
            obs = pearson(dr, lr)
            res = {"band": band, "n_electrodes": len(k), "n_subjects": int(k["sub"].nunique()), "rho_observed": obs}
            nulls = {}
            for name, within in (("within_subject", True), ("global", False)):
                nr = perm_rhos(dr, lr, subs, args.n_perm, rng, within)
                nulls[name] = nr
                res[f"p_two_sided_{name}"] = (1 + int((np.abs(nr) >= abs(obs)).sum())) / (1 + args.n_perm)
                res[f"p_one_sided_{name}"] = (1 + int((nr >= obs).sum())) / (1 + args.n_perm)
                res[f"null_rho_95th_{name}"] = float(np.quantile(np.abs(nr), 0.95))
            rows.append(res)
            fig_data.append((band, obs, nulls["within_subject"], res))
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(run, "gate1_permutation.csv"), index=False)
        print(f"\n{run}  ({args.n_perm} shuffles; p floor = {1 / (1 + args.n_perm):.5f})")
        print(df.round(5).to_string(index=False))

        fig, axes = plt.subplots(1, len(fig_data), figsize=(4.2 * len(fig_data), 3.6), squeeze=False)
        for ax, (band, obs, nr, res) in zip(axes[0], fig_data):
            ax.hist(nr, bins=50, color="0.7")
            ax.axvline(obs, color="C3", lw=2)
            p = res["p_two_sided_within_subject"]
            ptxt = f"p < {1 / (1 + args.n_perm):.4f}" if p <= 1 / (1 + args.n_perm) + 1e-12 else f"p = {p:.4f}"
            ax.set_title(f"{band}\nobserved rho = {obs:.2f}, {ptxt}\n(n = {res['n_electrodes']}, within-subject shuffle)", fontsize=8)
            ax.set_xlabel("rho under shuffled layer labels")
            ax.set_ylabel("shuffles")
        fig.suptitle(f"{os.path.basename(os.path.normpath(run))} | {args.n_perm} permutations", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(run, "gate1_permutation.png"), dpi=150, bbox_inches="tight")
        print("saved", os.path.join(run, "gate1_permutation.png"))


if __name__ == "__main__":
    main()
