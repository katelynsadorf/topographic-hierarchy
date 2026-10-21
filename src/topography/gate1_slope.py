"""Slope of best layer on distance from A1, with subject bootstrap and within-subject permutation p-values,
plus paired subject-bootstrap comparison of bands.
Usage: python gate1_slope.py --run figures/ecogprep/gate1_all15L_lg_mg_hg [--n_boot 10000 --n_perm 10000]
Reads <run>/electrodes_best_layer.csv, writes <run>/gate1_slope.csv and gate1_band_diff.csv."""
import argparse, os, numpy as np, pandas as pd
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--n_boot", type=int, default=10000)
ap.add_argument("--n_perm", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

t = pd.read_csv(os.path.join(a.run, "electrodes_best_layer.csv"))
t = t[t.significant & t.dist_to_A1_mm.notna()]
bands = ["high_gamma", "mid_gamma", "low_gamma"]
bands = [b for b in bands if b in set(t.band)]
subs = sorted(t["sub"].unique())

def slope(x, y):
    if len(x) < 3 or np.ptp(x) == 0: return np.nan
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())

def rho(x, y):
    if len(x) < 3: return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])

D = {}
for b in bands:
    k = t[t.band == b]
    D[b] = {s: (g.dist_to_A1_mm.to_numpy(), g.best_layer.to_numpy(float)) for s, g in k.groupby("sub")}

def stats(b, sample):
    parts = [D[b][s] for s in sample if s in D[b]]
    if not parts: return np.nan, np.nan
    x = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    return slope(x, y), rho(x, y)

# observed + subject bootstrap (same resampled subjects for all bands -> paired)
obs = {b: stats(b, subs) for b in bands}
boot = {b: np.full((a.n_boot, 2), np.nan) for b in bands}
for i in range(a.n_boot):
    sample = rng.choice(subs, len(subs), replace=True)
    for b in bands:
        boot[b][i] = stats(b, sample)

def p_boot(v):
    v = v[~np.isnan(v)]
    return float(min(1.0, 2 * (min((v <= 0).sum(), (v >= 0).sum()) + 1) / (len(v) + 1)))

rows = []
for b in bands:
    # within-subject permutation of best-layer labels
    ps = {s: D[b][s] for s in D[b]}
    xs = np.concatenate([v[0] for v in ps.values()]); ys = np.concatenate([v[1] for v in ps.values()])
    cuts = np.cumsum([len(v[0]) for v in ps.values()])[:-1]
    ysub = np.split(ys, cuts)
    xc = xs - xs.mean(); den = (xc ** 2).sum()
    null = np.empty(a.n_perm)
    for i in range(a.n_perm):
        yp = np.concatenate([rng.permutation(v) for v in ysub])
        null[i] = (xc * (yp - yp.mean())).sum() / den
    s_obs = obs[b][0]
    p_perm = (1 + (np.abs(null) >= abs(s_obs)).sum()) / (a.n_perm + 1)
    lo, hi = np.nanpercentile(boot[b][:, 0], [2.5, 97.5])
    rows.append(dict(band=b, n=len(xs), n_subjects=len(ps), slope_layers_per_10mm=10 * s_obs,
                     ci_lo=10 * lo, ci_hi=10 * hi, p_boot=p_boot(boot[b][:, 0]),
                     p_perm_within_subject=p_perm, null95_slope_per_10mm=10 * np.percentile(np.abs(null), 95)))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(a.run, "gate1_slope.csv"), index=False)
print(out.round(4).to_string(index=False))

dr = []
for i, b1 in enumerate(bands):
    for b2 in bands[i + 1:]:
        for j, name in enumerate(["slope_per_10mm", "rho"]):
            d = boot[b1][:, j] - boot[b2][:, j]
            sc = 10 if j == 0 else 1
            dobs = sc * (obs[b1][j] - obs[b2][j])
            lo, hi = np.nanpercentile(sc * d, [2.5, 97.5])
            dr.append(dict(comparison=f"{b1} - {b2}", metric=name, diff=dobs, ci_lo=lo, ci_hi=hi, p_boot=p_boot(d)))
dd = pd.DataFrame(dr)
dd.to_csv(os.path.join(a.run, "gate1_band_diff.csv"), index=False)
print(); print(dd.round(4).to_string(index=False))
