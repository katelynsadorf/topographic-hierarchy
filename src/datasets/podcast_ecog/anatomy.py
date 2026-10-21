"""Podcast ECoG (ds005574): electrode anatomy in the plan's HCP-MMP1 frame.

Usage: python -I anatomy.py <ds005574_dir> <glasser_dir> <out_dir>
"""
import sys, json, glob, os
import numpy as np, pandas as pd, nibabel as nib
from scipy.spatial import cKDTree

DS, ATL, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)

# Plan Step 2a stage -> HCP-MMP1 parcels (Glasser 2016 names, neuroparc numbering)
STAGES = {
    "1 Heschl / early auditory (phonemes)": [24, 173, 174, 124, 104],          # A1, MBelt, LBelt, PBelt, RI
    "2 posterior STG (syllables)":          [175, 125, 25, 28],                 # A4, A5, PSL, STV
    "3 STS / MTG (words)":                  [128, 129, 130, 176, 132, 133, 177],# STSda/dp/vp/va, TE1a/p/m
    "4 anterior temporal (sentences)":      [123, 131, 172, 134],               # STGa, TGd, TGv, TE2a
    "5 IFG (context)":                      [74, 75, 76, 77, 171, 82, 81],      # 44, 45, 47l, a47r, p47r, IFSa, IFSp
}
P2S = {p: s for s, ps in STAGES.items() for p in ps}

img = nib.load(f"{ATL}/Glasser_space-MNI152NLin6_res-1x1x1.nii.gz")
lab = np.asarray(img.dataobj).astype(int)
names = {int(k): v["label"] for k, v in json.load(open(f"{ATL}/Metadata-json/Glasser_space-MNI152NLin6_res-1x1x1.json"))["rois"].items()}
ijk = np.argwhere(lab > 0)
xyz = nib.affines.apply_affine(img.affine, ijk)
vox_lab = lab[tuple(ijk.T)]
tree = cKDTree(xyz)

# A1 centroid per hemisphere = origin for the hierarchy-gradient distance
a1 = {h: xyz[(vox_lab == 24) & (np.sign(xyz[:, 0]) == s)].mean(0) for h, s in (("L", -1), ("R", 1))}

rows = []
for f in sorted(glob.glob(f"{DS}/sub-*/ieeg/*space-MNI152NLin2009aSym_electrodes.tsv")):
    sub = os.path.basename(f).split("_")[0]
    el = pd.read_csv(f, sep="\t")
    ch = pd.read_csv(glob.glob(f"{DS}/{sub}/ieeg/*channels.tsv")[0], sep="\t")
    ch.columns = [c.strip("﻿") for c in ch.columns]
    el = el.merge(ch[["name", "type", "status", "status_description"]], on="name", how="left")
    el["sub"] = sub
    rows.append(el)
E = pd.concat(rows, ignore_index=True)
E["x"], E["y"], E["z"] = [pd.to_numeric(E[c], errors="coerce") for c in "xyz"]
E["localized"] = E[["x", "y", "z"]].notna().all(1)
E["good"] = (E["status"] == "good") & E["localized"]
E["hemi"] = np.where(E["x"] < 0, "L", "R")
kind = {"G": "grid", "S": "strip", "D": "depth", "EG": "hybrid grid"}
E["kind"] = E["group"].map(lambda g: kind.get(str(g), "other"))

# label: nearest labeled cortical voxel; surface contacts sit just off gray matter
G = E[E["good"]].copy()
d, i = tree.query(G[["x", "y", "z"]].to_numpy())
G["dist_to_cortex_mm"] = d
G["parcel_id"] = np.where(d <= 5, vox_lab[i], 0)          # 5 mm tolerance; else unlabeled
G["parcel"] = G["parcel_id"].map(lambda p: names.get(p, "unlabeled (>5 mm from cortex)") if p else "unlabeled (>5 mm from cortex)")
G["stage"] = G["parcel_id"].map(lambda p: P2S.get(p, "outside model's regions"))
G["dist_to_A1_mm"] = [np.linalg.norm(r[["x", "y", "z"]].to_numpy(float) - a1[r.hemi]) for _, r in G.iterrows()]

# readout-head geometry: nearest-neighbour spacing within each subject
nn = []
for s, g in G.groupby("sub"):
    t = cKDTree(g[["x", "y", "z"]].to_numpy())
    dd, _ = t.query(g[["x", "y", "z"]].to_numpy(), k=2)
    nn.append(pd.Series(dd[:, 1], index=g.index))
G["nn_spacing_mm"] = pd.concat(nn)

E.to_csv(f"{OUT}/electrodes_all.csv", index=False)
G.to_csv(f"{OUT}/electrodes_good_labeled.csv", index=False)

summary = {
    "n_total": int(len(E)), "n_localized": int(E.localized.sum()), "n_good": int(len(G)),
    "per_subject": G.groupby("sub").agg(n=("name", "size"), left=("hemi", lambda h: int((h == "L").sum())),
                                        right=("hemi", lambda h: int((h == "R").sum())),
                                        kinds=("kind", lambda k: dict(k.value_counts()))).to_dict("index"),
    "by_stage_hemi": G.groupby(["stage", "hemi"]).size().unstack(fill_value=0).to_dict("index"),
    "subjects_per_stage": G.groupby("stage")["sub"].nunique().to_dict(),
    "kind_counts": G["kind"].value_counts().to_dict(),
    "dist_to_cortex_median": float(G.dist_to_cortex_mm.median()),
    "unlabeled": int((G.parcel_id == 0).sum()),
    "nn_spacing_median_by_kind": G.groupby("kind").nn_spacing_mm.median().round(1).to_dict(),
    "top_parcels": G[G.parcel_id > 0].parcel.value_counts().head(15).to_dict(),
    "a1_centroids": {h: [round(float(v), 1) for v in c] for h, c in a1.items()},
}

# transcript timing on the model's 80 ms clock
tr = pd.read_csv(f"{DS}/stimuli/whisper-medium/transcript.tsv", sep="\t", index_col=0)
dur = tr["end"] - tr["start"]
gap = tr["start"].shift(-1) - tr["end"]
T = float(tr["end"].max())
frames = np.zeros(int(np.ceil(T / 0.08)) + 1, int)
for s in tr["start"]:
    frames[int(s // 0.08)] += 1
summary["transcript"] = {
    "n_words": int(len(tr)), "duration_s": round(T, 1), "words_per_min": round(len(tr) / T * 60, 1),
    "word_dur_median_ms": round(float(dur.median()) * 1000), "word_dur_p10_p90_ms": [round(float(dur.quantile(q)) * 1000) for q in (.1, .9)],
    "n_frames_80ms": int(len(frames)), "frac_frames_with_onset": round(float((frames > 0).mean()), 3),
    "frac_words_shorter_than_80ms": round(float((dur < 0.08).mean()), 3),
    "frac_onset_frames_with_2plus_words": round(float((frames >= 2).sum() / (frames > 0).sum()), 3),
    "pauses_over_1s": int((gap > 1).sum()),
}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
