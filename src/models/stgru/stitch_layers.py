"""Stitch Pavlos's per-segment .h5 activations into one continuous (frames x channels)
array per layer on the 80ms clock, so it lines up with the ECoG export (22,500 frames).

Each item i is a word or <SILENCE> segment starting at metadata/start_sec[i]. Its frame k
is placed at time start_sec[i] + k*0.08. Frames never covered stay NaN (see 'covered').

Usage (from the topographic folder):
  python stitch_layers.py data/models/fbanks_v295_podacast.h5 data/stitched
  python stitch_layers.py data/models/fbanks_v295_podacast.h5 data/stitched --layers 0 7 14
  python stitch_layers.py data/models/fbanks_v295_podacast.h5 data/stitched --arrays mems outer_residual
"""
import argparse
import os

import h5py
import numpy as np

DT = 0.08
LENGTHS_KEY = {
    "mems": "mem_lengths",
    "inner_residual": "inner_residual_lengths",
    "inner_residual_branch": "inner_residual_lengths",
    "outer_residual": "outer_residual_lengths",
    "outer_residual_skip": "outer_residual_lengths",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("out")
    ap.add_argument("--arrays", nargs="+", default=["mems"], choices=list(LENGTHS_KEY))
    ap.add_argument("--layers", nargs="+", type=int, default=list(range(15)))
    ap.add_argument("--n_frames", type=int, default=22500)
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        start = f["metadata/start_sec"][:]
        end = f["metadata/end_sec"][:]
        n_items = len(start)
        for layer in args.layers:
            g = f[f"snn_layer_{layer}"]
            for arr in args.arrays:
                ds = g[arr]
                max_t, n_ch = ds.shape[1], ds.shape[2]
                lens = np.minimum(g[LENGTHS_KEY[arr]][:], max_t)
                acc = np.zeros((args.n_frames, n_ch), dtype=np.float32)
                cnt = np.zeros(args.n_frames, dtype=np.int32)
                for i0 in range(0, n_items, args.chunk):
                    i1 = min(i0 + args.chunk, n_items)
                    block = ds[i0:i1]
                    for j in range(i1 - i0):
                        i = i0 + j
                        L = int(lens[i])
                        if L == 0:
                            continue
                        idx = int(np.floor(start[i] / DT + 0.5)) + np.arange(L)
                        ok = (idx >= 0) & (idx < args.n_frames)
                        acc[idx[ok]] += block[j, :L][ok].astype(np.float32)
                        cnt[idx[ok]] += 1
                covered = cnt > 0
                out = np.full_like(acc, np.nan)
                out[covered] = acc[covered] / cnt[covered, None]
                path = os.path.join(args.out, f"{arr}_layer{layer}.npz")
                np.savez_compressed(path, x=out, covered=covered, counts=cnt)
                too_long = int(((end - start) > max_t * DT + 1e-6).sum())
                print(
                    f"layer {layer:2d} {arr}: covered {covered.mean():.1%} of {args.n_frames} frames | "
                    f"zero-frame items {(lens == 0).sum()} | frames with >1 item {(cnt > 1).sum()} | "
                    f"items longer than {max_t * DT:.2f}s (truncated) {too_long} -> {path}"
                )


if __name__ == "__main__":
    main()
