# Data download

Nothing in `data/` is committed. Put inputs here (paths are relative to the project root).

## Podcast ECoG (OpenNeuro ds005574)
- Source: https://openneuro.org/datasets/ds005574 (version 1.0.2). Paper: bioRxiv 2025.02.14.638352.
- 9 subjects, one 30-minute This American Life episode, 1,330 electrodes, 512 Hz.
- Download with DataLad into `data/ds005574`:
  `datalad install https://github.com/OpenNeuroDatasets/ds005574.git data/ds005574`
  then `datalad get` the files you need, or use `openneuro-py`.
- Files used:
  - the unfiltered `ecogprep` derivative `sub-XX_task-podcast_ieeg.fif` per subject (spike removal, common average reference, 60/120/180/240 Hz notch). Use `export_bands_ecogprep.py --list` (with `--root` and `--electrodes`) to see what it finds.
  - the stimulus audio (`podcast.wav`)
  - electrode locations and the Destrieux atlas labels used to build `electrodes_good_labeled.csv`
- Do not build bands from the raw EDF files. They lack the notch filter, common average reference and spike removal, and this changed the envelope dry-run correlation from -0.327 to -0.205.

## Electrode table
`electrodes_good_labeled.csv` (in this repo, 1,268 electrodes) has each electrode's subject, name, distance to the A1 centroid (`dist_to_A1_mm`), anatomical parcel and stage.

## Model activations (stGRU, SpikeSpeechBrain)
- File: `fbanks_v295_podacast.h5` (4.1 GB), per-segment activations for the podcast audio, produced with the frozen model (no retraining). It is not public; request it from the authors.
- Place in `data/models/`. `stitch_layers.py` converts it to one array per layer on the 80 ms clock in `data/stitched/`.
- Current file is a placeholder inference run (30 s windows, 4 s padding). Each of the 15 layers has 1,024 channels of memory state (`mems`).

## Band export
`data/pavlos_handoff/ecog_bands_ecogprep.h5` is generated, not downloaded. It is created by `export_bands_ecogprep.py`. It holds, per subject, theta (4-8), beta (13-30), low gamma (30-60), mid gamma (60-90) and high gamma (90-150 Hz), each (n_electrodes, 22500) and z-scored on an 80 ms clock.
