# Runnable book: LF vs HF brain volume

Runs on Neurodesk. The `.md` chapters are the source; `jupyter-book build` executes them.

## Inputs (not in this repository)

`~/neurodesktop-storage/neurodesk_upload/` with `old/`, `new/`, `freesurfer/license.txt`.
Paths are set in `config.yml`.

## Once

```bash
bash setup.sh
```

Creates the Python environment `~/conda-envs/lfbook`, clones WMH-SynthSeg into `~/wmh` and
downloads its model weights into `~/wmh_model`.

## Run the whole book

Needs the GPU server (WMH-SynthSeg, Chapter 02).

```bash
mkdir -p ~/neurodesktop-storage/results/logs
nohup ~/conda-envs/lfbook/bin/jupyter-book build . > ~/neurodesktop-storage/results/logs/build.log 2>&1 &
tail -f ~/neurodesktop-storage/results/logs/build.log
```

- Results go to `~/neurodesktop-storage/results/`; command output of each tool to `results/logs/`.
- Finished steps are skipped, so after an interruption just run the same command again.
- To recompute a step, delete its output file or folder first.
- The HTML book is in `_build/html/`.

## Chapter → output

| Chapter | Writes |
|---|---|
| 01 | nothing (environment and data check) |
| 02 | `old/seg/SynthSeg/`, `old/seg/WMH_SynthSeg/`, `new/seg/WMH_SynthSeg/` |
| 03 | `old/reg/`, `new/reg/` (pairs, `.lta`), `old/dice/`, `new/dice/` |
| 04 | `old/volume/` |
| 05 | `new/analysis/` |
