# Runnable book: LF vs HF brain volume

The `.md` chapters are the source; `jupyter-book build` executes them. Two ways to run it:

- **Google Colab** (GPU works): open `colab_run.ipynb` in Colab and run it top to bottom.
  It installs everything, runs the book, and saves results, the built book and a fixed record
  of each successful run to `My Drive/lfbook/`. Needs `neurodesk_upload.zip` in the root of My Drive.
  Paths: `config_colab.yml`.
- **Neurodesk**: the rest of this README. Paths: `config.yml`. (On Neurodesk Play the A40
  server cannot run CUDA, so WMH-SynthSeg in Chapter 02 does not run there.)

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
