# Runnable book: LF vs HF brain volume

The `.md` chapters are the source; `jupyter-book build` executes them in order. The book runs on
the Neurodesk GPU server (NVIDIA A40).

## Before the first run

Only `~/neurodesktop-storage/` survives a Neurodesk restart, so the book lives in
`~/neurodesktop-storage/new-jupyterbook/`.

- Data: `~/neurodesktop-storage/neurodesk_upload/` with `old/`, `new/`, `freesurfer/license.txt`.
- Python: `~/conda-envs/lfbook` with `jupyter-book<2`, `pandas`, `nibabel`, `scipy`, `pyyaml`, `torch`.
- WMH-SynthSeg code and model: `wmh_code` and `wmh_model` in `config.yml`; Chapter 03 downloads
  and patches them if missing.

Paths are in `config.yml`. For a test run set `subjects: [sub-0011]` there; results then go to
`results_test/`. For the full run set `subjects: all`. The new dataset (Chapter 08) has its own
setting, `subjects_new` (e.g. `[sub-03]`, results in `new_test/`).

## Run

In a terminal, so the build continues when the browser is closed:

```bash
cd ~/neurodesktop-storage/new-jupyterbook && git pull
mkdir -p ~/neurodesktop-storage/lfbook/results/logs
nohup ~/conda-envs/lfbook/bin/jupyter-book build --all . > ~/neurodesktop-storage/lfbook/results/logs/build.log 2>&1 &
tail -f ~/neurodesktop-storage/lfbook/results/logs/build.log
```

- Always `--all`: without it Sphinx only rebuilds `.md` files that changed, so unchanged
  chapters are not executed again.
- Finished steps are skipped, so after an interruption run the same command again.
  Skipping looks only at whether the output file exists: after changing how a step computes,
  delete its output first, or the old result is kept.
- The build stops at the first failing chapter, with the traceback at the end of `build.log`.
- The HTML book is in `_build/html/`. `jupyter-book clean .` before a build removes pages and
  reports left over from earlier versions of the book.

Check that every chapter ran (00–08 each listed once; an error shows as `ExecutionError`):

```bash
grep -E "Executed|ExecutionError" ~/neurodesktop-storage/lfbook/results/logs/build.log
```

## Run or change one chapter interactively

In JupyterLab, kernel **Python [conda env:lfbook]**. The `.md` files are paired with `.ipynb`
copies (gitignored):

```bash
cd ~/neurodesktop-storage/new-jupyterbook && ~/conda-envs/lfbook/bin/jupytext --to ipynb 0*.md     # once
~/conda-envs/lfbook/bin/jupytext --sync 05_dice.ipynb                          # after editing
```

Finished steps are skipped here too, so re-running a chapter takes seconds. Do not run chapters
while a build is running. Edits reach the book only after `--sync` writes them back to the `.md`.

| Chapter | Writes (under `results/old/`; Chapter 08 under `results/new/`) |
|---|---|
| 01 Environment | nothing (checks the tools and the data) |
| 02 Input | `scans.csv`, `reg/pairs_{lowres,highres}.csv` |
| 03 Segmentation | `seg/SynthSeg/`, `seg/WMH_SynthSeg/` |
| 04 Registration | `reg/xfm_{lowres,highres}/`, `reg/qc/` |
| 05 Dice | `dice/` |
| 06 Volume | `volume/` |
| 07 Benchmark | nothing (reads the Dice and bias tables) |
| 08 New dataset | `seg/WMH_SynthSeg/`, `reg/`, `dice/`, `volume/` |

