# Runnable book: LF vs HF brain volume

The `.md` chapters are the source; `jupyter-book build` executes them in order. The book runs on
Neurodesk (CPU server); only WMH-SynthSeg runs on Google Colab (Appendix A).

## Before the first run

- Data: `~/neurodesktop-storage/neurodesk_upload/` with `old/`, `new/`, `freesurfer/license.txt`.
- Python: `~/conda-envs/lfbook` with `jupyter-book<2`, `pandas`, `nibabel`, `scipy`, `pyyaml`.
- WMH-SynthSeg results: run `appendix_wmh_colab.ipynb` on Colab (A100 GPU), download
  `My Drive/lfbook/WMH_SynthSeg_old.zip`, upload it to the Neurodesk home folder and unzip:

```bash
unzip -n ~/WMH_SynthSeg_old.zip -d ~/neurodesktop-storage/lfbook/results
```

Paths are in `config.yml`. For a test run set `subjects: [sub-0011]` there; results then go to
`results_test/`. For the full run set `subjects: all`.

## Run

In a terminal, so the build continues when the browser is closed:

```bash
cd ~/new-jupyterbook && git pull
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

Check that every chapter ran (00–07 each listed once; an error shows as `ExecutionError`):

```bash
grep -E "Executed|ExecutionError" ~/neurodesktop-storage/lfbook/results/logs/build.log
```

## Run or change one chapter interactively

In JupyterLab, kernel **Python [conda env:lfbook]**. The `.md` files are paired with `.ipynb`
copies (gitignored):

```bash
cd ~/new-jupyterbook && ~/conda-envs/lfbook/bin/jupytext --to ipynb 0*.md     # once
~/conda-envs/lfbook/bin/jupytext --sync 06_dice.ipynb                          # after editing
```

Finished steps are skipped here too, so re-running a chapter takes seconds. Do not run chapters
while a build is running. Edits reach the book only after `--sync` writes them back to the `.md`.

| Chapter | Writes (under `results/old/`) |
|---|---|
| 01 Environment | nothing (checks the tools and the data) |
| 02 Input | `scans.csv`, `reg/pairs_{lowres,highres}.csv` |
| 03 SynthSeg | `seg/SynthSeg/` |
| 04 WMH-SynthSeg | nothing (checks `seg/WMH_SynthSeg/` from Colab; copies it for a test run) |
| 05 Registration | `reg/xfm_{lowres,highres}/` |
| 06 Dice | `dice/` |
| 07 Volume | `volume/` |

The new dataset is not yet included.
