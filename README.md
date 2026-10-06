# Runnable book: LF vs HF brain volume

The `.md` chapters are the source; `jupyter-book build` executes them in order. The book runs on
Google Colab (GPU runtime) from start to end:

1. Upload `neurodesk_upload.zip` (`old/`, `new/`, `freesurfer/license.txt`) to the root of My Drive.
2. Open `colab_run.ipynb` in Colab, choose a GPU runtime, Run all.

Chapter 01 installs the tools; every later chapter saves its results to
`My Drive/lfbook/results/` as soon as they are computed. After a disconnect, Run all again:
finished steps are skipped. To recompute a step, delete its output on Drive first.

Paths are in `config.yml`. For a test run set `subjects: [sub-0011]` there; results then go to
`results_test/`.

| Chapter | Writes (under `results/old/`) |
|---|---|
| 01 Environment | nothing (installs and checks the tools) |
| 02 Input | `scans.csv`, `reg/pairs_{lowres,highres}.csv` |
| 03 SynthSeg | `seg/SynthSeg/` |
| 04 WMH-SynthSeg | `seg/WMH_SynthSeg/` |
| 05 Registration | `reg/xfm_{lowres,highres}/` |
| 06 Dice | `dice/` |
| 07 Volume | `volume/` |

The new dataset is not yet included.
