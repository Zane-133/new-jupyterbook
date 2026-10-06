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
mkdir -p ~/neurodesktop-storage/lfbook/results/logs
nohup ~/conda-envs/lfbook/bin/jupyter-book build . > ~/neurodesktop-storage/lfbook/results/logs/build.log 2>&1 &
tail -f ~/neurodesktop-storage/lfbook/results/logs/build.log
```

- Finished steps are skipped, so after an interruption run the same command again.
- To recompute a step, delete its output first.
- The HTML book is in `_build/html/`.

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
