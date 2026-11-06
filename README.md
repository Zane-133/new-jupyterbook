# Segmentation Reimagined for Low-Field Brain MRI

**Author:** Kaizhen Guo (supervisors: Dr David Waddington, Dr Thomas Boele), University of Sydney

An executable book (Jupyter Book) that compares automated brain segmentations of paired portable
64 mT and conventional 3 T MRI scans of the same subjects. Every scan is segmented with SynthSeg
and WMH-SynthSeg, each 64 mT scan is registered to its 3 T partner with SynthMorph, and the two
segmentations are compared structure by structure by overlap (Dice coefficient) and by volume
(volume bias). The book also checks data quality (scans cut off at the bottom, intracranial
volume) and repeats the analysis on a second, independent dataset. The aim is to identify
structural measures that can quantify improvements in low-field image quality.

The scans are not part of this repository. The public dataset is available from Zenodo:
van den Broek, Webb and Lena, *Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for
Neuroimaging Research*, version 5, <https://doi.org/10.5281/zenodo.20281403>. The USyd dataset
(Chapter 08) comes from a University of Sydney volunteer study (ethics 2025/HE000419) and is not
shared.

## Setup/Build/Install

The book runs on [Neurodesk](https://www.neurodesk.org): segmentation (WMH-SynthSeg) on an NVIDIA
A40 GPU, all other computations on the CPU. Only `~/neurodesktop-storage/` survives a Neurodesk
restart, so the book lives in `~/neurodesktop-storage/new-jupyterbook/`.

- **Data** in `~/neurodesktop-storage/neurodesk_upload/`:
  - `old/` – public dataset, `3T/sub-*/anat/*.nii.gz` and `64mT/sub-*/ses-*/anat/*.nii.gz`
    (BIDS, as distributed on Zenodo);
  - `new/` – USyd dataset, `3T/sub-*/anat/*.nii.gz` and `64mT/sub-*/anat/*.nii.gz`;
  - `freesurfer/license.txt` – FreeSurfer licence.
- **FreeSurfer 8.0.0** (Neurodesk module `freesurfer/8.0.0`): SynthSeg, SynthMorph and
  `mri_vol2vol`.
- **Python**: conda environment `~/conda-envs/lfbook` with `jupyter-book<2`, `jupytext`, `pandas`,
  `nibabel`, `scipy`, `pyyaml` and `torch`.
- **WMH-SynthSeg** code (commit `2bf9a42`) and model (`WMH-SynthSeg_v10_231110.pth`): paths
  `wmh_code` and `wmh_model` in `config.yml`; Chapter 03 downloads and patches them if missing.

All paths are set in `config.yml`; the chapters read them through `common.py`.

## Usage

**Minimum working example.** Set `subjects: [sub-0011]` in `config.yml` to run the whole book on
one subject of the public dataset; results then go to `results_test/`. For the USyd dataset,
`subjects_new: [sub-03]` writes to `new_test/`. For the full run set both to `all`.

**Build the book** in a terminal, so the build continues when the browser is closed:

```bash
cd ~/neurodesktop-storage/new-jupyterbook && git pull
mkdir -p ~/neurodesktop-storage/lfbook/results/logs
nohup ~/conda-envs/lfbook/bin/jupyter-book build --all . > ~/neurodesktop-storage/lfbook/results/logs/build.log 2>&1 &
tail -f ~/neurodesktop-storage/lfbook/results/logs/build.log
```

- Always use `--all`: without it Sphinx only rebuilds `.md` files that changed, so unchanged
  chapters are not executed again.
- Finished steps are skipped, so after an interruption run the same command again. Skipping looks
  only at whether the output file exists: after changing how a step computes, delete its output
  first, or the old result is kept.
- The build stops at the first failing chapter, with the traceback at the end of `build.log`.
- The HTML book is written to `_build/html/`. Run `jupyter-book clean .` before a build to remove
  pages and reports left over from earlier versions.

Check that every chapter ran (00–08 each listed once; an error shows as `ExecutionError`):

```bash
grep -E "Executed|ExecutionError" ~/neurodesktop-storage/lfbook/results/logs/build.log
```

**Run or change one chapter interactively** in JupyterLab, kernel **Python [conda env:lfbook]**.
Each `.md` chapter is paired with an `.ipynb` copy through Jupytext:

```bash
cd ~/neurodesktop-storage/new-jupyterbook && ~/conda-envs/lfbook/bin/jupytext --to ipynb 0*.md     # once
~/conda-envs/lfbook/bin/jupytext --sync 05_dice.ipynb                          # after editing
```

Finished steps are skipped here too, so re-running a chapter takes seconds. Do not run chapters
while a build is running. Edits reach the book only after `--sync` writes them back to the `.md`.

## Directory Structure

| Path | Contents |
|---|---|
| `00_intro.md` … `08_new_dataset.md` | The chapters, one step each; the source that `jupyter-book build` executes |
| `00_intro.ipynb` … `08_new_dataset.ipynb` | Jupytext-paired notebook copies of the chapters, committed with their last outputs |
| `common.py` | Paths and command runners shared by all chapters (no analysis code) |
| `config.yml` | Data, results and tool paths; subjects for test or full runs |
| `_config.yml`, `_toc.yml` | Jupyter Book settings and chapter order |
| `drafts/` | Early test of WMH-SynthSeg on the A40 GPU, not part of the book |

The book writes its results outside the repository, under `results/old/` (Chapter 08 under
`results/new/`) in the `results` path of `config.yml`:

| Chapter | Writes |
|---|---|
| 01 Environment | nothing (checks the tools and the data) |
| 02 Input | `scans.csv`, `reg/pairs_{lowres,highres}.csv` |
| 03 Segmentation | `seg/SynthSeg/`, `seg/WMH_SynthSeg/` |
| 04 Registration | `reg/xfm_{lowres,highres}/`, `reg/qc/` |
| 05 Dice | `dice/` |
| 06 Volume | `volume/` |
| 07 Benchmark | nothing (reads the Dice and bias tables) |
| 08 New dataset | `seg/WMH_SynthSeg/`, `reg/`, `dice/`, `volume/` |
