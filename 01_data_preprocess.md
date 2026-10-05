---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Data

Two datasets are used, in sequence and for different purposes.

| Dataset | Purpose |
|---|---|
| **Old** | Build the workflow, choose the segmentation tool, and measure the LF–HF gap (Ch. 01–04) |
| **New** | Check whether the findings replicate on an independent cohort (Ch. 05) |

## Old dataset

van den Broek R, Lena B, Webb A. *Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for
Neuroimaging Research.* Zenodo; 2025. [doi:10.5281/zenodo.15374450](https://doi.org/10.5281/zenodo.15374450)

Acquired at Leiden University Medical Center. Of 66 healthy volunteers scanned at 64mT,
**eleven** were also scanned at 3T; all eleven are used. Scanners: Hyperfine Swoop (64mT) and
Philips Achieva dStream (3T).

| | |
|---|---|
| Subjects | sub-0011, 0015, 0023, 0025, 0027, 0035, 0046, 0047, 0048, 0064, 0066 |
| Modalities | T1w, T2w, FLAIR |
| HF acquisitions | 66 — 11 subjects × 3 modalities × 2 resolutions |
| LF acquisitions | 61 over 20 sessions — FLAIR 21, T1w 20, T2w 20 |

3T was acquired at two resolutions: a clinical protocol (**highres**) and a protocol matched
to the low-field scanner (**lowres**).

| Modality | 3T highres | 3T lowres | LF 64mT |
|---|---|---|---|
| T1w | 0.49 × 0.49 × 1 | 1.38 × 1.38 × 5 | 1.6 × 1.6 × 5 |
| T2w | 0.22 × 0.22 × 3 | 1.38 × 1.38 × 5 | 1.6 × 1.6 × 5 |
| FLAIR | 0.43 × 0.43 × 5.5 | 1.38 × 1.38 × 5.5 | 1.7 × 1.7 × 5 |

## New dataset

Own acquisition: twelve healthy volunteers on a 3T Siemens MAGNETOM Cima.X and a 64mT Hyperfine
Swoop, 0–22 days apart. Delivered as DICOM and converted to NIfTI with dcm2niix; this book
starts from the NIfTI files.

| | |
|---|---|
| Subjects | sub-03, 04, 05, 06, 07, 10, 11, 12, 13, 14, 15, 17 |
| Modalities | T1w, T2w, FLAIR |
| HF | one ~1 mm isotropic scan per modality; **no lowres**. T1w is MP2RAGE INV2 |
| LF | one session per subject, 0.65–0.85 mm in-plane × 5 mm |

## Environment check

Everything in this book runs on Neurodesk. `setup.sh` has to be run once beforehand (see
`README.md`). The cells below check that the environment and the input data are complete;
later chapters assume they are.

```{code-cell} ipython3
import subprocess
import sys

import torch

from common import CFG, P, DATA, OLD_RAW, NEW_RAW, RESULTS, fs, sh

for k, v in P.items():
    print(f"{k:10s} {v}")
```

**GPU.** WMH-SynthSeg runs on the GPU (Chapter 02).

```{code-cell} ipython3
print("python", sys.executable)
print("torch ", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU   ", torch.cuda.get_device_name(0))
```

**FreeSurfer.** SynthSeg, SynthMorph and `mri_vol2vol` come from the FreeSurfer 8.0.0 module.

```{code-cell} ipython3
assert P["license"].is_file(), f"FreeSurfer licence not found: {P['license']}"
fs("which mri_synthseg mri_synthmorph mri_vol2vol && echo FS_LICENSE=$FS_LICENSE")
```

**WMH-SynthSeg.** Code from the same GitHub commit as the original runs, model path pointed to
the home folder, model weights downloaded.

```{code-cell} ipython3
inf = P["wmh_repo"] / "WMHSynthSeg" / "inference.py"
sh(f"git -C {P['wmh_repo']} log -1 --format='commit %h  %ad'")
sh(f"grep -n 'model_file = \\|torch.load(' {inf}")
assert P["wmh_model"].stat().st_size > 0
print(f"model {P['wmh_model'].name}: {P['wmh_model'].stat().st_size / 1e6:.0f} MB")
```

**Input data.** File counts per dataset and field strength; localizers are not used.

```{code-cell} ipython3
def nii(root):
    return sorted(f for f in root.rglob("*.nii.gz") if "localizer" not in f.name)

counts = {
    "old 3T":   (len(nii(OLD_RAW / "3T")), 66),
    "old 64mT": (len(nii(OLD_RAW / "64mT")), 61),
    "new 3T":   (len(nii(NEW_RAW / "3T")), 36),
    "new 64mT": (len(nii(NEW_RAW / "64mT")), 36),
}
for k, (n, expected) in counts.items():
    print(f"{k:9s} {n:3d} scans (expected {expected})")
assert all(n == e for n, e in counts.values()), "scan count does not match"
```
