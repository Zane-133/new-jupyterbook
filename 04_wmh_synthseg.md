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

# WMH-SynthSeg

The WMH-SynthSeg label maps are **not computed in this book**. They were computed on Google
Colab with a GPU ({doc}`appendix_wmh_colab`); this chapter reads them for Chapters 06–07.

## Why not on Neurodesk

WMH-SynthSeg resamples every input to 1 mm and pads the grid to a multiple of 32. No `--crop`
(it had an out-of-bounds bug), so the full field of view goes through the network at once.
Measured on the Neurodesk CPU server (8 CPUs, 32 GB RAM), `--device cpu --threads 8`, one scan:

| Scan | 1 mm grid | Peak RAM | Result |
|---|---|---|---|
| `sub-0035_acq-highres_T2w` | 224 × 224 × 160 = 8.0 M voxels | 27.0 GB | finished in 333 s |
| `sub-0064_ses-01_run-1_T1w` | 192 × 224 × 224 = 9.6 M voxels | > 32 GB | killed (out of memory) |

The first scan already needs 27 of the 32 GB, and 108 of the 127 scans have a 1 mm grid at least
as large. The Neurodesk GPU server (NVIDIA A40) cannot be used either: its AppArmor profile
blocks the abstract unix socket that the NVIDIA driver opens during CUDA initialisation, so
`cuInit` fails. WMH-SynthSeg was therefore run on a Colab A100 GPU.

## Read the results

The Colab output is unzipped into `wmh_results` (`config.yml`), in the same layout the book
uses: `{3T,64mT}/<subject>/[<session>/]<scan>_WMHseg.nii.gz` and `volumes_main.csv`, one row
per scan. `PROVENANCE.txt` records how they were computed.

```{code-cell} ipython3
import shutil

import pandas as pd

from common import P, OLD_OUT, seg_path

scans = pd.read_csv(OLD_OUT / 'scans.csv', keep_default_na=False)
SRC, WMH = P['wmh_results'], OLD_OUT / 'seg' / 'WMH_SynthSeg'
assert (SRC / 'volumes_main.csv').exists(), \
    f'WMH-SynthSeg results not found in {SRC}: run Appendix A on Colab and unzip its output there'
print((SRC / 'PROVENANCE.txt').read_text())
```

A test run writes to `results_test/`: the label maps and table rows of its subjects are copied
there. In a full run `wmh_results` is the results folder itself and nothing is copied.

```{code-cell} ipython3
if SRC != WMH:
    src_out = SRC.parents[1]   # <results>/old, so that seg_path() finds the label maps
    for r in scans.to_dict('records'):
        src, dst = seg_path(src_out, 'WMH_SynthSeg', r), seg_path(OLD_OUT, 'WMH_SynthSeg', r)
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
    v = pd.read_csv(SRC / 'volumes_main.csv', keep_default_na=False)
    v[v.scan.isin(scans.scan)].to_csv(WMH / 'volumes_main.csv', index=False)
    shutil.copy(SRC / 'PROVENANCE.txt', WMH / 'PROVENANCE.txt')
```

## Check

Every scan of the scan table needs a label map and a row in `volumes_main.csv`.

```{code-cell} ipython3
v = pd.read_csv(WMH / 'volumes_main.csv', keep_default_na=False)
no_map = [r['scan'] for r in scans.to_dict('records') if not seg_path(OLD_OUT, 'WMH_SynthSeg', r).exists()]
no_row = sorted(set(scans.scan) - set(v.scan))
assert not no_map and not no_row, f'missing label maps: {no_map}; missing table rows: {no_row}'
print(f"WMH-SynthSeg: {len(scans)} scans, label maps and volumes in {WMH}")
```

## New dataset

To be added.
