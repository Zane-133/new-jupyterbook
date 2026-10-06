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

# SynthSeg

Old dataset, all scans of the scan table. `mri_synthseg` reads a whole folder and loads its model
once per call, so the scans are grouped into 9 folders (3T highres, 3T lowres, 64mT × 3
modalities). SynthSeg resamples every input to 1 mm and writes the label map on that grid.

```{code-cell} ipython3
import os
import shutil
from pathlib import Path

import pandas as pd

from common import OLD_OUT, WORK, LOGS, THREADS, seg_path, fs

scans = pd.read_csv(OLD_OUT / 'scans.csv', keep_default_na=False)
scans['group'] = [f'3T_{a}_{m}' if f == '3T' else f'64mT_{m}' for f, a, m in zip(scans.fs, scans.acq, scans['mod'])]

SS = OLD_OUT / 'seg' / 'SynthSeg'
TABLES = SS / 'volumes'
TABLES.mkdir(parents=True, exist_ok=True)
```

## Segment and save

Per folder: link the scans into a staging folder on the Colab disk, run `mri_synthseg`, then save
to Drive the label maps and 1 mm images (`{3T,64mT}/<subject>/[<session>/]`) and, last, the
folder's volume and QC tables. A folder whose volume table is on Drive is finished and skipped.

```{code-cell} ipython3
for group, g in scans.groupby('group'):
    vol = TABLES / f'vol_{group}.csv'
    if vol.exists():
        continue
    print(f'=== {group}: {len(g)} scans')
    d_in, d_seg, d_res = (WORK / 'SynthSeg' / x / group for x in ('in', 'seg', 'res'))
    for d in (d_in, d_seg, d_res):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for src in g.src:
        (d_in / Path(src).name).symlink_to(src)

    fs(f'mri_synthseg --i {d_in} --o {d_seg} --resample {d_res} '
       f'--vol {d_seg}/vol.csv --qc {d_seg}/qc.csv --robust --threads {THREADS}',
       log=LOGS / 'synthseg_old.log')

    for r in g.to_dict('records'):
        out = seg_path(OLD_OUT, 'SynthSeg', r)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(d_seg / f"{r['scan']}_synthseg.nii.gz", out)
        res, = d_res.glob(f"{r['scan']}*.nii.gz")
        shutil.copy(res, out.parent / f"{r['scan']}_resampled.nii.gz")
    shutil.copy(d_seg / 'qc.csv', TABLES / f'qc_{group}.csv')
    shutil.copy(d_seg / 'vol.csv', vol)
```

## Merge the tables

The 9 volume tables and 9 QC tables are stacked into one table each, with the scan-table columns
added.

```{code-cell} ipython3
def stack(kind):
    d = pd.concat([pd.read_csv(f) for f in sorted(TABLES.glob(f'{kind}_*.csv'))], ignore_index=True)
    d = d.rename(columns={d.columns[0]: 'scan'})
    d['scan'] = (d.scan.astype(str).map(os.path.basename)
                 .str.replace(r'\.nii(\.gz)?$', '', regex=True).str.replace(r'_synthseg$', '', regex=True))
    return scans.drop(columns=['src', 'group']).merge(d, on='scan', validate='1:1')


for kind, name in [('vol', 'volumes_wide.csv'), ('qc', 'qc_all.csv')]:
    t = stack(kind)
    assert len(t) == len(scans), f'{name}: {len(t)} rows, expected {len(scans)}'
    t.to_csv(SS / name, index=False)

missing = [r['scan'] for r in scans.to_dict('records') if not seg_path(OLD_OUT, 'SynthSeg', r).exists()]
assert not missing, f'not segmented: {missing}'
print(f"SynthSeg: {len(scans)} scans -> {SS}/volumes_wide.csv, qc_all.csv")
```
