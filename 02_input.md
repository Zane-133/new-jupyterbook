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

# Input

Unzip the data, list every scan in a **scan table**, and pair each LF scan with its HF scan in a
**pair table**. All later chapters read these tables instead of searching for files.

```{code-cell} ipython3
import re

import pandas as pd

from common import P, DATA, OLD_RAW, OLD_OUT, SUBJECTS, seg_path, sh
```

## Unzip

From Drive to the Colab disk. Skipped if already unzipped in this runtime.

```{code-cell} ipython3
if not DATA.exists():
    sh(f"unzip -q {P['zip']} -d {DATA.parent}")
assert P['license'].is_file(), f"FreeSurfer licence not found: {P['license']}"
```

## Old dataset — scan table

11 subjects × 3 modalities. 3T: two resolutions, `<sub>_acq-{highres,lowres}_<mod>`. 64mT: all
sessions and runs, `<sub>_<ses>_run-N_<mod>`; localizers are left out. Columns not applicable to
a scan are `n/a` (3T session) or `-` (3T run, 64mT acq).

```{code-cell} ipython3
rows = []
for f in sorted(OLD_RAW.glob('3T/sub-*/anat/*.nii.gz')):
    m = re.fullmatch(r'(sub-\d+)_acq-(highres|lowres)_(T1w|T2w|FLAIR)\.nii\.gz', f.name)
    if m:
        sub, acq, mod = m.groups()
        rows.append(dict(scan=f.name[:-len('.nii.gz')], fs='3T', sub=sub, ses='n/a', acq=acq, run='-',
                         mod=mod, src=str(f)))
for f in sorted(OLD_RAW.glob('64mT/sub-*/ses-*/anat/*.nii.gz')):
    m = re.fullmatch(r'(sub-\d+)_(ses-\d+)_(run-\d+)_(T1w|T2w|FLAIR)\.nii\.gz', f.name)
    if m:
        sub, ses, run, mod = m.groups()
        rows.append(dict(scan=f.name[:-len('.nii.gz')], fs='64mT', sub=sub, ses=ses, acq='-', run=run,
                         mod=mod, src=str(f)))
scans = pd.DataFrame(rows)

n = scans.fs.value_counts()
assert (n['3T'], n['64mT']) == (66, 61) and scans.scan.is_unique, f'scan count does not match: {dict(n)}'
print(f"{n['3T']} 3T + {n['64mT']} 64mT scans, {scans['sub'].nunique()} subjects")
```

## Old dataset — pair tables

Each 64mT scan (moving) is paired with the 3T scan (fixed) of the same subject and modality, once
against 3T lowres and once against 3T highres. A pair also lists the label map each tool will
write (Chapters 03–04).

```{code-cell} ipython3
TOOLS = ['SynthSeg', 'WMH_SynthSeg']


def make_pairs(acq):
    hf = {(r['sub'], r['mod']): r for r in scans[(scans.fs == '3T') & (scans.acq == acq)].to_dict('records')}
    rows = []
    for lf in scans[scans.fs == '64mT'].to_dict('records'):
        h = hf[(lf['sub'], lf['mod'])]
        rows.append(dict(pair_id=f"{lf['scan']}__{acq}", subject=lf['sub'], session=lf['ses'],
                         run=lf['run'], modality=lf['mod'], mov_raw=lf['src'], fix_raw=h['src'])
                    | {f'mov_{t}': str(seg_path(OLD_OUT, t, lf)) for t in TOOLS}
                    | {f'fix_{t}': str(seg_path(OLD_OUT, t, h)) for t in TOOLS})
    return pd.DataFrame(rows)


pairs = {acq: make_pairs(acq) for acq in ['lowres', 'highres']}
assert all(len(p) == 61 for p in pairs.values())
```

## Save

A test run (`subjects` in `config.yml`) keeps only the listed subjects.

```{code-cell} ipython3
if SUBJECTS != 'all':
    scans = scans[scans['sub'].isin(SUBJECTS)]
    pairs = {acq: p[p.subject.isin(SUBJECTS)] for acq, p in pairs.items()}

(OLD_OUT / 'reg').mkdir(parents=True, exist_ok=True)
scans.to_csv(OLD_OUT / 'scans.csv', index=False)
for acq, p in pairs.items():
    p.to_csv(OLD_OUT / 'reg' / f'pairs_{acq}.csv', index=False)
print(f"{len(scans)} scans -> {OLD_OUT / 'scans.csv'}")
print(f"{len(pairs['lowres'])} + {len(pairs['highres'])} pairs -> {OLD_OUT / 'reg'}/pairs_{{lowres,highres}}.csv")
```

## New dataset

To be added.
