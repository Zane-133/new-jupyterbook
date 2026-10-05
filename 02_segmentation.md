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

# Segmentation

The old dataset is segmented with two tools, **SynthSeg** and **WMH-SynthSeg** (127 scans
each); the new dataset with WMH-SynthSeg only (72 scans).

| Step | SynthSeg | WMH-SynthSeg |
|---|---|---|
| 1. Environment | FreeSurfer 8.0.0 + licence | code and weights from `setup.sh`; GPU |
| 2. List the scans | found by file name; stop if the count is wrong | same |
| 3. Group | 9 folders (3 modalities × HF highres, HF lowres, LF) | 1 folder |
| 4. Segment | `mri_synthseg --robust --threads 8` | `inference.py --device cuda` |
| 5. Collect | label maps by subject; one volume table per folder | label maps by subject; one volume table |

Each tool reads a whole folder and loads its model once per call, which is why scans are
grouped into folders first. Both tools resample every input to 1 mm before segmenting and write
the label map on that 1 mm grid. Finished scans are skipped, so every cell can be re-run after
an interruption.

Output names keep the input name, including `run-1`, for both tools:
`<scan>_synthseg.nii.gz` and `<scan>_WMHseg.nii.gz`.

```{code-cell} ipython3
import os, re, shutil, sys

import pandas as pd
import torch

from common import P, OLD_RAW, NEW_RAW, OLD_OUT, NEW_OUT, LOGS, THREADS, fs, sh

SUBJECTS = ['sub-0011', 'sub-0015', 'sub-0023', 'sub-0025', 'sub-0027', 'sub-0035',
            'sub-0046', 'sub-0047', 'sub-0048', 'sub-0064', 'sub-0066']
MODS = ['T1w', 'T2w', 'FLAIR']
```

## Old dataset — SynthSeg

**Group.** One folder of links per scanner × resolution × modality (9 folders). LF scans of
all sessions and runs go into the LF folder of their modality.

```{code-cell} ipython3
SS = OLD_OUT / 'seg' / 'SynthSeg'
WORK = SS / '_work'

shutil.rmtree(WORK / 'in', ignore_errors=True)
for s in SUBJECTS:
    for m in MODS:
        for acq in ['highres', 'lowres']:
            d = WORK / 'in' / f'3T_{acq}_{m}'
            d.mkdir(parents=True, exist_ok=True)
            f = OLD_RAW / '3T' / s / 'anat' / f'{s}_acq-{acq}_{m}.nii.gz'
            (d / f.name).symlink_to(f)
        d = WORK / 'in' / f'64mT_{m}'
        d.mkdir(parents=True, exist_ok=True)
        # '_{m}.nii.gz' excludes localizers; 'run-*' keeps all runs
        for f in sorted((OLD_RAW / '64mT' / s).glob(f'ses-*/anat/{s}_ses-*_run-*_{m}.nii.gz')):
            (d / f.name).symlink_to(f)

# Stop if any scan is missing or a link is broken
links = sorted((WORK / 'in').glob('*/*.nii.gz'))
broken = [l for l in links if not l.exists()]
assert len(links) == 127 and not broken, f'{len(links)} links (expected 127); broken: {broken}'
```

**Segment.** One `mri_synthseg` call per folder. A folder whose label maps are all there is
skipped.

```{code-cell} ipython3
(SS / 'volumes').mkdir(parents=True, exist_ok=True)
for d in sorted((WORK / 'in').iterdir()):
    arm = d.name
    seg, res = WORK / 'seg' / arm, WORK / 'res' / arm
    seg.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    if len(list(seg.iterdir())) == len(list(d.iterdir())):
        continue
    print(f'=== {arm}')
    fs(f'mri_synthseg --i {d} --o {seg} --resample {res} '
       f'--vol {SS}/volumes/vol_{arm}.csv --qc {SS}/volumes/qc_{arm}.csv '
       f'--robust --threads {THREADS}', log=LOGS / 'synthseg_old.log')
```

**Collect.** Label maps and 1 mm resampled images are copied into
`SynthSeg/{3T,64mT}/<subject>/[<session>]/`.

```{code-cell} ipython3
for f in sorted(WORK.glob('seg/*/*.nii.gz')) + sorted(WORK.glob('res/*/*.nii.gz')):
    arm, b = f.parent.name, f.name
    if f.parent.parent.name == 'res' and not b.endswith('_resampled.nii.gz'):
        b = b[:-len('.nii.gz')] + '_resampled.nii.gz'
    sub = re.search(r'sub-\d+', b).group()
    ses = re.search(r'ses-\d+', b)
    dst = SS / arm.split('_')[0] / sub / (ses.group() if ses else '')
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(f, dst / b)

n = len(list(WORK.glob('seg/*/*_synthseg.nii.gz')))
assert n == 127, f'only {n} / 127 segmented'
print('SynthSeg, old dataset: 127 segmented')
```

## Old dataset — WMH-SynthSeg

**List the scans.** 3T: `3T/<sub>/anat/<sub>_acq-{highres,lowres}_<mod>.nii.gz`;
64mT: `64mT/<sub>/<ses>/anat/<sub>_<ses>_run-N_<mod>.nii.gz`, all runs, localizers excluded.

```{code-cell} ipython3
WMH_OLD = OLD_OUT / 'seg' / 'WMH_SynthSeg'

JOBS_OLD = []
for sub in SUBJECTS:
    for acq in ['highres', 'lowres']:
        for mod in MODS:
            stem = f'{sub}_acq-{acq}_{mod}'
            JOBS_OLD.append(dict(stem=stem, fs='3T', sub=sub, ses='n/a', acq=acq, run='-', mod=mod,
                                 src=f'{OLD_RAW}/3T/{sub}/anat/{stem}.nii.gz',
                                 out=f'{WMH_OLD}/3T/{sub}/{stem}_WMHseg.nii.gz'))
    for sesp in sorted((OLD_RAW / '64mT' / sub).glob('ses-*')):
        ses = sesp.name
        for mod in MODS:
            for src in sorted(sesp.glob(f'anat/{sub}_{ses}_run-*_{mod}.nii.gz')):
                stem = src.name[:-len('.nii.gz')]
                JOBS_OLD.append(dict(stem=stem, fs='64mT', sub=sub, ses=ses, acq='-',
                                     run=re.search(r'run-\d+', stem).group(), mod=mod,
                                     src=str(src), out=f'{WMH_OLD}/64mT/{sub}/{ses}/{stem}_WMHseg.nii.gz'))

# Stop if a scan is missing or two scans share a name
missing = [j['src'] for j in JOBS_OLD if not os.path.isfile(j['src'])]
assert not missing, f'missing: {missing}'
assert len(JOBS_OLD) == 127 and len({j['stem'] for j in JOBS_OLD}) == 127, f'{len(JOBS_OLD)} jobs, expected 127'
```

**Segment and collect.** Pending scans are linked into one flat folder and `inference.py` runs
once on it, so the model loads once. Outputs are moved back to `{3T,64mT}/<subject>/[<session>]/`
and the volumes are appended to `volumes_main.csv`. No `--crop`: it had an out-of-bounds bug, so
the full field of view is used. The same function is used for the new dataset below.

```{code-cell} ipython3
assert torch.cuda.is_available(), 'no GPU: start the NVIDIA A40 server'
WMH_CODE = P['wmh_repo'] / 'WMHSynthSeg'


def run_jobs(jobs, out_root, log):
    todo = [j for j in jobs if not os.path.exists(j['out'])]
    if todo:
        stage, segout = out_root / '_stage', out_root / '_segout'
        for d in (stage, segout):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        for j in todo:
            os.symlink(j['src'], stage / f"{j['stem']}.nii.gz")

        # expandable_segments reduces GPU memory fragmentation
        sh(f'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {sys.executable} inference.py '
           f'--i {stage} --o {segout} --csv_vols {segout}/vols.csv --device cuda',
           cwd=WMH_CODE, log=log)

        # Move outputs from the staging folder back to {3T,64mT}/<subject>/[<session>]/
        for j in todo:
            seg = segout / f"{j['stem']}_seg.nii.gz"
            if seg.exists():
                os.makedirs(os.path.dirname(j['out']), exist_ok=True)
                shutil.move(seg, j['out'])

        # Volume table: add metadata columns, merge with earlier results
        vols = pd.read_csv(segout / 'vols.csv')
        vols['stem'] = vols.pop('Input-file').map(lambda p: os.path.basename(p)[:-len('_seg.nii.gz')])
        tidy = pd.DataFrame(todo).drop(columns=['src', 'out']).merge(vols, on='stem')
        dst = out_root / 'volumes_main.csv'
        if dst.exists():
            tidy = pd.concat([pd.read_csv(dst), tidy], ignore_index=True).drop_duplicates('stem', keep='last')
        tidy.sort_values(['fs', 'sub', 'mod', 'acq']).to_csv(dst, index=False)

    missing = [j['stem'] for j in jobs if not os.path.exists(j['out'])]
    assert not missing, f'{len(missing)} not segmented (re-run the cell): {missing}'


run_jobs(JOBS_OLD, WMH_OLD, LOGS / 'wmh_old.log')
v = pd.read_csv(WMH_OLD / 'volumes_main.csv')
assert len(v) == 127
print('WMH-SynthSeg, old dataset: 127 segmented')
```

## New dataset — WMH-SynthSeg

**List the scans.** File names are `sub-XX_{T1w,T2w,FLAIR}.nii.gz` at both field strengths.
Because 3T and 64mT share file names, staging names carry a field prefix (`3T_sub-03_T2w`); the
outputs are sorted back into `3T/` and `64mT/`.

```{code-cell} ipython3
WMH_NEW = NEW_OUT / 'seg' / 'WMH_SynthSeg'

JOBS_NEW = []
for fs_ in ('3T', '64mT'):
    for src in sorted((NEW_RAW / fs_).glob('sub-*/anat/*.nii.gz')):
        name = src.name[:-len('.nii.gz')]
        sub, mod = re.fullmatch(r'(sub-\d+)_(T1w|T2w|FLAIR)', name).groups()
        JOBS_NEW.append(dict(stem=f'{fs_}_{name}', fs=fs_, sub=sub, acq='-', run=1, mod=mod,
                             src=str(src), out=f'{WMH_NEW}/{fs_}/{sub}/{name}_WMHseg.nii.gz'))
assert len(JOBS_NEW) == 72, f'{len(JOBS_NEW)} jobs, expected 72'
```

**Segment and collect.**

```{code-cell} ipython3
run_jobs(JOBS_NEW, WMH_NEW, LOGS / 'wmh_new.log')
v = pd.read_csv(WMH_NEW / 'volumes_main.csv')
assert len(v) == 72
print('WMH-SynthSeg, new dataset: 72 segmented')
```
