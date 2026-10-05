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

# Registration and Dice

Each LF scan is compared with the HF scan of the same subject and modality: LF is the
**moving** image, HF the **fixed** reference, so Dice is computed on the HF grid.

| Step | What it does |
|---|---|
| 1. Pair | each LF scan with the HF scan of the same subject and modality |
| 2. Register | `mri_synthmorph register -m rigid` on the grayscale images → one `.lta` per pair |
| 3. Apply | `mri_vol2vol --nearest` moves the LF label map onto the HF grid with that `.lta` |
| 4. Harmonise | left/right merged into the 17 regions; WMH counted as WM |
| 5. Dice | per pair and region |

The old dataset is run twice, against HF **lowres** (resolution matched to LF) and HF
**highres**; the highres arm is needed for the comparison with the new dataset in Chapter 05.
The same transform is applied to both tools' label maps, so registration error is identical for
the two. Finished transforms and Dice tables are skipped on a re-run.

```{code-cell} ipython3
import os, re

import nibabel as nib
import numpy as np
import pandas as pd

from common import OLD_RAW, NEW_RAW, OLD_OUT, NEW_OUT, LOGS, THREADS, fs
```

## 1. Pair

Old dataset: 61 pairs per HF arm. New dataset: 36 pairs. Each pair lists the raw images and the
label maps of every tool.

```{code-cell} ipython3
def pairs_old(acq):
    rows = []
    for f in sorted(OLD_RAW.glob('64mT/sub-*/ses-*/anat/*.nii.gz')):
        m = re.fullmatch(r'(sub-\d+)_(ses-\d+)_(run-\d+)_(T1w|T2w|FLAIR)\.nii\.gz', f.name)
        if not m:                                   # localizers
            continue
        sub, ses, run, mod = m.groups()
        stem, hf = f.name[:-len('.nii.gz')], f'{sub}_acq-{acq}_{mod}'
        # mov = 64mT (moving), fix = 3T (fixed reference)
        rows.append(dict(pair_id=f'{stem}__{acq}', subject=sub, session=ses, run=run, modality=mod,
                         mov_raw=str(f), fix_raw=f'{OLD_RAW}/3T/{sub}/anat/{hf}.nii.gz',
                         mov_SynthSeg=f'{OLD_OUT}/seg/SynthSeg/64mT/{sub}/{ses}/{stem}_synthseg.nii.gz',
                         fix_SynthSeg=f'{OLD_OUT}/seg/SynthSeg/3T/{sub}/{hf}_synthseg.nii.gz',
                         mov_WMH_SynthSeg=f'{OLD_OUT}/seg/WMH_SynthSeg/64mT/{sub}/{ses}/{stem}_WMHseg.nii.gz',
                         fix_WMH_SynthSeg=f'{OLD_OUT}/seg/WMH_SynthSeg/3T/{sub}/{hf}_WMHseg.nii.gz'))
    return rows


def pairs_new():
    rows = []
    for f in sorted(NEW_RAW.glob('64mT/sub-*/anat/*.nii.gz')):
        sub, mod = re.fullmatch(r'(sub-\d+)_(T1w|T2w|FLAIR)\.nii\.gz', f.name).groups()
        stem = f'{sub}_{mod}'
        rows.append(dict(pair_id=stem, subject=sub, modality=mod,
                         mov_raw=str(f), fix_raw=f'{NEW_RAW}/3T/{sub}/anat/{stem}.nii.gz',
                         mov_WMH_SynthSeg=f'{NEW_OUT}/seg/WMH_SynthSeg/64mT/{sub}/{stem}_WMHseg.nii.gz',
                         fix_WMH_SynthSeg=f'{NEW_OUT}/seg/WMH_SynthSeg/3T/{sub}/{stem}_WMHseg.nii.gz'))
    return rows


def save_pairs(rows, expected, out_csv):
    # Stop if a scan is missing or any paired file does not exist
    files = [r[k] for r in rows for k in r if k.startswith(('mov_', 'fix_'))]
    missing = [f for f in files if not os.path.exists(f)]
    assert len(rows) == expected, f'{len(rows)} pairs, expected {expected}'
    assert not missing, f'{len(missing)} missing files, e.g. {missing[:5]}'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'{len(rows)} pairs -> {out_csv}')
    return pd.DataFrame(rows)


PAIRS = {
    ('old', 'lowres'):  save_pairs(pairs_old('lowres'), 61, OLD_OUT / 'reg' / 'pairs_lowres.csv'),
    ('old', 'highres'): save_pairs(pairs_old('highres'), 61, OLD_OUT / 'reg' / 'pairs_highres.csv'),
    ('new', None):      save_pairs(pairs_new(), 36, NEW_OUT / 'reg' / 'pairs.csv'),
}
XFM = {
    ('old', 'lowres'):  OLD_OUT / 'reg' / 'xfm_lowres',
    ('old', 'highres'): OLD_OUT / 'reg' / 'xfm_highres',
    ('new', None):      NEW_OUT / 'reg' / 'xfm',
}
```

## 2. Register

Rigid, 6 DOF (3 rotations + 3 translations), estimated on the raw grayscale images, never on the
segmentations. In the new dataset the T1w pair is 64mT T1 Standard → 3T MP2RAGE INV2, a
different contrast; SynthMorph is contrast-agnostic.

```{code-cell} ipython3
for key, pairs in PAIRS.items():
    XFM[key].mkdir(parents=True, exist_ok=True)
    log = LOGS / f"register_{'_'.join(k for k in key if k)}.log"
    for r in pairs.itertuples():
        lta = XFM[key] / f'{r.pair_id}.lta'
        if lta.exists() and lta.stat().st_size > 0:
            continue
        print(r.pair_id)
        fs(f'export OMP_NUM_THREADS={THREADS} && '
           f'mri_synthmorph register -m rigid -t {lta} {r.mov_raw} {r.fix_raw}', log=log, show=False)
    n = len(list(XFM[key].glob('*.lta')))
    assert n == len(pairs), f'{key}: {n} / {len(pairs)} transforms'
    print(f'{key}: {n} transforms')
```

## 3–5. Apply, harmonise, Dice

The LF label map is moved onto the HF grid with its transform (nearest neighbour keeps labels
intact). Left/right labels are merged into regions **before** Dice; the merged maps are kept in
`work/` (`*_h.nii.gz`). A region absent from both maps is not reported.

```{code-cell} ipython3
# region -> (representative id, source labels); left/right merged.
# 5/44 (inf. lateral ventricle) merged into lateral ventricle: SynthSeg has it, WMH-SynthSeg does not.
REGIONS = {
    "WM":                (2,  [2, 41]),
    "Cortex":            (3,  [3, 42]),
    "Lateral-Ventricle": (4,  [4, 43, 5, 44]),
    "Cerebellum-WM":     (7,  [7, 46]),
    "Cerebellum-Cortex": (8,  [8, 47]),
    "Thalamus":          (10, [10, 49]),
    "Caudate":           (11, [11, 50]),
    "Putamen":           (12, [12, 51]),
    "Pallidum":          (13, [13, 52]),
    "3rd-Ventricle":     (14, [14]),
    "4th-Ventricle":     (15, [15]),
    "Brainstem":         (16, [16]),
    "Hippocampus":       (17, [17, 53]),
    "Amygdala":          (18, [18, 54]),
    "CSF":               (24, [24]),
    "Accumbens":         (26, [26, 58]),
    "VentralDC":         (28, [28, 60]),
    "WMH":               (77, [77]),          # WMH-SynthSeg only; also counted as WM
}
META = ['pair_id', 'subject', 'session', 'run', 'modality']


def harmonize(src, dst):
    """Save the merged label map (without 77) to dst; return {region: boolean mask}."""
    img = nib.load(src)
    v = np.rint(np.asarray(img.dataobj)).astype(np.int32)   # WMH-SynthSeg output is float32
    masks = {name: np.isin(v, labels) for name, (_, labels) in REGIONS.items()}
    masks["WM"] |= v == 77
    out = np.zeros_like(v)
    for name, (rid, _) in REGIONS.items():
        if name != "WMH":
            out[masks[name]] = rid
    o = nib.Nifti1Image(out, img.affine)     # affine only: avoid inheriting scl_slope from float32 headers
    o.set_data_dtype(np.int32)
    nib.save(o, dst)
    return masks


def dice(A, B):
    na, nb, i = int(A.sum()), int(B.sum()), int((A & B).sum())
    return dict(dice_np=2 * i / (na + nb), jaccard=i / (na + nb - i),
                voldiff=(nb - na) / ((na + nb) / 2), n_3T=na, n_64mT=nb)


def dice_table(pairs, tool, acq, xfm, work, out_csv, log):
    if out_csv.exists():
        return
    work.mkdir(parents=True, exist_ok=True)
    out = []
    for r in pairs.to_dict('records'):
        pid = r['pair_id']
        print(pid)
        moved = work / f'{pid}_64mT_native.nii.gz'
        fs(f"mri_vol2vol --mov {r['mov_' + tool]} --targ {r['fix_' + tool]} --lta {xfm}/{pid}.lta "
           f"--nearest --o {moved} --no-save-reg", log=log, show=False)
        A = harmonize(r['fix_' + tool], work / f'{pid}_3T_h.nii.gz')    # 3T reference
        B = harmonize(moved, work / f'{pid}_64mT_h.nii.gz')             # registered 64mT
        for name in REGIONS:
            if A[name].any() or B[name].any():                          # empty in both -> not reported
                out.append({k: r[k] for k in META if k in r} | dict(tool=tool)
                           | (dict(acq=acq) if acq else {}) | dict(region=name, **dice(A[name], B[name])))
    pd.DataFrame(out).to_csv(out_csv, index=False)
    print(f'{len(out)} rows -> {out_csv}')


for acq in ['lowres', 'highres']:
    for tool in ['SynthSeg', 'WMH_SynthSeg']:
        dice_table(PAIRS[('old', acq)], tool, acq, XFM[('old', acq)],
                   OLD_OUT / 'dice' / 'work' / f'{tool}_{acq}',
                   OLD_OUT / 'dice' / f'dice_{tool}_{acq}.csv', LOGS / 'dice_old.log')
dice_table(PAIRS[('new', None)], 'WMH_SynthSeg', None, XFM[('new', None)],
           NEW_OUT / 'dice' / 'work', NEW_OUT / 'dice' / 'dice_WMH_SynthSeg.csv', LOGS / 'dice_new.log')
```
