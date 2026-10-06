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

# Dice

Per pair and tool: move the 64mT label map onto the 3T grid with the pair's `.lta`, merge
left/right labels into regions, and compute Dice per region. The same transform is used for
both tools, so registration error is identical for the two.

```{code-cell} ipython3
import os

import nibabel as nib
import numpy as np
import pandas as pd

from common import OLD_OUT, LOGS, fs

REG, DICE = OLD_OUT / 'reg', OLD_OUT / 'dice'
TOOLS = ['SynthSeg', 'WMH_SynthSeg']
```

## Regions

17 regions, left/right merged; WMH also counted as WM. A region absent from both maps is not
reported.

```{code-cell} ipython3
# region -> (representative id, source labels)
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
```

## Compute and save

One Dice table per tool and 3T resolution. The moved and merged label maps are kept in
`dice/work/`. An existing table is skipped.

```{code-cell} ipython3
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
                out.append({k: r[k] for k in META} | dict(tool=tool, acq=acq, region=name)
                           | dice(A[name], B[name]))
    pd.DataFrame(out).to_csv(out_csv, index=False)
    print(f'{len(out)} rows -> {out_csv}')


for acq in ['lowres', 'highres']:
    pairs = pd.read_csv(REG / f'pairs_{acq}.csv', keep_default_na=False)
    files = [f for t in TOOLS for f in pairs[f'mov_{t}']] + [f for t in TOOLS for f in pairs[f'fix_{t}']]
    missing = [f for f in files if not os.path.exists(f)]
    assert not missing, f'{len(missing)} label maps missing, e.g. {missing[:3]}'
    for tool in TOOLS:
        dice_table(pairs, tool, acq, REG / f'xfm_{acq}', DICE / 'work' / f'{tool}_{acq}',
                   DICE / f'dice_{tool}_{acq}.csv', LOGS / 'dice_old.log')
```

## New dataset

To be added.
