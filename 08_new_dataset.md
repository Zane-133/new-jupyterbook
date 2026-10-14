---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python [conda env:lfbook]
  language: python
  name: conda-env-lfbook-py
---

# New Dataset

The analysis of Chapters 03–07 repeated on the new dataset, WMH-SynthSeg only, and compared with
the old dataset.

New dataset: 12 healthy volunteers, 3T Siemens MAGNETOM Cima.X and 64mT Hyperfine Swoop, one
scan per subject, modality and field strength (72 scans). For a test run set `subjects_new` in
`config.yml` (e.g. `[sub-03]`); results then go to `new_test/` instead of `new/`.

Differences from the old dataset:

- 3T has one ~1 mm isotropic scan per modality and **no lowres**, so 64mT is compared with this
  scan. The old dataset is compared with 3T lowres (Chapters 05–07).
- 3T T1w is MP2RAGE INV2, a different contrast from the 64mT T1w.
- 64mT in-plane voxels are 0.65–0.85 mm (old: 1.6–1.7 mm); slices are 5 mm in both.
- One 64mT scan per subject: no averaging over sessions, n = 12.

```{code-cell} ipython3
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel.processing import resample_from_to

from common import P, NEW_RAW, NEW_OUT, OLD_OUT, WORK, LOGS, THREADS, SUBJECTS_NEW, fs, sh

SEG, REG, DICE, VOL = (NEW_OUT / x for x in ('seg/WMH_SynthSeg', 'reg', 'dice', 'volume'))
for d in (SEG, REG, DICE, VOL):
    d.mkdir(parents=True, exist_ok=True)
MODS = ['T2w', 'T1w', 'FLAIR']
```

## Scans and pairs

Each 64mT scan (moving) is paired with the 3T scan (fixed) of the same subject and modality.

```{code-cell} ipython3
rows = []
for f in sorted(NEW_RAW.glob('64mT/sub-*/anat/*.nii.gz')):
    sub, mod = re.fullmatch(r'(sub-\d+)_(T1w|T2w|FLAIR)\.nii\.gz', f.name).groups()
    stem = f'{sub}_{mod}'
    rows.append(dict(pair_id=stem, subject=sub, modality=mod,
                     mov_raw=str(f), fix_raw=str(NEW_RAW / '3T' / sub / 'anat' / f.name),
                     mov_seg=str(SEG / '64mT' / sub / f'{stem}_WMHseg.nii.gz'),
                     fix_seg=str(SEG / '3T' / sub / f'{stem}_WMHseg.nii.gz')))
pairs = pd.DataFrame(rows)
assert len(pairs) == 36, f'{len(pairs)} pairs, expected 36'
if SUBJECTS_NEW != 'all':
    pairs = pairs[pairs.subject.isin(SUBJECTS_NEW)].reset_index(drop=True)

assert len(pairs), f"no pairs for subjects_new = {SUBJECTS_NEW!r}: use 'all' or a list such as [sub-03]"
missing = [f for c in ['mov_raw', 'fix_raw'] for f in pairs[c] if not os.path.exists(f)]
assert not missing, f'{len(missing)} scans missing, e.g. {missing[:3]}'

pairs.to_csv(REG / 'pairs.csv', index=False)
print(f"{len(pairs)} pairs, {pairs.subject.nunique()} subjects -> {REG / 'pairs.csv'}")
```

## Segmentation

WMH-SynthSeg on the A40 with the code, model and settings of Chapter 03. 3T and 64mT share file
names, so each scan gets an `id` with the field strength in front (`3T_sub-03_T1w`), used for
staging and in the volume table. Label maps: `seg/WMH_SynthSeg/{3T,64mT}/<sub>/<sub>_<mod>_WMHseg.nii.gz`.

```{code-cell} ipython3
WMH_REPO, MODEL = P['wmh_code'], P['wmh_model']
INF = WMH_REPO / 'WMHSynthSeg' / 'inference.py'
assert INF.exists() and MODEL.exists(), 'WMH-SynthSeg code or model missing: run Chapter 03 first'
assert torch.cuda.is_available(), 'no GPU: start Neurodesk on the A40 server'

# one row per scan: 3T = fixed side of the pairs, 64mT = moving side
scans = pd.concat([pd.DataFrame({'id': f'{field}_' + pairs.pair_id, 'scan': pairs.pair_id, 'fs': field,
                                 'sub': pairs.subject, 'mod': pairs.modality,
                                 'src': pairs[f'{side}_raw'], 'seg': pairs[f'{side}_seg']})
                   for field, side in [('3T', 'fix'), ('64mT', 'mov')]], ignore_index=True)
print(f"{len(scans)} scans, {scans['sub'].nunique()} subjects")
```

Voxel size and grid in R × A × S order, as `inference.py` sees them, with the 1 mm grid and the
padded grid the network runs on (see Chapter 03).

```{code-cell} ipython3
def grid_ras(src):
    """Voxel size (mm) and grid size in R, A, S order, as inference.py sees them (header only)."""
    img = nib.load(src)
    axis = nib.io_orientation(img.affine)[:, 0].astype(int)   # input axis i -> RAS axis axis[i]
    vox, shape = np.zeros(3), np.zeros(3, int)
    vox[axis], shape[axis] = img.header.get_zooms()[:3], img.shape[:3]
    return vox, shape


rows = []
for r in scans.to_dict('records'):
    vox, shape = grid_ras(r['src'])
    new = np.round(shape * vox).astype(int)                    # factor = voxel size / 1 mm
    pad = (np.ceil(new / 32) * 32).astype(int)
    rows.append(dict(fs=r['fs'], mod=r['mod'],
                     voxel_mm=' × '.join(f'{x:.2f}' for x in vox),
                     grid=' × '.join(map(str, shape)),
                     grid_1mm=' × '.join(map(str, new)),
                     padded=' × '.join(map(str, pad))))
(pd.DataFrame(rows).groupby(list(rows[0]), sort=False).size().rename('scans')
 .reset_index().sort_values(['fs', 'mod'], ignore_index=True))
```

Scans without a label map are segmented in one run of `inference.py`; label maps are moved to the
results folder and their volumes added to `volumes_main.csv`, as in Chapter 03.

```{code-cell} ipython3
LOG = LOGS / 'wmh_synthseg_new.log'
todo = [r for r in scans.to_dict('records') if not os.path.exists(r['seg'])]
if todo:
    print(f'{len(todo)} scans to segment')
    stage, segout = WORK / 'WMH_SynthSeg_new' / 'stage', WORK / 'WMH_SynthSeg_new' / 'segout'
    for d in (stage, segout):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for r in todo:
        (stage / f"{r['id']}.nii.gz").symlink_to(r['src'])

    # expandable_segments reduces GPU memory fragmentation
    sh(f'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {sys.executable} inference.py '
       f'--i {stage} --o {segout} --csv_vols {segout}/vols.csv --device cuda',
       cwd=INF.parent, log=LOG, show=False)

    for r in todo:
        seg = segout / f"{r['id']}_seg.nii.gz"
        if seg.exists():
            Path(r['seg']).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(seg, r['seg'])

    # volume table: scan columns + volumes, merged with earlier results
    vols = pd.read_csv(segout / 'vols.csv')
    vols['id'] = vols.pop('Input-file').map(lambda p: os.path.basename(p)[:-len('_seg.nii.gz')])
    tidy = pd.DataFrame(todo).drop(columns=['src', 'seg']).merge(vols, on='id')
    dst = SEG / 'volumes_main.csv'
    if dst.exists():
        tidy = pd.concat([pd.read_csv(dst, keep_default_na=False), tidy],
                         ignore_index=True).drop_duplicates('id', keep='last')
    tidy.sort_values(['fs', 'sub', 'mod']).to_csv(dst, index=False)

    commit = subprocess.run(['git', '-C', str(WMH_REPO), 'log', '-1', '--format=%h %ad'],
                            capture_output=True, text=True).stdout.strip()
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    (SEG / 'PROVENANCE.txt').write_text(f"""WMH-SynthSeg results, new dataset ({len(scans)} scans)
Computed on Neurodesk, Chapter 08 of the book; written {time.strftime('%Y-%m-%d %H:%M %Z')}
GPU: {gpu} | torch {torch.__version__}
Code: lasopablo/freesurfer-freesurfer-dev-mri_WMHsynthseg, commit {commit}
Model: {MODEL.name} ({MODEL.stat().st_size / 1e6:.0f} MB)
Settings: --device cuda, no --crop
""")

missing = [r['id'] for r in scans.to_dict('records') if not os.path.exists(r['seg'])]
if missing:
    sh(f"grep 'error occurred' {LOG} | sort | uniq -c || true")
    raise RuntimeError(f'{len(missing)} scans not segmented: {missing}')
vol_new = pd.read_csv(SEG / 'volumes_main.csv', keep_default_na=False)
no_row = sorted(set(scans.id) - set(vol_new.id))
assert not no_row, f'missing rows in volumes_main.csv: {no_row}'
print((SEG / 'PROVENANCE.txt').read_text())
print(f'WMH-SynthSeg: {len(scans)} scans -> {SEG}')
```

## Registration

`mri_synthmorph register`, 64mT raw image onto 3T raw image, one `.lta` per pair. Existing
transforms are skipped. `REG_MODEL` sets the transform: `rigid` (6 parameters: rotation and
translation, as Chapter 04) or `affine` (12 parameters: also scaling and shear). Affine is used
here because the 64mT brain is smaller than the 3T brain in every scan (ICV, see Volume below);
a rigid transform cannot correct that. Each model keeps its own transforms and Dice table, so
switching back to `rigid` reuses the earlier results.

```{code-cell} ipython3
REG_MODEL = 'affine'                     # 'rigid' or 'affine'
SUFFIX = '' if REG_MODEL == 'rigid' else f'_{REG_MODEL}'
XFM = REG / f'xfm{SUFFIX}'
XFM.mkdir(exist_ok=True)
for r in pairs.itertuples():
    lta = XFM / f'{r.pair_id}.lta'
    if lta.exists() and lta.stat().st_size > 0:
        continue
    print(r.pair_id)
    fs(f'export OMP_NUM_THREADS={THREADS} && '
       f'mri_synthmorph register -m {REG_MODEL} -t {lta} {r.mov_raw} {r.fix_raw}',
       log=LOGS / f'register_new{SUFFIX}.log', show=False)

n = sum((XFM / f'{p}.lta').exists() for p in pairs.pair_id)
assert n == len(pairs), f'{n} / {len(pairs)} transforms'
print(f'{REG_MODEL}: {n} transforms -> {XFM}')
```

Scaling in each transform: how much the 64mT image is stretched along its R, A and S axes to
fit the 3T image, and the volume factor (determinant). 1.00 = no scaling; above 1 = the 64mT
image is smaller than the 3T image. For a rigid transform all are 1.00.

```{code-cell} ipython3
def lta_ras2ras(path):
    """4x4 transform of an .lta file in scanner (RAS) coordinates, moving -> fixed."""
    lines = Path(path).read_text().splitlines()
    kind = int(next(l for l in lines if l.strip().startswith('type')).split('=')[1].split()[0])
    i = lines.index('1 4 4')
    M = np.loadtxt(lines[i + 1:i + 5])
    if kind == 1:                                    # LINEAR_RAS_TO_RAS
        return M

    def vox2ras(tag):                                # volume geometry stored in the .lta
        j = next(k for k, l in enumerate(lines) if l.startswith(f'{tag} volume info'))
        g = {k.strip(): v for k, v in (l.split('=', 1) for l in lines[j + 1:j + 12] if '=' in l)}
        vs, dims = (np.array(g[k].split(), float) for k in ('voxelsize', 'volume'))
        R = np.column_stack([np.array(g[k].split(), float) for k in ('xras', 'yras', 'zras')]) * vs
        A = np.eye(4)
        A[:3, :3], A[:3, 3] = R, np.array(g['cras'].split(), float) - R @ (dims / 2)
        return A
    return vox2ras('dst') @ M @ np.linalg.inv(vox2ras('src'))   # LINEAR_VOX_TO_VOX


rows = []
for r in pairs.itertuples():
    L = lta_ras2ras(XFM / f'{r.pair_id}.lta')[:3, :3]
    sR, sA, sS = np.linalg.norm(L, axis=0)           # stretch of the 64mT R, A, S axes
    rows.append(dict(subject=r.subject, modality=r.modality, R=sR, A=sA, S=sS, volume=np.linalg.det(L)))
scale = pd.DataFrame(rows)
(scale.groupby('modality')[['R', 'A', 'S', 'volume']]
 .agg(lambda x: f'{x.median():.3f} [{x.min():.3f}, {x.max():.3f}]').loc[MODS])
```

## Dice

As Chapter 05: move the 64mT label map onto the 3T label map with the pair's `.lta`, merge
left/right labels into 17 regions (WMH also counted as WM), Dice per region.

```{code-cell} ipython3
# region -> (representative id, source labels), as Chapter 05
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
    "WMH":               (77, [77]),          # also counted as WM
}
LABEL = {name: rid for name, (rid, _) in REGIONS.items() if name != 'WMH'}
SIZE_ORDER = ['WM', 'Cortex', 'Cerebellum-Cortex', 'Cerebellum-WM', 'Brainstem', 'Lateral-Ventricle',
              'CSF', 'Thalamus', 'Putamen', 'Hippocampus', 'Caudate', 'VentralDC', 'Pallidum',
              'Amygdala', '3rd-Ventricle', '4th-Ventricle', 'Accumbens']      # large -> small


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
    o = nib.Nifti1Image(out, img.affine)
    o.set_data_dtype(np.int32)
    nib.save(o, dst)
    return masks


def dice(A, B):
    na, nb, i = int(A.sum()), int(B.sum()), int((A & B).sum())
    return dict(dice_np=2 * i / (na + nb), jaccard=i / (na + nb - i),
                voldiff=(nb - na) / ((na + nb) / 2), n_3T=na, n_64mT=nb)


DICE_CSV, WORK = DICE / f'dice_WMH_SynthSeg{SUFFIX}.csv', DICE / f'work{SUFFIX}'
if not DICE_CSV.exists():
    WORK.mkdir(parents=True, exist_ok=True)
    out = []
    for r in pairs.to_dict('records'):
        pid = r['pair_id']
        print(pid)
        moved = WORK / f'{pid}_64mT_native.nii.gz'
        fs(f"mri_vol2vol --mov {r['mov_seg']} --targ {r['fix_seg']} --lta {XFM}/{pid}.lta "
           f"--nearest --o {moved} --no-save-reg", log=LOGS / 'dice_new.log', show=False)
        A = harmonize(r['fix_seg'], WORK / f'{pid}_3T_h.nii.gz')       # 3T reference
        B = harmonize(moved, WORK / f'{pid}_64mT_h.nii.gz')            # registered 64mT
        for name in REGIONS:
            if A[name].any() or B[name].any():
                out.append(dict(pair_id=pid, subject=r['subject'], modality=r['modality'], region=name)
                           | dice(A[name], B[name]))
    pd.DataFrame(out).to_csv(DICE_CSV, index=False)

dice_new = pd.read_csv(DICE_CSV)
print(f'{len(dice_new)} rows, {dice_new.pair_id.nunique()} pairs, '
      f'{dice_new.region.nunique()} regions -> {DICE_CSV}')
```

Median Dice per region and sequence, 12 subjects:

```{code-cell} ipython3
d17 = dice_new[dice_new.region.isin(LABEL)]
(d17.pivot_table(index='region', columns='modality', values='dice_np', aggfunc='median')
 .loc[SIZE_ORDER, MODS].round(2))
```

```{code-cell} ipython3
def plot_dist(d, value, ylabel, ylim, title, zero=False):
    """Box plot per region and sequence (one value per subject)."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, mod in zip(axes, MODS):
        x = d[d.modality == mod]
        ax.boxplot([x[x.region == r][value].dropna() for r in SIZE_ORDER], widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor='lightgray'), medianprops=dict(color='k'),
                   flierprops=dict(markersize=3))
        if zero:
            ax.axhline(0, color='0.5', lw=0.8)
        ax.set(title=f'{title} {mod}', ylabel=ylabel, ylim=ylim)
        ax.grid(axis='y', alpha=0.3)
    axes[-1].set_xticks(range(1, len(SIZE_ORDER) + 1), SIZE_ORDER, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


plot_dist(d17, 'dice_np', 'Dice', (0, 1), 'New dataset, 64mT vs 3T:')
```

## Volume

As Chapter 06: bias (%) = (64mT − 3T) / 3T × 100, from `volumes_main.csv`; left/right summed,
WMH counted as WM, no ICV normalisation.

```{code-cell} ipython3
# region -> WMH-SynthSeg columns (summed), as Chapter 06
VCOLS = {
    'WM':                ['left-white-matter(2)', 'right-white-matter(41)', 'WMH(77)'],
    'Cortex':            ['left-cortex(3)', 'right-cortex(42)'],
    'Cerebellum-Cortex': ['left-cerebellum-cortex(8)', 'right-cerebellum-cortex(47)'],
    'Cerebellum-WM':     ['left-cerebellum-white-matter(7)', 'right-cerebellum-white-matter(46)'],
    'Brainstem':         ['brainstem(16)'],
    'Lateral-Ventricle': ['left-lateral-ventricle(4)', 'right-lateral-ventricle(43)'],
    'CSF':               ['extracerebral_CSF(24)'],
    'Thalamus':          ['left-thalamus(10)', 'right-thalamus(49)'],
    'Putamen':           ['left-putamen(12)', 'right-putamen(51)'],
    'Hippocampus':       ['left-hippocampus(17)', 'right-hippocampus(53)'],
    'Caudate':           ['left-caudate(11)', 'right-caudate(50)'],
    'VentralDC':         ['left-ventral-DC(28)', 'right-ventral-DC(60)'],
    'Pallidum':          ['left-pallidum(13)', 'right-pallidum(52)'],
    'Amygdala':          ['left-amygdala(18)', 'right-amygdala(54)'],
    '3rd-Ventricle':     ['3rd-ventricle(14)'],
    '4th-Ventricle':     ['4th-ventricle(15)'],
    'Accumbens':         ['left-accumbens(26)', 'right-accumbens(58)'],
    'ICV':               ['Intracranial-volume'],
}

v = vol_new[vol_new['sub'].isin(pairs.subject)]
vol = pd.DataFrame({r: v[c].astype(float).sum(axis=1) for r, c in VCOLS.items()})
vol[['subject', 'fs', 'modality']] = v[['sub', 'fs', 'mod']]
key = ['subject', 'modality']
hf = vol[vol.fs == '3T'].set_index(key)[list(VCOLS)]
lf = vol[vol.fs == '64mT'].set_index(key)[list(VCOLS)]
bias_new = (((lf - hf.reindex(lf.index)) / hf.reindex(lf.index) * 100)
            .reset_index().melt(id_vars=key, var_name='region', value_name='pct_diff'))
bias_new.to_csv(VOL / 'bias_long.csv', index=False)
print(f"{len(bias_new)} rows ({bias_new.subject.nunique()} subjects × 3 sequences × {len(VCOLS)} regions) "
      f"-> {VOL / 'bias_long.csv'}")
```

Median [Q1, Q3] of volume bias (%), 12 subjects. ICV is listed for reference.

```{code-cell} ipython3
def bias_table(b, regions):
    q = b.groupby(['region', 'modality']).pct_diff.quantile([0.25, 0.5, 0.75]).unstack()
    cell = q.apply(lambda r: f'{r[0.5]:+.0f} [{r[0.25]:+.0f}, {r[0.75]:+.0f}]', axis=1)
    return cell.unstack('modality').loc[regions, MODS]


bias_table(bias_new, SIZE_ORDER + ['ICV'])
```

The same bias after removing the scaling of each pair's transform (64mT volume × volume factor
above). With rigid registration nothing changes. If the shrinkage is uniform, ICV should move to
about 0 %.

```{code-cell} ipython3
vf = scale.set_index(['subject', 'modality']).volume
lf_corr = lf.mul(vf.reindex(lf.index), axis=0)
bias_corr = (((lf_corr - hf.reindex(lf.index)) / hf.reindex(lf.index) * 100)
             .reset_index().melt(id_vars=key, var_name='region', value_name='pct_diff'))
bias_table(bias_corr, SIZE_ORDER + ['ICV'])
```

```{code-cell} ipython3
plot_dist(bias_new[bias_new.region.isin(SIZE_ORDER)], 'pct_diff', 'Volume bias (%)', (-105, 105),
          'New dataset, 64mT vs 3T:', zero=True)
```

## Benchmark structures

As Chapter 07: structure size against Dice, then the three groups.

```{code-cell} ipython3
def plot_size_dice(d, title):
    g = d.groupby(['modality', 'region'])
    t = pd.DataFrame(dict(vol=g.n_3T.median() / 1000, q1=g.dice_np.quantile(0.25),
                          dice=g.dice_np.median(), q3=g.dice_np.quantile(0.75))).reset_index()
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    for ax, mod in zip(axes, MODS):
        x = t[t.modality == mod]
        ax.errorbar(x.vol, x.dice, yerr=[x.dice - x.q1, x.q3 - x.dice], fmt='o', ms=5,
                    color='#2a78d6', lw=1, capsize=2)
        for _, r in x.iterrows():
            ax.annotate(r.region, (r.vol, r.dice), xytext=(5, 2), textcoords='offset points', fontsize=8)
        ax.set(xscale='log', title=mod, ylabel='Dice, 64mT vs 3T', xlim=(0.5, 800),
               ylim=(min(0.5, t.q1.min() - 0.05), 1))
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].set_xlabel('Structure size: 3T volume (cm³, log scale)')
    fig.suptitle(title, x=0.01, ha='left')
    plt.tight_layout()
    plt.show()


plot_size_dice(d17, 'New dataset: structure size vs Dice (median of 12 subjects, bar = Q1–Q3)')
```

```{code-cell} ipython3
def group_table(regions):
    rows = {}
    for r in regions:
        row = {}
        for mod in MODS:
            d = d17[(d17.region == r) & (d17.modality == mod)].dice_np.quantile([0.25, 0.5, 0.75])
            b = bias_new[(bias_new.region == r) & (bias_new.modality == mod)].pct_diff.quantile([0.25, 0.5, 0.75])
            row[('Dice', mod)] = f'{d[0.5]:.2f} ({d[0.75] - d[0.25]:.2f})'
            row[('Volume bias %', mod)] = f'{b[0.5]:+.0f} [{b[0.25]:+.0f}, {b[0.75]:+.0f}]'
        rows[r] = row
    return pd.DataFrame(rows).T


def show_overlay(regions, mod='T2w', n_slices=4, zoom=False):
    """As Chapter 07; background = 3T raw image resampled onto the label-map grid.
    Red = 3T only, blue = 64mT only, purple = both."""
    ids = [LABEL[r] for r in regions]
    d = d17[(d17.modality == mod) & d17.region.isin(regions)].groupby('pair_id').dice_np.mean()
    pid = (d - d.median()).abs().idxmin()
    hf_ = nib.as_closest_canonical(nib.load(WORK / f'{pid}_3T_h.nii.gz'))
    lf_ = nib.as_closest_canonical(nib.load(WORK / f'{pid}_64mT_h.nii.gz'))
    img = nib.load(pairs.set_index('pair_id').loc[pid, 'fix_raw'])
    bg = resample_from_to(img, hf_, order=1).get_fdata()
    bg = (bg / np.percentile(bg, 99)).clip(0, 1)
    A, B = np.isin(np.asarray(hf_.dataobj), ids), np.isin(np.asarray(lf_.dataobj), ids)

    xs, ys, _ = np.where((A | B) if zoom else (bg > 0.05))
    m = 15 if zoom else 0
    crop = np.s_[max(xs.min() - m, 0):xs.max() + m, max(ys.min() - m, 0):ys.max() + m]
    z = np.where((A | B).any(axis=(0, 1)))[0]
    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4.6))
    for ax, k in zip(axes, z[np.arange(1, n_slices + 1) * len(z) // (n_slices + 1)]):
        a, b = A[:, :, k], B[:, :, k]
        over = np.zeros(a.shape + (4,))
        over[a & ~b] = [0.85, 0.1, 0.1, 0.7]
        over[b & ~a] = [0.1, 0.35, 0.9, 0.7]
        over[a & b] = [0.6, 0.2, 0.75, 0.7]
        ax.imshow(np.rot90(bg[:, :, k][crop]), cmap='gray')
        ax.imshow(np.rot90(over[crop]))
        sd = 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
        ax.set_title(f'slice {k}, Dice {sd:.2f}', fontsize=9)
        ax.axis('off')
    fig.suptitle(f"{pid}: {', '.join(regions)}   (whole-structure Dice {d[pid]:.2f}; R on right)",
                 x=0.01, ha='left', fontsize=10)
    plt.tight_layout()
    plt.show()


HIGH = ['Thalamus', 'Brainstem']
SMALL = ['3rd-Ventricle', '4th-Ventricle', 'Accumbens', 'Pallidum']
LARGE_LOW = ['Cortex', 'CSF']
```

### 1. High Dice: thalamus and brainstem

```{code-cell} ipython3
group_table(HIGH)
```

```{code-cell} ipython3
show_overlay(HIGH)
```

### 2. Small structures

```{code-cell} ipython3
group_table(SMALL)
```

```{code-cell} ipython3
show_overlay(SMALL, zoom=True)
```

### 3. Large, low but stable: cortex and CSF

```{code-cell} ipython3
group_table(LARGE_LOW)
```

```{code-cell} ipython3
show_overlay(['Cortex'])
```

```{code-cell} ipython3
show_overlay(['CSF'])
```

## Old vs new

Old: 64mT vs 3T lowres, repeated scans averaged per subject (n = 11; Chapters 05–06).
New: 64mT vs 3T ~1 mm (n = 12). Bar = median, error bar = Q1–Q3.

```{code-cell} ipython3
dice_old = pd.read_csv(OLD_OUT / 'dice' / 'dice_WMH_SynthSeg_lowres.csv')
dice_old = (dice_old[dice_old.region.isin(LABEL)]
            .groupby(['subject', 'modality', 'region'], as_index=False).dice_np.mean())
bias_old = pd.read_csv(OLD_OUT / 'volume' / 'bias_long.csv').query("tool == 'WMH_SynthSeg'")
assert set(SIZE_ORDER) <= set(bias_old.region), 'old bias_long.csv is not the Chapter 06 version: run Chapter 06'
both_dice = pd.concat({'old': dice_old, 'new': d17[['subject', 'modality', 'region', 'dice_np']]},
                      names=['dataset']).reset_index(level=0)
both_bias = pd.concat({'old': bias_old[['subject', 'modality', 'region', 'pct_diff']],
                       'new': bias_new[bias_new.region.isin(SIZE_ORDER)]},
                      names=['dataset']).reset_index(level=0)
print(both_dice.groupby('dataset').subject.nunique().to_dict(), 'subjects (Dice)')
print(both_bias.groupby('dataset').subject.nunique().to_dict(), 'subjects (volume)')
```

```{code-cell} ipython3
COLORS = {'old': '#c4c4c4', 'new': '#2a78d6'}


def plot_old_new(d, value, ylabel, ylim):
    x = np.arange(len(SIZE_ORDER))
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for ax, mod in zip(axes, MODS):
        for i, ds in enumerate(['old', 'new']):
            g = d[(d.dataset == ds) & (d.modality == mod)].groupby('region')[value]
            q = g.quantile([0.25, 0.5, 0.75]).unstack().reindex(SIZE_ORDER)
            ax.bar(x + (i - 0.5) * 0.4, q[0.5], 0.4, color=COLORS[ds], label=ds,
                   yerr=[q[0.5] - q[0.25], q[0.75] - q[0.5]],
                   error_kw=dict(lw=0.8, capsize=2, ecolor='0.3'))
        ax.axhline(0, color='0.5', lw=0.8)
        ax.set(title=mod, ylabel=ylabel, ylim=ylim)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper right', ncol=2, frameon=False)
    axes[-1].set_xticks(x, SIZE_ORDER, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
```

### Dice

```{code-cell} ipython3
plot_old_new(both_dice, 'dice_np', 'Dice', (0, 1))
```

```{code-cell} ipython3
def side_by_side(d, value, fmt):
    """Median [Q1, Q3] per region and sequence, old next to new."""
    q = d.groupby(['region', 'modality', 'dataset'])[value].quantile([0.25, 0.5, 0.75]).unstack()
    cell = q.apply(lambda r: f'{r[0.5]:{fmt}} [{r[0.25]:{fmt}}, {r[0.75]:{fmt}}]', axis=1)
    t = cell.unstack(['modality', 'dataset'])
    return t.loc[SIZE_ORDER, [(m, ds) for m in MODS for ds in ['old', 'new']]]


side_by_side(both_dice, 'dice_np', '.2f')
```

### Volume bias

```{code-cell} ipython3
plot_old_new(both_bias, 'pct_diff', 'Volume bias (%)', (-60, 60))
```

```{code-cell} ipython3
side_by_side(both_bias, 'pct_diff', '+.0f')
```

## Discussion

To be written after the results.

Limitations to keep in mind when reading the comparison:

- **No 3T lowres.** The new 64mT scans are compared with ~1 mm 3T scans, the old ones with 3T
  lowres. The new comparison therefore contains a resolution gap as well as a field-strength
  gap, and is expected to give lower Dice than the old one.
- **3T T1w is MP2RAGE INV2**, a different contrast from the 64mT T1w.
- **64mT in-plane resolution** is higher than in the old dataset (0.65–0.85 mm vs 1.6–1.7 mm);
  slice thickness is the same (5 mm).
- **n = 12, one session each**; the old dataset averages repeated scans per subject (n = 11).
