---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python [conda env:lfbook]
  language: python
  name: conda-env-lfbook-py
---

# Benchmark Structures

Which brain structures are suitable as a benchmark for judging whether an image-enhancement
pipeline improves 64mT scans? A good benchmark shows the gap between 64mT and 3T clearly, and in
the same way for every subject, so that an improvement would be visible.

## Why WMH-SynthSeg

- **Dice (Chapter 05)**: WMH-SynthSeg agrees better with 3T on FLAIR and T1w, and equally on T2w.
- **Volume (Chapter 06)**: SynthSeg systematically underestimates deep and lower-brain structures
  on 64mT, with segmentation failures (zero volume).

From here on only WMH-SynthSeg is used: 64mT compared with 3T lowres, repeated scans averaged per
subject (n = 11), the same 17 regions as Chapters 05 and 06.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to

from common import OLD_OUT

DICE, REG, VOL = OLD_OUT / 'dice', OLD_OUT / 'reg', OLD_OUT / 'volume'
MODS = ['T2w', 'T1w', 'FLAIR']
# region -> merged label id in the Chapter 05 label maps
LABEL = {'WM': 2, 'Cortex': 3, 'Lateral-Ventricle': 4, 'Cerebellum-WM': 7, 'Cerebellum-Cortex': 8,
         'Thalamus': 10, 'Caudate': 11, 'Putamen': 12, 'Pallidum': 13, '3rd-Ventricle': 14,
         '4th-Ventricle': 15, 'Brainstem': 16, 'Hippocampus': 17, 'Amygdala': 18, 'CSF': 24,
         'Accumbens': 26, 'VentralDC': 28}

pairs = pd.read_csv(REG / 'pairs_lowres.csv', keep_default_na=False).set_index('pair_id')
dice = pd.read_csv(DICE / 'dice_WMH_SynthSeg_lowres.csv')
dice = dice[dice.region.isin(LABEL)]
# one value per subject: average repeated sessions / runs first
subj = dice.groupby(['subject', 'modality', 'region'], as_index=False)[['dice_np', 'n_3T']].mean()
bias = pd.read_csv(VOL / 'bias_long.csv').query("tool == 'WMH_SynthSeg'")    # Chapter 06, per subject
```

## How Dice and volume relate

Registration has been checked (Chapter 04), so the two label maps are aligned. Then:

- **High Dice → small volume difference.** Two maps that overlap well cannot differ much in size.
- **Small volume difference → Dice can still be low.** Two maps can have the same volume and
  still disagree on where the boundary is. This happens most for thin, folded structures.

So volume alone can hide a gap that Dice shows. Dice also depends on the structure itself: its
size, and how much of it lies on a boundary.

## Structure size and Dice

Each point is one region: median Dice of 11 subjects, bar = Q1–Q3 (spread between subjects);
x = size of the structure on 3T.

```{code-cell} ipython3
def plot_size_dice():
    g = subj.groupby(['modality', 'region'])
    t = pd.DataFrame(dict(vol=g.n_3T.median() / 1000, q1=g.dice_np.quantile(0.25),
                          dice=g.dice_np.median(), q3=g.dice_np.quantile(0.75))).reset_index()
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    for ax, mod in zip(axes, MODS):
        x = t[t.modality == mod]
        ax.errorbar(x.vol, x.dice, yerr=[x.dice - x.q1, x.q3 - x.dice], fmt='o', ms=5,
                    color='#2a78d6', lw=1, capsize=2)
        for _, r in x.iterrows():
            ax.annotate(r.region, (r.vol, r.dice), xytext=(5, 2), textcoords='offset points', fontsize=8)
        ax.set(xscale='log', title=mod, ylabel='Dice, 64mT vs 3T lowres', xlim=(0.5, 800), ylim=(0.5, 1))
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].set_xlabel('Structure size: 3T volume (cm³, log scale)')
    fig.suptitle('WMH-SynthSeg: structure size vs Dice (median of 11 subjects, bar = Q1–Q3)', x=0.01, ha='left')
    plt.tight_layout()
    plt.show()


plot_size_dice()
```

Three groups stand out:

1. **High Dice**: thalamus and brainstem, at the top.
2. **Small structures**: 3rd and 4th ventricle, accumbens, pallidum, on the left; on FLAIR they
   drop lowest and spread most.
3. **Large, low but stable**: cortex and CSF, bottom right, with short bars.

The remaining regions sit in between and are not discussed further.

## The three groups

For each group: Dice as median (IQR) of 11 subjects; volume bias (Chapter 06) as median % [Q1, Q3];
and one subject on T2w, chosen as the pair whose Dice is closest to the group median.
Red = 3T only, blue = 64mT only, purple = both, on the 3T image.

```{code-cell} ipython3
def group_table(regions):
    rows = {}
    for r in regions:
        row = {}
        for mod in MODS:
            d = subj[(subj.region == r) & (subj.modality == mod)].dice_np.quantile([0.25, 0.5, 0.75])
            b = bias[(bias.region == r) & (bias.modality == mod)].pct_diff.quantile([0.25, 0.5, 0.75])
            row[('Dice', mod)] = f'{d[0.5]:.2f} ({d[0.75] - d[0.25]:.2f})'
            row[('Volume bias %', mod)] = f'{b[0.5]:+.0f} [{b[0.25]:+.0f}, {b[0.75]:+.0f}]'
        rows[r] = row
    return pd.DataFrame(rows).T


def show_overlay(regions, mod='T2w', n_slices=4, zoom=False):
    """zoom=True crops to the structures instead of the whole head."""
    ids = [LABEL[r] for r in regions]
    d = dice[(dice.modality == mod) & dice.region.isin(regions)].groupby('pair_id').dice_np.mean()
    pid = (d - d.median()).abs().idxmin()
    work = DICE / 'work' / 'WMH_SynthSeg_lowres'
    hf = nib.as_closest_canonical(nib.load(work / f'{pid}_3T_h.nii.gz'))
    lf = nib.as_closest_canonical(nib.load(work / f'{pid}_64mT_h.nii.gz'))
    img = nib.load(pairs.loc[pid, 'fix_SynthSeg'].replace('_synthseg.nii.gz', '_resampled.nii.gz'))
    bg = resample_from_to(img, hf, order=1).get_fdata()
    bg = (bg / np.percentile(bg, 99)).clip(0, 1)
    A, B = np.isin(np.asarray(hf.dataobj), ids), np.isin(np.asarray(lf.dataobj), ids)

    xs, ys, _ = np.where((A | B) if zoom else (bg > 0.05))            # crop to structures or head
    m = 15 if zoom else 0
    crop = np.s_[max(xs.min() - m, 0):xs.max() + m, max(ys.min() - m, 0):ys.max() + m]
    z = np.where((A | B).any(axis=(0, 1)))[0]                         # slices containing the structures
    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4.6))
    for ax, k in zip(axes, z[np.arange(1, n_slices + 1) * len(z) // (n_slices + 1)]):
        a, b = A[:, :, k], B[:, :, k]
        over = np.zeros(a.shape + (4,))
        over[a & ~b] = [0.85, 0.1, 0.1, 0.7]                          # 3T only: red
        over[b & ~a] = [0.1, 0.35, 0.9, 0.7]                          # 64mT only: blue
        over[a & b] = [0.6, 0.2, 0.75, 0.7]                           # both: purple
        ax.imshow(np.rot90(bg[:, :, k][crop]), cmap='gray')
        ax.imshow(np.rot90(over[crop]))
        sd = 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
        ax.set_title(f'slice {k}, Dice {sd:.2f}', fontsize=9)
        ax.axis('off')
    fig.suptitle(f"{pid.split('__')[0]}: {', '.join(regions)}   (whole-structure Dice {d[pid]:.2f}; R on right)",
                 x=0.01, ha='left', fontsize=10)
    plt.tight_layout()
    plt.show()
```

### 1. High Dice: thalamus and brainstem

```{code-cell} ipython3
HIGH = ['Thalamus', 'Brainstem']
group_table(HIGH)
```

```{code-cell} ipython3
show_overlay(HIGH)
```

Dice is close to 0.9 and the volume bias is small. There is little room left for an enhancement
method to show improvement. The high score may also come partly from the tool's prior: these
structures have a fixed position and a predictable shape, so they can be placed well even when
the image carries little information.

### 2. Small structures

```{code-cell} ipython3
SMALL = ['3rd-Ventricle', '4th-Ventricle', 'Accumbens', 'Pallidum']
group_table(SMALL)
```

```{code-cell} ipython3
show_overlay(SMALL, zoom=True)
```

A few voxels make a large difference in both Dice and volume, so results jump between subjects
and sequences (widest on FLAIR). A real improvement would be hard to tell apart from this noise.

### 3. Large, low but stable: cortex and CSF

```{code-cell} ipython3
LARGE_LOW = ['Cortex', 'CSF']
group_table(LARGE_LOW)
```

```{code-cell} ipython3
show_overlay(['Cortex'])
```

```{code-cell} ipython3
show_overlay(['CSF'])
```

Dice is low (cortex about 0.7, CSF about 0.6) but nearly the same in every subject, so there is
room to improve and an improvement would be visible. The disagreement sits along the thin,
folded boundary — where a loss of contrast at low field should show. The volume bias is small,
so volume alone would miss the gap.

## Conclusion

A structure suitable as a benchmark must:

1. **Have a low Dice** — room to improve, without being a segmentation failure;
2. **Be stable across subjects** — the gap is about the same for everyone, so an improvement is
   visible;
3. **Not be too small** — otherwise a few voxels dominate the result.

**Cortex and CSF meet all three.** They should be judged by **Dice, not volume**: an enhancement
model can generate plausible sulci in the wrong place while the volume still matches; Dice
penalises misplaced boundaries, volume does not. CSF on FLAIR should be used with care, as FLAIR
suppresses CSF signal by design.

**Next: a second dataset.** Chapter 08 repeats the analysis on a new dataset (12 subjects, a
different 3T scanner) to check whether cortex and CSF again show a gap to 3T that is stable
across subjects and leaves room for an enhancement method to improve.
