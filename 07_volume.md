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

# Volume

Volume bias between 3T and 64mT on the old dataset, per tool, modality and region.

Two comparisons, same subject and modality:

| | Reference → target | Contains |
|---|---|---|
| **A** | 3T highres → 3T lowres | resolution |
| **B** | 3T lowres → 64mT | field strength |

- Left/right labels summed into one region; no ICV normalisation.
- 64mT subjects with several sessions: bias per session, then averaged per subject.
- Zero volumes are kept and flagged (segmentation failure, not missing data).

```{code-cell} ipython3
import numpy as np
import pandas as pd
from scipy import stats

from common import OLD_OUT

SEG = OLD_OUT / 'seg'
OUT = OLD_OUT / 'volume'
OUT.mkdir(parents=True, exist_ok=True)
```

## Bias per subject

SynthSeg reads `volumes_wide.csv` (Chapter 03); WMH-SynthSeg reads `volumes_main.csv` (Chapter 04).
The two tools name their columns differently; the tables below map both onto the same regions.

```{code-cell} ipython3
# region -> ((SynthSeg left, right), (WMH-SynthSeg left, right)); None = label absent
PAIRED = {
    'cerebral white matter':    (('left cerebral white matter', 'right cerebral white matter'),
                                 ('left-white-matter(2)', 'right-white-matter(41)')),
    'cerebral cortex':          (('left cerebral cortex', 'right cerebral cortex'),
                                 ('left-cortex(3)', 'right-cortex(42)')),
    'lateral ventricle':        (('left lateral ventricle', 'right lateral ventricle'),
                                 ('left-lateral-ventricle(4)', 'right-lateral-ventricle(43)')),
    'inferior lateral ventricle': (('left inferior lateral ventricle', 'right inferior lateral ventricle'),
                                 None),
    'cerebellum white matter':  (('left cerebellum white matter', 'right cerebellum white matter'),
                                 ('left-cerebellum-white-matter(7)', 'right-cerebellum-white-matter(46)')),
    'cerebellum cortex':        (('left cerebellum cortex', 'right cerebellum cortex'),
                                 ('left-cerebellum-cortex(8)', 'right-cerebellum-cortex(47)')),
    'thalamus':                 (('left thalamus', 'right thalamus'),
                                 ('left-thalamus(10)', 'right-thalamus(49)')),
    'caudate':                  (('left caudate', 'right caudate'),
                                 ('left-caudate(11)', 'right-caudate(50)')),
    'putamen':                  (('left putamen', 'right putamen'),
                                 ('left-putamen(12)', 'right-putamen(51)')),
    'pallidum':                 (('left pallidum', 'right pallidum'),
                                 ('left-pallidum(13)', 'right-pallidum(52)')),
    'hippocampus':              (('left hippocampus', 'right hippocampus'),
                                 ('left-hippocampus(17)', 'right-hippocampus(53)')),
    'amygdala':                 (('left amygdala', 'right amygdala'),
                                 ('left-amygdala(18)', 'right-amygdala(54)')),
    'accumbens area':           (('left accumbens area', 'right accumbens area'),
                                 ('left-accumbens(26)', 'right-accumbens(58)')),
    'ventral DC':               (('left ventral DC', 'right ventral DC'),
                                 ('left-ventral-DC(28)', 'right-ventral-DC(60)')),
}
# region -> (SynthSeg column, WMH-SynthSeg column)
SINGLE = {
    '3rd ventricle':      ('3rd ventricle',      '3rd-ventricle(14)'),
    '4th ventricle':      ('4th ventricle',      '4th-ventricle(15)'),
    'brain-stem':         ('brain-stem',         'brainstem(16)'),
    'csf':                ('csf',                'extracerebral_CSF(24)'),
    'total intracranial': ('total intracranial', 'Intracranial-volume'),
    'WMH':                (None,                 'WMH(77)'),
    'optic chiasm':       (None,                 'optic-chiasm(85)'),
}

META = ['subject', 'field_strength', 'acq', 'session', 'run', 'modality']


def load(csv, tool):
    """Read a volume table -> META + merged region columns."""
    df = pd.read_csv(csv, keep_default_na=False)
    idx = 0 if tool == 'SynthSeg' else 1
    df = df.rename(columns={'sub': 'subject', 'fs': 'field_strength', 'ses': 'session', 'mod': 'modality'})
    for c in META:   # unify 'not applicable' ('-', '', 'nan' -> 'n/a')
        df[c] = df[c].astype(str).str.strip().replace({'-': 'n/a', '': 'n/a', 'nan': 'n/a'})

    num = lambda c: pd.to_numeric(df[c], errors='coerce')
    out = df[META].copy()
    for region, spec in PAIRED.items():
        if spec[idx]:
            out[region] = num(spec[idx][0]) + num(spec[idx][1])
    for region, spec in SINGLE.items():
        if spec[idx]:
            out[region] = num(spec[idx])
    return out


def pair_bias(df, tool):
    """Per-subject bias for groups A and B (long format)."""
    regs = [c for c in df.columns if c not in META]
    rows = []
    for modality in sorted(df.modality.unique()):
        m = df[df.modality == modality]
        hi  = m[(m.field_strength == '3T') & (m.acq == 'highres')]
        lo  = m[(m.field_strength == '3T') & (m.acq == 'lowres')]
        low = m[m.field_strength == '64mT']

        for subject in sorted(m.subject.unique()):
            h, l, s = hi[hi.subject == subject], lo[lo.subject == subject], low[low.subject == subject]

            # Group A: 3T highres (ref) -> 3T lowres
            if len(h) == 1 and len(l) == 1:
                for r in regs:
                    ref, tgt = h.iloc[0][r], l.iloc[0][r]
                    rows.append(dict(tool=tool, comparison='A_3Thigh_to_3Tlow',
                                     modality=modality, subject=subject, region=r,
                                     ref_mm3=ref, target_mm3=tgt, diff_mm3=tgt - ref,
                                     pct_diff=np.nan if ref == 0 else (tgt - ref) / ref * 100,
                                     n_sessions=1, has_zero=bool(ref == 0 or tgt == 0)))

            # Group B: 3T lowres (ref) -> 64mT; average over sessions
            if len(l) == 1 and len(s) >= 1:
                for r in regs:
                    ref = l.iloc[0][r]
                    tg  = s[r].to_numpy(dtype=float)
                    pct = np.full(len(tg), np.nan) if ref == 0 else (tg - ref) / ref * 100
                    rows.append(dict(tool=tool, comparison='B_3Tlow_to_64mT',
                                     modality=modality, subject=subject, region=r,
                                     ref_mm3=ref, target_mm3=float(np.mean(tg)),
                                     diff_mm3=float(np.mean(tg - ref)),
                                     pct_diff=np.nan if np.all(np.isnan(pct)) else float(np.mean(pct)),
                                     n_sessions=len(tg), has_zero=bool(ref == 0 or (tg == 0).any())))
    return pd.DataFrame(rows)


S = load(SEG / 'SynthSeg' / 'volumes_wide.csv', 'SynthSeg')
W = load(SEG / 'WMH_SynthSeg' / 'volumes_main.csv', 'WMH_SynthSeg')
long = pd.concat([pair_bias(S, 'SynthSeg'), pair_bias(W, 'WMH_SynthSeg')], ignore_index=True)
long.to_csv(OUT / 'bias_long.csv', index=False)
print(f"bias_long.csv {len(long)} rows")
```

## Summary per region and save

Per tool × comparison × modality × region: Bland–Altman mean and 95% limits of agreement, ICC(2,1)
(absolute agreement), one-sample t-test against zero, Benjamini–Hochberg FDR within each
tool × comparison × modality.

Grade:

| Grade | Rule |
|---|---|
| usable | \|bias\| < 5% and ICC ≥ 0.90 |
| correctable | \|bias\| ≥ 5% and ICC ≥ 0.90 (systematic scaling; subject ranking preserved) |
| unusable | ICC < 0.90 |

```{code-cell} ipython3
BIAS_TOL, ICC_TOL = 5.0, 0.90


def icc21(ref, tgt):
    """ICC(2,1), absolute agreement, two measurements."""
    Y = np.column_stack([ref, tgt]).astype(float)
    Y = Y[~np.isnan(Y).any(axis=1)]
    n, k = len(Y), 2
    if n < 3:
        return np.nan
    grand = Y.mean()
    MSR = k * ((Y.mean(axis=1) - grand) ** 2).sum() / (n - 1)          # between subjects
    MSC = n * ((Y.mean(axis=0) - grand) ** 2).sum() / (k - 1)          # between measurements
    MSE = ((Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0) + grand) ** 2).sum() / ((n - 1) * (k - 1))
    den = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    return np.nan if den == 0 else (MSR - MSE) / den


def summarize(long):
    """Bland-Altman stats, ICC, one-sample t-test, BH-FDR per tool x comparison x modality."""
    rows = []
    keys = ['tool', 'comparison', 'modality', 'region']
    for k, g in long.groupby(keys, sort=False):
        pct = g.pct_diff.dropna().to_numpy()
        n = len(pct)
        mean = pct.mean() if n else np.nan
        sd = pct.std(ddof=1) if n > 1 else np.nan
        t, p = stats.ttest_1samp(pct, 0) if (n > 1 and sd > 0) else (np.nan, np.nan)
        rows.append(dict(zip(keys, k)) | dict(
            n_subjects=n, n_with_zero=int(g.has_zero.sum()),
            mean_pct_diff=mean, sd_pct_diff=sd,
            loa_lower=mean - 1.96 * sd, loa_upper=mean + 1.96 * sd,
            mean_diff_mm3=g.diff_mm3.mean(), ref_mean_mm3=g.ref_mm3.mean(),
            icc=icc21(g.ref_mm3, g.target_mm3), t_stat=t, p_value=p))
    out = pd.DataFrame(rows)

    # Benjamini-Hochberg within each tool x comparison x modality
    out['p_fdr'] = np.nan
    for _, idx in out.groupby(['tool', 'comparison', 'modality']).groups.items():
        p = out.loc[idx, 'p_value'].dropna().sort_values()
        adj = p.to_numpy() * len(p) / np.arange(1, len(p) + 1)
        out.loc[p.index, 'p_fdr'] = np.minimum.accumulate(adj[::-1])[::-1].clip(0, 1)
    return out


def grade(r):
    if pd.isna(r.icc):
        return 'n/a'
    if r.icc < ICC_TOL:
        return 'unusable'
    return 'usable' if abs(r.mean_pct_diff) < BIAS_TOL else 'correctable'


summ = summarize(long)
summ['grade'] = summ.apply(grade, axis=1)
summ.to_csv(OUT / 'bias_summary.csv', index=False)
print(f"bias_summary.csv {len(summ)} rows")
```

## The two tools on shared regions

Group B only: bias gap between the tools and whether they give the same grade.

```{code-cell} ipython3
B = summ[summ.comparison == 'B_3Tlow_to_64mT']
cols = ['region', 'modality', 'mean_pct_diff', 'icc', 'grade']
common = (B[B.tool == 'SynthSeg'][cols]
          .merge(B[B.tool == 'WMH_SynthSeg'][cols], on=['region', 'modality'], suffixes=('_syn', '_wmh')))
common['bias_gap'] = (common.mean_pct_diff_syn - common.mean_pct_diff_wmh).abs()
common['same_grade'] = common.grade_syn == common.grade_wmh
common.to_csv(OUT / 'cross_tool_comparison.csv', index=False)
print(f"cross_tool_comparison.csv {len(common)} rows")
```

## New dataset

To be added.
