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

# New Dataset: Do the Findings Replicate?

New dataset, 3T vs 64mT: volume bias, Dice, and the same comparison recomputed on the old
dataset.

The method follows Chapter 04 (left/right summed, no ICV normalisation, Bland–Altman, ICC(2,1),
one-sample t-test + BH-FDR, same grading), with two differences:

- one comparison only: **3T highres → 64mT** (the new dataset has no 3T lowres, so resolution
  and field strength are combined);
- WMH-SynthSeg only.

The old dataset is recomputed the same way, 3T highres → 64mT, so the two datasets are compared
like for like.

```{code-cell} ipython3
import numpy as np
import pandas as pd
from scipy import stats

from common import OLD_OUT, NEW_OUT

NEW_VOL  = NEW_OUT / 'seg' / 'WMH_SynthSeg' / 'volumes_main.csv'
NEW_DICE = NEW_OUT / 'dice' / 'dice_WMH_SynthSeg.csv'
OLD_VOL  = OLD_OUT / 'seg' / 'WMH_SynthSeg' / 'volumes_main.csv'
OLD_DICE = OLD_OUT / 'dice' / 'dice_WMH_SynthSeg_highres.csv'
OUT = NEW_OUT / 'analysis'
OUT.mkdir(parents=True, exist_ok=True)
```

## Regions and helpers

```{code-cell} ipython3
# region -> volume-table columns (left + right); last item is the region name used for Dice
PAIRED = {
    'cerebral white matter':   ('left-white-matter(2)', 'right-white-matter(41)', 'WM'),
    'cerebral cortex':         ('left-cortex(3)', 'right-cortex(42)', 'Cortex'),
    'lateral ventricle':       ('left-lateral-ventricle(4)', 'right-lateral-ventricle(43)', 'Lateral-Ventricle'),
    'cerebellum white matter': ('left-cerebellum-white-matter(7)', 'right-cerebellum-white-matter(46)', 'Cerebellum-WM'),
    'cerebellum cortex':       ('left-cerebellum-cortex(8)', 'right-cerebellum-cortex(47)', 'Cerebellum-Cortex'),
    'thalamus':                ('left-thalamus(10)', 'right-thalamus(49)', 'Thalamus'),
    'caudate':                 ('left-caudate(11)', 'right-caudate(50)', 'Caudate'),
    'putamen':                 ('left-putamen(12)', 'right-putamen(51)', 'Putamen'),
    'pallidum':                ('left-pallidum(13)', 'right-pallidum(52)', 'Pallidum'),
    'hippocampus':             ('left-hippocampus(17)', 'right-hippocampus(53)', 'Hippocampus'),
    'amygdala':                ('left-amygdala(18)', 'right-amygdala(54)', 'Amygdala'),
    'accumbens area':          ('left-accumbens(26)', 'right-accumbens(58)', 'Accumbens'),
    'ventral DC':              ('left-ventral-DC(28)', 'right-ventral-DC(60)', 'VentralDC'),
}
SINGLE = {
    '3rd ventricle':      ('3rd-ventricle(14)', '3rd-Ventricle'),
    '4th ventricle':      ('4th-ventricle(15)', '4th-Ventricle'),
    'brain-stem':         ('brainstem(16)', 'Brainstem'),
    'csf':                ('extracerebral_CSF(24)', 'CSF'),
    'total intracranial': ('Intracranial-volume', None),
    'WMH':                ('WMH(77)', 'WMH'),
    'optic chiasm':       ('optic-chiasm(85)', None),
}
DICE2REG = {v[2]: k for k, v in PAIRED.items()} | {v[1]: k for k, v in SINGLE.items() if v[1]}
REGIONS = list(PAIRED) + list(SINGLE)
MODS = ['T1w', 'T2w', 'FLAIR']


def regional(df):
    out = df[['fs', 'sub', 'mod']].copy()
    for r, (L, R, _) in PAIRED.items():
        out[r] = df[L] + df[R]
    for r, (c, _) in SINGLE.items():
        out[r] = df[c]
    return out


def pair_bias(v, dataset):
    """3T (reference) -> 64mT. 64mT with multiple sessions: bias per session, then averaged."""
    rows = []
    for (sub, mod), g in v.groupby(['sub', 'mod']):
        ref = g[g.fs == '3T']; low = g[g.fs == '64mT']
        if len(ref) != 1 or len(low) == 0:
            continue
        for r in REGIONS:
            a = float(ref.iloc[0][r]); t = low[r].to_numpy(float)
            pct = np.full(len(t), np.nan) if a == 0 else (t - a) / a * 100
            rows.append(dict(dataset=dataset, subject=sub, modality=mod, region=r,
                             ref_mm3=a, target_mm3=t.mean(), diff_mm3=(t - a).mean(),
                             pct_diff=np.nanmean(pct) if not np.isnan(pct).all() else np.nan,
                             n_sessions=len(t)))
    return pd.DataFrame(rows)


def add_icv_adjusted(long):
    """Supplementary: divide 64mT volumes by the ICV ratio (64mT/3T) of the same subject and
    modality, removing global geometric scaling."""
    icv = long[long.region == 'total intracranial'].set_index(['dataset', 'subject', 'modality'])
    s = (icv.target_mm3 / icv.ref_mm3).rename('icv_ratio')
    long = long.join(s, on=['dataset', 'subject', 'modality'])
    adj = long.target_mm3 / long.icv_ratio
    long['target_adj_mm3'] = adj
    long['pct_diff_adj'] = np.where(long.ref_mm3 == 0, np.nan, (adj - long.ref_mm3) / long.ref_mm3 * 100)
    return long


def icc31(ref, tgt):
    """ICC(3,1) consistency: ignores a global offset; only subject ordering/proportion matters."""
    Y = np.column_stack([ref, tgt]).astype(float)
    Y = Y[~np.isnan(Y).any(axis=1)]
    n, k = Y.shape
    if n < 3:
        return np.nan
    g = Y.mean()
    MSR = k * ((Y.mean(1) - g) ** 2).sum() / (n - 1)
    MSE = ((Y - Y.mean(1, keepdims=True) - Y.mean(0) + g) ** 2).sum() / ((n - 1) * (k - 1))
    return (MSR - MSE) / (MSR + (k - 1) * MSE)


def icc21(ref, tgt):
    """ICC(2,1) absolute agreement (same as Chapter 04)."""
    Y = np.column_stack([ref, tgt]).astype(float)
    Y = Y[~np.isnan(Y).any(axis=1)]
    n, k = Y.shape
    if n < 3:
        return np.nan
    g = Y.mean()
    MSR = k * ((Y.mean(1) - g) ** 2).sum() / (n - 1)
    MSC = n * ((Y.mean(0) - g) ** 2).sum() / (k - 1)
    MSE = ((Y - Y.mean(1, keepdims=True) - Y.mean(0) + g) ** 2).sum() / ((n - 1) * (k - 1))
    den = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    return np.nan if den == 0 else (MSR - MSE) / den


def bh(p):
    p = pd.Series(p); s = p.dropna().sort_values(); m = len(s)
    adj = (s.to_numpy() * m / np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1].clip(0, 1)
    out = pd.Series(np.nan, index=p.index); out[s.index] = adj
    return out


def grade(bias, icc):
    """Grade as in Chapter 04: usable if |bias|<5% and ICC>=0.90;
    correctable if ICC>=0.90 with larger bias; otherwise unusable."""
    if np.isnan(icc) or icc < 0.90:
        return 'unusable'
    return 'usable' if abs(bias) < 5 else 'correctable'


def summarize(long):
    rows = []
    for (ds, mod, r), g in long.groupby(['dataset', 'modality', 'region'], sort=False):
        pct = g.pct_diff.dropna().to_numpy(); n = len(pct)
        sd = pct.std(ddof=1); mean = pct.mean()
        t, p = stats.ttest_1samp(pct, 0) if sd > 0 else (np.nan, np.nan)
        icc = icc21(g.ref_mm3, g.target_mm3)
        adj = g.pct_diff_adj.dropna()
        rows.append(dict(dataset=ds, modality=mod, region=r, n_subjects=n,
                         mean_pct_diff=mean, sd_pct_diff=sd,
                         loa_lower=mean - 1.96 * sd, loa_upper=mean + 1.96 * sd,
                         ref_mean_mm3=g.ref_mm3.mean(), icc=icc, t_stat=t, p_value=p,
                         icc_consistency=icc31(g.ref_mm3, g.target_mm3),
                         adj_pct_diff=adj.mean(), adj_sd=adj.std(ddof=1),
                         adj_icc=icc21(g.ref_mm3, g.target_adj_mm3)))
    s = pd.DataFrame(rows)
    s['p_fdr'] = np.nan
    for _, idx in s.groupby(['dataset', 'modality']).groups.items():
        s.loc[idx, 'p_fdr'] = bh(s.loc[idx, 'p_value']).to_numpy()
    s['grade'] = [grade(b, i) for b, i in zip(s.mean_pct_diff, s.icc)]
    s['adj_grade'] = [grade(b, i) for b, i in zip(s.adj_pct_diff, s.adj_icc)]
    s.loc[s.region == 'total intracranial', ['adj_pct_diff', 'adj_sd', 'adj_icc', 'adj_grade']] = np.nan
    return s


def dice_summary(d, dataset):
    d = d[d.tool == 'WMH_SynthSeg'] if 'tool' in d else d
    # old dataset has multiple sessions: average per subject first so each subject weighs equally
    per_sub = d.groupby(['subject', 'modality', 'region']).dice_np.mean().reset_index()
    s = per_sub.groupby(['modality', 'region']).dice_np.agg(
        dice_median='median', dice_q1=lambda x: x.quantile(.25),
        dice_q3=lambda x: x.quantile(.75), dice_min='min', n='count').reset_index()
    s['region'] = s.region.map(DICE2REG)
    s.insert(0, 'dataset', dataset)
    return s
```

## Bias, Dice and grade per region

```{code-cell} ipython3
new_v = regional(pd.read_csv(NEW_VOL))
old_raw = pd.read_csv(OLD_VOL)
old_raw = old_raw[(old_raw.fs == '64mT') | (old_raw.acq == 'highres')]   # 3T: highres only
old_v = regional(old_raw)

long = pd.concat([pair_bias(new_v, 'new'), pair_bias(old_v, 'old')], ignore_index=True)
long = add_icv_adjusted(long)
summ = summarize(long)
dice = pd.concat([dice_summary(pd.read_csv(NEW_DICE), 'new'),
                  dice_summary(pd.read_csv(OLD_DICE), 'old')], ignore_index=True)

both = summ.merge(dice, on=['dataset', 'modality', 'region'], how='left')
new = both[both.dataset == 'new'].drop(columns='dataset')

# modality ranking: excludes ICV, optic chiasm (no Dice) and WMH (reported separately)
core = new[~new.region.isin(['total intracranial', 'optic chiasm', 'WMH'])]
rank = core.groupby('modality').agg(
    n_usable=('grade', lambda g: (g == 'usable').sum()),
    n_correctable=('grade', lambda g: (g == 'correctable').sum()),
    n_unusable=('grade', lambda g: (g == 'unusable').sum()),
    median_abs_bias=('mean_pct_diff', lambda x: x.abs().median()),
    median_icc=('icc', 'median'),
    n_usable_adj=('adj_grade', lambda g: (g == 'usable').sum()),
    median_abs_bias_adj=('adj_pct_diff', lambda x: x.abs().median()),
    median_icc_adj=('adj_icc', 'median'),
    median_dice=('dice_median', 'median')).reindex(MODS)

ovn = both.pivot_table(index=['modality', 'region'], columns='dataset',
                       values=['mean_pct_diff', 'icc', 'dice_median', 'adj_pct_diff', 'adj_icc'])
ovn.columns = [f'{a}_{b}' for a, b in ovn.columns]
g = both.pivot_table(index=['modality', 'region'], columns='dataset', values='grade', aggfunc='first')
ovn = ovn.join(g.add_prefix('grade_')).reset_index()

long[long.dataset == 'new'].to_csv(OUT / 'bias_long.csv', index=False)
summ[summ.dataset == 'new'].to_csv(OUT / 'bias_summary.csv', index=False)
dice[dice.dataset == 'new'].to_csv(OUT / 'dice_summary.csv', index=False)
new.to_csv(OUT / 'region_verdict.csv', index=False)
rank.to_csv(OUT / 'sequence_ranking.csv')
ovn.to_csv(OUT / 'old_vs_new.csv', index=False)
print(f'6 tables -> {OUT}')
```

## Do the old-dataset findings hold?

Same comparison for both datasets: WMH-SynthSeg, 3T highres → 64mT.

| Check | Criterion |
|---|---|
| Stability class | p_fdr < 0.05; between-subject SD < 10 percentage points |
| Prior collapse | between-subject CV ratio 64mT / 3T; Levene test |
| Effect size | \|bias\| / SD; n needed to detect a 50% reduction of the bias (α = 0.05, power 0.80) |

```{code-cell} ipython3
Z = 1.959964 + 0.8416212          # z(0.975) + z(0.80): two-sided alpha 0.05, power 0.80


def stability(long):
    rows = []
    for (ds, mod, reg), g in long.groupby(['dataset', 'modality', 'region']):
        pct = g.pct_diff.dropna().to_numpy(); n = len(pct)
        bias, sd = pct.mean(), pct.std(ddof=1)
        t, p = stats.ttest_1samp(pct, 0) if sd > 0 else (np.nan, np.nan)
        sign = max((pct > 0).mean(), (pct < 0).mean())
        rows.append(dict(dataset=ds, modality=mod, region=reg, n=n, bias=bias, sd=sd, p=p,
                         sign_consist=sign, gain=(1 - sd / np.hypot(bias, sd)) * 100))
    R = pd.DataFrame(rows)
    for _, idx in R.groupby(['dataset', 'modality']).groups.items():
        R.loc[idx, 'p_fdr'] = bh(R.loc[idx, 'p']).to_numpy()
    sig, tight = R.p_fdr < 0.05, R.sd < 10
    R['verdict'] = np.select([~sig & tight, sig & tight, sig & ~tight],
                             ['1_no_bias', '2_stable_bias', '3_unstable_bias'], '4_noise')
    R['effect_size'] = R.bias.abs() / R.sd
    R['n_for_50pct'] = np.ceil(Z**2 / (R.effect_size / 2)**2)
    return R


def collapse(v, dataset):
    rows = []
    for mod in MODS:
        a_ = v[(v.fs == '3T') & (v['mod'] == mod)]
        b_ = v[(v.fs == '64mT') & (v['mod'] == mod)].groupby('sub', as_index=False).first()  # first session
        for reg in REGIONS:
            a, b = a_[reg].to_numpy(float), b_[reg].to_numpy(float)
            a, b = a[a > 0], b[b > 0]
            if len(a) < 5 or len(b) < 5:
                continue
            cv3, cv6 = a.std(ddof=1) / a.mean() * 100, b.std(ddof=1) / b.mean() * 100
            _, p = stats.levene(a / a.mean(), b / b.mean())
            rows.append(dict(dataset=dataset, modality=mod, region=reg, cv_3T=cv3, cv_64mT=cv6,
                             ratio=cv6 / cv3, p_levene=p))
    return pd.DataFrame(rows)


long2 = pd.read_csv(OUT / 'bias_long.csv').assign(dataset='new')
long2 = pd.concat([long2, pair_bias(old_v, 'old')], ignore_index=True)
long2 = long2[~long2.region.isin(['optic chiasm'])]

R = stability(long2)
R.to_csv(OUT / 'old_findings_check.csv', index=False)
C = pd.concat([collapse(new_v, 'new'), collapse(old_v, 'old')], ignore_index=True)
C.to_csv(OUT / 'prior_collapse_new_vs_old.csv', index=False)
print(f'old_findings_check.csv, prior_collapse_new_vs_old.csv -> {OUT}')
```
