---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  language: python
  name: conda-env-lfbook-py
  display_name: Python [conda env:lfbook]
---

# Volume

Volume bias between 64mT and 3T lowres on the old dataset, per tool, sequence and region:

bias (%) = (64mT − 3T lowres) / 3T lowres × 100

- Same 17 regions as Chapter 05: left/right summed, inferior lateral ventricle counted as
  lateral ventricle, WMH counted as WM (WMH-SynthSeg labels lesions separately; SynthSeg does not).
- Subjects with several 64mT scans: bias per scan, then averaged per subject (n = 11).
- No ICV normalisation. Zero volumes are kept (segmentation failure gives −100%).
- Some scans are cut off at the bottom of the scanned volume; see *Limitation* below.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd

from common import OLD_OUT

SEG = OLD_OUT / 'seg'
OUT = OLD_OUT / 'volume'
OUT.mkdir(parents=True, exist_ok=True)
MODS = ['T2w', 'T1w', 'FLAIR']
```

## Regions

SynthSeg reads `volumes_wide.csv`, WMH-SynthSeg reads `volumes_main.csv` (both from Chapter 03).
The two tools name their columns differently; the table below maps both onto the same regions,
ordered from large to small.

```{code-cell} ipython3
FILES = {'SynthSeg':     SEG / 'SynthSeg' / 'volumes_wide.csv',
         'WMH_SynthSeg': SEG / 'WMH_SynthSeg' / 'volumes_main.csv'}
TOOLS = list(FILES)


def lr(name):
    return [f'left {name}', f'right {name}']


# region -> (SynthSeg columns, WMH-SynthSeg columns); columns are summed
REGIONS = {
    'WM':                (lr('cerebral white matter'),   ['left-white-matter(2)', 'right-white-matter(41)', 'WMH(77)']),
    'Cortex':            (lr('cerebral cortex'),         ['left-cortex(3)', 'right-cortex(42)']),
    'Cerebellum-Cortex': (lr('cerebellum cortex'),       ['left-cerebellum-cortex(8)', 'right-cerebellum-cortex(47)']),
    'Cerebellum-WM':     (lr('cerebellum white matter'), ['left-cerebellum-white-matter(7)', 'right-cerebellum-white-matter(46)']),
    'Brainstem':         (['brain-stem'],                ['brainstem(16)']),
    'Lateral-Ventricle': (lr('lateral ventricle') + lr('inferior lateral ventricle'),
                          ['left-lateral-ventricle(4)', 'right-lateral-ventricle(43)']),
    'CSF':               (['csf'],                       ['extracerebral_CSF(24)']),
    'Thalamus':          (lr('thalamus'),                ['left-thalamus(10)', 'right-thalamus(49)']),
    'Putamen':           (lr('putamen'),                 ['left-putamen(12)', 'right-putamen(51)']),
    'Hippocampus':       (lr('hippocampus'),             ['left-hippocampus(17)', 'right-hippocampus(53)']),
    'Caudate':           (lr('caudate'),                 ['left-caudate(11)', 'right-caudate(50)']),
    'VentralDC':         (lr('ventral DC'),              ['left-ventral-DC(28)', 'right-ventral-DC(60)']),
    'Pallidum':          (lr('pallidum'),                ['left-pallidum(13)', 'right-pallidum(52)']),
    'Amygdala':          (lr('amygdala'),                ['left-amygdala(18)', 'right-amygdala(54)']),
    '3rd-Ventricle':     (['3rd ventricle'],             ['3rd-ventricle(14)']),
    '4th-Ventricle':     (['4th ventricle'],             ['4th-ventricle(15)']),
    'Accumbens':         (lr('accumbens area'),          ['left-accumbens(26)', 'right-accumbens(58)']),
}
```

## Bias per subject

```{code-cell} ipython3
def load(tool):
    """Volume table -> subject, field strength, acq, modality + one column per region (mm³)."""
    df = pd.read_csv(FILES[tool], keep_default_na=False)
    df = df.rename(columns={'sub': 'subject', 'fs': 'field_strength', 'mod': 'modality'})
    i = TOOLS.index(tool)
    vol = pd.DataFrame({r: df[cols[i]].astype(float).sum(axis=1) for r, cols in REGIONS.items()})
    return pd.concat([df[['subject', 'field_strength', 'acq', 'modality']], vol], axis=1)


def bias(vol):
    """Per subject x sequence: bias (%) of each 64mT scan against 3T lowres, averaged over scans."""
    key = ['subject', 'modality']
    ref = vol[vol.acq == 'lowres'].set_index(key)[list(REGIONS)]
    lf = vol[vol.field_strength == '64mT'].set_index(key)[list(REGIONS)]
    ref = ref.reindex(lf.index)
    return ((lf - ref) / ref * 100).groupby(level=key).mean()


vols = {t: load(t) for t in TOOLS}
long = (pd.concat({t: bias(v) for t, v in vols.items()}, names=['tool'])
        .reset_index().melt(id_vars=['tool', 'subject', 'modality'], var_name='region', value_name='pct_diff'))
long.to_csv(OUT / 'bias_long.csv', index=False)
```

## Volume bias result

Two tables, one per tool: 64mT compared with 3T lowres.

**Input of volume bias: 2 tools**

```{code-cell} ipython3
pd.DataFrame([
    dict(tool=t, file=FILES[t].name, scans_3T_lowres=(v.acq == 'lowres').sum(),
         scans_64mT=(v.field_strength == '64mT').sum(), subjects=v.subject.nunique())
    for t, v in vols.items()
]).set_index('tool')
```

```{code-cell} ipython3
def bias_table(tool):
    """Median [Q1, Q3] of volume bias (%) per region and sequence, 11 subjects."""
    q = long[long.tool == tool].groupby(['region', 'modality']).pct_diff.quantile([0.25, 0.5, 0.75]).unstack()
    cell = q.apply(lambda r: f'{r[0.5]:+.0f} [{r[0.25]:+.0f}, {r[0.75]:+.0f}]', axis=1)
    return cell.unstack('modality').loc[list(REGIONS), MODS]


def plot_bias(tool):
    """Box plot per region (one value per subject): line = median, box = Q1-Q3, dots = outliers."""
    d = long[long.tool == tool]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, mod in zip(axes, MODS):
        x = d[d.modality == mod]
        ax.boxplot([x[x.region == r].pct_diff.dropna() for r in REGIONS], widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor='lightgray'), medianprops=dict(color='k'),
                   flierprops=dict(markersize=3))
        ax.axhline(0, color='0.5', lw=0.8)
        ax.set(title=f'{tool} {mod}: 64mT vs 3T lowres', ylabel='Volume bias (%)', ylim=(-105, 105))
        ax.grid(axis='y', alpha=0.3)
    axes[-1].set_xticks(range(1, len(REGIONS) + 1), list(REGIONS), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
```

### SynthSeg

Median [Q1, Q3] of volume bias (%), 11 subjects.

```{code-cell} ipython3
bias_table('SynthSeg')
```

```{code-cell} ipython3
plot_bias('SynthSeg')
```

### WMH-SynthSeg

Median [Q1, Q3] of volume bias (%), 11 subjects.

```{code-cell} ipython3
bias_table('WMH_SynthSeg')
```

```{code-cell} ipython3
plot_bias('WMH_SynthSeg')
```

## Resolution or field strength?

The 3T scans of the old dataset exist at two resolutions. Comparing 3T lowres with 3T highres
(same scanner, same contrast) shows the effect of resolution alone; comparing 64mT with 3T lowres
(similar voxel size) shows mainly the effect of the field strength. WMH-SynthSeg, median volume
bias (%) over 11 subjects; ICV added. The FLAIR highres scans are also 5.5 mm thick, so for FLAIR
only the in-plane resolution differs.

```{code-cell} ipython3
raw = pd.read_csv(FILES['WMH_SynthSeg'], keep_default_na=False)
v = vols['WMH_SynthSeg'].assign(ICV=raw['Intracranial-volume'].astype(float))
cols, key = list(REGIONS) + ['ICV'], ['subject', 'modality']
lo = v[v.acq == 'lowres'].set_index(key)[cols]
hi = v[v.acq == 'highres'].set_index(key)[cols]
lf = v[v.field_strength == '64mT'].set_index(key)[cols]
res = (lo - hi.reindex(lo.index)) / hi.reindex(lo.index) * 100                           # resolution only
fld = ((lf - lo.reindex(lf.index)) / lo.reindex(lf.index) * 100).groupby(level=key).mean()  # field strength
med = pd.concat({'resolution': res.groupby(level='modality').median().T,
                 'field strength': fld.groupby(level='modality').median().T}, axis=1).swaplevel(axis=1)
med[[(m, k) for m in MODS for k in ['resolution', 'field strength']]].loc[cols].round(0)
```

Resolution alone changes most volumes by less than 5 %; the exceptions are the third ventricle,
the cerebellar white matter and the CSF on T2w. The bias of 64mT against 3T lowres is often larger
and of different sign, so the gap measured in this chapter reflects mainly the field strength.
The ICV row also shows that the two scanners image the head at the same size (64mT within a few
percent of 3T lowres).

## Limitation: scans cut off at the bottom

In the old dataset the lower cerebellum and brainstem lie outside the scanned volume on many
scans (Chapter 05, *Scans cut off at the bottom*). A volume is measured in each scan's own
volume, so a structure that is cut off simply loses its missing part:

- **Why it matters**: the volume bias compares 64mT with 3T, so what counts is how the two cuts
  relate.
  - **Cut at different heights**: the strip scanned by only one of the two enters the comparison
    as a difference. The bias is too negative when the 64mT scan is cut higher, too positive when
    the 3T scan is cut higher.
  - **Cut at the same height**: the comparison is fair but incomplete. The missing part is in
    neither volume, so the bias describes only the scanned part of the structure.
- **Regions affected**: cerebellum cortex and white matter, brainstem (above all the medulla), and
  the CSF around them; the 4th ventricle only on the worst scans; the lowest temporal and
  occipital cortex marginally.
- **Not affected**: cortex as a whole, white matter, deep grey matter and the lateral and 3rd
  ventricles, which lie inside the scanned volume on every scan. ICV is hardly affected: the
  missing strip is small next to the whole intracranial volume.

## Conclusion

- **SynthSeg** systematically underestimates deep and lower-brain structures on 64mT scans:
  worst on FLAIR and T1w, still visible on T2w, with segmentation failures (zero volume, −100%).
  SynthSeg is therefore not considered further.
- Volume bias alone cannot tell which regions are reliable enough to serve as a benchmark:
  a small volume bias does not mean the region is segmented in the same place.
  Chapter 07 combines the WMH-SynthSeg volume bias with the Dice of Chapter 05 to answer this.

```{code-cell} ipython3

```
