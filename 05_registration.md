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

# Registration

Each pair: the 64mT scan (moving) is registered to the 3T scan (fixed) with
`mri_synthmorph register -m rigid`, 6 DOF (3 rotations + 3 translations), on the raw grayscale
images, never on the label maps. One `.lta` per pair, against 3T lowres and 3T highres.

```{code-cell} ipython3
import pandas as pd

from common import OLD_OUT, LOGS, THREADS, fs

REG = OLD_OUT / 'reg'
```

## Register and save

Each `.lta` is written to Drive as soon as it is computed; existing ones are skipped.

```{code-cell} ipython3
for acq in ['lowres', 'highres']:
    pairs = pd.read_csv(REG / f'pairs_{acq}.csv', keep_default_na=False)
    xfm = REG / f'xfm_{acq}'
    xfm.mkdir(parents=True, exist_ok=True)
    for r in pairs.itertuples():
        lta = xfm / f'{r.pair_id}.lta'
        if lta.exists() and lta.stat().st_size > 0:
            continue
        print(r.pair_id)
        fs(f'export OMP_NUM_THREADS={THREADS} && '
           f'mri_synthmorph register -m rigid -t {lta} {r.mov_raw} {r.fix_raw}',
           log=LOGS / f'register_old_{acq}.log', show=False)

    n = sum((xfm / f'{p}.lta').exists() for p in pairs.pair_id)
    assert n == len(pairs), f'{acq}: {n} / {len(pairs)} transforms'
    print(f'{acq}: {n} transforms -> {xfm}')
```

## New dataset

To be added.
