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

# WMH-SynthSeg

Old dataset, all scans of the scan table, on the GPU. Like SynthSeg, it resamples every input to
1 mm and writes the label map on that grid. No `--crop`: it had an out-of-bounds bug, so the full
field of view is used.

```{code-cell} ipython3
import os
import shutil
import sys

import pandas as pd
import torch

from common import P, OLD_OUT, WORK, LOGS, seg_path, sh

assert torch.cuda.is_available(), 'no GPU: Runtime -> Change runtime type -> GPU'
scans = pd.read_csv(OLD_OUT / 'scans.csv', keep_default_na=False)
WMH = OLD_OUT / 'seg' / 'WMH_SynthSeg'
```

## Segment and save

Scans without a label map on Drive are linked into one staging folder on the Colab disk, and
`inference.py` runs once on it, so the model loads once. Label maps are moved to
`{3T,64mT}/<subject>/[<session>/]` on Drive and their volumes are added to `volumes_main.csv`.
Finished scans are skipped.

```{code-cell} ipython3
def run_wmh(scans, out, log):
    todo = [r for r in scans.to_dict('records') if not seg_path(out, 'WMH_SynthSeg', r).exists()]
    if todo:
        stage, segout = WORK / 'WMH' / 'stage', WORK / 'WMH' / 'segout'
        for d in (stage, segout):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        for r in todo:
            os.symlink(r['src'], stage / f"{r['scan']}.nii.gz")

        # expandable_segments reduces GPU memory fragmentation
        sh(f'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {sys.executable} inference.py '
           f'--i {stage} --o {segout} --csv_vols {segout}/vols.csv --device cuda',
           cwd=P['wmh_repo'] / 'WMHSynthSeg', log=log)

        for r in todo:
            dst = seg_path(out, 'WMH_SynthSeg', r)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(segout / f"{r['scan']}_seg.nii.gz", dst)

        # volume table: scan-table columns + volumes, merged with earlier results
        vols = pd.read_csv(segout / 'vols.csv')
        vols['scan'] = vols.pop('Input-file').map(lambda p: os.path.basename(p)[:-len('_seg.nii.gz')])
        tidy = pd.DataFrame(todo).drop(columns='src').merge(vols, on='scan')
        dst = out / 'seg' / 'WMH_SynthSeg' / 'volumes_main.csv'
        if dst.exists():
            tidy = pd.concat([pd.read_csv(dst, keep_default_na=False), tidy],
                             ignore_index=True).drop_duplicates('scan', keep='last')
        tidy.sort_values(['fs', 'sub', 'mod', 'acq']).to_csv(dst, index=False)


run_wmh(scans, OLD_OUT, LOGS / 'wmh_old.log')
```

## Check

```{code-cell} ipython3
v = pd.read_csv(WMH / 'volumes_main.csv', keep_default_na=False)
missing = set(scans.scan) - set(v.scan)
assert not missing, f'not segmented: {sorted(missing)}'
print(f"WMH-SynthSeg: {len(scans)} scans -> {WMH}/volumes_main.csv")
```

## New dataset

To be added.
