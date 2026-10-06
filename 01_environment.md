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

# Environment

The book runs on Neurodesk (8 CPUs, 32 GB RAM, no GPU). Nothing is installed here: FreeSurfer
comes from the Neurodesk module, the Python packages from the `lfbook` environment. The cells
below check that the tools and the input data are in place; later chapters assume they are.

```{code-cell} ipython3
import sys
from pathlib import Path

from common import P, OLD_RAW, RESULTS, THREADS, fs

for k, v in P.items():
    print(f'{k:12s} {v}')
print(f"{'results now':12s} {RESULTS}")
```

**Resources.** CPU and memory limits of this Neurodesk session. SynthSeg and SynthMorph use
all CPUs.

```{code-cell} ipython3
mem = Path('/sys/fs/cgroup/memory.max').read_text().strip()
print('python', sys.executable)
print(f'{THREADS} CPUs | memory limit', mem if mem == 'max' else f'{int(mem) / 1e9:.0f} GB')
```

**FreeSurfer.** SynthSeg, SynthMorph and `mri_vol2vol` from the FreeSurfer 8.0.0 module.

```{code-cell} ipython3
assert P['license'].is_file(), f"FreeSurfer licence not found: {P['license']}"
fs('which mri_synthseg mri_synthmorph mri_vol2vol && echo FS_LICENSE=$FS_LICENSE')
```

**Input data.** Old dataset, localizers not counted.

```{code-cell} ipython3
def nii(root):
    return sorted(f for f in root.rglob('*.nii.gz') if 'localizer' not in f.name)


counts = {'old 3T': (len(nii(OLD_RAW / '3T')), 66), 'old 64mT': (len(nii(OLD_RAW / '64mT')), 61)}
for k, (n, expected) in counts.items():
    print(f'{k:9s} {n:3d} scans (expected {expected})')
assert all(n == e for n, e in counts.values()), 'scan count does not match'
```
