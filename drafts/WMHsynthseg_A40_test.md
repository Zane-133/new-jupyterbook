---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: Python [conda env:lfbook]
  name: conda-env-lfbook-py
  language: python
---

# WMH-SynthSeg on the Neurodesk A40 server

Step-by-step test of WMH-SynthSeg on the Neurodesk GPU server (NVIDIA A40), from the driver up to
the full script, to show where it fails. Run on the A40 server with the lfbook kernel. Test
scans: sub-03 of the new dataset, one 64mT and one 3T T1w. Outputs go to `work/A40_test/`.

```{code-cell} ipython3
import ctypes
import resource
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from common import P, NEW_RAW, WORK, THREADS, sh

OUT = WORK / 'A40_test'
OUT.mkdir(parents=True, exist_ok=True)
INF = P['wmh_code'] / 'WMHSynthSeg' / 'inference.py'
SCANS = {'64mT': NEW_RAW / '64mT/sub-03/anat/sub-03_T1w.nii.gz',
         '3T':   NEW_RAW / '3T/sub-03/anat/sub-03_T1w.nii.gz'}
summary = []
```

## 1. Server

CPUs and memory this container may use, and the GPU as the driver reports it.

```{code-cell} ipython3
mem = Path('/sys/fs/cgroup/memory.max').read_text().strip()
print(f"CPUs: {THREADS} | memory limit: {int(mem) / 2**30:.0f} GB" if mem != 'max' else f'CPUs: {THREADS} | memory: no limit')
sh('nvidia-smi')
summary.append(('1. GPU visible (nvidia-smi)', 'yes'))
```

## 2. CUDA driver

`cuInit` is the first call of every CUDA program (PyTorch included). 0 means success.

```{code-cell} ipython3
cuda = ctypes.CDLL('libcuda.so.1')
version = ctypes.c_int()
cuda.cuDriverGetVersion(ctypes.byref(version))
err = cuda.cuInit(0)
name = ctypes.c_char_p()
cuda.cuGetErrorName(err, ctypes.byref(name))
print(f'CUDA driver API version: {version.value // 1000}.{version.value % 1000 // 10}')
print(f'cuInit(0) = {err} ({name.value.decode()})')
summary.append(('2. CUDA initialisation (cuInit)', f'failed: {err} {name.value.decode()}' if err else 'ok'))
```

## 3. PyTorch

```{code-cell} ipython3
import torch

print('torch', torch.__version__, '| built for CUDA', torch.version.cuda)
ok = torch.cuda.is_available()
print('torch.cuda.is_available():', ok)
summary.append(('3. PyTorch sees the GPU', 'yes' if ok else 'no'))
```

## 4. WMH-SynthSeg code and model

Commit `2bf9a42`, the same as on Colab, with two patches: model path and `weights_only=False`.

```{code-cell} ipython3
sh(f"git -C {P['wmh_code']} log -1 --format='commit %h  %ad'")
sh(f"grep -n 'model_file = \\|torch.load(' {INF}")
print(f"model: {P['wmh_model'].name}, {P['wmh_model'].stat().st_size / 1e6:.0f} MB")
for fs, f in SCANS.items():
    img = nib.load(f)
    print(f"{fs:5s} {f.name}: grid {img.shape[:3]}, voxel {np.round(img.header.get_zooms()[:3], 2)} mm")
summary.append(('4. Code, patches, model', 'ok'))
```

## 5. WMH-SynthSeg on the GPU

The full script with `--device cuda`, as on Colab, on the 64mT scan.

```{code-cell} ipython3
:tags: [raises-exception]

summary.append(('5. WMH-SynthSeg --device cuda', 'failed (see error below)'))
sh(f"{sys.executable} -u inference.py --i {SCANS['64mT']} --o {OUT / 'gpu_64mT_seg.nii.gz'} --device cuda",
   cwd=INF.parent)
```

## 6–7. WMH-SynthSeg on the CPU

The same script with `--device cpu` and all CPUs, first the 64mT scan, then the 3T scan. The 3T
scan takes much longer (about 27 minutes in an earlier run).

```{code-cell} ipython3
def run_cpu(fs):
    out = OUT / f'cpu_{fs}_seg.nii.gz'
    t0 = time.time()
    sh(f"{sys.executable} -u inference.py --i {SCANS[fs]} --o {out} --device cpu --threads {THREADS}",
       cwd=INF.parent)
    mins = (time.time() - t0) / 60
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20   # largest so far, GB
    print(f'\n{fs}: {mins:.1f} min, peak memory {peak:.1f} GB, output {out.name}')
    return mins


summary.append(('6. WMH-SynthSeg --device cpu, 3T', f'ok, {run_cpu("3T"):.1f} min'))
```

```{code-cell} ipython3
summary.append(('7. WMH-SynthSeg --device cpu, 3T', f'ok, {run_cpu("3T"):.1f} min'))
```

## 8. Summary

```{code-cell} ipython3
pd.DataFrame(summary, columns=['step', 'result'])
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
