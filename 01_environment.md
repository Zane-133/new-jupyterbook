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

Install the tools, then check that each one works. Installed tools stay on the Colab disk until
the runtime ends; on a re-run in the same runtime the installs are skipped.

```{code-cell} ipython3
import json
import os
import subprocess
import sys

import torch

from common import P, fs, sh
```

## Install

**Neurodesk** (official Colab setup, several minutes): FreeSurfer 8.0.0 with SynthSeg,
SynthMorph and `mri_vol2vol`. The environment it sets up is saved to a file, which every later
chapter reloads through `common.py`.

```{code-cell} ipython3
if not P['neurodesk_env'].exists():
    setup = '/content/googlecolab_setup.sh'
    subprocess.run(['curl', '-fsSL', 'https://raw.githubusercontent.com/NeuroDesk/neurocommand/main/googlecolab_setup.sh',
                    '-o', setup], check=True)
    with open('/content/neurodesk-setup.log', 'w') as log:
        r = subprocess.run(['bash', setup], env=dict(os.environ, NEURODESK_COLAB_PYTHON=sys.executable),
                           stdout=log, stderr=subprocess.STDOUT)
    if r.returncode:
        print(open('/content/neurodesk-setup.log').read()[-6000:])
        r.check_returncode()
os.environ.update(json.loads(P['neurodesk_env'].read_text()))
print('Neurodesk ready')
```

**Python packages** not included in Colab.

```{code-cell} ipython3
sh(f'{sys.executable} -m pip install -q nibabel')
```

**WMH-SynthSeg**: code from the same GitHub commit as the original runs. Two patches: the model
path (hard-coded to `/app/models`) points to the Drive folder, and `weights_only=False` for
PyTorch ≥ 2.6. The model weights (790 MB) are downloaded once and kept on Drive.

```{code-cell} ipython3
repo, model = P['wmh_repo'], P['wmh_model']
inf = repo / 'WMHSynthSeg' / 'inference.py'
if not repo.exists():
    sh(f'git clone -q https://github.com/lasopablo/freesurfer-freesurfer-dev-mri_WMHsynthseg.git {repo} '
       f'&& git -C {repo} checkout -q 2bf9a42')
    sh(f"""sed -i "s#'/app/models'#'{model.parent}'#" {inf}""")
    sh(f'sed -i "s/torch.load(model_file, map_location=device)/'
       f'torch.load(model_file, map_location=device, weights_only=False)/" {inf}')
if not (model.exists() and model.stat().st_size > 0):
    model.parent.mkdir(parents=True, exist_ok=True)
    sh(f'wget -q -c -O {model} https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/WMH-SynthSeg/{model.name}')
```

## Check

```{code-cell} ipython3
assert torch.cuda.is_available(), 'no GPU: Runtime -> Change runtime type -> GPU'
print('torch', torch.__version__, '|', torch.cuda.get_device_name(0), '|', os.cpu_count(), 'CPUs')

fs('which mri_synthseg mri_synthmorph mri_vol2vol')

sh(f"git -C {repo} log -1 --format='WMH-SynthSeg commit %h  %ad'")
sh(f"grep -n 'model_file = \\|torch.load(' {inf}")
print(f'model {model.name}: {model.stat().st_size / 1e6:.0f} MB')
```
