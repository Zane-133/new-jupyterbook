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

# Appendix A: WMH-SynthSeg on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zane-133/new-jupyterbook/blob/main/appendix_wmh_colab.ipynb)

The script that computes the WMH-SynthSeg label maps read by Chapter 04. It does not run as part
of the book (it needs a GPU, see Chapter 04); open it in Colab with the button above.

Before you start: upload `neurodesk_upload.zip` (`old/`, `new/`, `freesurfer/`) to the root of
My Drive, and choose Runtime → Change runtime type → **A100 GPU** (a T4 runs out of memory).
Every label map is saved to Drive as soon as it is computed; after a disconnect, run all cells
again and finished scans are skipped.

At the end the results are zipped to `My Drive/lfbook/WMH_SynthSeg_old.zip`. On Neurodesk,
upload the zip to the home folder and unzip it into the results folder:

```bash
unzip -n ~/WMH_SynthSeg_old.zip -d ~/neurodesktop-storage/lfbook/results
```

## Settings

```{code-cell} ipython3
import os, re, shutil, subprocess, sys, time
from collections import deque
from pathlib import Path

import pandas as pd

ZIP       = Path('/content/drive/MyDrive/neurodesk_upload.zip')   # input data on Drive
DATA      = Path('/content/neurodesk_upload')                     # unzipped on the Colab disk
EXPORT    = Path('/content/drive/MyDrive/lfbook/wmh_export')      # results on Drive (zip root)
WMH       = EXPORT / 'old' / 'seg' / 'WMH_SynthSeg'                # same layout as the book
ZIP_OUT   = Path('/content/drive/MyDrive/lfbook/WMH_SynthSeg_old.zip')
WORK      = Path('/content/work')                                 # staging folders
WMH_REPO  = Path('/content/wmh')                                  # WMH-SynthSeg code
COMMIT    = '2bf9a42'
WMH_MODEL = Path('/content/drive/MyDrive/wmh_model/WMH-SynthSeg_v10_231110.pth')  # kept on Drive
INF       = WMH_REPO / 'WMHSynthSeg' / 'inference.py'


def sh(cmd, log=None, cwd=None):
    """Run a bash command, print its output and append it to `log`. Stop on failure."""
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, 'a') if log else None
    p = subprocess.Popen(['bash', '-lc', cmd], cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='')
        if fh:
            fh.write(line)
    p.wait()
    if fh:
        fh.close()
    if p.returncode:
        raise RuntimeError(f'exit code {p.returncode}: {cmd}')
```

## Drive, GPU and data

```{code-cell} ipython3
from google.colab import drive
drive.mount('/content/drive')
assert ZIP.exists(), f'upload neurodesk_upload.zip to {ZIP.parent}'

import torch
assert torch.cuda.is_available(), 'no GPU: Runtime -> Change runtime type -> A100 GPU'
print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))

if not DATA.exists():
    sh(f'unzip -q {ZIP} -d {DATA.parent}')
```

## WMH-SynthSeg

Code from commit `2bf9a42`. Two patches: the model path (hard-coded to `/app/models`) points to
the Drive folder, and `weights_only=False` for PyTorch ≥ 2.6. The model weights (790 MB) are
downloaded once and kept on Drive.

```{code-cell} ipython3
sh(f'{sys.executable} -m pip install -q nibabel')
if not WMH_REPO.exists():
    sh(f'git clone -q https://github.com/lasopablo/freesurfer-freesurfer-dev-mri_WMHsynthseg.git {WMH_REPO} '
       f'&& git -C {WMH_REPO} checkout -q {COMMIT}')
    sh(f"""sed -i "s#'/app/models'#'{WMH_MODEL.parent}'#" {INF}""")
    sh(f'sed -i "s/torch.load(model_file, map_location=device)/'
       f'torch.load(model_file, map_location=device, weights_only=False)/" {INF}')
if not (WMH_MODEL.exists() and WMH_MODEL.stat().st_size > 0):
    WMH_MODEL.parent.mkdir(parents=True, exist_ok=True)
    sh(f'wget -q -c -O {WMH_MODEL} https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/WMH-SynthSeg/{WMH_MODEL.name}')
sh(f"grep -n 'model_file = \\|torch.load(' {INF}")
```

## Scan table

Same as Chapter 02: 3T `<sub>_acq-{highres,lowres}_<mod>`, 64mT `<sub>_<ses>_run-N_<mod>`,
localizers left out; 66 + 61 = 127 scans.

```{code-cell} ipython3
rows = []
for f in sorted((DATA / 'old').glob('3T/sub-*/anat/*.nii.gz')):
    m = re.fullmatch(r'(sub-\d+)_acq-(highres|lowres)_(T1w|T2w|FLAIR)\.nii\.gz', f.name)
    if m:
        sub, acq, mod = m.groups()
        rows.append(dict(scan=f.name[:-7], fs='3T', sub=sub, ses='n/a', acq=acq, run='-', mod=mod, src=str(f)))
for f in sorted((DATA / 'old').glob('64mT/sub-*/ses-*/anat/*.nii.gz')):
    m = re.fullmatch(r'(sub-\d+)_(ses-\d+)_(run-\d+)_(T1w|T2w|FLAIR)\.nii\.gz', f.name)
    if m:
        sub, ses, run, mod = m.groups()
        rows.append(dict(scan=f.name[:-7], fs='64mT', sub=sub, ses=ses, acq='-', run=run, mod=mod, src=str(f)))
scans = pd.DataFrame(rows)
n = scans.fs.value_counts()
assert (n['3T'], n['64mT']) == (66, 61) and scans.scan.is_unique, f'scan count does not match: {dict(n)}'


def seg_path(r):
    """<WMH>/<fs>/<sub>/[<ses>/]<scan>_WMHseg.nii.gz, as seg_path() in the book's common.py"""
    d = WMH / r['fs'] / r['sub']
    if r['ses'].startswith('ses-'):
        d = d / r['ses']
    return d / f"{r['scan']}_WMHseg.nii.gz"
```

## Segment and save

Scans without a label map on Drive are linked into one staging folder, and `inference.py` runs
once on it, so the model loads once. Label maps are moved to Drive and their volumes added to
`volumes_main.csv`. `inference.py` does not stop when a scan fails; those messages are shown if
any scan is left without a label map.

```{code-cell} ipython3
LOG = WMH / 'logs' / 'main.log'
todo = [r for r in scans.to_dict('records') if not seg_path(r).exists()]
if todo:
    stage, segout = WORK / 'stage', WORK / 'segout'
    for d in (stage, segout):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for r in todo:
        os.symlink(r['src'], stage / f"{r['scan']}.nii.gz")

    # expandable_segments reduces GPU memory fragmentation
    sh(f'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {sys.executable} inference.py '
       f'--i {stage} --o {segout} --csv_vols {segout}/vols.csv --device cuda',
       cwd=INF.parent, log=LOG)

    for r in todo:
        seg = segout / f"{r['scan']}_seg.nii.gz"
        if seg.exists():
            seg_path(r).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(seg, seg_path(r))

    # volume table: scan-table columns + volumes, merged with earlier results
    vols = pd.read_csv(segout / 'vols.csv')
    vols['scan'] = vols.pop('Input-file').map(lambda p: os.path.basename(p)[:-len('_seg.nii.gz')])
    tidy = pd.DataFrame(todo).drop(columns='src').merge(vols, on='scan')
    dst = WMH / 'volumes_main.csv'
    if dst.exists():
        tidy = pd.concat([pd.read_csv(dst, keep_default_na=False), tidy],
                         ignore_index=True).drop_duplicates('scan', keep='last')
    tidy.sort_values(['fs', 'sub', 'mod', 'acq']).to_csv(dst, index=False)

missing = [r['scan'] for r in scans.to_dict('records') if not seg_path(r).exists()]
if missing:
    sh(f"grep 'error occurred' {LOG} | sort | uniq -c || true")
    raise RuntimeError(f'{len(missing)} scans not segmented: {missing}')
print(f'WMH-SynthSeg: {len(scans)} scans done')
```

## Provenance and zip

```{code-cell} ipython3
commit = subprocess.run(['git', '-C', str(WMH_REPO), 'log', '-1', '--format=%h %ad'],
                        capture_output=True, text=True).stdout.strip()
(WMH / 'PROVENANCE.txt').write_text(f"""WMH-SynthSeg results, old dataset ({len(scans)} scans)
Computed on Google Colab, Appendix A of the book; written {time.strftime('%Y-%m-%d %H:%M %Z')}
GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}
Code: lasopablo/freesurfer-freesurfer-dev-mri_WMHsynthseg, commit {commit}
Model: {WMH_MODEL.name} ({WMH_MODEL.stat().st_size / 1e6:.0f} MB)
Settings: --device cuda, no --crop
""")
print((WMH / 'PROVENANCE.txt').read_text())

ZIP_OUT.unlink(missing_ok=True)
sh(f'zip -q -r {ZIP_OUT} old', cwd=EXPORT)
print(f'{ZIP_OUT}: {ZIP_OUT.stat().st_size / 1e6:.0f} MB')
```
