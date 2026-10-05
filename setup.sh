#!/usr/bin/env bash
# One-time setup on Neurodesk (any server type). Safe to re-run: finished steps are skipped.
#   bash setup.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

# 1. Python environment (on the persistent home disk)
ENV=$HOME/conda-envs/lfbook
CONDA=$(command -v mamba || command -v conda)
[ -x "$ENV/bin/python" ] || "$CONDA" env create -y -q -p "$ENV" -f "$HERE/environment.yml"
# pip packages. torch (with CUDA libraries) is several GB: no pip cache, and unpack on the
# home disk instead of /tmp, so the install does not run out of memory.
TMP=$HOME/.pip-tmp
mkdir -p "$TMP"
"$ENV/bin/python" -c "import torch" 2>/dev/null || TMPDIR=$TMP "$ENV/bin/pip" install --no-cache-dir -q torch
[ -x "$ENV/bin/jupyter-book" ] || TMPDIR=$TMP "$ENV/bin/pip" install --no-cache-dir -q "jupyter-book<2"   # v2 drops _config.yml / _toc.yml
rm -rf "$TMP"

# 2. WMH-SynthSeg code: same repository and commit as the original Colab runs
REPO=$HOME/wmh
if [ ! -d "$REPO" ]; then
  git clone -q https://github.com/lasopablo/freesurfer-freesurfer-dev-mri_WMHsynthseg.git "$REPO"
  git -C "$REPO" checkout -q 2bf9a42
fi
INF=$REPO/WMHSynthSeg/inference.py
# The model path is hard-coded to /app/models; point it to ~/wmh_model
sed -i "s#'/app/models'#'$HOME/wmh_model'#" "$INF"
# Same patch as on Colab: PyTorch>=2.6 defaults torch.load(weights_only=True), which breaks model loading
sed -i "s/torch.load(model_file, map_location=device)/torch.load(model_file, map_location=device, weights_only=False)/" "$INF"

# 3. Model weights
MODEL=$HOME/wmh_model/WMH-SynthSeg_v10_231110.pth
mkdir -p "$(dirname "$MODEL")"
[ -s "$MODEL" ] || wget -q -c -O "$MODEL" https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/WMH-SynthSeg/WMH-SynthSeg_v10_231110.pth

echo "setup done"
grep -n "model_file = \|torch.load(" "$INF"
ls -la "$MODEL"
"$ENV/bin/python" -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
