"""Paths and command runners shared by all chapters. No analysis code here."""
import json
import os
import subprocess
from collections import deque
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config.yml").read_text())
P = {k: Path(v) for k, v in CFG["paths"].items()}

# Every chapter runs in a new kernel: reload the Neurodesk environment set up in Chapter 01
if P["neurodesk_env"].exists():
    os.environ.update(json.loads(P["neurodesk_env"].read_text()))

SUBJECTS = CFG["subjects"]
RESULTS = P["results"] if SUBJECTS == "all" else P["results"].with_name(P["results"].name + "_test")
DATA, WORK = P["data"], P["work"]
OLD_RAW, NEW_RAW = DATA / "old", DATA / "new"
OLD_OUT, NEW_OUT = RESULTS / "old", RESULTS / "new"
LOGS = RESULTS / "logs"
THREADS = os.cpu_count()

# Same as typing `ml freesurfer/8.0.0` and `export FS_LICENSE=...` in a Neurodesk terminal
FS_INIT = f"module load {CFG['freesurfer_module']} && export FS_LICENSE={P['license']}"

SEG_SUFFIX = {"SynthSeg": "synthseg", "WMH_SynthSeg": "WMHseg"}


def seg_path(out, tool, scan):
    """Label map of one scan (a row of scans.csv):
    <out>/seg/<tool>/<fs>/<sub>/[<ses>/]<scan>_<suffix>.nii.gz"""
    d = out / "seg" / tool / scan["fs"] / scan["sub"]
    if scan["ses"].startswith("ses-"):
        d = d / scan["ses"]
    return d / f"{scan['scan']}_{SEG_SUFFIX[tool]}.nii.gz"


def sh(cmd, log=None, cwd=None, show=True):
    """Run a bash command and append its output to `log`. show=False keeps the output out of
    the notebook (only the last lines are printed if the command fails). Stop on failure."""
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "a") if log else None
    tail = deque(maxlen=30)
    p = subprocess.Popen(["bash", "-lc", cmd], cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        tail.append(line)
        if show:
            print(line, end="")
        if fh:
            fh.write(line)
    p.wait()
    if fh:
        fh.close()
    if p.returncode:
        if not show:
            print("".join(tail), end="")
        raise RuntimeError(f"exit code {p.returncode}: {cmd}")


def fs(cmd, log=None, cwd=None, show=True):
    """Run a command with FreeSurfer loaded."""
    sh(f"{FS_INIT} && {cmd}", log=log, cwd=cwd, show=show)
