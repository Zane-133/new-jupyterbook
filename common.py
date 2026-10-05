"""Paths and command runners shared by all chapters. No analysis code here."""
import os
import subprocess
from collections import deque
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config.yml").read_text())
P = {k: Path(os.path.expanduser(v)) for k, v in CFG["paths"].items()}

DATA, RESULTS = P["data"], P["results"]
OLD_RAW, NEW_RAW = DATA / "old", DATA / "new"
OLD_OUT, NEW_OUT = RESULTS / "old", RESULTS / "new"
LOGS = RESULTS / "logs"
THREADS = CFG["threads"]

# Same as typing `ml freesurfer/8.0.0` and `export FS_LICENSE=...` in a Neurodesk terminal
FS_INIT = f"module load {CFG['freesurfer_module']} && export FS_LICENSE={P['license']}"


def sh(cmd, log=None, cwd=None, show=True):
    """Run a bash command and append its output to `log`. show=False keeps the output out of
    the notebook (only the last lines are printed if the command fails). Stop on failure."""
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "a", buffering=1) if log else None      # line-buffered: the log is always current
    tail = deque(maxlen=30)
    env = os.environ | {"PYTHONUNBUFFERED": "1"}      # Python tools: write output line by line
    p = subprocess.Popen(["bash", "-lc", cmd], cwd=cwd, env=env, stdout=subprocess.PIPE,
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
