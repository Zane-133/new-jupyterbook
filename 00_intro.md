---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python [conda env:lfbook]
  language: python
  name: conda-env-lfbook-py
---

# LF vs HF Brain Volume

Low-field (64mT) and high-field (3T) scans of the same subjects are segmented, registered and
compared by Dice and regional volume.

The book runs on the Neurodesk GPU server (NVIDIA A40) from start to end: `jupyter-book build`
executes the chapters in order. Every chapter saves its results to
`~/neurodesktop-storage/lfbook/results/` as soon as they are computed; finished steps are skipped
on a re-run.

| Ch | Step | Reads | Writes |
|---|---|---|---|
| 01 | Environment | — | — |
| 02 | Input | data folder | scan table, pair tables |
| 03 | Segmentation | scan table | label maps and volume tables (SynthSeg, WMH-SynthSeg) |
| 04 | Registration | pair tables | one `.lta` per pair |
| 05 | Dice | pair tables, `.lta`, label maps | Dice tables |
| 06 | Volume | volume tables | bias and agreement tables |
| 07 | Benchmark | Dice and bias tables | — |

The old dataset is processed in full. The new dataset is not yet included.
