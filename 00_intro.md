---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# LF vs HF Brain Volume

Low-field (64mT) and high-field (3T) scans of the same subjects are segmented, registered and
compared by Dice and regional volume.

The book runs on Neurodesk (8 CPUs, 32 GB RAM, no GPU) from start to end: `jupyter-book build`
executes the chapters in order. Every chapter saves its results to
`~/neurodesktop-storage/lfbook/results/` as soon as they are computed; finished steps are skipped
on a re-run.

One step is the exception. WMH-SynthSeg needs more memory than this server has and a GPU that
works, so it was run on Google Colab; the script is in Appendix A, and Chapter 04 reads its
results.

| Ch | Step | Reads | Writes |
|---|---|---|---|
| 01 | Environment | — | — |
| 02 | Input | data folder | scan table, pair tables |
| 03 | SynthSeg | scan table | label maps, volume table |
| 04 | WMH-SynthSeg | scan table, Colab results (Appendix A) | — (checks them) |
| 05 | Registration | pair tables | one `.lta` per pair |
| 06 | Dice | pair tables, `.lta`, label maps | Dice tables |
| 07 | Volume | volume tables | bias and agreement tables |

The old dataset is processed in full. The new dataset is not yet included.
