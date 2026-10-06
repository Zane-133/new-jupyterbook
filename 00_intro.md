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

The book runs on Google Colab from start to end: open `colab_run.ipynb`, choose a GPU runtime and
run all cells. Every chapter saves its results to `My Drive/lfbook/results/` as soon as they are
computed; finished steps are skipped on a re-run.

| Ch | Step | Reads | Writes |
|---|---|---|---|
| 01 | Environment | — | — |
| 02 | Input | `neurodesk_upload.zip` | scan table, pair tables |
| 03 | SynthSeg | scan table | label maps, volume table |
| 04 | WMH-SynthSeg | scan table | label maps, volume table |
| 05 | Registration | pair tables | one `.lta` per pair |
| 06 | Dice | pair tables, `.lta`, label maps | Dice tables |
| 07 | Volume | volume tables | bias and agreement tables |

The old dataset is processed in full. The new dataset is not yet included.
