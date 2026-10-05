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

# Aim

## The question

Low-field MRI (LF, 64mT) is cheap and portable but lower in image quality than high-field (HF, 3T). Several AI methods claim to close this gap. To check them, we first need to know what the gap looks like without AI.

This book asks two questions:

1. How large is the LF–HF gap in Dice and volume, and which brain structures show it?
2. How much of that gap comes from the lower field, rather than from resolution or the processing pipeline?

A gap that is large, stable, and clearly due to the lower field is a candidate metric: if an enhancement method works, it should shrink.

## Measurement chain

```
HF (3T)  ─┐
          ├─→  segmentation tool  ─→  regional volumes  ─→  HF vs LF gap
LF (64mT)─┘
```

## What is measured

Two quantities, per structure and per modality:

| Quantity | Definition | Captures |
|---|---|---|
| **Volume difference** | $(V_{LF} - V_{HF}) / V_{HF}$, per subject | how far LF volume departs from HF |
| **Dice** | spatial overlap of the two segmentations | whether the structure is found in the same place |

## Roadmap

Chapters 01–04 use the old dataset only; the new dataset enters at 05.

| Ch | Content |
|---|---|
| 01 | The two datasets |
| 02 | Segmentation and choice of tool |
| 03 | Registration and Dice |
| 04 | The LF–HF gap in Dice and volume (WMH-SynthSeg) |
| 05 | New dataset: do Chapter 04's findings replicate? |
| 06 | Selected structures and limitations |
