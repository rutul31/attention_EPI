# EPINTLM Cross-Attention Placement — Presentation

Polished academic slide deck for the EPINTLM placement-ablation work.

## Contents

```
presentation/
├── slides.md              ← the deck (Marp markdown, edit this)
├── generate_figures.py    ← regenerates all matplotlib figures from runs/
├── build.sh               ← renders slides.md → PDF / PPTX / HTML
├── figures/               ← all images embedded in the deck
│   ├── fig_paper_baseline.png   ← Paper Table 2 bar chart
│   ├── fig_architecture.png     ← post_gru / pre_gru / dual diagram
│   ├── fig_results_bars.png     ← our val AUC + AUPR bars
│   ├── fig_relative_lift.png    ← Δ vs baseline
│   ├── fig_training_curves.png  ← convergence comparison
│   ├── heat_post_gru_cross_enh.png
│   ├── heat_pre_gru_cross_enh.png
│   ├── heat_dual_pre_enh.png
│   └── heat_dual_post_enh.png
└── out/
    ├── slides.pdf
    ├── slides.pptx
    └── slides.html
```

## Rebuild

```bash
# Regenerate figures (after changing run data)
conda run -n Bioinfomatrics python presentation/generate_figures.py

# Re-render the deck
cd presentation && bash build.sh
```

## Edit

`slides.md` is plain Marp markdown — edit text, list bullets, tables, or `![h:NN](figures/…)` image directives directly. Run `build.sh` to refresh outputs.

## Slide-by-slide map (20 slides)

| # | Section | Title |
|--:|---|---|
| 1 | Title | Improving Enhancer-Promoter Prediction with Cross-Attention Placement |
| 2 | Introduction | Outline |
| 3 | Introduction | The Problem: EPIs |
| 4 | Introduction | Why It Matters — Applications |
| 5 | State of the art | Current SOTA: EPINTLM |
| 6 | State of the art | Baseline Performance — Paper Table 2 |
| 7 | Motivation | Where the Baseline Can Be Improved |
| 8 | Method | What We Changed: Placement Variants |
| 9 | Method | Why pre-GRU Should Help |
| 10 | Experiments | Experimental Setup |
| 11 | Experiments | Datasets — TargetFinder Statistics |
| 12 | Results | Placement Ablation |
| 13 | Results | Improvement Over Baseline |
| 14 | Results | Training Dynamics |
| 15 | Discussion | Why It Works — Heatmaps (post vs pre) |
| 16 | Discussion | Why It Works — Dual Placement |
| 17 | Discussion | What This Tells Us |
| 18 | Conclusion | Conclusion |
| 19 | Conclusion | Future Work |
| 20 | Acknowledgements | References |

## Key numbers shown

**Paper baselines (Table 2, HeLa-S3):** AUROC 0.970, AUPR 0.865
**Our placement ablation (val on bin-based split):**

| Placement | val AUROC | val AUPR |
|---|---:|---:|
| post_gru (baseline) | 0.7790 | 0.4945 |
| pre_gru | 0.8452 | 0.5031 |
| dual | 0.8111 | 0.5459 |

Lift: pre_gru +0.066 AUROC, dual +0.051 AUPR.
