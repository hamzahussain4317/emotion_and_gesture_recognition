# Report and slides

Two Markdown sources:

- [report.md](report.md) — formal project report.
- [slides.md](slides.md) — 13-slide deck for the ~15-minute presentation.

Both include live numbers from the trained models (read directly from `artifacts/metrics/*.json`).

## Produce PDFs — easiest path (no pandoc required)

```bash
pip install markdown
python report/build_html.py
```

This generates [report.html](report.html) and [slides.html](slides.html), with every figure embedded as base64 (one self-contained file each).

Then:

1. Open the `.html` file in **Chrome** or **Edge**.
2. Press **Ctrl+P** → Destination: **Save as PDF** → Save.
3. For slides, set *Layout: Landscape* and *Paper size: A4* (or Letter).

## Produce PDFs — with pandoc (alternative)

If you already have a LaTeX distribution installed:

```bash
pandoc report.md -o report.pdf --pdf-engine=xelatex
pandoc slides.md -o slides.pdf -t beamer -V theme=default
```

## Figures

Both documents reference figures in `../artifacts/figures/`. Run the training pipeline first (`python -m src.run_all`) so every referenced file exists, then rebuild the HTML/PDFs.
