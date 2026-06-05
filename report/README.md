# Course project report (LaTeX)

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Full report: abstract, methodology, greedy vs beam, samples, limitations |
| `generate_figures.py` | Optional PNG/PDF plots (training, length histogram, beam comparison) |
| `figures/` | Put Colab screenshots here (see below) |

## Compile PDF

Requires **XeLaTeX** and an Ethiopic font (Noto Sans Ethiopic recommended).

```bash
cd report
xelatex main.tex
xelatex main.tex   # second pass for references
```

Or: `make`

Output: `report/main.pdf`

## Add Colab images

In the notebook, right-click plot outputs → **Save image as…**

Suggested names in `report/figures/`:

- `training_colab.png` — loss / accuracy curves from training
- `bleu_histogram_colab.png` — length distribution + exact match bar chart
- `sample_predictions_colab.png` — optional screenshot of SRC/PRED/TGT prints

Then add to `main.tex` (appendix or replace Figure 2):

```latex
\includegraphics[width=\linewidth]{figures/bleu_histogram_colab.png}
```

The report already includes **TikZ fallbacks** if PNGs are missing.

## Optional: generate figures locally

```bash
pip install matplotlib
python3 report/generate_figures.py
```

This writes `figures/training_curves.pdf`, `kufale_length_distribution.pdf`, etc.

## Customize

Edit title/author in `main.tex`:

```latex
\author{Your Name \\ Course: ... \\ Institution: ...}
```
