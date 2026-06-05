# English → French Neural Machine Translation

BiLSTM encoder–decoder with attention, trained on an English–French parallel corpus. Includes a Streamlit web app for interactive translation.

**Notebooks:** `en_fr_baseline.ipynb` (EN→FR) · `gez_to_amh_blstm.ipynb` (Ge'ez→Amharic)

---

## Model artifacts

```
en_fr_artifacts/
├── best_model.keras
├── src_tokenizer.pkl
├── tgt_tokenizer.pkl
└── meta.json
```


---

## Run locally

1. Clone the repository.
2. Unzip or copy model artifacts into `en_fr_artifacts/`.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Or with [uv](https://github.com/astral-sh/uv):

   ```bash
   uv sync
   ```

4. Start the Streamlit app:

   ```bash
   streamlit run main.py
   ```

5. Open http://localhost:8501 in your browser.

The app loads `en_fr_artifacts/` by default. Use the sidebar to change the artifacts path or upload files manually.

---

## Deploy with Streamlit Cloud

1. Push **code only** to GitHub (`main.py`, `inference.py`, `requirements.txt`, notebooks, report, etc.).
2. Do **not** add model weights to the repo — they are listed in `.gitignore`.
3. Connect the repo at [share.streamlit.io](https://share.streamlit.io) and set **Main file path** to `main.py`.

### Providing model files on Streamlit Cloud

Because weights are not in the repo, use one of these approaches:

| Approach | When to use |
|----------|-------------|
| **Sidebar upload** | Quick demos — upload `best_model.keras`, tokenizers, and `meta.json` via the app (already built in). |
| **Hugging Face Hub** | Best for a permanent public app — host artifacts on HF and download on startup. |
| **Google Drive / GitHub Releases** | Share a download link; load files into a cache directory when the app starts. |

For a course submission, sharing a zip or Drive link alongside the repo is usually enough; reviewers can run locally or use the upload option.

---

## Project layout

```
├── main.py              # Streamlit app
├── inference.py         # Load artifacts, beam/greedy decode
├── requirements.txt
├── en_fr_baseline.ipynb # Training & evaluation (EN→FR)
├── gez_to_amh_blstm.ipynb
├── en_fr_artifacts/     # Local only — not in git
└── report/              # LaTeX report and figures
```

---

## Training

See `en_fr_baseline.ipynb` for data prep, model architecture, training on Colab (T4 GPU), and evaluation (corpus BLEU ~44, chrF ~62.5).

After training, export artifacts to `en_fr_artifacts/` for inference and deployment.
