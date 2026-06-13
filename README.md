---
title: English → French Translator
emoji: 🇫🇷
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
python_version: "3.10"
---

# English → French Neural Machine Translation

BiLSTM encoder–decoder with attention, trained on an English–French parallel corpus.

**Notebooks:** `en_fr_baseline.ipynb` (EN→FR) · `gez_to_amh_blstm.ipynb` (Ge'ez→Amharic)

---

## Model artifacts

Weights are **not** in this Git repo (~400 MB). Host them on **Hugging Face Hub**:

```
your-username/en-fr-translator/
├── best_model.keras
├── src_tokenizer.pkl
├── tgt_tokenizer.pkl
└── meta.json
```

### Upload artifacts to Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login
python scripts/upload_artifacts_to_hf.py YOUR_USERNAME/en-fr-translator
```

---

## Deploy on Hugging Face Spaces (recommended)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - **SDK:** Gradio
   - **Hardware:** CPU Basic (free) is enough for inference
2. Push this repo (or connect GitHub) — entry point is **`app.py`**
3. In Space **Settings → Variables and secrets**, add:

   | Name | Value |
   |------|--------|
   | `HF_MODEL_REPO` | `YOUR_USERNAME/en-fr-translator` |

4. The Space downloads weights from Hub on first run, then caches them.

The YAML block at the top of this README configures the Space automatically when this file is `README.md` in the Space repo.

---

## Run locally (Gradio)

1. Place artifacts in `en_fr_artifacts/`, **or** set `HF_MODEL_REPO=YOUR_USERNAME/en-fr-translator`
2. Install and run:

   ```bash
   pip install -r requirements.txt
   python app.py
   ```

---

## Run locally (Streamlit, optional)

```bash
pip install -r requirements-streamlit.txt
streamlit run main.py
```

---

## Project layout

```
├── app.py               # Hugging Face Space (Gradio)
├── main.py              # Optional Streamlit UI
├── inference.py         # Load artifacts, beam/greedy decode
├── requirements.txt     # HF Space dependencies
├── en_fr_baseline.ipynb
├── en_fr_artifacts/     # Local only — not in git
└── report/
```

---

## Training

See `en_fr_baseline.ipynb` for training on Colab (T4 GPU) and evaluation (corpus BLEU ~44, chrF ~62.5). Export artifacts to `en_fr_artifacts/`, then upload to Hugging Face Hub.
