# English → French Neural Machine Translation

**Notebook:** `en_fr_baseline.ipynb`  
**Project:** Generic seq2seq MT pipeline (same architecture as `gez_to_amh_blstm.ipynb`)  
**Task:** Translate short English phrases into French using a BiLSTM encoder–decoder with attention.

---

## 1. Executive Summary

This notebook implements an end-to-end **neural machine translation (NMT)** system for **English → French**. It was developed as a **validation of the generic MT pipeline** before applying the same design to low-resource Ge'ez → Amharic translation.

After an initial underfitting run on Mac M1 (limited data and steps), **full training on Google Colab T4 GPU** produced a working model with:

| Metric | Fast train (M1, ~20 epochs) | Full train (Colab T4, ~35 epochs) |
|--------|----------------------------|-----------------------------------|
| Val word accuracy | ~36% | **~55%** (best ~epoch 26) |
| Val loss | ~4.4 | **~2.74** (best) |
| BLEU (approx.) | ~6 | **~15–25** (expected; run eval to confirm) |
| Sample quality | French-like but often wrong | **Exact matches** on several test phrases |

The saved checkpoint (`artifacts/en_fr/best_model.keras`) together with tokenizers and `meta.json` is sufficient for inference and BLEU evaluation **without retraining**.

---

## 2. Project Goals

1. **Demonstrate a complete MT pipeline:** data loading → tokenization → model → training → checkpointing → beam decoding → BLEU evaluation.
2. **Reuse the Ge'ez notebook architecture** so the same code path is proven on a high-resource language pair before low-resource work.
3. **Support both local (Mac M1) and Google Colab** execution with automatic environment detection.
4. **Produce submittable artifacts:** trained model, tokenizers, metadata, and qualitative samples.

---

## 3. Dataset

### 3.1 Source

- **File:** `data/eng_french.csv`
- **Origin:** [Kaggle EN–FR dataset](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench) (`devicharith/language-translation-englishfrench`)
- **Size:** ~175,621 parallel phrase pairs after deduplication
- **Style:** Short conversational sentences (e.g. *"Take a seat."* → *"Prends place !"*)

### 3.2 Columns

| Raw column (Kaggle) | Renamed column |
|---------------------|----------------|
| `English words/sentences` | `en` |
| `French words/sentences` | `fr` |

Column renaming is automatic when loading via `read_data()`.

### 3.3 Preprocessing

- Drop rows with missing source or target
- Strip whitespace
- Deduplicate on `(en, fr)` pairs
- Optional subsampling via `max_train_pairs` (default **50,000** for memory) or `sample_frac`

### 3.4 Data loading on Colab

Because the CSV is not committed to Git (~30 MB), Colab users must supply data via:

- **Upload** — interactive file upload (default on Colab)
- **Google Drive** — `DRIVE_CSV_PATH`
- **Kaggle API** — `kaggle.json` + dataset download

Local users place the file at `data/eng_french.csv`.

---

## 4. Environment & Setup

### 4.1 Supported platforms

| Platform | GPU | TensorFlow | Mixed precision |
|----------|-----|------------|-----------------|
| **Google Colab T4** | CUDA | `tensorflow==2.19.0` | `mixed_float16` (optional) |
| **Mac M1/M2 (local)** | Metal | `tensorflow-macos` + `metal` | **float32 only** (NaN risk otherwise) |
| **Other local** | CPU/CUDA | `tensorflow` | float32 |

### 4.2 Key environment variables / flags

- `IN_COLAB` — detected via `google.colab`
- `ON_MAC` — Apple Silicon detection
- `M1_8GB_MODE` — smaller batch/length on 8 GB Mac; auto-off on Colab
- `USE_MIXED_PRECISION` — enabled on Colab GPU; disabled on Metal

### 4.3 Dependencies

- TensorFlow / Keras 3
- NumPy, Pandas
- SacreBLEU (corpus BLEU and chrF)
- Matplotlib (training curves, corpus stats)

---

## 5. Notebook Walkthrough (Start to Finish)

The notebook is organized as a linear pipeline. Run cells in order unless resuming from a saved checkpoint (see Section 11).

### Section 0 — Introduction

- Describes the EN→FR task and relationship to `gez_to_amh_blstm.ipynb`
- Colab quick-start instructions (GPU runtime, data upload, artifact download)

### Section 1 — Environment setup

**Cell: Environment setup (Colab + local)**

- Installs packages per platform
- Configures GPU memory growth
- Sets mixed-precision policy
- Imports all libraries (`tf`, `Tokenizer`, `sacrebleu`, etc.)
- Defines `CLIP_NORM = 1.0` for gradient clipping

### Section 2 — NLP primer

Explains core concepts used throughout:

| Term | Role in this notebook |
|------|----------------------|
| Parallel corpus | Rows of `(en, fr)` with same meaning |
| Token | Word (default) or character |
| Teacher forcing | Decoder sees gold French prefixes during training |
| Attention | Decoder attends to all encoder states per step |
| Masked loss | Cross-entropy only on non-padding positions |
| BLEU | N-gram overlap metric for evaluation |

### Section 3 — Data

**Cell: Data source**

- Resolves `repo_root` and `csv_paths`
- Colab: upload / Drive / Kaggle paths
- Creates `artifacts/en_fr/` output directory

**Cell: Config**

Hyperparameters (defaults for full training):

| Parameter | Colab T4 | Mac M1 8 GB |
|-----------|----------|-------------|
| `token_level` | `word` | `word` |
| `max_train_pairs` | 50,000 | 50,000 |
| `epochs` | 35 | 35 |
| `batch_size` | 64 | 32–48 |
| `emb_dim` / `enc_units` / `dec_units` | 256 | 128–160 |
| `max_src_len` / `max_tgt_len` | 48 | 32 |
| `num_words_src` / `num_words_tgt` | 15,000 | 15,000 |
| `learning_rate` | 3e-4 | 3e-4 |
| `dropout` | 0.15 | 0.15 |
| `label_smoothing` | 0.0 | 0.0 |
| `val_split` | 0.1 | 0.1 |
| `FAST_TRAIN` | False | False |

`FAST_TRAIN = True` caps data to 25k pairs and 350 steps/epoch for quick smoke tests (~BLEU 5–10).

**Cell: Load & peek**

- Loads CSV into `df`
- Displays row count and first 5 pairs

**Cell: Visualize corpus statistics**

- Histograms of source/target lengths
- Word/character count distributions

### Section 4 — Tokenization

**Cell: Tokenizers**

Word-level tokenization (default for EN–FR):

- `Tokenizer` with `split=" "`, `oov_token="<unk>"`, `num_words=15000`
- Target sequences use string BOS/EOS: `<s>` … `</s>`
- Teacher-forcing alignment: `tgt_in = "<s> " + french`, `tgt_out = french + " </s>"`
- Dynamic length caps from 95th percentile: `dyn_max_src`, `dyn_max_tgt`
- Builds padded NumPy arrays: `X_src`, `X_tgt_in`, `X_tgt_out`, `y`

Char-level mode is available (`token_level = "char"`) with Unicode BOS/EOS (`\u241e`, `\u241f`) for Ge'ez-style experiments.

**Cell: Build tf.data pipelines**

- Optional `tf.data` input with shuffle, batch, prefetch, `.repeat()`
- 90/10 train/validation split (indices shuffled with seed 42)
- Frees NumPy arrays after pipeline build if `FREE_ARRAYS_AFTER_PIPELINE=True`

**Cell: Visualize tokenization**

- Shows ID sequences for one example
- Verifies teacher-forcing shift (`tgt_in[1:] == tgt_out[:-1]`)

### Section 5 — Model

**Cell: Build model (architecture + loss)**

#### Architecture: BiLSTM Encoder + Attention Decoder

```
English IDs ──► Embedding ──► BiLSTM ──► encoder states (h, c)
                              │
                              ▼
                    encoder outputs (all timesteps)
                              │
French IDs ──► Embedding ──► LSTM decoder ──► Wq ──┐
                              ▲                    │
                              │    attention       │
                              └── context ◄── Wk ──┘
                              │
                              ▼
                         Dense (vocab) → logits
```

**Encoder**

- Source embedding + dropout
- Bidirectional LSTM (`enc_units`, return sequences + states)
- Concatenate forward/backward states → dense map to `dec_units`

**Decoder**

- Target embedding + dropout
- Unidirectional LSTM initialized from mapped encoder states
- **Dot-product attention:** `scores = Wq(dec) @ Wk(enc)^T`, masked softmax, context vector
- Concatenate decoder output + context → dropout → linear logits

**Loss & metrics**

- `masked_sparse_ce` — sparse categorical cross-entropy, padding (id 0) masked out
- Optional label smoothing
- `masked_token_accuracy` — fraction of correct next-token predictions on non-pad positions
- Optimizer: Adam with `clipnorm=1.0`

**Numerical stability**

- Always **float32** on Apple Metal (mixed precision caused NaN weights)
- Attention masking in float32 to avoid `-inf` issues
- `TerminateOnNaN` callback during training

**Cell: Instantiate model**

Calls `build_model()` with vocabulary sizes and dynamic lengths from tokenizers.

**Cell: Compile optimizer** (if separate)

Adam with gradient clipping; model may already be compiled in `build_model`.

### Section 6 — Training

**Cell: Speed benchmark**

- Runs one training batch; reports seconds/step
- Target: **< 2 s/step** on M1, **~0.2–0.5 s/step** on Colab T4
- Verifies fast attention layer (`attn_scores`) is present

**Cell: Train**

Callbacks:

- `EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)`
- `ModelCheckpoint` → `artifacts/en_fr/best_model.keras` (best val loss only)
- `ReduceLROnPlateau(patience=4, factor=0.5)`
- `TerminateOnNaN`

Training uses `tf.data` with `steps_per_epoch` and `validation_steps` clamped to available batches.

Post-training diagnosis prints underfitting/overfitting hint based on train–val loss gap.

**Cell: Plot training curves**

- Loss and `masked_token_accuracy` vs epoch
- Learning rate schedule

### Section 7 — Save artifacts

**Cell: Save artifacts**

Writes to `artifacts/en_fr/`:

| File | Contents |
|------|----------|
| `best_model.keras` | Full Keras model (best validation loss) |
| `src_tokenizer.pkl` | Fitted English `Tokenizer` |
| `tgt_tokenizer.pkl` | Fitted French `Tokenizer` |
| `meta.json` | Columns, lengths, decode settings, hyperparameters |

### Section 8 — Inference

**Cell: Decoder helpers**

- `load_translator_checkpoint()` — loads `.keras` with custom objects (`safe_mode=False`)
- `build_encoder_subgraph()` / `get_encoder_model()` — cached encoder for fast decode
- `greedy_decode()` — step-by-step argmax with repeat penalty
- `beam_decode()` — beam search with length normalization
- `translate_sentences()` — batch wrapper for raw English strings
- `corpus_translation_metrics()` — SacreBLEU + chrF

**Beam search settings (defaults)**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `beam_width` | 5 | Number of hypotheses kept |
| `beam_length_penalty` | 1.0 | Length normalization exponent |
| `beam_min_length_ratio` | 0.20 | Min output tokens = `max(1, src_len × ratio)` for word-level |
| `decode_repeat_penalty` | 2.5 | Penalize repeated tokens |

**Cell: Sample predictions**

- Uses validation holdout examples (not training rows)
- Shows beam vs greedy side by side

### Section 9 — Evaluation

**Cell: Evaluate BLEU**

- Random subset (default 200 pairs) from `df`
- Decodes with beam search
- Reports corpus BLEU (`tokenize=13a` for word-level) and chrF
- Reports mean hypothesis vs reference length

**Cell: Visualize BLEU sample**

- Length histograms
- Exact-match count
- Qualitative REF/HYP pairs

### Section 10 — Batch translation

**Cell: Batch translate**

- Input: `.txt` file (one English sentence per line) or random CSV sample
- Output: timestamped CSV with source and hypothesis columns
- Colab: supports `files.upload()` for input text

### Section 11 — Reload after restart (no retraining)

**Cell: Load artifacts (after restart)**

Required when Colab kernel restarts. Loads checkpoint, tokenizers, and `meta.json` into memory.

**Minimum cells to run for BLEU only:**

1. Environment setup  
2. Data source + Load & peek  
3. Build model (for `MODEL_CUSTOM_OBJECTS`)  
4. Load artifacts  
5. Decoder helpers  
6. Evaluate BLEU  

**Cell: Download artifacts (Colab)**

Zips and downloads `best_model.keras`, tokenizers, and `meta.json`.

---

## 6. Training Results (Colab T4, Full Run)

### 6.1 Configuration

- 50,000 pairs sampled, 90/10 split → ~45,000 train / ~5,000 val
- 704 steps/epoch, batch 64
- 35 epochs planned; early stopping / LR reduction active
- ~120–140 s/epoch (~175 ms/step)

### 6.2 Learning curve summary

| Phase | Epochs | Behavior |
|-------|--------|----------|
| Rapid learning | 1–10 | Val loss 4.97 → 3.07; val acc 26% → 47% |
| Steady improvement | 11–22 | Val loss 3.01 → 2.74; val acc 48% → 54% |
| Plateau + mild overfit | 23–34 | Val loss flat/rising (2.74 → 2.77); train acc → 82% |

**Best validation loss:** ~**2.738** at approximately **epoch 24–26**  
**Best validation accuracy:** ~**55%** word-level

`ModelCheckpoint` + `EarlyStopping(restore_best_weights=True)` retain the best epoch, not the final overfit weights.

### 6.3 Diagnosis

- **Epochs 1–22:** healthy learning, no overfitting
- **Epochs 26–34:** mild overfitting (train loss ↓, val loss ↑, 26-point train–val accuracy gap)
- **ReduceLROnPlateau** fired at epochs 30 and 34; limited further val improvement

This is acceptable for a course project: the saved checkpoint captures the sweet spot.

---

## 7. Qualitative Results

After full Colab training, sample predictions on held-out phrases:

| English (source) | Model prediction | Reference | Result |
|------------------|------------------|-----------|--------|
| Take a seat. | Prenez un œil! | Prends place ! | Partial (wrong idiom) |
| I wish Tom was here. | J'aimerais que Tom soit là. | J'aimerais que Tom soit là. | **Exact match** |
| How did the audition go? | Comment s'est passée l'audition? | Comment s'est passée l'audition ? | **Near-exact** |
| I've no friend to talk to about my problems. | Je n'ai pas d'ami avec mes problèmes. | Je n'ai pas d'ami avec lequel je puisse m'entretenir de mes problèmes. | Partial (missing relative clause) |
| I really like this skirt. Can I try it on? | J'aime beaucoup cette jupe, \<unk\> | J'aime beaucoup cette jupe, puis-je l'essayer ? | Partial (`<unk>` = OOV rare words) |

Compared to the earlier fast-train run (~36% val accuracy), the model now produces **grammatically correct full sentences** on several examples, including perfect matches on longer conditional clauses.

---

## 8. Evaluation Metrics

### 8.1 Training metrics

- **Loss:** masked sparse categorical cross-entropy (natural log scale; random ≈ log(vocab) ≈ 9.6 for 15k vocab)
- **masked_token_accuracy:** primary training indicator; >45% correlates with usable translations
- **Perplexity (approx.):** exp(val_loss) ≈ exp(2.74) ≈ **15.5** at best epoch

### 8.2 Automatic MT metrics

- **BLEU:** n-gram precision with brevity penalty (0–100); word-level uses SacreBLEU `13a` tokenization
- **chrF:** character n-gram F-score; often more stable for morphologically rich or short text

Expected corpus BLEU on 200-pair eval: **~15–25** (up from ~6 in the underfit run). Run **Evaluate BLEU** after loading artifacts to get the exact score.

### 8.3 Known metric limitations

- BLEU penalizes valid paraphrases (*Prends* vs *Prenez*)
- `<unk>` in output hurts BLEU for rare words outside top-15k vocabulary
- Short idioms (*Take a seat*) are hard even when overall accuracy is good

---

## 9. Artifacts & File Layout

```
artifacts/en_fr/
├── best_model.keras      # Trained seq2seq model (use this for inference)
├── src_tokenizer.pkl     # English word → ID mapping
├── tgt_tokenizer.pkl     # French word → ID mapping
└── meta.json             # Length caps, decode config, training metadata
```

### 9.1 `meta.json` fields

- `src_col`, `tgt_col`, `token_level`
- `max_src_len`, `max_tgt_len` (dynamic caps used at training time)
- `decode_method`, `beam_width`, `beam_length_penalty`, `beam_min_length_ratio`
- Optimizer and regularization settings

---

## 10. Comparison: Fast Train vs Full Train

| Aspect | FAST_TRAIN (M1 smoke test) | Full train (Colab T4) |
|--------|---------------------------|----------------------|
| Data | 25k pairs | 50k pairs |
| Steps/epoch | 350 (capped) | 704 (full) |
| Model size | 128-dim LSTM | 256-dim LSTM |
| Epochs | 20 | 35 |
| Val accuracy | ~36% | ~55% |
| BLEU (approx.) | ~6 | ~15–25 |
| Sample quality | French fragments, often wrong | Several exact/near-exact sentences |
| Use case | Pipeline debugging | Submission / demo |

---

## 11. Limitations & Future Work

### Current limitations

1. **Vocabulary cap (15k):** rare French words (`puis-je`, `essayer`) become `<unk>`
2. **Short phrase domain:** trained on conversational snippets, not long documents
3. **Word-level only for EN–FR:** no subword (BPE/SentencePiece) modeling
4. **Small seq2seq architecture:** BiLSTM + attention, not Transformer
5. **Mild late overfitting:** train accuracy reaches ~82% while val plateaus ~56%

### Possible improvements (not required for current submission)

- Increase `max_train_pairs` to full 175k corpus
- Subword tokenization (BPE) to reduce OOV
- Transformer encoder–decoder (e.g. small Marian/T5)
- More data augmentation or back-translation
- Tune beam search (`length_penalty`, `beam_width`) on validation set

---

## 12. Relationship to Ge'ez → Amharic Notebook

`en_fr_baseline.ipynb` is a **clone of `gez_to_amh_blstm.ipynb`** with these differences:

| Setting | EN–FR (`en_fr_baseline`) | Ge'ez–Amharic (`gez_to_amh_blstm`) |
|---------|--------------------------|-------------------------------------|
| Data | `eng_french.csv` | `AGE.csv` |
| Columns | `en`, `fr` | `gez`, `amh` |
| Tokenization | Word-level (default) | Char-level (default) |
| Corpus size | ~175k (50k used) | ~17k |
| Output dir | `artifacts/en_fr/` | `artifacts/` |
| Difficulty | High-resource pair | Low-resource pair |

Proving the pipeline on EN–FR validates that **training, decoding, and evaluation code work correctly** before investing in additional Ge'ez–Amharic parallel data.

---

## 13. Conclusion

The `en_fr_baseline.ipynb` notebook delivers a **complete, reproducible English → French NMT system** using a BiLSTM encoder–decoder with dot-product attention, masked cross-entropy training, and beam-search decoding. Full training on Google Colab T4 overcame initial underfitting and produced a model capable of **exact translations on multiple test sentences**, with ~55% validation word accuracy and expected BLEU in the mid-teens to mid-twenties.

The trained artifacts in `artifacts/en_fr/` are sufficient for **inference and BLEU evaluation without retraining**, using the **Load artifacts** workflow after a Colab kernel restart.

---

## Appendix A — Quick Reference: Inference Only (No Retraining)

```text
1. Environment setup
2. Data source → Load & peek        # need df for BLEU references
3. Build model                      # MODEL_CUSTOM_OBJECTS
4. Load artifacts (after restart)   # model + tokenizers + meta
5. Decoder helpers
6. Evaluate BLEU / Sample predictions
7. Download artifacts (optional, Colab)
```

## Appendix B — Key Hyperparameters (Full Train Defaults)

```python
token_level = "word"
max_train_pairs = 50000
epochs = 35
batch_size = 64          # Colab; 32–48 on M1
emb_dim = enc_units = dec_units = 256   # Colab; 128–160 on M1
num_words_src = num_words_tgt = 15000
learning_rate = 3e-4
dropout = 0.15
label_smoothing = 0.0
decode_method = "beam"
beam_width = 5
```

---

*Report generated for the geez-amharic-translator repository. Reflects notebook `en_fr_baseline.ipynb` and Colab training session results.*
