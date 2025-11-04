
import os, json, pickle, tempfile, time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="BLSTM Translator", page_icon="🌐", layout="centered")

st.title("BLSTM Translator (Seq2Seq + Attention)")

st.markdown(
    "Load your trained artifacts (`best_model.keras`, `src_tokenizer.pkl`, "
    "`tgt_tokenizer.pkl`, `meta.json`) and translate Amharic/Ge'ez/English sentences."
)

# ---------- Sidebar: load artifacts ----------
with st.sidebar:
    st.header("Artifacts")
    artifacts_dir = st.text_input("Artifacts directory", value="artifacts")
    use_uploads = st.checkbox("Upload artifacts instead of directory", value=False)

    model_path = None
    src_tok = None
    tgt_tok = None
    meta = None

    if not use_uploads:
        if st.button("Load from directory"):
            try:
                model_path = os.path.join(artifacts_dir, "best_model.keras")
                with open(os.path.join(artifacts_dir, "src_tokenizer.pkl"), "rb") as f:
                    src_tok = pickle.load(f)
                with open(os.path.join(artifacts_dir, "tgt_tokenizer.pkl"), "rb") as f:
                    tgt_tok = pickle.load(f)
                with open(os.path.join(artifacts_dir, "meta.json"), "r", encoding="utf-8") as f:
                    meta = json.load(f)
                st.success("Loaded artifacts from directory.")
            except Exception as e:
                st.error(f"Failed to load artifacts: {e}")
    else:
        up_model = st.file_uploader("best_model.keras", type=["keras", "h5"])
        up_src = st.file_uploader("src_tokenizer.pkl", type=["pkl"])
        up_tgt = st.file_uploader("tgt_tokenizer.pkl", type=["pkl"])
        up_meta = st.file_uploader("meta.json", type=["json"])
        if st.button("Load uploaded artifacts"):
            try:
                tmpdir = tempfile.mkdtemp()
                model_path = os.path.join(tmpdir, "best_model.keras")
                with open(model_path, "wb") as f:
                    f.write(up_model.read())
                src_tok = pickle.load(up_src)
                tgt_tok = pickle.load(up_tgt)
                meta = json.load(up_meta)
                st.success("Loaded uploaded artifacts.")
            except Exception as e:
                st.error(f"Failed to load uploaded artifacts: {e}")

# Cache model loading
@st.cache_resource(show_spinner=True)
def load_model_cached(path):
    return tf.keras.models.load_model(path, compile=False)

def ensure_artifacts_loaded():
    if not model_path or not src_tok or not tgt_tok or not meta:
        st.stop()

# ---------- Greedy decoding using the full model ----------
def greedy_translate(model, src_text, src_tok, tgt_tok, max_src_len, max_tgt_len):
    # Prepare tokens
    start_id = tgt_tok.word_index.get("<s>")
    end_id = tgt_tok.word_index.get("</s>")
    if start_id is None or end_id is None:
        return "[Error: target tokenizer must contain <s> and </s>]"

    src_seq = src_tok.texts_to_sequences([str(src_text).strip()])[0]
    if not src_seq:
        return ""

    src_seq = pad_sequences([src_seq], maxlen=max_src_len, padding="post")
    tgt_seq = np.zeros((1, max_tgt_len), dtype="int32")
    tgt_seq[0, 0] = start_id

    out_tokens = []
    for t in range(1, max_tgt_len):
        # Predict full sequence probs and read the current step
        probs = model.predict([src_seq, tgt_seq], verbose=0)
        next_id = int(np.argmax(probs[0, t-1]))
        if next_id == end_id:
            break
        out_tokens.append(next_id)
        tgt_seq[0, t] = next_id

    index2word = {i: w for w, i in tgt_tok.word_index.items()}
    index2word[0] = "<pad>"
    words = [index2word.get(i, "<unk>") for i in out_tokens]
    # collapse space before punctuation (basic)
    detok = " ".join(words).replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?").strip()
    return detok

# ---------- Main UI ----------
tab1, tab2 = st.tabs(["🔤 Single Sentence", "📄 Batch Translate (TXT→CSV)"])

with tab1:
    st.subheader("Single Sentence")
    direction = st.text_input("Direction (for reference only)", value="amh → eng")
    src_text = st.text_area("Source text", height=120, placeholder="Type a sentence in the source language...")

    if st.button("Translate"):
        ensure_artifacts_loaded()
        model = load_model_cached(model_path)
        max_src_len = int(meta["max_src_len"])
        max_tgt_len = int(meta["max_tgt_len"])
        with st.spinner("Translating..."):
            pred = greedy_translate(model, src_text, src_tok, tgt_tok, max_src_len, max_tgt_len)
        st.success("Done")
        st.markdown("**Prediction:**")
        st.write(pred)

with tab2:
    st.subheader("Batch Translate")
    st.markdown("Upload a `.txt` file with **one source sentence per line**.")
    txt = st.file_uploader("Upload TXT", type=["txt"], key="txtu")
    if st.button("Run batch translate"):
        ensure_artifacts_loaded()
        model = load_model_cached(model_path)
        max_src_len = int(meta["max_src_len"])
        max_tgt_len = int(meta["max_tgt_len"])
        if not txt:
            st.warning("Please upload a TXT file first.")
        else:
            lines = [ln.strip() for ln in txt.read().decode("utf-8").splitlines() if ln.strip()]
            preds = []
            prog = st.progress(0.0)
            for i, line in enumerate(lines):
                preds.append(greedy_translate(model, line, src_tok, tgt_tok, max_src_len, max_tgt_len))
                if (i+1) % 5 == 0:
                    prog.progress((i+1)/len(lines))
            prog.progress(1.0)
            df = pd.DataFrame({"src": lines, "pred": preds})
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="translations.csv", mime="text/csv")
