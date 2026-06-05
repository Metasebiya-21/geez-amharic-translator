"""Streamlit app: English → French translation using trained artifacts."""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from inference import load_artifacts, translate

ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = ROOT / "en_fr_artifacts"

st.set_page_config(page_title="EN → FR Translator", page_icon="🇫🇷", layout="centered")

st.title("English → French Translator")
st.caption("BiLSTM seq2seq + attention · beam search decoding")


@st.cache_resource(show_spinner="Loading model…")
def _load_cached(artifacts_dir: str):
    return load_artifacts(artifacts_dir)


def _try_load_directory(path: Path) -> dict | None:
    try:
        return _load_cached(str(path.resolve()))
    except Exception as exc:
        st.sidebar.error(f"Failed to load: {exc}")
        return None


with st.sidebar:
    st.header("Model artifacts")
    artifacts_dir = st.text_input(
        "Artifacts folder",
        value=str(DEFAULT_ARTIFACTS),
        help="Folder containing best_model.keras, tokenizers, and meta.json",
    )
    use_upload = st.checkbox("Upload artifacts instead", value=False)

    bundle = None

    if use_upload:
        up_model = st.file_uploader("best_model.keras", type=["keras"])
        up_src = st.file_uploader("src_tokenizer.pkl", type=["pkl"])
        up_tgt = st.file_uploader("tgt_tokenizer.pkl", type=["pkl"])
        up_meta = st.file_uploader("meta.json", type=["json"])
        if st.button("Load uploaded files") and all([up_model, up_src, up_tgt, up_meta]):
            tmp = Path(tempfile.mkdtemp(prefix="en_fr_"))
            (tmp / "best_model.keras").write_bytes(up_model.getvalue())
            (tmp / "src_tokenizer.pkl").write_bytes(up_src.getvalue())
            (tmp / "tgt_tokenizer.pkl").write_bytes(up_tgt.getvalue())
            (tmp / "meta.json").write_text(up_meta.getvalue().decode("utf-8"), encoding="utf-8")
            bundle = _try_load_directory(tmp)
    else:
        path = Path(artifacts_dir).expanduser()
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if path.is_dir():
            bundle = _try_load_directory(path)
        else:
            st.warning(f"Folder not found: {path}")

    decode_method = "beam"
    if bundle:
        meta = bundle["meta"]
        st.success("Model loaded")
        st.write(f"**Path:** `{bundle['artifacts_dir']}`")
        st.write(f"**Max lengths:** src={meta['max_src_len']}, tgt={meta['max_tgt_len']}")
        decode_method = st.selectbox(
            "Decode method",
            options=["beam", "greedy"],
            index=0 if meta.get("decode_method", "beam") == "beam" else 1,
        )

tab1, tab2 = st.tabs(["Single sentence", "Batch (TXT → CSV)"])

with tab1:
    st.subheader("Translate English to French")
    examples = [
        "I wish Tom was here.",
        "How did the audition go?",
        "Take a seat.",
    ]
    example = st.selectbox("Try an example", [""] + examples)
    src_text = st.text_area(
        "English text",
        value=example,
        height=120,
        placeholder="Enter an English sentence…",
    )

    if st.button("Translate", type="primary"):
        if bundle is None:
            st.error("Load model artifacts in the sidebar first.")
        elif not src_text.strip():
            st.warning("Enter some English text.")
        else:
            with st.spinner("Translating…"):
                pred = translate(bundle, src_text, decode=decode_method)
            st.markdown("**French translation:**")
            st.info(pred)

with tab2:
    st.subheader("Batch translate")
    st.markdown("Upload a `.txt` file with **one English sentence per line**.")
    txt_file = st.file_uploader("Upload TXT", type=["txt"], key="batch_txt")

    if st.button("Run batch translate"):
        if bundle is None:
            st.error("Load model artifacts in the sidebar first.")
        elif txt_file is None:
            st.warning("Upload a TXT file first.")
        else:
            lines = [
                ln.strip()
                for ln in txt_file.read().decode("utf-8").splitlines()
                if ln.strip()
            ]
            preds = []
            prog = st.progress(0.0)
            for i, line in enumerate(lines):
                preds.append(translate(bundle, line, decode=decode_method))
                prog.progress((i + 1) / len(lines))
            df = pd.DataFrame({"en": lines, "fr": preds})
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="translations_en_fr.csv",
                mime="text/csv",
            )
