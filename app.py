"""Hugging Face Space: English → French translation (Gradio)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd

from inference import load_artifacts, resolve_artifacts_dir, translate

ROOT = Path(__file__).resolve().parent
EXAMPLES = [
    ["I wish Tom was here.", "beam"],
    ["How did the audition go?", "beam"],
    ["Take a seat.", "beam"],
]


@lru_cache(maxsize=1)
def _bundle():
    art = resolve_artifacts_dir(
        local_dir=os.environ.get("ARTIFACTS_DIR", str(ROOT / "en_fr_artifacts")),
        hub_repo=os.environ.get("HF_MODEL_REPO"),
    )
    return load_artifacts(art)


def translate_sentence(text: str, decode_method: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return translate(_bundle(), text, decode=decode_method)


def translate_batch(file_obj, decode_method: str):
    if file_obj is None:
        raise gr.Error("Upload a .txt file with one English sentence per line.")

    path = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise gr.Error("The file is empty.")

    preds = [translate(_bundle(), line, decode=decode_method) for line in lines]
    df = pd.DataFrame({"en": lines, "fr": preds})
    out = ROOT / "translations_en_fr.csv"
    df.to_csv(out, index=False)
    return df, str(out)


def build_ui() -> gr.Blocks:
    hub_repo = os.environ.get("HF_MODEL_REPO", "")
    subtitle = (
        f"BiLSTM seq2seq + attention · weights from [`{hub_repo}`](https://huggingface.co/{hub_repo})"
        if hub_repo
        else "BiLSTM seq2seq + attention · beam search decoding"
    )

    with gr.Blocks(title="EN → FR Translator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# English → French Translator")
        gr.Markdown(subtitle)

        with gr.Tab("Single sentence"):
            with gr.Row():
                src = gr.Textbox(
                    label="English",
                    placeholder="Enter an English sentence…",
                    lines=4,
                )
                tgt = gr.Textbox(label="French", lines=4, interactive=False)
            method = gr.Radio(["beam", "greedy"], value="beam", label="Decode method")
            btn = gr.Button("Translate", variant="primary")
            gr.Examples(
                examples=[[ex[0]] for ex in EXAMPLES],
                inputs=src,
                label="Examples",
            )
            btn.click(translate_sentence, inputs=[src, method], outputs=tgt)
            src.submit(translate_sentence, inputs=[src, method], outputs=tgt)

        with gr.Tab("Batch (TXT → CSV)"):
            gr.Markdown("Upload a `.txt` file with **one English sentence per line**.")
            batch_file = gr.File(label="Upload TXT", file_types=[".txt"])
            batch_method = gr.Radio(["beam", "greedy"], value="beam", label="Decode method")
            batch_btn = gr.Button("Run batch translate", variant="primary")
            batch_table = gr.Dataframe(label="Translations", interactive=False)
            batch_download = gr.File(label="Download CSV")
            batch_btn.click(
                translate_batch,
                inputs=[batch_file, batch_method],
                outputs=[batch_table, batch_download],
            )

    return demo


if __name__ == "__main__":
    build_ui().launch()
