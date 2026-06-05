"""Load EN→FR artifacts and run greedy / beam decoding."""

from __future__ import annotations

import json
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Input,
    Lambda,
    Layer,
    LSTM,
    TimeDistributed,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences


class NoMask(Layer):
    def call(self, x):
        return x

    def compute_mask(self, inputs, mask=None):
        return None


def _apply_src_mask_fp32(scores, src_mask):
    s32 = tf.cast(scores, tf.float32)
    m32 = tf.cast(src_mask, tf.float32)
    m32 = tf.expand_dims(m32, axis=1)
    return s32 + (1.0 - m32) * (-1e4)


def _softmax_fp32(logits, axis=-1):
    return tf.nn.softmax(tf.cast(logits, tf.float32), axis=axis)


def _attn_scores(inputs):
    return tf.matmul(inputs[0], inputs[1], transpose_b=True)


def _not_zero(x):
    return tf.not_equal(x, 0)


def _apply_src_mask(inputs):
    return _apply_src_mask_fp32(inputs[0], inputs[1])


def _context_matmul(inputs):
    return tf.matmul(inputs[0], inputs[1])


def _mask_context(args):
    ctx, qmask = args
    qmask = tf.cast(qmask, ctx.dtype)
    return ctx * tf.expand_dims(qmask, axis=-1)


def build_en_fr_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    *,
    emb_dim: int = 256,
    enc_units: int = 256,
    dec_units: int = 256,
    max_src_len: int = 11,
    max_tgt_len: int = 15,
    dropout: float = 0.15,
    emb_dropout: float = 0.1,
    recurrent_dropout: float = 0.1,
) -> Model:
    src_in = Input(shape=(max_src_len,), name="src_in")
    tgt_in = Input(shape=(max_tgt_len,), name="tgt_in")

    enc_emb = Dropout(emb_dropout, name="src_emb_dropout")(
        Embedding(src_vocab_size, emb_dim, mask_zero=False, name="src_emb")(src_in)
    )
    dec_emb = Dropout(emb_dropout, name="tgt_emb_dropout")(
        Embedding(tgt_vocab_size, emb_dim, mask_zero=False, name="tgt_emb")(tgt_in)
    )

    enc_blstm, f_h, f_c, b_h, b_c = Bidirectional(
        LSTM(
            enc_units,
            return_sequences=True,
            return_state=True,
            implementation=2,
            dropout=0.0,
            recurrent_dropout=recurrent_dropout,
            name="enc_lstm",
        ),
        name="bilstm",
    )(enc_emb)

    state_h = Concatenate(name="enc_state_h")([f_h, b_h])
    state_c = Concatenate(name="enc_state_c")([f_c, b_c])
    state_h = Dense(dec_units, activation="tanh", name="map_h")(state_h)
    state_c = Dense(dec_units, activation="tanh", name="map_c")(state_c)

    dec_out, _, _ = LSTM(
        dec_units,
        return_sequences=True,
        return_state=True,
        implementation=2,
        dropout=0.0,
        recurrent_dropout=recurrent_dropout,
        name="dec_lstm",
    )(dec_emb, initial_state=[state_h, state_c])

    Wq = TimeDistributed(Dense(dec_units, use_bias=False), name="Wq")(dec_out)
    Wk = TimeDistributed(Dense(dec_units, use_bias=False), name="Wk")(enc_blstm)
    e = Lambda(_attn_scores, name="attn_scores")([Wq, Wk])

    tgt_mask = Lambda(_not_zero, name="tgt_mask")(tgt_in)
    src_mask = Lambda(_not_zero, name="src_mask")(src_in)
    e_masked_fp32 = Lambda(_apply_src_mask, name="apply_src_mask")([e, src_mask])
    alphas = Lambda(lambda x: _softmax_fp32(x, axis=-1), name="alphas")(e_masked_fp32)
    context = Lambda(_context_matmul, name="context")([alphas, enc_blstm])
    context = Lambda(_mask_context, name="mask_context")([context, tgt_mask])

    comb = Concatenate(name="attn_concat")([dec_out, context])
    comb = Dropout(dropout, name="dropout")(comb)
    comb = NoMask(name="drop_mask")(comb)
    logits = TimeDistributed(Dense(tgt_vocab_size, dtype="float32"), name="logits")(comb)
    return Model([src_in, tgt_in], logits, name="en_fr_seq2seq")


def _layer_config_map(keras_path: Path) -> dict[str, dict]:
    with zipfile.ZipFile(keras_path) as zf:
        config = json.loads(zf.read("config.json"))
    return {layer["name"]: layer for layer in config["config"]["layers"]}


def _arch_from_keras(keras_path: Path, meta: dict) -> dict[str, Any]:
    layers = _layer_config_map(keras_path)
    src_emb = layers["src_emb"]["config"]
    tgt_emb = layers["tgt_emb"]["config"]
    enc_lstm = layers["bilstm"]["config"]["layer"]["config"]
    dec_lstm = layers["dec_lstm"]["config"]
    return {
        "src_vocab_size": int(src_emb["input_dim"]),
        "tgt_vocab_size": int(tgt_emb["input_dim"]),
        "emb_dim": int(src_emb["output_dim"]),
        "enc_units": int(enc_lstm["units"]),
        "dec_units": int(dec_lstm["units"]),
        "max_src_len": int(meta["max_src_len"]),
        "max_tgt_len": int(meta["max_tgt_len"]),
        "dropout": float(meta.get("dropout", 0.15)),
        "emb_dropout": float(meta.get("emb_dropout", 0.1)),
        "recurrent_dropout": float(meta.get("recurrent_dropout", 0.1)),
    }


def _load_weights_from_keras(model: Model, keras_path: Path) -> None:
    try:
        model.load_weights(str(keras_path))
        return
    except Exception:
        pass
    with zipfile.ZipFile(keras_path) as zf:
        weights_bytes = zf.read("model.weights.h5")
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
        tmp.write(weights_bytes)
        tmp_path = tmp.name
    try:
        model.load_weights(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def load_artifacts(artifacts_dir: str | Path) -> dict[str, Any]:
    art = Path(artifacts_dir)
    keras_path = art / "best_model.keras"
    for name in ("best_model.keras", "src_tokenizer.pkl", "tgt_tokenizer.pkl", "meta.json"):
        if not (art / name).exists():
            raise FileNotFoundError(f"Missing {art / name}")

    with open(art / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open(art / "src_tokenizer.pkl", "rb") as f:
        src_tok = pickle.load(f)
    with open(art / "tgt_tokenizer.pkl", "rb") as f:
        tgt_tok = pickle.load(f)

    arch = _arch_from_keras(keras_path, meta)
    model = build_en_fr_model(**arch)
    _load_weights_from_keras(model, keras_path)
    enc_model = build_encoder_subgraph(model)

    return {
        "model": model,
        "enc_model": enc_model,
        "src_tok": src_tok,
        "tgt_tok": tgt_tok,
        "meta": meta,
        "artifacts_dir": str(art.resolve()),
    }


def target_bos_eos_ids(tgt_tok):
    if getattr(tgt_tok, "char_level", False):
        bos = tgt_tok.word_index.get("\u241e")
        eos = tgt_tok.word_index.get("\u241f")
    else:
        bos = tgt_tok.word_index.get("<s>")
        eos = tgt_tok.word_index.get("</s>")
    return bos, eos


def format_word_translation(text: str) -> str:
    text = re.sub(r"\s+([.,!?;:»”'%\)])", r"\1", text)
    text = re.sub(r"([(«“\"'])\s+", r"\1", text)
    return text.strip()


def _ids_to_tgt_text(tgt_tok, token_ids) -> str:
    index2word = {i: w for w, i in tgt_tok.word_index.items()}
    words = [index2word.get(i, "<unk>") for i in token_ids if i != 0]
    if getattr(tgt_tok, "char_level", False):
        return "".join(words)
    return format_word_translation(" ".join(words))


def _infer_src_shape(model):
    input_spec = model.input_shape
    if isinstance(input_spec, list):
        for spec in input_spec:
            if spec is not None:
                return spec[1:]
    if input_spec is None:
        raise ValueError("Model input shape is None.")
    return input_spec[1:]


def build_encoder_subgraph(model):
    src_shape = _infer_src_shape(model)
    src_input = tf.keras.Input(shape=src_shape, dtype="int32", name="enc_src_input")
    enc_emb = model.get_layer("src_emb")(src_input)
    enc_blstm, f_h, f_c, b_h, b_c = model.get_layer("bilstm")(enc_emb)
    state_h = model.get_layer("enc_state_h")([f_h, b_h])
    state_c = model.get_layer("enc_state_c")([f_c, b_c])
    map_h = model.get_layer("map_h")(state_h)
    map_c = model.get_layer("map_c")(state_c)
    return tf.keras.Model(src_input, [enc_blstm, map_h, map_c], name="encoder_subgraph")


def _decode_argmax(logits_1d, forbidden_ids, penalty_ids=None, penalty_amount=0.0):
    logits = np.asarray(logits_1d, dtype=np.float32).reshape(-1).copy()
    for i in forbidden_ids:
        if i is not None and 0 <= i < logits.shape[0]:
            logits[i] = -1e9
    if penalty_ids and penalty_amount > 0:
        for i in penalty_ids:
            if i is not None and 0 <= i < logits.shape[0]:
                logits[i] -= penalty_amount
    return int(np.argmax(logits))


def _get_decode_layers(model):
    return {
        "dec_emb": model.get_layer("tgt_emb"),
        "dec_lstm": model.get_layer("dec_lstm"),
        "Wq": model.get_layer("Wq"),
        "Wk": model.get_layer("Wk"),
        "attn_scores": model.get_layer("attn_scores"),
        "apply_mask": model.get_layer("apply_src_mask"),
        "alphas": model.get_layer("alphas"),
        "context": model.get_layer("context"),
        "attn_concat": model.get_layer("attn_concat"),
        "dropout": model.get_layer("dropout"),
        "logits_td": model.get_layer("logits"),
    }


def _decoder_step(layers, y, h, c, keys, enc_outputs, src_mask):
    y_emb = layers["dec_emb"](y)
    dec_out, h_new, c_new = layers["dec_lstm"](y_emb, initial_state=[h, c])
    queries = layers["Wq"](dec_out)
    scores = layers["attn_scores"]([queries, keys])
    masked_scores = layers["apply_mask"]([scores, src_mask])
    alphas = layers["alphas"](masked_scores)
    context = layers["context"]([alphas, enc_outputs])
    comb = layers["attn_concat"]([dec_out, context])
    comb = layers["dropout"](comb, training=False)
    logits = layers["logits_td"](comb).numpy()[:, 0, :]
    return logits, h_new, c_new


def _log_softmax_np(logits):
    x = np.asarray(logits, dtype=np.float32).reshape(-1)
    x = x - np.max(x)
    return x - np.log(np.sum(np.exp(x)) + 1e-12)


def _length_norm_score(log_prob, length, alpha):
    if alpha is None or alpha <= 0 or length <= 0:
        return log_prob
    return log_prob / (((5.0 + length) ** alpha) / ((5.0 + 1.0) ** alpha))


def _beam_min_output_len(src_token_len, tgt_tok, ratio):
    if getattr(tgt_tok, "char_level", False):
        return max(4, int(src_token_len * ratio))
    return max(1, int(src_token_len * ratio))


def greedy_decode(model, src_seq, src_tok, tgt_tok, max_src_len, max_tgt_len, enc_model, cfg):
    if not src_seq:
        src_seq = [src_tok.word_index.get("<unk>", 0)]
    src_padded = pad_sequences([src_seq], maxlen=max_src_len, padding="post", dtype="int32")
    enc_outputs, h, c = enc_model(src_padded, training=False)
    enc_outputs = tf.convert_to_tensor(enc_outputs)
    keys = model.get_layer("Wk")(enc_outputs)
    src_mask = model.get_layer("src_mask")(tf.convert_to_tensor(src_padded))
    layers = _get_decode_layers(model)
    start_id, end_id = target_bos_eos_ids(tgt_tok)

    repeat_penalty = float(cfg.get("decode_repeat_penalty", 2.5))
    repeat_window = int(cfg.get("decode_repeat_window", 12))
    stop_repeat = int(cfg.get("decode_stop_repeat", 3))

    y = np.array([[start_id]], dtype="int32")
    out_tokens = []
    for _ in range(max_tgt_len):
        logits, h, c = _decoder_step(layers, y, h, c, keys, enc_outputs, src_mask)
        forbidden = {0, start_id}
        pen_ids = list(set(out_tokens[-repeat_window:])) if out_tokens and repeat_window > 0 else []
        next_id = _decode_argmax(logits[0], forbidden, pen_ids, repeat_penalty)
        if next_id in (end_id, 0):
            break
        if stop_repeat > 1 and len(out_tokens) >= stop_repeat:
            if len(set(out_tokens[-stop_repeat:])) == 1:
                break
        out_tokens.append(next_id)
        y = np.array([[next_id]], dtype="int32")
    return _ids_to_tgt_text(tgt_tok, out_tokens)


def beam_decode(model, src_seq, src_tok, tgt_tok, max_src_len, max_tgt_len, enc_model, cfg):
    beam_width = int(cfg.get("beam_width", 5))
    length_penalty = float(cfg.get("beam_length_penalty", 1.0))
    repeat_penalty = float(cfg.get("decode_repeat_penalty", 2.5))
    repeat_window = int(cfg.get("decode_repeat_window", 12))
    min_ratio = float(cfg.get("beam_min_length_ratio", 0.2))

    if not src_seq:
        src_seq = [src_tok.word_index.get("<unk>", 0)]
    min_len = _beam_min_output_len(len(src_seq), tgt_tok, min_ratio)
    src_padded = pad_sequences([src_seq], maxlen=max_src_len, padding="post", dtype="int32")
    enc_outputs, h0, c0 = enc_model(src_padded, training=False)
    enc_outputs = tf.convert_to_tensor(enc_outputs)
    layers = _get_decode_layers(model)
    keys = layers["Wk"](enc_outputs)
    src_mask = model.get_layer("src_mask")(tf.convert_to_tensor(src_padded))
    start_id, end_id = target_bos_eos_ids(tgt_tok)
    forbidden = {0, start_id}

    hyps = [(0.0, [], h0, c0)]
    completed = []
    for _ in range(max_tgt_len):
        if not hyps:
            break
        all_candidates = []
        for score, toks, h_t, c_t in hyps:
            last_id = start_id if not toks else toks[-1]
            y = np.array([[last_id]], dtype="int32")
            logits, h_new, c_new = _decoder_step(layers, y, h_t, c_t, keys, enc_outputs, src_mask)
            log_probs = _log_softmax_np(logits[0])
            for i in forbidden:
                if i is not None and 0 <= i < log_probs.shape[0]:
                    log_probs[i] = -1e9
            if toks and repeat_penalty > 0 and repeat_window > 0:
                for tid in set(toks[-repeat_window:]):
                    if 0 <= tid < log_probs.shape[0]:
                        log_probs[tid] -= repeat_penalty
            top_idx = np.argpartition(log_probs, -beam_width)[-beam_width:]
            for nid in top_idx:
                lp = float(log_probs[nid])
                if nid == end_id:
                    if len(toks) >= min_len:
                        completed.append((score + lp, toks))
                else:
                    all_candidates.append((score + lp, toks + [int(nid)], h_new, c_new))
        if not all_candidates:
            break
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        hyps = all_candidates[:beam_width]

    pool = completed if completed else [(s, t) for s, t, _, _ in hyps]
    _, best_toks = max(pool, key=lambda x: _length_norm_score(x[0], len(x[1]), length_penalty))
    return _ids_to_tgt_text(tgt_tok, best_toks)


def translate(
    bundle: dict[str, Any],
    text: str,
    decode: str | None = None,
) -> str:
    meta = bundle["meta"]
    src_tok = bundle["src_tok"]
    tgt_tok = bundle["tgt_tok"]
    model = bundle["model"]
    enc_model = bundle["enc_model"]
    max_src = int(meta["max_src_len"])
    max_tgt = int(meta["max_tgt_len"])
    decode = decode or meta.get("decode_method", "beam")

    seqs = src_tok.texts_to_sequences([str(text).strip()])
    seq = seqs[0] if seqs and seqs[0] else [src_tok.word_index.get("<unk>", 0)]
    fn = beam_decode if decode == "beam" else greedy_decode
    return fn(model, seq, src_tok, tgt_tok, max_src, max_tgt, enc_model, meta)
