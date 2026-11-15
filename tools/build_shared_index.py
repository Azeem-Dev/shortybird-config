"""Builds ShortyBird's shared story index: titles, synopses and bge-small vectors.

Matches the app exactly: embedding text = "<title>. <synopsis>" cut to 700 UTF-16 units,
192-token cap, [CLS] pooling + L2 norm, and vectorHash = Java hashCode("2|" + text).

usage: python build_shared_index.py <device_story_index.db> <embed.onnx> <tokenizer.json> <out.db>
"""
import sqlite3
import struct
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_VERSION = 2
SKIP_PROVIDERS = {"Stremio", "Dailymotion"}  # movies / noisy uploads, not short dramas


def utf16_take(text: str, n: int) -> str:
    raw = text.encode("utf-16-le")[: n * 2]
    return raw.decode("utf-16-le", errors="ignore")


def java_hash(text: str) -> int:
    h = 0
    raw = text.encode("utf-16-le")
    for i in range(0, len(raw), 2):
        unit = raw[i] | (raw[i + 1] << 8)
        h = (31 * h + unit) & 0xFFFFFFFF
    return h - (1 << 32) if h >= (1 << 31) else h


def main(src, model, tok_path, out):
    rows = sqlite3.connect(src).execute(
        "SELECT url, provider, title, body, thumb FROM story WHERE body IS NOT NULL AND length(body) > 20",
    ).fetchall()
    rows = [r for r in rows if r[1] not in SKIP_PROVIDERS]
    print("stories with synopsis:", len(rows))

    tok = Tokenizer.from_file(tok_path)
    tok.enable_truncation(max_length=192)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    names = {i.name for i in session.get_inputs()}

    texts = [utf16_take(f"{title}. {body or ''}", 700) for (_, _, title, body, _) in rows]
    vectors = []
    for start in range(0, len(texts), 32):
        enc = tok.encode_batch(texts[start:start + 32])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = session.run(None, feed)[0]
        cls = hidden[:, 0, :]
        cls = cls / np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-9)
        vectors.extend(cls.astype(np.float32))
        print(f"\rembedded {min(start + 32, len(texts))}/{len(texts)}", end="")
    print()

    db = sqlite3.connect(out)
    db.executescript(
        """
        DROP TABLE IF EXISTS story; DROP TABLE IF EXISTS story_vec; DROP TABLE IF EXISTS meta;
        CREATE TABLE story(url TEXT PRIMARY KEY, provider TEXT NOT NULL, title TEXT NOT NULL, body TEXT, thumb TEXT);
        CREATE TABLE story_vec(url TEXT PRIMARY KEY, hash INTEGER NOT NULL, vec BLOB NOT NULL);
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
        """,
    )
    for (url, provider, title, body, thumb), text, vec in zip(rows, texts, vectors):
        db.execute("INSERT OR REPLACE INTO story VALUES (?,?,?,?,?)", (url, provider, title, body, thumb))
        blob = struct.pack(f"<{len(vec)}f", *vec.tolist())
        db.execute("INSERT OR REPLACE INTO story_vec VALUES (?,?,?)", (url, java_hash(f"{MODEL_VERSION}|{text}"), blob))
    db.execute("INSERT INTO meta VALUES ('model_version', ?)", (str(MODEL_VERSION),))
    db.execute("INSERT INTO meta VALUES ('count', ?)", (str(len(rows)),))
    db.commit()
    db.execute("VACUUM")
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:5])
