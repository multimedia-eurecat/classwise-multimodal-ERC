"""
Extract RoBERTa text embeddings from annotated CSV.

Reads inference.csv, builds contextual utterance strings (context_window=2,
speaker-prefixed), then embeds with all-roberta-large-v1.
"""

import os
from .eval_metrics import phase, check_embeddings
import pickle

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

CSV_PATH = os.environ["EVAL_CSV_PATH"]
OUTPUT_PKL = os.environ["EVAL_TEXT_PKL"]

MODEL_ID = os.environ["EVAL_TEXT_MODEL"]
MAX_LENGTH     = 128
BATCH_SIZE     = 16
CONTEXT_WINDOW = 2
CONTEXT_SEP    = " [SEP] "
DEVICE = os.environ.get("EVAL_DEVICE", "auto")
if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def build_contextual_string(rows: list[dict], target_idx: int,
                             context_window: int, context_sep: str) -> str:
    """
    Builds the encoder input for one utterance, prepending the previous
    `context_window` utterances from the same dialogue.

    Each segment is prefixed with "Speaker{id}: " for speaker identity.
    Example:
      "Speaker0: I can't believe it [SEP] Speaker1: Why? [SEP] Speaker0: This is wrong."
    """
    start = max(0, target_idx - context_window)
    window = rows[start: target_idx + 1]
    parts = []
    for row in window:
        spk  = row["speaker"]
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        parts.append(f"Speaker{spk}: {text}")
    return context_sep.join(parts)


@torch.no_grad()
def embed_batch(texts: list[str], tokenizer, model, device: str) -> np.ndarray:
    """Mean-pool RoBERTa embeddings over the attention mask. Returns (B, 1024)."""
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    last = out.last_hidden_state                          # (B, T, H)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    emb  = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return emb.detach().cpu().float().numpy().astype(np.float32)



def main():
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)

    # Load CSV
    df = pd.read_csv(CSV_PATH, sep=";")
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} utterances from {df['video_file'].nunique()} videos")

    # Assign dialogue_id and utterance_id (same scheme as extract_audio.py)
    video_files  = sorted(df["video_file"].unique())
    video_to_did = {v: i for i, v in enumerate(video_files)}
    df["dialogue_id"]  = df["video_file"].map(video_to_did)
    df["utterance_id"] = df.groupby("dialogue_id").cumcount()

    # Load model
    print(f"\nLoading {MODEL_ID} ...")
    use_fp16  = DEVICE.startswith("cuda") and torch.cuda.is_available()
    dtype     = torch.float16 if use_fp16 else torch.float32
    with phase('model_load'):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        model     = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype).to(DEVICE).eval()
    print(f"Model loaded on {DEVICE}.")

    # Extract embeddings
    embeddings = {}  # {dialogue_id: {utterance_id: np.ndarray}}

    with phase('encoder_inference'):
        for did, group in df.groupby("dialogue_id"):
            group = group.sort_values("utterance_id").reset_index(drop=True)
            rows  = group.to_dict("records")

            # Build contextual strings for this dialogue
            texts = []
            for i, row in enumerate(rows):
                txt = build_contextual_string(rows, i, CONTEXT_WINDOW, CONTEXT_SEP)
                texts.append(txt)

            video_name = rows[0]["video_file"]
            print(f"  [{video_name}]  {len(texts)} utterances", flush=True)

            # Embed in batches
            all_embs = []
            for b_start in range(0, len(texts), BATCH_SIZE):
                batch = texts[b_start: b_start + BATCH_SIZE]
                embs  = embed_batch(batch, tokenizer, model, DEVICE)
                all_embs.append(embs)

            all_embs = np.concatenate(all_embs, axis=0)  # (N, 1024)

            for i, row in enumerate(rows):
                uid = int(row["utterance_id"])
                embeddings.setdefault(int(did), {})[uid] = all_embs[i]

            print(f"    dim={all_embs.shape[1]}")

    check_embeddings(embeddings, 1024)

    # Save
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_dialogues = len(embeddings)
    n_utts = sum(len(v) for v in embeddings.values())
    any_d  = next(iter(embeddings))
    any_u  = next(iter(embeddings[any_d]))
    dim    = embeddings[any_d][any_u].shape[0]
    print(f"\nSaved: {OUTPUT_PKL}")
    print(f"  dialogues={n_dialogues}  utterances={n_utts}  dim={dim}")


if __name__ == "__main__":
    main()
