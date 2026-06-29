# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
Thin inference wrapper around the V-SPLADE retriever from `train/models/`.

This file used to declare its own SPLADE head, Li-LSR query encoder, pooling
helper, and special-token mask, duplicating the training code. The unified
architecture now lives in `train/models/` and is reused here:

    model = build_model(hf_dir, mode='inference_only')  # one source of truth

This wrapper only provides the convenience API needed for a single-image /
single-query demo:

    vs = VSPLADEInference.from_pretrained(hf_dir)
    img_vec   = vs.encode_image(pil_image)        # (vocab_size,)
    query_vec = vs.encode_query("some text")      # (vocab_size,)
    score     = vs.similarity(query_vec, img_vec)
    top       = vs.decode_topk(img_vec, k=10)     # [(token_str, weight), ...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor

# Make the unified `train/models/` package importable.
_TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from models import build_model  # noqa: E402


class VSPLADEInference:
    """Convenience wrapper for single-image / single-query inference."""

    def __init__(self, model, processor, tokenizer, device, dtype):
        self.model     = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device    = device
        self.dtype     = dtype
        # Pre-build the Li-LSR vocab lookup table so the first query is fast.
        self.model.query_encoder.to(device, dtype=dtype)
        self.model.query_encoder.build_lookup_table()

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, hf_dir: str | Path,
                        device: str = "cuda",
                        dtype: torch.dtype = torch.bfloat16) -> "VSPLADEInference":
        hf_dir = Path(hf_dir)
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(hf_dir, trust_remote_code=True)
        model = build_model(str(hf_dir), mode="inference_only", dtype=dtype).to(device)
        return cls(model, processor, processor.tokenizer, device, dtype)

    # ── Image side ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode one PIL image into a (vocab_size,) sparse vector."""
        chat = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": ""}]}]
        prompt = self.processor.apply_chat_template(chat, add_generation_prompt=True)
        inputs = self.processor(images=[image.convert("RGB")], text=prompt,
                                return_tensors="pt")
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v
                  for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        return self.model.encode_passage(**inputs)[0]

    # ── Query side ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_query(self, text: str) -> torch.Tensor:
        """Encode a text query via the inference-free Li-LSR lookup."""
        tok = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return self.model.encode_query(
            tok["input_ids"].to(self.device),
            tok["attention_mask"].to(self.device),
        )[0]

    # ── Utilities ───────────────────────────────────────────────────────────

    def decode_topk(self, vec: torch.Tensor, k: int = 10) -> List[Tuple[str, float]]:
        """Return [(token_str, weight)] for the k highest-weighted dimensions."""
        v = vec.float().cpu()
        top_w, top_ids = torch.topk(v, k=k)
        return [(self.tokenizer.decode([int(i)]).strip(), float(w))
                for i, w in zip(top_ids, top_w)]

    @staticmethod
    def similarity(query_vec: torch.Tensor, doc_vec: torch.Tensor) -> float:
        """Sparse dot product between query and document vectors."""
        return float((query_vec.float() * doc_vec.float()).sum())
