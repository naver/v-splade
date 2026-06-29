# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
V-SPLADE quickstart — one-image, one-query end-to-end demo.

Loads a V-SPLADE HuggingFace export and shows:

  (1) sparse embedding for an arbitrary image (PyTorch tensor)
  (2) the top-10 activated vocabulary tokens with their weights
  (3) similarity score with one or more text queries

Run:

    python examples/quickstart.py \
        --hf_dir <path/to/v-splade-quality> \
        --image  examples/sample_page.png \
        --queries "annual revenue 2024" "carbon emissions" "vacation policy"
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from vsplade_inference import VSPLADEInference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_dir",  required=True,
                   help="Path to the V-SPLADE HF export directory "
                        "(local dir or HuggingFace Hub repo id, "
                        "e.g. <HF_ORG>/v-splade-quality)")
    p.add_argument("--image",   default="examples/sample_page.png",
                   help="Path to a single page image")
    p.add_argument("--queries", nargs="+",
                   default=["financial summary", "market share", "carbon emissions"],
                   help="Text queries to score against the image")
    p.add_argument("--device",  default="cuda")
    p.add_argument("--topk",    type=int, default=10)
    args = p.parse_args()

    # ── 1) Load model ────────────────────────────────────────────────────────
    print(f"[1/3] Loading V-SPLADE from {args.hf_dir}")
    model = VSPLADEInference.from_pretrained(args.hf_dir, device=args.device)

    # ── 2) Encode image -> sparse embedding ──────────────────────────────────
    print(f"[2/3] Encoding image: {args.image}")
    image = Image.open(args.image)
    doc_vec = model.encode_image(image)
    nnz = int((doc_vec > 0).sum())
    print(f"      sparse vector shape={tuple(doc_vec.shape)}  nnz={nnz}  "
          f"max={float(doc_vec.max()):.3f}")
    print(f"      Top-{args.topk} activated tokens:")
    for tok, w in model.decode_topk(doc_vec, k=args.topk):
        print(f"        {w:7.3f}   {tok!r}")

    # ── 3) Encode queries and score against the image ────────────────────────
    print(f"[3/3] Query-image similarity scores")
    for q in args.queries:
        q_vec = model.encode_query(q)
        score = model.similarity(q_vec, doc_vec)
        # show which query tokens contributed
        contrib = (q_vec.float() * doc_vec.float()).cpu()
        top_w, top_ids = torch.topk(contrib, k=5)
        contribs = ", ".join(
            f"{model.tokenizer.decode([int(i)]).strip()}({float(w):.3f})"
            for i, w in zip(top_ids, top_w) if float(w) > 0
        )
        print(f"        score={score:7.3f}   query={q!r}")
        if contribs:
            print(f"          top matches: {contribs}")


if __name__ == "__main__":
    main()
