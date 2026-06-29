# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
Download the public datasets used by V-SPLADE from the HuggingFace Hub.

Three families are needed:

1. **ColPali training set** — 118K (page-image, query) pairs. This is
   the base image-side dataset. The ColPali corpus does *not* ship with
   captions; captions are generated locally.

2. **RLHN 680K text retrieval pairs** — used to mix text-only batches
   into image-side training at a 3:1 text-to-image ratio (the same
   recipe as the BiModernVBERT backbone). 300K rows are sampled at
   training start; the rest is unused.

3. **ViDoRe v2 (BEIR-format)** — the evaluation benchmark. Four
   corpora are reported: ``esg_reports_v2``,
   ``biomedical_lectures_v2``, ``economics_reports_v2``, and
   ``esg_reports_human_labeled_v2``.

Usage:

    DATA_ROOT=/path/to/data python scripts/download_data.py
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset


COLPALI_HF_ID = "vidore/colpali_train_set"
RLHN_HF_ID = os.environ.get("RLHN_HF_ID", "rlhn/rlhn-680K")

VIDORE_V2_CORPORA = [
    "esg_reports_v2",
    "biomedical_lectures_v2",
    "economics_reports_v2",
    "esg_reports_human_labeled_v2",
]
VIDORE_V2_SPLITS = ["corpus", "queries", "qrels"]


def _save(ds, out_path: Path):
    out_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_path))
    print(f"  saved -> {out_path}")


def download_colpali(data_root: Path):
    out = data_root / "colpali_train_set"
    if out.exists():
        print(f"[colpali] already at {out}, skipping")
        return
    print(f"[colpali] downloading {COLPALI_HF_ID} ...")
    ds = load_dataset(COLPALI_HF_ID, split="train")
    _save(ds, out)


def download_rlhn(data_root: Path):
    out = data_root / "rlhn_680k"
    if out.exists():
        print(f"[rlhn] already at {out}, skipping")
        return
    print(f"[rlhn] downloading {RLHN_HF_ID} ...")
    ds = load_dataset(RLHN_HF_ID, split="train")
    _save(ds, out)


def download_vidore_v2(data_root: Path):
    base = data_root / "vidore_v2"
    for corpus in VIDORE_V2_CORPORA:
        for split in VIDORE_V2_SPLITS:
            out = base / corpus / split
            if out.exists():
                print(f"[vidore_v2/{corpus}/{split}] already present, skipping")
                continue
            hf_id = f"vidore/{corpus}"
            print(f"[vidore_v2/{corpus}/{split}] downloading {hf_id} ({split}) ...")
            try:
                ds = load_dataset(hf_id, data_dir=split, split="test")
            except Exception:
                ds = load_dataset(hf_id, name=split, split="test")
            _save(ds, out)


def main():
    parser = argparse.ArgumentParser(
        description="Download V-SPLADE training and evaluation datasets from the HuggingFace Hub.",
    )
    parser.add_argument(
        "--data_root", type=str,
        default=os.environ.get("DATA_ROOT", "./data"),
        help="Destination directory. Each dataset writes a sub-folder.",
    )
    parser.add_argument(
        "--skip", type=str, nargs="*", default=[],
        choices=["colpali", "rlhn", "vidore_v2"],
        help="Datasets to skip (e.g. --skip rlhn).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    print(f"DATA_ROOT = {data_root}")

    if "colpali" not in args.skip:
        download_colpali(data_root)
    if "rlhn" not in args.skip:
        download_rlhn(data_root)
    if "vidore_v2" not in args.skip:
        download_vidore_v2(data_root)

    print("Done.")


if __name__ == "__main__":
    main()
