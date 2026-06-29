# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
CLI to convert the upstream ModernVBERT release into a V-SPLADE-compatible backbone.

Training auto-converts on the fly (see models.convert.ensure_compatible_backbone),
so this script is only needed if you want to materialize / inspect / double-check
the converted backbone yourself.

Usage:
    # download upstream, convert, write to ./modernvbert-vsplade-base
    python scripts/convert_modernvbert_backbone.py --out ./modernvbert-vsplade-base

    # additionally double-check the result against a known-good reference dir
    python scripts/convert_modernvbert_backbone.py --out ./out \
        --reference /path/to/open_datasets/modernvbert
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from models.convert import convert_backbone, double_check, UPSTREAM_REPO, TOKENIZER_REPO  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output backbone directory")
    ap.add_argument("--repo", default=UPSTREAM_REPO, help="upstream ModernVBERT Hub repo id")
    ap.add_argument("--tokenizer-repo", default=TOKENIZER_REPO,
                    help="text base repo to source tokenizer files from")
    ap.add_argument("--reference", default=None,
                    help="known-good backbone dir to double-check the output against")
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="skip tokenizer/processor files (weights+config only)")
    args = ap.parse_args()

    print(f"[*] converting {args.repo} -> {args.out}")
    convert_backbone(args.repo, args.out, tokenizer_repo=args.tokenizer_repo,
                     with_tokenizer=not args.no_tokenizer)
    print(f"[*] done -> {args.out}")

    if args.reference:
        raise SystemExit(0 if double_check(args.out, args.reference) else 1)


if __name__ == "__main__":
    main()
