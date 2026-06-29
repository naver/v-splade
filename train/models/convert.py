# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
Backbone conversion: upstream ModernVBERT -> V-SPLADE-compatible layout.

The public ``ModernVBERT/modernvbert`` checkpoint stores its embeddings/LM-head
in a layout the decoupled-embedding model can't load (combined 50408 embedding,
plain LM head, double-nested vision keys, no decoupled-vocab config). This
module re-packages it.

It is used two ways:
  * ``ensure_compatible_backbone(ref)`` — called automatically by the training
    loader; converts on the fly and caches, so ``from_scratch`` "just works"
    with the upstream Hub id.
  * ``scripts/convert_modernvbert_backbone.py`` — a thin CLI over the same code.

Verified transform (every tensor reproduced bit-identically from upstream):

  tok_embeddings.weight                            <- upstream[:50368]
  tok_embeddings.additional_embedding.weight       <- upstream[50368:50408]
  lm_head.decoder.weight / .bias                   <- upstream lm_head.weight/bias[:50368]
  additional_fc.weight                             <- upstream lm_head.weight[50368:50408]
  lm_head.head.dense.weight                        <- upstream projection_head.dense.weight
  lm_head.head.norm.weight                         <- upstream projection_head.norm.weight
  model.vision_model.X                             <- upstream model.vision_model.vision_model.X
  model.connector.modality_projection.proj.weight  <- upstream ...modality_projection.weight
  (all other model.text_model.* keys copied unchanged)
"""

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

MAIN_VOCAB = 50368
FULL_VOCAB = 50408
ADDITIONAL_VOCAB = FULL_VOCAB - MAIN_VOCAB   # 40

UPSTREAM_REPO = "ModernVBERT/modernvbert"
TOKENIZER_REPO = "jhu-clsp/ettin-encoder-150m"

PREPROCESSOR_CONFIG = {
    "do_convert_rgb": True, "do_image_splitting": True, "do_normalize": True,
    "do_pad": True, "do_rescale": True, "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    "image_processor_type": "Idefics3ImageProcessor",
    "processor_class": "Idefics3Processor",
    "max_image_size": {"longest_edge": 512}, "resample": 1,
    "rescale_factor": 0.00392156862745098, "size": {"longest_edge": 2048},
}
PROCESSOR_CONFIG = {"image_seq_len": 64, "processor_class": "Idefics3Processor"}


# ──────────────────────────────────────────────────────────────────────────────
#  Core transform
# ──────────────────────────────────────────────────────────────────────────────
def transform_state_dict(u: dict) -> dict:
    """Apply the verified upstream -> V-SPLADE backbone weight recipe."""
    out = {}
    tok = u["model.text_model.embeddings.tok_embeddings.weight"]
    assert tok.shape[0] == FULL_VOCAB, f"unexpected embedding rows: {tuple(tok.shape)}"
    out["model.text_model.embeddings.tok_embeddings.weight"] = tok[:MAIN_VOCAB].clone()
    out["model.text_model.embeddings.tok_embeddings.additional_embedding.weight"] = \
        tok[MAIN_VOCAB:FULL_VOCAB].clone()

    lw, lb = u["lm_head.weight"], u["lm_head.bias"]
    out["lm_head.decoder.weight"] = lw[:MAIN_VOCAB].clone()
    out["lm_head.decoder.bias"]   = lb[:MAIN_VOCAB].clone()
    out["additional_fc.weight"]   = lw[MAIN_VOCAB:FULL_VOCAB].clone()

    out["lm_head.head.dense.weight"] = u["projection_head.dense.weight"].clone()
    out["lm_head.head.norm.weight"]  = u["projection_head.norm.weight"].clone()

    out["model.connector.modality_projection.proj.weight"] = \
        u["model.connector.modality_projection.weight"].clone()

    vp = "model.vision_model.vision_model."
    for k, v in u.items():
        if k.startswith(vp):
            out["model.vision_model." + k[len(vp):]] = v.clone()

    for k, v in u.items():
        if k.startswith("model.text_model.") and "tok_embeddings" not in k:
            out[k] = v.clone()
    return out


def build_config(u_cfg: dict) -> dict:
    """Patch the upstream config into the decoupled-embedding V-SPLADE layout."""
    cfg = json.loads(json.dumps(u_cfg))
    cfg["vocab_size"] = MAIN_VOCAB
    cfg["additional_vocab_size"] = ADDITIONAL_VOCAB
    cfg["freeze_config"] = {
        "freeze_lm_head": True, "freeze_text_layers": True, "freeze_vision_layers": True,
    }
    cfg.setdefault("architectures", ["BiModernVBert"])
    if "text_config" in cfg:
        cfg["text_config"]["vocab_size"] = MAIN_VOCAB
    return cfg


def is_compatible_config(cfg: dict) -> bool:
    """True if the config already uses the decoupled-embedding V-SPLADE layout."""
    return cfg.get("freeze_config") is not None and cfg.get("additional_vocab_size") is not None


# ──────────────────────────────────────────────────────────────────────────────
#  Conversion + auto-ensure
# ──────────────────────────────────────────────────────────────────────────────
def _load_config(ref: str) -> dict:
    if os.path.isdir(ref):
        return json.load(open(os.path.join(ref, "config.json")))
    from huggingface_hub import hf_hub_download
    return json.load(open(hf_hub_download(ref, "config.json")))


def convert_backbone(ref: str, out_dir, tokenizer_repo: str = TOKENIZER_REPO,
                     with_tokenizer: bool = True) -> str:
    """Convert backbone ``ref`` (Hub id or local dir) into ``out_dir``. Returns out_dir."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    if os.path.isdir(ref):
        sd_path = os.path.join(ref, "model.safetensors")
        cfg = json.load(open(os.path.join(ref, "config.json")))
    else:
        from huggingface_hub import hf_hub_download
        sd_path = hf_hub_download(ref, "model.safetensors")
        cfg = json.load(open(hf_hub_download(ref, "config.json")))

    save_file(transform_state_dict(load_file(sd_path)),
              str(out / "model.safetensors"), metadata={"format": "pt"})
    json.dump(build_config(cfg), open(out / "config.json", "w"), indent=2)

    if with_tokenizer:
        from huggingface_hub import hf_hub_download
        for fn in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
            try:
                shutil.copy2(hf_hub_download(tokenizer_repo, fn), out / fn)
            except Exception:
                pass
        json.dump(PREPROCESSOR_CONFIG, open(out / "preprocessor_config.json", "w"), indent=2)
        json.dump(PROCESSOR_CONFIG, open(out / "processor_config.json", "w"), indent=2)
    return str(out)


def _cache_root() -> Path:
    root = os.environ.get("VSPLADE_BACKBONE_CACHE")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "v-splade" / "backbones"


def ensure_compatible_backbone(ref: str, tokenizer_repo: str = TOKENIZER_REPO,
                               verbose: bool = True) -> str:
    """Return a local path to a V-SPLADE-compatible backbone for ``ref``.

    If ``ref`` already uses the decoupled layout, it is returned unchanged
    (``from_pretrained`` will download a Hub id as usual). Otherwise the upstream
    checkpoint is converted once into a cache directory and that path is returned,
    so ``from_scratch`` training works directly from the raw upstream Hub id.
    """
    try:
        cfg = _load_config(ref)
    except Exception:
        return ref   # can't introspect (offline/unknown) — let from_pretrained handle it
    if is_compatible_config(cfg):
        return ref

    out = _cache_root() / ref.replace("/", "__")
    if (out / "model.safetensors").is_file() and (out / "config.json").is_file():
        if verbose:
            print(f"[convert] using cached converted backbone: {out}")
        return str(out)
    if verbose:
        print(f"[convert] '{ref}' is an upstream-layout backbone; "
              f"converting once -> {out}")
    return convert_backbone(ref, out, tokenizer_repo=tokenizer_repo)


# ──────────────────────────────────────────────────────────────────────────────
#  Double-check (used by the CLI)
# ──────────────────────────────────────────────────────────────────────────────
def double_check(out_dir, ref_dir) -> bool:
    out_dir, ref_dir = Path(out_dir), Path(ref_dir)
    print(f"\n[verify] comparing {out_dir}  vs  reference {ref_dir}")
    o = load_file(out_dir / "model.safetensors")
    r = load_file(ref_dir / "model.safetensors")
    ok = True
    if set(o) != set(r):
        ok = False
        print(f"  [FAIL] key sets differ "
              f"(only_out={sorted(set(o)-set(r))[:3]} only_ref={sorted(set(r)-set(o))[:3]})")
    else:
        print(f"  [ok] key sets match ({len(o)} tensors)")
    mismatched = [k for k in set(o) & set(r)
                  if o[k].shape != r[k].shape or not torch.equal(o[k].float(), r[k].float())]
    if mismatched:
        ok = False
        print(f"  [FAIL] {len(mismatched)} tensor(s) differ, e.g. {mismatched[:5]}")
    else:
        print(f"  [ok] all {len(set(o) & set(r))} shared tensors bit-identical")
    oc, rc = json.load(open(out_dir / "config.json")), json.load(open(ref_dir / "config.json"))
    for f in ["vocab_size", "additional_vocab_size", "freeze_config"]:
        if oc.get(f) != rc.get(f):
            ok = False; print(f"  [FAIL] config.{f}: {oc.get(f)} != {rc.get(f)}")
        else:
            print(f"  [ok] config.{f} == {oc.get(f)}")
    print(f"\n[verify] {'PASSED' if ok else 'FAILED'}.")
    return ok
