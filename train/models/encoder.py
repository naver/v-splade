# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
Encoder Layer — backbone model that produces hidden states.

V_SPLADE uses the VBert (BiModernVBERT) encoder as its sole backbone.
The encoder exposes a unified API:
    encode_passage(inputs) -> (hidden_states, attention_mask)
    encode_text(inputs)    -> (hidden_states, attention_mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Optional, Tuple


class EncoderType(Enum):
    VBERT = "vbert"


# --------------------------------------------------------------
# Abstract Base
# --------------------------------------------------------------

class BaseEncoder(nn.Module):
    """Abstract encoder base.

    Unified API:
        encode_passage(inputs) -> (hidden_states, attention_mask)
        encode_text(inputs)    -> (hidden_states, attention_mask)
    """

    vocab_size: int = 0
    hidden_size: int = 0

    def encode_passage(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def encode_text(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        pass

    def get_text_embeddings(self) -> Optional[nn.Embedding]:
        """Return the text embedding layer (for query-encoder initialization)."""
        return None


# --------------------------------------------------------------
# MLM head used by VBert
# --------------------------------------------------------------

class ModernVBertMLMHead(nn.Module):
    """MLM head: dense(768->768) -> GELU -> LayerNorm(768) -> decoder(768->50368)."""

    def __init__(self, hidden_size: int = 768, vocab_size: int = 50368):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.dense(hidden_states)
        h = F.gelu(h)
        h = self.norm(h)
        h = self.decoder(h)
        return h

    @classmethod
    def from_safetensors(cls, safetensors_path: str, **kwargs):
        from safetensors import safe_open
        head = cls(**kwargs)
        with safe_open(safetensors_path, framework="pt") as f:
            head.dense.weight.data.copy_(f.get_tensor("lm_head.head.dense.weight"))
            head.norm.weight.data.copy_(f.get_tensor("lm_head.head.norm.weight"))
            head.decoder.weight.data.copy_(f.get_tensor("lm_head.decoder.weight"))
            head.decoder.bias.data.copy_(f.get_tensor("lm_head.decoder.bias"))
        return head


# --------------------------------------------------------------
# VBert Encoder (BiModernVBERT)
# --------------------------------------------------------------

class VBertEncoder(BaseEncoder):
    """BiModernVBERT encoder + external MLM head, with optional LoRA."""

    def __init__(
        self,
        model_name: str = "ModernVBERT/bimodernvbert",
        lm_head_model: str = "ModernVBERT/ModernVBERT",
        lm_head_lora_r: int = 32,
        encoder_lora_r: int = 32,
        lm_head_full: bool = False,
        **kwargs,
    ):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from models.convert import ensure_compatible_backbone

        # 0. Auto-convert the backbone if it uses the upstream ModernVBERT layout.
        #    Compatible backbones (local or Hub) pass through unchanged; the raw
        #    upstream checkpoint is downloaded + converted once (cached) so that
        #    from_scratch training works directly from the Hub id.
        model_name = ensure_compatible_backbone(model_name)
        lm_head_model = ensure_compatible_backbone(lm_head_model) if lm_head_model else model_name

        # 1. Load encoder backbone.
        model_cls = self._resolve_model_cls(model_name)
        self.encoder = model_cls.from_pretrained(model_name, dtype=torch.bfloat16)

        # Disable compiled_mlp - FX tracing in gradient_checkpointing traces
        # both branches of the if/else in ModernBertEncoderLayer.forward(),
        # hitting compiled_mlp even when reference_compile is None/False.
        def _set_reference_compile_false(module):
            if hasattr(module, "config") and hasattr(module.config, "reference_compile"):
                module.config.reference_compile = False
        for m in self.encoder.modules():
            _set_reference_compile_false(m)

        # 2. Merge any existing LoRA adapters into base weights.
        has_lora = any("lora" in k for k in self.encoder.state_dict().keys())
        if has_lora:
            from peft.tuners.lora.layer import Linear as LoraLinear
            for _, mod in self.encoder.named_modules():
                if isinstance(mod, LoraLinear) and hasattr(mod, "merge"):
                    mod.merge()

        # 3. Apply a fresh full LoRA on encoder (all layers: attn + mlp).
        if encoder_lora_r > 0:
            self.encoder.model.text_model = get_peft_model(
                self.encoder.model.text_model,
                LoraConfig(
                    r=encoder_lora_r, lora_alpha=encoder_lora_r,
                    target_modules=["Wqkv", "Wo", "Wi"],
                    bias="none",
                ),
            )

        # 4. Load MLM head - from same model dir or separate model.
        import os as _os
        encoder_sf = _os.path.join(model_name, "model.safetensors")
        has_lm_head_in_encoder = False
        if _os.path.isfile(encoder_sf):
            from safetensors import safe_open as _safe_open
            with _safe_open(encoder_sf, framework="pt") as _f:
                has_lm_head_in_encoder = any("lm_head" in k for k in _f.keys())
        if has_lm_head_in_encoder:
            self.mlm_head = ModernVBertMLMHead.from_safetensors(
                encoder_sf, hidden_size=768, vocab_size=50368,
            ).to(torch.bfloat16)
        else:
            safetensors_path = self._find_safetensors(lm_head_model)
            self.mlm_head = ModernVBertMLMHead.from_safetensors(
                safetensors_path, hidden_size=768, vocab_size=50368,
            ).to(torch.bfloat16)

        # 5. Apply LoRA to MLM head (dense + decoder).
        if lm_head_lora_r > 0 and not lm_head_full:
            self.mlm_head = get_peft_model(self.mlm_head, LoraConfig(
                r=lm_head_lora_r, lora_alpha=lm_head_lora_r,
                target_modules=["dense", "decoder"], bias="none",
            ))

        self.vocab_size = 50368
        self.hidden_size = 768

        # Freeze base weights, keep LoRA trainable.
        for name, param in self.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Optional full-parameter tuning for the MLM head (no LoRA).
        if lm_head_full:
            for param in self.mlm_head.parameters():
                param.requires_grad = True

    @classmethod
    def from_hf_export(cls, hf_dir: str, dtype: torch.dtype = torch.bfloat16) -> "VBertEncoder":
        """Build an empty VBertEncoder shell from a V-SPLADE HF export.

        Constructs `BiModernVBert(config)` + `ModernVBertMLMHead(...)` with
        randomly-initialized weights — the caller is expected to populate them
        via :func:`models.load_hf_export`. Used by `build_model(mode='inference_only')`.
        """
        from colpali_engine.models import BiModernVBert
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        config = BiModernVBert.config_class.from_pretrained(hf_dir)
        instance.encoder  = BiModernVBert(config).to(dtype=dtype)
        instance.mlm_head = ModernVBertMLMHead(
            hidden_size=config.hidden_size, vocab_size=50368,
        ).to(dtype=dtype)
        instance.vocab_size = 50368
        instance.hidden_size = config.hidden_size
        # No grad needed at inference; trainer-side flags are not touched.
        for p in instance.parameters():
            p.requires_grad = False
        return instance

    @staticmethod
    def _resolve_model_cls(model_name: str):
        import json, os
        from colpali_engine.models import BiModernVBert

        config_path = os.path.join(model_name, "config.json")
        adapter_config_path = os.path.join(model_name, "adapter_config.json")

        if os.path.isfile(adapter_config_path):
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            base_path = adapter_cfg.get("base_model_name_or_path", "")
            base_config = os.path.join(base_path, "config.json")
            if os.path.isfile(base_config):
                config_path = base_config

        if os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            archs = cfg.get("architectures", [])
            # V_SPLADE only uses the bidirectional encoder variant.
            if "BiModernVBert" in archs:
                return BiModernVBert

        return BiModernVBert

    @staticmethod
    def _find_safetensors(model_name: str) -> str:
        import os
        local = os.path.join(model_name, "model.safetensors")
        if os.path.isfile(local):
            return local
        from huggingface_hub import hf_hub_download
        return hf_hub_download(model_name, "model.safetensors")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        kwargs = gradient_checkpointing_kwargs or {"use_reentrant": False}
        text_model = self.encoder.model.text_model
        if hasattr(text_model, "gradient_checkpointing_enable"):
            text_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)

    def _get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if pixel_values is not None:
            kw["pixel_values"] = pixel_values
        if pixel_attention_mask is not None:
            kw["pixel_attention_mask"] = pixel_attention_mask
        outputs = self.encoder.model(**kw)
        return outputs[0]

    def encode_passage(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self._get_hidden_states(
            kwargs["input_ids"], kwargs["attention_mask"],
            kwargs.get("pixel_values"), kwargs.get("pixel_attention_mask"),
        )
        return hidden, kwargs["attention_mask"]

    def encode_text(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self._get_hidden_states(kwargs["input_ids"], kwargs["attention_mask"])
        return hidden, kwargs["attention_mask"]

    def get_lm_head(self):
        return self.mlm_head

    def get_text_embeddings(self) -> Optional[nn.Module]:
        return self.encoder.model.text_model.get_input_embeddings()

    @property
    def image_token_id(self) -> int:
        return 50407  # BiModernVBERT <image> token


# --------------------------------------------------------------
# Factory
# --------------------------------------------------------------

def build_encoder(encoder_type: str, **kwargs) -> BaseEncoder:
    """Build encoder by type string. V_SPLADE only ships the vbert backbone."""
    if encoder_type == "vbert":
        return VBertEncoder(**kwargs)
    raise ValueError(f"Unknown encoder_type: {encoder_type}. Choose: vbert")
