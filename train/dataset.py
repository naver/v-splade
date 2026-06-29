# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
V-SPLADE — Dataset & Collator.

VisualDataset                : Unified visual retrieval dataset
                               (text query + image document + caption).
RLHNDataset                  : Text-only retrieval dataset (RLHN 680K)
                               for mixed-modality training.
TaskAwareBatchSampler        : Interleaves visual and text batches at
                               a fixed ratio so each batch is homogeneous.
RetrieverCollator            : Generic VLM bi-encoder collator
                               (text-query / image-passage / caption).
ModernVBertRetrieverCollator : BiModernVBERT-specific collator.
"""

import io
import random
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset, Sampler
from datasets import load_from_disk, load_dataset
from typing import Dict, List, Any


def _ensure_pil(img):
    """Convert image dict (decode=False) or path/bytes to a PIL Image."""
    if isinstance(img, PILImage.Image):
        return img
    if isinstance(img, dict) and "bytes" in img:
        return PILImage.open(io.BytesIO(img["bytes"])).convert("RGB")
    if isinstance(img, (str, bytes)):
        return PILImage.open(img).convert("RGB")
    return img


# ──────────────────────────────────────────────────────────────
# VisualDataset
# ──────────────────────────────────────────────────────────────

class VisualDataset(Dataset):
    """Unified visual retrieval dataset.

    Each row provides a text query, an image document, and a caption.
    In-batch negatives only (no hard-negative mining required).

    Args:
        dataset_path:    HuggingFace dataset ID or local disk path.
        caption_column:  caption column name (e.g. "caption").
        min_query_length: drop rows whose query is too short.
    """

    def __init__(
        self,
        dataset_path: str,
        caption_column: str = "caption",
        min_query_length: int = 5,
    ):
        import os, logging
        logger = logging.getLogger(__name__)
        self.rng = random.Random()

        if os.path.isdir(dataset_path):
            self.dataset = load_from_disk(dataset_path)
        else:
            self.dataset = load_dataset(dataset_path, split="train")

        if "image" in self.dataset.column_names and \
                not self.dataset.features["image"].decode:
            from datasets import Image as HFImage
            self.dataset = self.dataset.cast_column("image", HFImage(decode=True))

        self.caption_column = caption_column

        # Filter short queries
        if min_query_length > 0:
            queries = self.dataset["query"]
            self._valid_indices = [
                i for i, q in enumerate(queries)
                if len(q.strip()) > min_query_length
            ]
            n_filtered = len(self.dataset) - len(self._valid_indices)
            logger.info(
                f"VisualDataset: {n_filtered}/{len(self.dataset)} "
                f"short queries removed"
            )
        else:
            self._valid_indices = None

    def __len__(self) -> int:
        if self._valid_indices is not None:
            return len(self._valid_indices)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self._valid_indices[idx] if self._valid_indices is not None else idx
        sample = self.dataset[real_idx]

        result = {
            "query": sample["query"],
            "image": _ensure_pil(sample["image"]),
        }

        if self.caption_column in sample:
            cap = sample[self.caption_column]
            if cap and isinstance(cap, str) and len(cap.strip()) > 0:
                result["caption"] = cap
            else:
                raise ValueError(
                    f"Empty caption at index {idx} "
                    f"(column={self.caption_column})."
                )

        return result


# ──────────────────────────────────────────────────────────────
# RLHN (text-only) Dataset — mixed-modality training partner
# ──────────────────────────────────────────────────────────────

class RLHNDataset(Dataset):
    """RLHN 680K text-only retrieval dataset.

    Each row provides a query, one positive passage, and several hard
    negative passages. A subset is sampled at startup to match the paper's
    3:1 image-to-text training ratio.
    """

    def __init__(
        self,
        dataset_path: str,
        num_samples: int = 300_000,
        num_hard_negatives: int = 2,
        seed: int = 42,
    ):
        import logging
        logger = logging.getLogger(__name__)
        ds = load_from_disk(dataset_path)
        self.dataset = ds["train"] if "train" in ds else ds

        rng = random.Random(seed)
        total = len(self.dataset)
        if num_samples < total:
            self._indices = sorted(rng.sample(range(total), num_samples))
        else:
            self._indices = list(range(total))
        logger.info(f"RLHN: sampled {len(self._indices)}/{total}")

        self.num_hard_negatives = num_hard_negatives
        self.rng = random.Random()

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self._indices[idx]
        sample = self.dataset[real_idx]

        positives = sample["positive_passages"]
        pos = self.rng.choice(positives)

        negatives = sample["negative_passages"]
        n = min(self.num_hard_negatives, len(negatives))
        chosen_negs = self.rng.sample(negatives, n)

        return {
            "query": sample["query"],
            "text": pos["text"],
            "hard_neg_texts": [neg["text"] for neg in chosen_negs],
            "task_type": "rlhn",
        }


# ──────────────────────────────────────────────────────────────
# Task-aware batch sampler
# ──────────────────────────────────────────────────────────────

class TaskAwareBatchSampler(Sampler):
    """Interleave two task datasets at a fixed ratio with homogeneous batches.

    Used with ``ConcatDataset([rlhn_ds, visual_ds])``: every yielded batch
    contains samples from only one task. DDP-aware: each rank gets a
    disjoint subset of the (shuffled) per-task indices while preserving
    the task ratio within the rank.
    """

    def __init__(
        self,
        rlhn_size: int,
        colpali_size: int,
        rlhn_offset: int,
        colpali_offset: int,
        batch_size: int = 12,
        ratio_text_to_image: int = 2,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.rlhn_size = rlhn_size
        self.colpali_size = colpali_size
        self.rlhn_offset = rlhn_offset
        self.colpali_offset = colpali_offset
        self.batch_size = batch_size
        self.ratio = ratio_text_to_image
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = random.Random(self.seed + self.epoch)

        rlhn_indices = [self.rlhn_offset + i for i in range(self.rlhn_size)]
        colpali_indices = [self.colpali_offset + i for i in range(self.colpali_size)]
        g.shuffle(rlhn_indices)
        g.shuffle(colpali_indices)

        if self.world_size > 1:
            def _pad_divisible(indices, ws):
                rem = len(indices) % ws
                if rem != 0:
                    indices = indices + indices[: ws - rem]
                return indices
            rlhn_indices = _pad_divisible(rlhn_indices, self.world_size)
            colpali_indices = _pad_divisible(colpali_indices, self.world_size)
            rlhn_indices = rlhn_indices[self.rank :: self.world_size]
            colpali_indices = colpali_indices[self.rank :: self.world_size]

        rlhn_batches = [
            rlhn_indices[i : i + self.batch_size]
            for i in range(0, len(rlhn_indices), self.batch_size)
        ]
        colpali_batches = [
            colpali_indices[i : i + self.batch_size]
            for i in range(0, len(colpali_indices), self.batch_size)
        ]

        ri, ci = 0, 0
        while ri < len(rlhn_batches) and ci < len(colpali_batches):
            for _ in range(self.ratio):
                if ri < len(rlhn_batches):
                    yield rlhn_batches[ri]
                    ri += 1
            if ci < len(colpali_batches):
                yield colpali_batches[ci]
                ci += 1

        while ri < len(rlhn_batches):
            yield rlhn_batches[ri]
            ri += 1
        while ci < len(colpali_batches):
            yield colpali_batches[ci]
            ci += 1

    def __len__(self):
        rlhn_per_rank = (self.rlhn_size + self.world_size - 1) // self.world_size
        colpali_per_rank = (self.colpali_size + self.world_size - 1) // self.world_size
        rlhn_batches = (rlhn_per_rank + self.batch_size - 1) // self.batch_size
        colpali_batches = (colpali_per_rank + self.batch_size - 1) // self.batch_size
        return rlhn_batches + colpali_batches


# ──────────────────────────────────────────────────────────────
# Generic VLM Collator (text query + image passage + caption)
# ──────────────────────────────────────────────────────────────

class RetrieverCollator:
    """Bi-encoder collator for generic VLM retrieval training.

    Handles text queries, image documents, captions, and hard negatives.
    """

    def __init__(
        self,
        processor,
        query_type: str = "text",
        doc_type: str = "image",
        num_hard_negatives: int = 0,
        query_field: str = "query",
        doc_field: str = "image",
        caption_field: str = "caption",
        query_text_prompt: str = "{text}\nSummarize the query above in one word:",
        doc_image_prompt: str = "Summarize the document image above in one word:",
        cap_text_prompt: str = "{text}",
        query_max_length: int = 512,
        doc_max_length: int = 1024,
    ):
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"
        self.query_type = query_type
        self.doc_type = doc_type
        self.num_hard_negatives = num_hard_negatives

        self.query_field = query_field
        self.doc_field = doc_field
        self.caption_field = caption_field

        self.query_text_prompt = query_text_prompt
        self.doc_image_prompt = doc_image_prompt
        self.cap_text_prompt = cap_text_prompt
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length

    def _get_effective_max_length(self, max_length: int) -> int:
        """Cap to the tokenizer's true model_max_length when smaller."""
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        tok_max = getattr(tokenizer, "model_max_length", max_length)
        return min(max_length, tok_max) \
            if isinstance(tok_max, int) and tok_max < 100_000 else max_length

    def _encode_text(self, texts: List[str], max_length: int,
                     prompt: str = "{text}") -> Dict:
        formatted = [prompt.format(text=t) for t in texts]
        actual_max = self._get_effective_max_length(max_length)
        return self.processor.tokenizer(
            formatted, max_length=actual_max, truncation=True,
            padding="longest", return_tensors="pt",
        )

    def _encode_caption(self, texts: List[str]) -> Dict:
        formatted = [self.cap_text_prompt.format(text=t) for t in texts]
        cap_max = self._get_effective_max_length(self.doc_max_length)
        return self.processor.tokenizer(
            formatted, max_length=cap_max, truncation=True,
            padding="longest", return_tensors="pt",
        )

    def _encode_images(self, images: List) -> Dict:
        # Ensure all images are RGB PIL
        from PIL import Image as _PILImage
        images = [
            img.convert("RGB") if isinstance(img, _PILImage.Image)
            and img.mode != "RGB"
            else (_PILImage.fromarray(img).convert("RGB")
                  if not isinstance(img, _PILImage.Image) else img)
            for img in images
        ]
        convs = [
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self.doc_image_prompt},
            ]}]
            for _ in images
        ]
        try:
            texts = [
                self.processor.apply_chat_template(
                    c, tokenize=False, add_generation_prompt=False
                )
                for c in convs
            ]
        except (ValueError, AttributeError):
            texts = [self.doc_image_prompt] * len(images)
        return self.processor(
            text=texts, images=images,
            padding=True, truncation=False, return_tensors="pt",
        )

    def _encode_rlhn_passage(self, texts: List[str]) -> Dict:
        """Encode RLHN text passages with the caption text prompt."""
        formatted = [self.cap_text_prompt.format(text=t) for t in texts]
        doc_max = self._get_effective_max_length(self.doc_max_length)
        return self.processor.tokenizer(
            formatted, max_length=doc_max, truncation=True,
            padding="longest", return_tensors="pt",
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        result = {}
        task_type = batch[0].get("task_type", "colpali")
        result["task_type"] = task_type

        # Query
        q = self._encode_text(
            [item["query"] for item in batch],
            max_length=self.query_max_length,
            prompt=self.query_text_prompt,
        )
        result["query_input_ids"] = q["input_ids"]
        result["query_attention_mask"] = q.get(
            "attention_mask", torch.ones_like(q["input_ids"])
        )

        if task_type == "rlhn":
            # RLHN: text-only passages (no image, no caption).
            texts = [item["text"] for item in batch]
            p = self._encode_rlhn_passage(texts)
            result["passage_input_ids"] = p["input_ids"]
            result["passage_attention_mask"] = p.get(
                "attention_mask", torch.ones_like(p["input_ids"])
            )

            # Hard-negative texts
            if "hard_neg_texts" in batch[0]:
                flat_hn = [t for item in batch for t in item["hard_neg_texts"]]
                hn = self._encode_rlhn_passage(flat_hn)
                result["hard_neg_passage_input_ids"] = hn["input_ids"]
                result["hard_neg_passage_attention_mask"] = hn.get(
                    "attention_mask", torch.ones_like(hn["input_ids"])
                )
            return result

        # Passage (image)
        images = [item["image"] for item in batch]
        p = self._encode_images(images)
        result["passage_input_ids"] = p["input_ids"]
        result["passage_attention_mask"] = p.get(
            "attention_mask", torch.ones_like(p["input_ids"])
        )
        if "pixel_values" in p and p["pixel_values"] is not None:
            result["passage_pixel_values"] = p["pixel_values"]
        if "image_grid_thw" in p and p["image_grid_thw"] is not None:
            result["passage_image_grid_thw"] = p["image_grid_thw"]

        # Caption (caption-gated token supervision)
        if self.caption_field in batch[0]:
            cap_texts = [item[self.caption_field] for item in batch]
            cap = self._encode_caption(cap_texts)
            result["caption_input_ids"] = cap["input_ids"]
            result["caption_attention_mask"] = cap.get(
                "attention_mask", torch.ones_like(cap["input_ids"])
            )
            # If cap prompt wraps the caption, also tokenize the plain
            # caption text for BOW masking (so prompt tokens do not pollute
            # cap_bow).
            if self.cap_text_prompt != "{text}":
                cap_max = self._get_effective_max_length(self.doc_max_length)
                cap_plain = self.processor.tokenizer(
                    cap_texts, max_length=cap_max, truncation=True,
                    padding="longest", return_tensors="pt",
                )
                result["caption_bow_input_ids"] = cap_plain["input_ids"]
                result["caption_bow_attention_mask"] = cap_plain.get(
                    "attention_mask",
                    torch.ones_like(cap_plain["input_ids"])
                )

        # Hard-negative images
        if self.num_hard_negatives > 0 and "hard_neg_images" in batch[0]:
            flat_hn = [img for item in batch for img in item["hard_neg_images"]]
            hn = self._encode_images(flat_hn)
            result["hard_neg_passage_input_ids"] = hn["input_ids"]
            result["hard_neg_passage_attention_mask"] = hn.get(
                "attention_mask", torch.ones_like(hn["input_ids"])
            )
            if "pixel_values" in hn and hn["pixel_values"] is not None:
                result["hard_neg_passage_pixel_values"] = hn["pixel_values"]
            if "image_grid_thw" in hn and hn["image_grid_thw"] is not None:
                result["hard_neg_passage_image_grid_thw"] = hn["image_grid_thw"]

        return result


# ──────────────────────────────────────────────────────────────
# ModernVBERT Collator
# ──────────────────────────────────────────────────────────────

class ModernVBertRetrieverCollator:
    """Collator for ModernVBERT (BiModernVBertProcessor).

    Query   : process_queries(texts) -> {input_ids, attention_mask}
    Passage : process_images(images) -> {pixel_values, pixel_attention_mask,
                                         input_ids, attention_mask}
    Caption : tokenizer(texts)       -> {input_ids, attention_mask}
    """

    def __init__(
        self,
        processor,
        num_hard_negatives: int = 0,
        caption_field: str = "caption",
        max_caption_length: int = 1024,
    ):
        self.processor = processor
        self.num_hard_negatives = num_hard_negatives
        self.caption_field = caption_field
        self.max_caption_length = max_caption_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        result = {}
        task_type = batch[0].get("task_type", "colpali")
        result["task_type"] = task_type

        # Query
        queries = [item["query"] for item in batch]
        q = self.processor.process_queries(queries)
        result["query_input_ids"] = q["input_ids"]
        result["query_attention_mask"] = q["attention_mask"]

        if task_type == "rlhn":
            # RLHN: text-only passages (no image, no caption).
            texts = [item["text"] for item in batch]
            p = self.processor.tokenizer(
                texts, max_length=512, truncation=True,
                padding="longest", return_tensors="pt",
            )
            result["passage_input_ids"] = p["input_ids"]
            result["passage_attention_mask"] = p["attention_mask"]

            # Hard-negative texts
            if "hard_neg_texts" in batch[0]:
                flat_hn = [t for item in batch for t in item["hard_neg_texts"]]
                hn = self.processor.tokenizer(
                    flat_hn, max_length=512, truncation=True,
                    padding="longest", return_tensors="pt",
                )
                result["hard_neg_passage_input_ids"] = hn["input_ids"]
                result["hard_neg_passage_attention_mask"] = hn["attention_mask"]
            return result

        # Passage (image)
        images = [item["image"] for item in batch]
        p = self.processor.process_images(images)
        result["passage_input_ids"] = p["input_ids"]
        result["passage_attention_mask"] = p["attention_mask"]
        if "pixel_values" in p:
            result["passage_pixel_values"] = p["pixel_values"]
        if "pixel_attention_mask" in p:
            result["passage_pixel_attention_mask"] = p["pixel_attention_mask"]

        # Caption
        if self.caption_field in batch[0]:
            captions = [item.get(self.caption_field, "") for item in batch]
            cap = self.processor.tokenizer(
                captions, max_length=self.max_caption_length,
                truncation=True, padding="longest", return_tensors="pt",
            )
            result["caption_input_ids"] = cap["input_ids"]
            result["caption_attention_mask"] = cap["attention_mask"]

        # Hard-negative images
        if self.num_hard_negatives > 0 and "hard_neg_images" in batch[0]:
            flat_hn = [img for item in batch for img in item["hard_neg_images"]]
            hn = self.processor.process_images(flat_hn)
            result["hard_neg_passage_input_ids"] = hn["input_ids"]
            result["hard_neg_passage_attention_mask"] = hn["attention_mask"]
            if "pixel_values" in hn:
                result["hard_neg_passage_pixel_values"] = hn["pixel_values"]
            if "pixel_attention_mask" in hn:
                result["hard_neg_passage_pixel_attention_mask"] = hn["pixel_attention_mask"]

        return result
