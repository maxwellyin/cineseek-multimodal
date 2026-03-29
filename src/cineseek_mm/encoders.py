from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from cineseek_mm.config import CLIP_MODEL_NAME, get_device


def load_clip():
    device = get_device()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()
    return model, processor, device


def _feature_tensor(output: Any, model: CLIPModel, modality: str) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    embed_attr = f"{modality}_embeds"
    if hasattr(output, embed_attr):
        embeds = getattr(output, embed_attr)
        if embeds is not None:
            return embeds

    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        pooled = output.pooler_output
        if (
            modality == "text"
            and hasattr(model, "text_projection")
            and pooled.shape[-1] == model.text_projection.in_features
        ):
            return model.text_projection(pooled)
        if (
            modality == "image"
            and hasattr(model, "visual_projection")
            and pooled.shape[-1] == model.visual_projection.in_features
        ):
            return model.visual_projection(pooled)
        return pooled

    if isinstance(output, tuple) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first

    raise TypeError(f"Unsupported CLIP {modality} output type: {type(output)!r}")


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = F.normalize(tensor, dim=-1)
    return tensor.detach().cpu().numpy().astype("float32")


def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    model, processor, device = load_clip()
    outputs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
            batch = texts[start : start + batch_size]
            inputs = processor(text=batch, padding=True, truncation=True, return_tensors="pt").to(device)
            features = _feature_tensor(model.get_text_features(**inputs), model, "text")
            outputs.append(_to_numpy(features))
    return np.vstack(outputs) if outputs else np.empty((0, model.config.projection_dim), dtype="float32")


def _load_image(path: str | Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def encode_images(paths: list[str | Path], batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    model, processor, device = load_clip()
    outputs = []
    valid_mask = []
    with torch.no_grad():
        for start in tqdm(range(0, len(paths), batch_size), desc="Encoding images"):
            batch_paths = paths[start : start + batch_size]
            images = []
            batch_valid = []
            for path in batch_paths:
                image = _load_image(path)
                if image is None:
                    batch_valid.append(False)
                    continue
                images.append(image)
                batch_valid.append(True)
            valid_mask.extend(batch_valid)
            if not images:
                continue
            inputs = processor(images=images, return_tensors="pt").to(device)
            features = _feature_tensor(model.get_image_features(**inputs), model, "image")
            outputs.append(_to_numpy(features))

    valid_embeddings = np.vstack(outputs) if outputs else np.empty((0, model.config.projection_dim), dtype="float32")
    full = np.zeros((len(paths), model.config.projection_dim), dtype="float32")
    full[np.array(valid_mask, dtype=bool)] = valid_embeddings
    return full, np.array(valid_mask, dtype=bool)


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (matrix / norms).astype("float32")


def fuse_embeddings(text_embeddings: np.ndarray, image_embeddings: np.ndarray, image_weight: float = 0.35) -> np.ndarray:
    if text_embeddings.shape != image_embeddings.shape:
        raise ValueError(f"Shape mismatch: {text_embeddings.shape} vs {image_embeddings.shape}")
    fused = (1.0 - image_weight) * text_embeddings + image_weight * image_embeddings
    return normalize_matrix(fused)
