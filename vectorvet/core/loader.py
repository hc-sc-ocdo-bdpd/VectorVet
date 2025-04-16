"""Load embedding matrices from disk.

Supported formats:  ▸ .npy  ▸ .pt (torch tensors)  ▸ .pkl (pickled numpy array)
Return type: Dict[str, np.ndarray] where keys are model names.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch

__all__ = ["load_embeddings", "load_multiple_embeddings"]


def _load_single(path: str | Path) -> np.ndarray:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".pt":
        return torch.load(path, map_location="cpu").numpy()
    if ext == ".pkl":
        with open(path, "rb") as f:
            arr = pickle.load(f)
        if isinstance(arr, torch.Tensor):
            arr = arr.numpy()
        return arr
    raise ValueError(f"Unsupported embedding file extension: {ext}")


def load_embeddings(file_path: str | Path) -> np.ndarray:
    return _load_single(file_path)


def load_multiple_embeddings(files: Mapping[str, str | Path]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, fp in files.items():
        arr = _load_single(fp)
        if arr.ndim != 2:
            raise ValueError(f"Embeddings for {name} must be 2‑D (n×d). Got shape {arr.shape}.")
        out[name] = arr.astype(np.float32, copy=False)
    return out