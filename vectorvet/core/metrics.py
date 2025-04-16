# Directory: vectorvet/core/metrics.py
"""Intrinsic metrics for embedding quality."""

from __future__ import annotations

import warnings
from typing import Dict, Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# IsoScore: pip install IsoScore  (capital “I” in package name)
# ---------------------------------------------------------------------
try:
    from IsoScore.IsoScore import IsoScore as _IsoScoreFn  
except ImportError:  # graceful degradation
    _IsoScoreFn = None
    print("⚠️  IsoScore package not found – isotropy metric will be NaN.")

# Hubness analysis (pip install scikit-hubness)
from skhubness.analysis import Hubness

__all__ = [
    "compute_isotropy",
    "compute_hubness",
    "compute_clustering",
    "pairwise_similarity_stats",
    "run_all_metrics",
]


# ---------------------------------------------------------------------
# Isotropy
# ---------------------------------------------------------------------
def compute_isotropy(X: np.ndarray, sample: int | None = 5000) -> float:
    """
    Return IsoScore ∈ [0, 1].

    IsoScore expects shape **(dim, n_points)**, so we transpose.
    Optionally subsample to *sample* points for speed / memory.
    """
    if _IsoScoreFn is None:
        return float("nan")

    if sample is not None and X.shape[0] > sample:
        X = X[np.random.default_rng(0).choice(X.shape[0], size=sample, replace=False)]
    # Transpose to (d, n)
    return float(_IsoScoreFn(X.T))


# ---------------------------------------------------------------------
# Hubness
# ---------------------------------------------------------------------
from skhubness.analysis import Hubness  
  
def compute_hubness(X: np.ndarray, k: int = 10) -> Dict[str, float]:  
    h = Hubness(k=k, metric="cosine", return_value='all')  
    h.fit(X)  
    hubness_stats = h.score()  
  
    return {  
        "skewness": float(hubness_stats["k_skewness"]),  
        "robin_hood": float(hubness_stats["robinhood"]),  
        "antihub_rate": float(hubness_stats["antihub_occurrence"]),  
    }  


# ---------------------------------------------------------------------
# Clustering quality
# ---------------------------------------------------------------------
def compute_clustering(X: np.ndarray, n_clusters: int | None = None) -> Dict[str, float]:
    n_samples = X.shape[0]
    if n_clusters is None:
        n_clusters = max(2, min(100, int(np.sqrt(n_samples))))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit(X)

    labels = km.labels_
    sil = silhouette_score(X, labels, metric="cosine")
    db = davies_bouldin_score(X, labels)
    return {"silhouette": float(sil), "davies_bouldin": float(db)}


# ---------------------------------------------------------------------
# Pairwise cosine statistics
# ---------------------------------------------------------------------
def pairwise_similarity_stats(X: np.ndarray, sample_size: int = 20_000) -> Dict[str, float]:
    n = X.shape[0]
    idx = np.random.default_rng(42).choice(n, size=min(sample_size, n), replace=False)
    sub = X[idx]
    norm = np.linalg.norm(sub, axis=1, keepdims=True)
    sub_norm = sub / np.clip(norm, 1e-8, None)
    sims = sub_norm @ sub_norm.T
    iu = np.triu_indices_from(sims, k=1)
    vals = sims[iu]
    return {"cos_mean": float(vals.mean()), "cos_std": float(vals.std())}


# ---------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------
def run_all_metrics(X: np.ndarray) -> Dict[str, Any]:
    out = {}
    print("Calculating Isotropy...")
    out["IsoScore"] = compute_isotropy(X)
    print("Calculating Hubness...")
    out.update(compute_hubness(X))
    print("Calculating Clustering Quality...")
    out.update(compute_clustering(X))
    print("Calculating Pairwise Cosine Similarity...")
    out.update(pairwise_similarity_stats(X))
    return out
