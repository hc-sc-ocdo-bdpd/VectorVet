"""Aggregate metric results and export helpers."""
from __future__ import annotations

from typing import Mapping, Any
import pandas as pd

__all__ = ["summarize_to_dataframe"]


def summarize_to_dataframe(results: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    return df.sort_index()