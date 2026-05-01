from __future__ import annotations

import numpy as np


def distribution_consistent(
    y: np.ndarray, min_n0: int = 10, min_n1: int = 3, eps: float = 1e-6
):
    """计算两时期分布的统计一致性Z值。"""
    if y.ndim != 3:
        raise ValueError(f"y must be (T,H,W), got {y.shape}")
    T, H, W = y.shape
    if T != 16:
        raise ValueError(f"y must be 16 years, got {T}")

    y = y.astype(np.float32, copy=False)

    valid_time_mask = np.all(np.isfinite(y), axis=0)

    y0 = y[:13]
    y1 = y[13:]

    mean0 = np.full((H, W), np.nan, dtype=np.float32)
    std0 = np.full((H, W), np.nan, dtype=np.float32)
    mean1 = np.full((H, W), np.nan, dtype=np.float32)

    if valid_time_mask.any():
        mean0[valid_time_mask] = np.nanmean(y0[:, valid_time_mask], axis=0)
        std0[valid_time_mask] = np.nanstd(y0[:, valid_time_mask], axis=0, ddof=0)
        mean1[valid_time_mask] = np.nanmean(y1[:, valid_time_mask], axis=0)

    prod_diff = mean1 - mean0

    gate = valid_time_mask & np.isfinite(std0) & (std0 > eps)

    z = np.full((H, W), np.nan, dtype=np.float32)
    z[gate] = prod_diff[gate] / (std0[gate] / np.sqrt(3.0))

    return z, prod_diff
