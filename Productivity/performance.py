from __future__ import annotations

import numpy as np


def calculate_90_quantile(
    mean_npp: np.ndarray, lceu_id: np.ndarray, nodata_lceu: int = 0
) -> np.ndarray:
    """计算每个LCEU分区的90分位数NPP值。"""
    if mean_npp.ndim != 2:
        raise ValueError(f"mean_npp must be (H,W), got {mean_npp.shape}")
    if lceu_id.shape != mean_npp.shape:
        raise ValueError("lceu_id shape must match mean_npp")

    H, W = mean_npp.shape
    mean_flat = mean_npp.reshape(-1).astype(np.float32, copy=False)
    lceu_flat = lceu_id.reshape(-1).astype(np.int32, copy=False)

    valid = (lceu_flat != nodata_lceu) & np.isfinite(mean_flat)
    mean_v = mean_flat[valid]
    lceu_v = lceu_flat[valid]

    p90_by_gid = {}
    for gid in np.unique(lceu_v):
        vals = mean_v[lceu_v == gid]
        if len(vals) >= 2:
            vals_valid = vals[np.isfinite(vals)]
            if len(vals_valid) >= 2:
                p90_by_gid[int(gid)] = float(np.nanpercentile(vals_valid, 90))

    p90_flat = np.full(mean_flat.shape, np.nan, dtype=np.float32)
    for gid, p90 in p90_by_gid.items():
        p90_flat[lceu_flat == gid] = p90

    return p90_flat.reshape(H, W)


def performance_evaluation(
    y: np.ndarray,
    lceu_id: np.ndarray,
    nodata_lceu: int = 0,
):
    """评估土地生产力性能指标。"""
    if y.ndim != 3:
        raise ValueError(f"y must be (T,H,W), got {y.shape}")

    T, H, W = y.shape
    y_float = y.astype(np.float32, copy=False)

    valid_time_mask = np.all(np.isfinite(y_float), axis=0)

    mean_npp = np.full((H, W), np.nan, dtype=np.float32)
    if valid_time_mask.any():
        mean_npp[valid_time_mask] = np.nanmean(y_float[:, valid_time_mask], axis=0)

    quantile_npp = calculate_90_quantile(mean_npp, lceu_id, nodata_lceu=nodata_lceu)

    ratio = mean_npp / quantile_npp
    ratio[~np.isfinite(ratio)] = np.nan
    ratio[lceu_id == nodata_lceu] = np.nan
    return ratio.astype(np.float32)
