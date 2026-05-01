from __future__ import annotations

import numpy as np


def theil_sen_slope_block(y: np.ndarray) -> np.ndarray:
    """计算Theil-Sen斜率估计值。"""
    T, h, w = y.shape
    y = y.astype(np.float32, copy=False)
    valid_all = np.all(np.isfinite(y), axis=0)

    out = np.full((h, w), np.nan, dtype=np.float32)
    if not np.any(valid_all):
        return out

    slopes = []
    for i in range(T - 1):
        yi = y[i]
        for j in range(i + 1, T):
            yj = y[j]
            s = (yj - yi) / (j - i)
            slopes.append(s)

    S = np.stack(slopes, axis=0)

    med = np.median(S[:, valid_all], axis=0).astype(np.float32)
    out[valid_all] = med
    return out


def mann_kendall_S(y: np.ndarray) -> np.ndarray:
    """计算Mann-Kendall检验的S统计量。"""
    T, h, w = y.shape
    y = y.astype(np.float32, copy=False)
    valid_all = np.all(np.isfinite(y), axis=0)

    out = np.full((h, w), np.nan, dtype=np.float32)
    if not np.any(valid_all):
        return out

    sign = []
    for i in range(T - 1):
        yi = y[i]
        for j in range(i + 1, T):
            yj = y[j]
            s = np.sign(yj - yi)
            sign.append(s)

    S = np.stack(sign, axis=0)

    med = np.sum(S[:, valid_all], axis=0).astype(np.float32)
    out[valid_all] = med
    return out


def mann_kendall_z(y: np.ndarray) -> np.ndarray:
    """计算Mann-Kendall检验的Z值（标准化统计量）。"""
    if y.ndim != 3:
        raise ValueError(f"y must be (T,H,W), got {y.shape}")
    T, h, w = y.shape
    y = y.astype(np.float32, copy=False)
    valid_all = np.all(np.isfinite(y), axis=0)
    z = np.full((h, w), np.nan, dtype=np.float32)
    s = mann_kendall_S(y)
    base_num = T * (T - 1) * (2 * T + 5)
    denom = 18.0

    varS = np.full((h, w), np.nan, dtype=np.float32)
    for r, c in np.argwhere(valid_all):
        vals = y[:, r, c]
        _, counts = np.unique(vals, return_counts=True)
        tie_sum = 0.0
        for t in counts:
            if t > 1:
                tie_sum += t * (t - 1) * (2 * t + 5)
        varS[r, c] = (base_num - tie_sum) / denom

    ok = valid_all & (varS > 0)
    z[ok] = s[ok] / np.sqrt(varS[ok])
    return z


def kendall_tau_b_z(y: np.ndarray) -> np.ndarray:
    """计算Kendall tau-b相关系数并转换为z-score。"""
    if y.ndim != 3:
        raise ValueError(f"y must be (T,H,W), got {y.shape}")
    T, h, w = y.shape
    y = y.astype(np.float32, copy=False)

    valid_all = np.all(np.isfinite(y), axis=0)
    out = np.full((h, w), np.nan, dtype=np.float32)
    if not np.any(valid_all):
        return out

    S = np.zeros((h, w), dtype=np.float32)
    for i in range(T - 1):
        yi = y[i]
        for j in range(i + 1, T):
            S += np.sign(y[j] - yi).astype(np.float32)

    P = T * (T - 1) / 2.0

    Ty = np.zeros((h, w), dtype=np.float32)
    for r, c in np.argwhere(valid_all):
        vals = y[:, r, c]
        _, counts = np.unique(vals, return_counts=True)
        Ty_rc = 0.0
        for t in counts:
            if t > 1:
                Ty_rc += t * (t - 1) / 2.0
        Ty[r, c] = Ty_rc

    denom = np.sqrt(P * (P - Ty))
    ok = valid_all & (denom > 0)
    tau_b = np.zeros((h, w), dtype=np.float32)
    tau_b[ok] = S[ok] / denom[ok]

    N = float(T)
    numerator = 3.0 * tau_b * np.sqrt(N * (N - 1.0))
    denominator = 2.0 * (2.0 * N + 5.0)

    out[ok] = numerator[ok] / denominator
    return out
