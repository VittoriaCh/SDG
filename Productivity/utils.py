from __future__ import annotations

import logging
import math
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
)
from tqdm import tqdm

from .types import BasicArgs, MaskArgs, NppArgs


def initalize_args(metrics: str, reporting_year: int) -> BasicArgs:
    """初始化评估参数配置。"""
    name_field = "NAME_1"
    dst_crs = "EPSG:6933"
    dst_res = 250
    z_edges = [-np.inf, -1.96, -0.5, 0.5, 1.96, np.inf]
    trend_args = MaskArgs(mask=False, slope_threshold=0.025)
    state_args = MaskArgs(mask=False, slope_threshold=0.025)
    npp_path = Path("../data/NPP_PROXY/NPP_2000_2025.tif")
    npp_args = prepare_profile(npp_path, base_year=2000, reporting_year=reporting_year)

    out_dir = Path(f"../output/{metrics}/{reporting_year}/")
    out_dir.mkdir(parents=True, exist_ok=True)

    shp_path = Path("../abu_dhabi_all/gadm41_ARE_1.shp")
    return BasicArgs(
        metrics=metrics,
        name_field=name_field,
        dst_crs=dst_crs,
        dst_res=dst_res,
        z_edges=z_edges,
        reporting_year=reporting_year,
        out_dir=out_dir,
        shp_path=shp_path,
        mask_low_productivity=False,
        npp_args=npp_args,
        trend_args=trend_args,
        state_args=state_args,
    )


def prepare_profile(
    npp_path: Path, base_year: int = 2000, reporting_year: int = 2015
) -> NppArgs:
    """从NPP栅格文件中提取元数据并创建NppArgs对象。"""
    with rasterio.open(npp_path) as src:
        height = src.height
        width = src.width
        nodata = src.nodata
        npp_bands = src.count
        npp_args = NppArgs(
            npp_path=npp_path,
            height=height,
            width=width,
            nodata=nodata,
            npp_bands=npp_bands,
            base_year=base_year,
            reporting_year=reporting_year,
            npp_profile=src.profile.copy(),
        )
    return npp_args


def mask_low_productivity(y_prod: np.ndarray) -> np.ndarray:
    """生成低生产力像元的掩膜。"""
    valid_prod = np.isfinite(y_prod)
    N_prod = valid_prod.sum(axis=0)

    mean_prod = np.nanmean(y_prod, axis=0)
    p90_prod = np.nanpercentile(y_prod, 90, axis=0)

    valid_mean = mean_prod[np.isfinite(mean_prod)]
    valid_p90 = p90_prod[np.isfinite(p90_prod)]

    mean_thr = float(np.nanpercentile(valid_mean, 30))
    p90_thr = float(np.nanpercentile(valid_p90, 30))

    valid_years_min = 12

    prod_mask = (N_prod >= valid_years_min) & (mean_prod >= mean_thr) & (p90_prod >= p90_thr)

    print("productive mask %:", float(np.mean(prod_mask) * 100))
    print("mean_thr:", mean_thr, "p90_thr:", p90_thr, "valid_years_min:", valid_years_min)
    return prod_mask


def mask_small_diff(s: np.ndarray, threshold: float = 0.0025) -> np.ndarray:
    """掩膜差异值较小的像元。"""
    return s > threshold


def mask_small_slope(s: np.ndarray, threshold: float) -> np.ndarray:
    """掩膜斜率绝对值较小的像元。"""
    mask = np.abs(s) >= threshold
    return mask


def classify_bins(z: np.ndarray, Z_EDGES) -> np.ndarray:
    """将Z值数组按照指定的分箱边界进行分类。"""
    bin_id = np.zeros(z.shape, dtype=np.int16)

    gate = np.isfinite(z)
    bin_id[~gate] = 99

    idx = np.digitize(z[gate], Z_EDGES)
    bin_id[gate] = idx.astype(np.int16)

    return bin_id


def read_block_to_float(src, win, nodata):
    """从栅格文件中读取指定窗口的数据块并转换为浮点数。"""
    data = src.read(window=win).astype(np.float32)
    if nodata is not None and not np.isnan(nodata):
        data = np.where(data == nodata, np.nan, data)
    return data


def reproject_to_equal_area(
    src_path: Path,
    dst_path: Path,
    dst_crs: str,
    dst_res: float,
    resampling: Resampling = Resampling.bilinear,
    *,
    force_float32: bool = True,
    dst_nodata: float = np.nan,
    copy_band_descriptions: bool = True,
    block_size: int = 512,
    compress: str = "lzw",
) -> None:
    """将栅格（单波段或多波段）重投影到等面积坐标系并写入文件。"""
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=dst_res
        )

        out_dtype = "float32" if force_float32 else src.dtypes[0]

        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            compress=compress,
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size,
            dtype=out_dtype,
            nodata=dst_nodata,
            count=src.count,
        )

        src_nodata = src.nodata
        src_has_finite_nodata = src_nodata is not None and np.isfinite(src_nodata)

        with rasterio.open(dst_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                src_data = src.read(b)

                if force_float32:
                    src_data = src_data.astype(np.float32, copy=False)

                if src_has_finite_nodata:
                    src_data = np.where(src_data == src_nodata, np.nan, src_data).astype(
                        np.float32 if force_float32 else src_data.dtype,
                        copy=False,
                    )

                dst_data = np.full((height, width), dst_nodata, dtype=np.float32 if force_float32 else src_data.dtype)

                reproject(
                    source=src_data,
                    destination=dst_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    src_nodata=np.nan if force_float32 else src_nodata,
                    dst_nodata=dst_nodata,
                )

                dst.write(dst_data, b)

            if copy_band_descriptions and src.descriptions:
                dst.descriptions = src.descriptions


def pixel_area_km2(src: rasterio.io.DatasetReader) -> float:
    """计算栅格像元的面积（平方公里）。"""
    px_w = src.transform.a
    px_h = src.transform.e
    if abs(px_w) < 0.1 or abs(px_h) < 0.1:
        raise ValueError(
            "Raster appears to be in degrees (geographic CRS). "
            "For accurate area, reproject to an equal-area projection first."
        )
    return abs(px_w * px_h) / 1e6


def setup_logger(
    name: str = "app",
    log_file: str | Path = "app.log",
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
) -> logging.Logger:
    """创建同时输出到控制台和滚动日志文件的日志记录器。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if getattr(logger, "_configured", False):
        return logger

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    fh = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.propagate = False
    logger._configured = True
    return logger


