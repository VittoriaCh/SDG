from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling
from rasterio.windows import Window
from tqdm import tqdm

from .performance import performance_evaluation
from .state import distribution_consistent
from .trend import mann_kendall_z, theil_sen_slope_block
from .types import BasicArgs
from .utils import (
    classify_bins,
    mask_low_productivity,
    mask_small_slope,
    pixel_area_km2,
    read_block_to_float,
    reproject_to_equal_area,
)


# ---------------------------------------------------------------------------
# generate_z_or_ratio_results — split into per-indicator functions
# ---------------------------------------------------------------------------

def generate_trend_results(args: BasicArgs) -> Tuple[Path, Path]:
    """生成trend指标的zscore.tif和slope.tif。"""
    out_dir = args.out_dir
    out_z_path = out_dir / "zscore.tif"
    out_slope_path = out_dir / "slope.tif"

    tif_path = args.npp_args.npp_path
    npp_args = args.npp_args
    profile = npp_args.npp_profile
    nodata = npp_args.nodata
    height = npp_args.height
    width = npp_args.width

    # 从26波段中取对应16年窗口
    start_band = args.reporting_year - args.npp_args.base_year - 15  # 0-indexed
    bands = list(range(start_band, start_band + 16))

    profile_float = profile.copy()
    profile_float.update(count=1, dtype="float32", nodata=np.nan)

    block_h, block_w = 512, 512
    n_rows = math.ceil(height / block_h)
    n_cols = math.ceil(width / block_w)

    with rasterio.open(tif_path) as src:
        with rasterio.open(out_z_path, "w", **profile_float) as dst_z:
            with rasterio.open(out_slope_path, "w", **profile_float) as dst_slope:
                for br in tqdm(range(n_rows), desc="generate trend results"):
                    for bc in range(n_cols):
                        r0 = br * block_h
                        c0 = bc * block_w
                        h = min(block_h, height - r0)
                        w = min(block_w, width - c0)
                        win = Window(c0, r0, w, h)

                        y = read_block_to_float(src, win, nodata)[bands]
                        slope = theil_sen_slope_block(y)
                        z = mann_kendall_z(y)

                        dst_z.write(z, 1, window=win)
                        dst_slope.write(slope, 1, window=win)

    return out_z_path, out_slope_path


def generate_state_results(args: BasicArgs) -> Tuple[Path, Path]:
    """生成state指标的zscore.tif和prod_diff.tif。"""
    out_dir = args.out_dir
    out_z_path = out_dir / "zscore.tif"
    out_prod_diff_path = out_dir / "prod_diff.tif"

    tif_path = args.npp_args.npp_path
    npp_args = args.npp_args
    profile = npp_args.npp_profile
    nodata = npp_args.nodata
    height = npp_args.height
    width = npp_args.width

    # 从26波段中取对应16年窗口
    start_band = args.reporting_year - args.npp_args.base_year - 15  # 0-indexed
    bands = list(range(start_band, start_band + 16))

    profile_float = profile.copy()
    profile_float.update(count=1, dtype="float32", nodata=np.nan)

    block_h, block_w = 512, 512
    n_rows = math.ceil(height / block_h)
    n_cols = math.ceil(width / block_w)

    with rasterio.open(tif_path) as src:
        with rasterio.open(out_z_path, "w", **profile_float) as dst_z:
            with rasterio.open(out_prod_diff_path, "w", **profile_float) as dst_prod:
                for br in tqdm(range(n_rows), desc="generate state results"):
                    for bc in range(n_cols):
                        r0 = br * block_h
                        c0 = bc * block_w
                        h = min(block_h, height - r0)
                        w = min(block_w, width - c0)
                        win = Window(c0, r0, w, h)

                        y = read_block_to_float(src, win, nodata)[bands]
                        z, prod_diff = distribution_consistent(y)

                        dst_z.write(z, 1, window=win)
                        dst_prod.write(prod_diff, 1, window=win)

    return out_z_path, out_prod_diff_path


def generate_performance_results(args: BasicArgs) -> Path:
    """生成performance指标的performance_ratio.tif。"""
    out_dir = args.out_dir
    out_ratio_path = out_dir / "performance_ratio.tif"

    tif_path = args.npp_args.npp_path
    npp_args = args.npp_args
    profile = npp_args.npp_profile
    nodata = npp_args.nodata
    height = npp_args.height
    width = npp_args.width

    # 从26波段中取对应16年窗口
    start_band = args.reporting_year - args.npp_args.base_year - 15  # 0-indexed
    bands = list(range(start_band, start_band + 16))

    profile_float = profile.copy()
    profile_float.update(count=1, dtype="float32", nodata=np.nan)

    block_h, block_w = 512, 512
    n_rows = math.ceil(height / block_h)
    n_cols = math.ceil(width / block_w)

    with rasterio.open(tif_path) as src:
        with rasterio.open(args.lecu_path, "r") as zsrc:
            with rasterio.open(out_ratio_path, "w", **profile_float) as dst_ratio:
                for br in tqdm(range(n_rows), desc="generate performance results"):
                    for bc in range(n_cols):
                        r0 = br * block_h
                        c0 = bc * block_w
                        h = min(block_h, height - r0)
                        w = min(block_w, width - c0)
                        win = Window(c0, r0, w, h)

                        y = read_block_to_float(src, win, nodata)[bands]
                        x = zsrc.read(1, window=win).astype(np.float32, copy=False)
                        ratio = performance_evaluation(y, x)
                        dst_ratio.write(ratio, 1, window=win)

    return out_ratio_path


def generate_z_or_ratio_results(args: BasicArgs) -> Union[Path, Tuple[Path, Path]]:
    """根据metrics类型生成相应的Z值或性能比率结果文件。

    薄分发器，调用对应的指标专用函数。
    """
    metrics = args.metrics

    if metrics == "trend":
        return generate_trend_results(args)
    elif metrics == "state":
        return generate_state_results(args)
    elif metrics == "performance":
        return generate_performance_results(args)
    else:
        raise ValueError(f"Unknown metrics: {metrics}")


# ---------------------------------------------------------------------------
# generate_report_tables — split masking + shared table generation
# ---------------------------------------------------------------------------

def _apply_metric_masks(
    z: np.ndarray,
    npp_src: rasterio.io.DatasetReader,
    mask_eq,
    args: BasicArgs,
) -> np.ndarray:
    """应用指标特定的掩膜，返回更新后的valid掩膜。"""
    metrics = args.metrics
    valid = np.isfinite(z)

    if args.mask_low_productivity:
        start_band = args.reporting_year - args.npp_args.base_year - 15
        bands = list(range(start_band, start_band + 16))
        prod_mask = mask_low_productivity(npp_src.read(bands))
        valid = valid & prod_mask

    if metrics == "trend":
        mask_slope = args.trend_args.mask
        if mask_slope:
            slope_threshold = args.trend_args.slope_threshold
            if mask_eq is None:
                raise ValueError(
                    "trend_args.mask=True 需要提供 slope_eq（用于 mask_small_slope）。"
                )
            with rasterio.open(mask_eq) as ssrc:
                s = ssrc.read(1).astype(np.float32)
                print("slope min", np.nanmin(s))
                print("slope max", np.nanmax(s))
                print("slope p1", np.nanpercentile(s[np.isfinite(s)], 1))
                prod_slop_mask = mask_small_slope(s, slope_threshold)
                valid = valid & prod_slop_mask.astype(bool)

    elif metrics == "state":
        mask_small_diff = args.state_args.mask
        if mask_small_diff:
            if mask_eq is None:
                raise ValueError("state_args.mask=True 需要提供 mask_eq。")
            with rasterio.open(mask_eq) as mask_src:
                diff_arr = mask_src.read(1).astype(np.float32)
                print("state diff min", np.nanmin(diff_arr))
                print("state diff max", np.nanmax(diff_arr))
                print("state diff p95", np.nanpercentile(diff_arr, 95))
                print("state diff p99", np.nanpercentile(diff_arr, 99))
                threshold = args.state_args.slope_threshold
                prod_diff_mask = mask_small_slope(diff_arr, threshold)
                valid = valid & prod_diff_mask.astype(bool)

    return valid


def _generate_tables_and_files(
    new_z: np.ndarray,
    zsrc,
    npp_src,
    px_area_km2: float,
    args: BasicArgs,
) -> tuple:
    """共享的表格生成和文件写入逻辑。"""
    metrics = args.metrics
    reporting_year = args.reporting_year
    out_dir = args.out_dir
    shp_path = args.shp_path
    dst_crs = args.dst_crs
    NAME_FIELD = args.name_field
    Z_EDGES = args.z_edges

    out_degrad_path = out_dir / "degrading.tif"
    out_improve_path = out_dir / "improving.tif"
    out_bins_path = out_dir / "bins.tif"
    out_overall_path = out_dir / "overall.tif"
    out_emirates_table_path = out_dir / f"{reporting_year}_emirates_table.csv"
    out_land_cover_table_path = out_dir / f"{reporting_year}_land_cover_table.csv"

    profile_output = zsrc.profile.copy()
    profile_output.update(
        dtype="uint8",
        nodata=0,
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    profile_bins = zsrc.profile.copy()
    profile_bins.update(
        dtype="int16",
        nodata=99,
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    bins = classify_bins(new_z, Z_EDGES)
    degrading = bins == 1
    improving = bins == len(Z_EDGES) - 1
    stable = (bins >= 2) & (bins < len(Z_EDGES) - 1) & (bins != 99)
    no_data = (bins == 99) | ~np.isfinite(new_z)
    nodata_code = np.uint8(99)
    overall = np.full(bins.shape, nodata_code, dtype=np.uint8)

    overall[degrading & ~no_data] = np.uint8(2)
    overall[stable & ~no_data] = np.uint8(0)
    overall[improving & ~no_data] = np.uint8(1)

    emirates = gpd.read_file(shp_path).to_crs(dst_crs)
    emirates["geom_area_km2"] = emirates.geometry.area / 1e6

    rows = []
    total_degradation = 0
    total_improvement = 0

    for _, row in tqdm(emirates.iterrows(), total=len(emirates), desc="Emirates"):
        from rasterio.features import geometry_mask
        reg_mask = geometry_mask(
            [row.geometry],
            out_shape=(zsrc.height, zsrc.width),
            transform=zsrc.transform,
            invert=True,
        )
        deg_km2 = float(np.count_nonzero(degrading & reg_mask) * px_area_km2)
        imp_km2 = float(np.count_nonzero(improving & reg_mask) * px_area_km2)
        total_degradation += deg_km2
        total_improvement += imp_km2

        rows.append({
            "Emirate": row[NAME_FIELD],
            "Degrading land (km²)": deg_km2,
            "Improving land (km²)": imp_km2,
        })

    total_mask = geometry_mask(
        emirates.geometry.tolist(),
        out_shape=(zsrc.height, zsrc.width),
        transform=zsrc.transform,
        invert=True,
    )

    improved_area_km2 = float(np.count_nonzero(improving & total_mask) * px_area_km2)
    stable_area_km2 = float(np.count_nonzero(stable & total_mask) * px_area_km2)
    degraded_area_km2 = float(np.count_nonzero(degrading & total_mask) * px_area_km2)
    no_data_area_km2 = float(np.count_nonzero(no_data & total_mask) * px_area_km2)

    total_area_km2 = improved_area_km2 + stable_area_km2 + degraded_area_km2 + no_data_area_km2

    if total_area_km2 > 0:
        improved_percent = (improved_area_km2 / total_area_km2) * 100
        stable_percent = (stable_area_km2 / total_area_km2) * 100
        degraded_percent = (degraded_area_km2 / total_area_km2) * 100
        no_data_percent = (no_data_area_km2 / total_area_km2) * 100
    else:
        improved_percent = 0.0
        stable_percent = 0.0
        degraded_percent = 0.0
        no_data_percent = 0.0

    land_cover_table = pd.DataFrame({
        "Land cover change category": [
            "Land area with improved land cover",
            "Land area with stable land cover",
            "Land area with degraded land cover",
            "Land area with no land cover data",
        ],
        "Area (km²)": [
            round(improved_area_km2, 1),
            round(stable_area_km2, 1),
            round(degraded_area_km2, 1),
            round(no_data_area_km2, 1),
        ],
        "Percent of total land area (%)": [
            round(improved_percent, 1),
            round(stable_percent, 1),
            round(degraded_percent, 1),
            round(no_data_percent, 1),
        ],
    })

    result = {
        "Target year": reporting_year,
        "Area of Degrading Land (km²)": total_degradation,
        "Area of Improving Land (km²)": total_improvement,
    }
    table36 = pd.DataFrame(rows).sort_values("Emirate").reset_index(drop=True)
    print(table36)
    table36.to_csv(out_emirates_table_path, index=False)
    land_cover_table.to_csv(out_land_cover_table_path, index=False)

    print("\n" + "=" * 80)
    print("Land Cover Change Statistics")
    print("=" * 80)
    print(land_cover_table.to_string(index=False))
    print("=" * 80)

    with rasterio.open(out_degrad_path, "w", **profile_output) as dst_output:
        dst_output.write(degrading.astype(np.uint8), 1)
    with rasterio.open(out_improve_path, "w", **profile_output) as dst_output:
        dst_output.write(improving.astype(np.uint8), 1)

    with rasterio.open(out_bins_path, "w", **profile_bins) as dst_output:
        dst_output.write(bins.astype(np.int16), 1)
    with rasterio.open(out_overall_path, "w", **profile_bins) as dst_output:
        dst_output.write(overall.astype(np.uint8), 1)

    return degrading, improving, result


def generate_report_tables(
    z_eq, npp_eq, mask_eq, args: BasicArgs
) -> tuple:
    """生成评估报告表格和分类结果。"""
    with rasterio.open(z_eq) as zsrc, rasterio.open(npp_eq) as npp_src:
        z = zsrc.read(1).astype(np.float32)
        px_area_km2 = pixel_area_km2(zsrc)

        z = np.where(np.isfinite(z), z, np.nan)

        gate = _apply_metric_masks(z, npp_src, mask_eq, args)

        new_z = np.where(gate == 1, z, np.nan).astype(np.float32, copy=False)

        return _generate_tables_and_files(new_z, zsrc, npp_src, px_area_km2, args)


# ---------------------------------------------------------------------------
# generate_report_period — orchestrator
# ---------------------------------------------------------------------------

def generate_report_period(metrics: str, args: BasicArgs) -> dict:
    """生成完整的评估报告周期数据。"""
    dst_crs = args.dst_crs
    dst_res = args.dst_res

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    npp_path = args.npp_args.npp_path
    npp_eq = out_dir / (npp_path.stem + "NPP_PROXY_eqarea6933.tif")
    reproject_to_equal_area(npp_path, npp_eq, dst_crs, dst_res, Resampling.bilinear)

    assert metrics in ("trend", "state", "performance")
    output = generate_z_or_ratio_results(args)
    s_or_prod_diff_eq = None

    if isinstance(output, tuple):
        out_z, out_slope_or_prod_diff_or_ratio = output
        z_eq = out_dir / (out_z.stem + "_eqarea6933.tif")
        reproject_to_equal_area(out_z, z_eq, dst_crs, dst_res, Resampling.bilinear)
        s_or_prod_diff_eq = out_dir / (out_slope_or_prod_diff_or_ratio.stem + "_slope_eqarea6933.tif")
        reproject_to_equal_area(out_slope_or_prod_diff_or_ratio, s_or_prod_diff_eq, dst_crs, dst_res, Resampling.bilinear)
    elif isinstance(output, Path):
        out_ratio = output
        z_eq = out_dir / (out_ratio.stem + "_eqarea6933.tif")
        reproject_to_equal_area(out_ratio, z_eq, dst_crs, dst_res, Resampling.bilinear)

    degrading, improving, result = generate_report_tables(z_eq, npp_eq, s_or_prod_diff_eq, args)

    return result
