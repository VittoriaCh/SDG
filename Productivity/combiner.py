from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform_bounds,
)


def combine_metrics_degradation(
    trend_degrading_path: Path,
    state_degrading_path: Path,
    performance_degrading_path: Path,
    output_path: Path,
    table_version: str = "4-4",
) -> Path:
    """组合三个指标的退化判断结果，根据查找表判断每个像素是否最终退化。"""
    with rasterio.open(trend_degrading_path) as src_trend:
        trend_degrading = src_trend.read(1).astype(np.uint8)
        profile = src_trend.profile.copy()
        trend_nodata = src_trend.nodata if src_trend.nodata is not None else 99

        with rasterio.open(state_degrading_path) as src_state:
            state_degrading = src_state.read(1).astype(np.uint8)
            state_nodata = src_state.nodata if src_state.nodata is not None else 99

            with rasterio.open(performance_degrading_path) as src_perf:
                performance_degrading = src_perf.read(1).astype(np.uint8)
                perf_nodata = src_perf.nodata if src_perf.nodata is not None else 99

                if (
                    trend_degrading.shape != state_degrading.shape
                    or trend_degrading.shape != performance_degrading.shape
                ):
                    raise ValueError(
                        f"Degrading arrays must have the same shape. "
                        f"Got: trend {trend_degrading.shape}, "
                        f"state {state_degrading.shape}, "
                        f"performance {performance_degrading.shape}"
                    )

                if trend_nodata is not None:
                    trend_valid = trend_degrading != trend_nodata
                else:
                    trend_valid = np.ones_like(trend_degrading, dtype=bool)

                if state_nodata is not None:
                    state_valid = state_degrading != state_nodata
                else:
                    state_valid = np.ones_like(state_degrading, dtype=bool)

                if perf_nodata is not None:
                    perf_valid = performance_degrading != perf_nodata
                else:
                    perf_valid = np.ones_like(performance_degrading, dtype=bool)

                valid_pixels = trend_valid & state_valid & perf_valid

                trend_y = (trend_degrading == 2) & valid_pixels
                state_y = (state_degrading == 2) & valid_pixels
                perf_y = (performance_degrading == 2) & valid_pixels

                trend_improving = (trend_degrading == 1) & valid_pixels
                state_improving = (state_degrading == 1) & valid_pixels
                perf_improving = (performance_degrading == 1) & valid_pixels

                final_class = np.full(trend_degrading.shape, 99, dtype=np.uint8)

                if valid_pixels.any():
                    t_y = trend_y[valid_pixels]
                    s_y = state_y[valid_pixels]
                    p_y = perf_y[valid_pixels]

                    t_i = trend_improving[valid_pixels]
                    s_i = state_improving[valid_pixels]
                    p_i = perf_improving[valid_pixels]

                    if table_version == "4-4":
                        degraded_mask = (
                            (t_y & s_y & p_y)
                            | (t_y & s_y & ~p_y)
                            | (t_y & ~s_y & p_y)
                            | (t_y & ~s_y & ~p_y)
                            | (~t_y & s_y & p_y)
                        )
                    elif table_version == "4-5":
                        degraded_mask = (
                            (t_y & s_y & p_y)
                            | (t_y & s_y & ~p_y)
                            | (t_y & ~s_y & p_y)
                            | (~t_y & s_y & p_y)
                            | (~t_y & ~s_y & p_y)
                        )
                    else:
                        raise ValueError(f"Unknown table_version: {table_version}. Must be '4-4' or '4-5'")

                    non_degraded = ~degraded_mask
                    improving_mask = non_degraded & (t_i & s_i & p_i)
                    stable_mask = non_degraded & (~improving_mask)

                    out = np.empty_like(degraded_mask, dtype=np.uint8)
                    out[stable_mask] = 0
                    out[improving_mask] = 1
                    out[degraded_mask] = 2

                    final_class[valid_pixels] = out

                output_profile = profile.copy()
                output_profile.update(
                    dtype="uint8",
                    nodata=99,
                    compress="lzw",
                    tiled=True,
                    blockxsize=512,
                    blockysize=512,
                )

                output_path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(output_path, "w", **output_profile) as dst:
                    dst.write(final_class, 1)

    return output_path


def productivity_land_cover_classification(
    land_cover_path: Path,
    productivity_bins: Path,
    land_cover_class_names: Optional[dict[int, str]] = None,
    dst_crs: str = "EPSG:6933",
    dst_res: float = 250.0,
) -> pd.DataFrame:
    """按土地覆盖类别统计生产力退化/非退化/无数据面积。"""
    with rasterio.open(land_cover_path) as lc_src, rasterio.open(productivity_bins) as bins_src:
        lc_bounds = transform_bounds(lc_src.crs, dst_crs, *lc_src.bounds)
        bins_bounds = transform_bounds(bins_src.crs, dst_crs, *bins_src.bounds)

        union_bounds = (
            min(lc_bounds[0], bins_bounds[0]),
            min(lc_bounds[1], bins_bounds[1]),
            max(lc_bounds[2], bins_bounds[2]),
            max(lc_bounds[3], bins_bounds[3]),
        )

        try:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                dst_crs, dst_crs,
                width=None, height=None,
                left=union_bounds[0], bottom=union_bounds[1],
                right=union_bounds[2], top=union_bounds[3],
                resolution=dst_res,
            )
        except Exception:
            dst_transform = from_bounds(
                union_bounds[0], union_bounds[1], union_bounds[2], union_bounds[3],
                int((union_bounds[2] - union_bounds[0]) / dst_res),
                int((union_bounds[3] - union_bounds[1]) / dst_res),
            )
            dst_width = int((union_bounds[2] - union_bounds[0]) / dst_res)
            dst_height = int((union_bounds[3] - union_bounds[1]) / dst_res)

        if dst_transform is None:
            raise ValueError(f"Failed to create transform for CRS {dst_crs} with resolution {dst_res}")

        lc_nodata = lc_src.nodata if lc_src.nodata is not None else -32768
        bins_nodata = bins_src.nodata if bins_src.nodata is not None else 99

    with rasterio.open(land_cover_path) as lc_src:
        lc_data = lc_src.read(1)
        lc_data_reprojected = np.full((dst_height, dst_width), lc_nodata, dtype=lc_src.dtypes[0])

        reproject(
            source=lc_data,
            destination=lc_data_reprojected,
            src_transform=lc_src.transform,
            src_crs=lc_src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=lc_src.nodata,
            dst_nodata=lc_nodata,
        )

    with rasterio.open(productivity_bins) as bins_src:
        bins_data_src = bins_src.read(1).astype(np.int16)
        bins_data_reprojected = np.full((dst_height, dst_width), bins_nodata, dtype=np.int16)

        reproject(
            source=bins_data_src,
            destination=bins_data_reprojected,
            src_transform=bins_src.transform,
            src_crs=bins_src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=bins_src.nodata,
            dst_nodata=bins_nodata,
        )

        valid_bins = bins_data_reprojected[bins_data_reprojected != bins_nodata]
        if len(valid_bins) == 0:
            max_bin_id = 1
        else:
            max_bin_id = int(np.max(valid_bins))

    if dst_transform is None:
        raise ValueError(f"Transform is None. Cannot calculate pixel area.")

    px_w = dst_transform.a
    px_h = dst_transform.e

    if px_w is None or px_h is None:
        raise ValueError(
            f"Invalid transform: {dst_transform}. "
            f"px_w={px_w}, px_h={px_h}. "
            f"CRS: {dst_crs}, Resolution: {dst_res}"
        )

    px_w_abs = abs(float(px_w))
    px_h_abs = abs(float(px_h))

    if px_w_abs < 0.1 or px_h_abs < 0.1:
        raise ValueError(
            f"Raster appears to be in degrees (geographic CRS). "
            f"Transform pixel size: {px_w_abs} x {px_h_abs}. "
            f"For accurate area, reproject to an equal-area projection first."
        )

    px_area_km2 = (px_w_abs * px_h_abs) / 1e6

    if px_area_km2 is None or not np.isfinite(px_area_km2):
        raise ValueError(f"Failed to calculate pixel area.")

    unique_lc_classes = np.unique(lc_data_reprojected)
    unique_lc_classes = unique_lc_classes[unique_lc_classes != lc_nodata]
    print(unique_lc_classes)

    if land_cover_class_names is None:
        land_cover_class_names = {cls: f"Class {cls}" for cls in unique_lc_classes}

    results = []

    for lc_class in unique_lc_classes:
        lc_mask = lc_data_reprojected == lc_class
        lc_bins = bins_data_reprojected[lc_mask]

        degraded_mask = lc_bins == 1
        degraded_count = np.count_nonzero(degraded_mask)
        degraded_km2 = float(degraded_count) * float(px_area_km2)

        non_degraded_mask = (lc_bins >= 2) & (lc_bins < 99)
        non_degraded_count = np.count_nonzero(non_degraded_mask)
        non_degraded_km2 = float(non_degraded_count) * float(px_area_km2)

        no_data_mask = lc_bins == 99
        no_data_count = np.count_nonzero(no_data_mask)
        no_data_km2 = float(no_data_count) * float(px_area_km2)

        class_name = land_cover_class_names.get(lc_class, f"Class {lc_class}")

        results.append({
            "Land Cover Class": class_name,
            "Degraded (km²)": round(degraded_km2, 2),
            "Non-Degraded (km²)": round(non_degraded_km2, 2),
            "No Data (km²)": round(no_data_km2, 2),
        })

    df = pd.DataFrame(results)
    return df


def land_cover_conversion_productivity(
    initial_land_cover_path: Path,
    final_land_cover_path: Path,
    productivity_bins: Path,
    land_cover_class_names: Optional[dict[int, str]] = None,
    output_dir: Optional[Path] = None,
    dst_crs: str = "EPSG:6933",
    dst_res: float = 250.0,
    bin_labels: Optional[dict[int, str]] = None,
) -> pd.DataFrame:
    """分析土地覆盖转换及其对应的生产力动态。"""
    if bin_labels is None:
        bin_labels = {
            1: "Declining",
            2: "Moderate Decline",
            3: "Stressed",
            4: "Stable",
            5: "Increasing",
        }

    with (
        rasterio.open(initial_land_cover_path) as lc1_src,
        rasterio.open(final_land_cover_path) as lc2_src,
        rasterio.open(productivity_bins) as bins_src,
    ):
        lc1_bounds = transform_bounds(lc1_src.crs, dst_crs, *lc1_src.bounds)
        lc2_bounds = transform_bounds(lc2_src.crs, dst_crs, *lc2_src.bounds)
        bins_bounds = transform_bounds(bins_src.crs, dst_crs, *bins_src.bounds)

        union_bounds = (
            min(lc1_bounds[0], lc2_bounds[0], bins_bounds[0]),
            min(lc1_bounds[1], lc2_bounds[1], bins_bounds[1]),
            max(lc1_bounds[2], lc2_bounds[2], bins_bounds[2]),
            max(lc1_bounds[3], lc2_bounds[3], bins_bounds[3]),
        )

        try:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                dst_crs, dst_crs,
                width=None, height=None,
                left=union_bounds[0], bottom=union_bounds[1],
                right=union_bounds[2], top=union_bounds[3],
                resolution=dst_res,
            )
        except Exception:
            dst_transform = from_bounds(
                union_bounds[0], union_bounds[1], union_bounds[2], union_bounds[3],
                int((union_bounds[2] - union_bounds[0]) / dst_res),
                int((union_bounds[3] - union_bounds[1]) / dst_res),
            )
            dst_width = int((union_bounds[2] - union_bounds[0]) / dst_res)
            dst_height = int((union_bounds[3] - union_bounds[1]) / dst_res)

        if dst_transform is None:
            raise ValueError(f"Failed to create transform for CRS {dst_crs} with resolution {dst_res}")

        lc1_nodata = lc1_src.nodata if lc1_src.nodata is not None else -32768
        lc2_nodata = lc2_src.nodata if lc2_src.nodata is not None else -32768
        bins_nodata = bins_src.nodata if bins_src.nodata is not None else 99

    with rasterio.open(initial_land_cover_path) as lc1_src:
        lc1_data = lc1_src.read(1)
        lc1_data_reprojected = np.full((dst_height, dst_width), lc1_nodata, dtype=lc1_src.dtypes[0])

        reproject(
            source=lc1_data,
            destination=lc1_data_reprojected,
            src_transform=lc1_src.transform,
            src_crs=lc1_src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=lc1_src.nodata,
            dst_nodata=lc1_nodata,
        )

    with rasterio.open(final_land_cover_path) as lc2_src:
        lc2_data = lc2_src.read(1)
        lc2_data_reprojected = np.full((dst_height, dst_width), lc2_nodata, dtype=lc2_src.dtypes[0])

        reproject(
            source=lc2_data,
            destination=lc2_data_reprojected,
            src_transform=lc2_src.transform,
            src_crs=lc2_src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=lc2_src.nodata,
            dst_nodata=lc2_nodata,
        )

    with rasterio.open(productivity_bins) as bins_src:
        bins_data_src = bins_src.read(1).astype(np.int16)
        bins_data_reprojected = np.full((dst_height, dst_width), bins_nodata, dtype=np.int16)

        reproject(
            source=bins_data_src,
            destination=bins_data_reprojected,
            src_transform=bins_src.transform,
            src_crs=bins_src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=bins_src.nodata,
            dst_nodata=bins_nodata,
        )

    px_w = abs(float(dst_transform.a))
    px_h = abs(float(dst_transform.e))

    if px_w < 0.1 or px_h < 0.1:
        raise ValueError(
            f"Raster appears to be in degrees (geographic CRS). "
            f"Transform pixel size: {px_w} x {px_h}. "
            f"For accurate area, reproject to an equal-area projection first."
        )

    px_area_km2 = (px_w * px_h) / 1e6

    if px_area_km2 is None or not np.isfinite(px_area_km2):
        raise ValueError(f"Failed to calculate pixel area.")

    valid_mask = (
        (lc1_data_reprojected != lc1_nodata)
        & (lc2_data_reprojected != lc2_nodata)
        & (bins_data_reprojected != bins_nodata)
        & (bins_data_reprojected != 99)
    )

    unique_lc1_classes = np.unique(lc1_data_reprojected[valid_mask])
    unique_lc1_classes = unique_lc1_classes[unique_lc1_classes != lc1_nodata]

    unique_lc2_classes = np.unique(lc2_data_reprojected[valid_mask])
    unique_lc2_classes = unique_lc2_classes[unique_lc2_classes != lc2_nodata]

    if land_cover_class_names is None:
        all_classes = np.unique(np.concatenate([unique_lc1_classes, unique_lc2_classes]))
        land_cover_class_names = {cls: f"Class {cls}" for cls in all_classes}

    results = []

    valid_bins_arr = bins_data_reprojected[valid_mask]
    unique_bins = np.unique(valid_bins_arr)
    unique_bins = unique_bins[unique_bins != bins_nodata]
    unique_bins = unique_bins[unique_bins != 99]

    for lc1_class in unique_lc1_classes:
        for lc2_class in unique_lc2_classes:
            conversion_mask = (
                valid_mask
                & (lc1_data_reprojected == lc1_class)
                & (lc2_data_reprojected == lc2_class)
            )

            if not np.any(conversion_mask):
                continue

            conversion_bins = bins_data_reprojected[conversion_mask]

            bin_areas = {}
            for bin_id in unique_bins:
                bin_mask = conversion_bins == bin_id
                bin_count = np.count_nonzero(bin_mask)
                bin_areas[bin_id] = float(bin_count) * float(px_area_km2)

            from_name = land_cover_class_names.get(lc1_class, f"Class {lc1_class}")
            to_name = land_cover_class_names.get(lc2_class, f"Class {lc2_class}")

            result_row = {
                "From": from_name,
                "To": to_name,
            }

            for bin_id in sorted(unique_bins):
                label = bin_labels.get(bin_id, f"Bin {bin_id}")
                area = round(bin_areas.get(bin_id, 0.0), 1)
                result_row[f"{label} (km²)"] = area

            has_data = any(bin_areas.get(bin_id, 0.0) > 0 for bin_id in unique_bins)
            if has_data:
                results.append(result_row)

    if len(results) == 0:
        print("Warning: No valid land cover conversions found.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    column_order = ["From", "To"]
    for bin_id in sorted(unique_bins):
        label = bin_labels.get(bin_id, f"Bin {bin_id}")
        column_name = f"{label} (km²)"
        if column_name in df.columns:
            column_order.append(column_name)

    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]

    if len(df) == 0:
        print("Warning: No valid land cover conversions found.")
        return df

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        initial_name = initial_land_cover_path.stem
        final_name = final_land_cover_path.stem
        bins_name = productivity_bins.stem

        output_file = output_dir / f"land_conversion_{initial_name}_to_{final_name}_{bins_name}.csv"
        df.to_csv(output_file, index=False)
        print(f"Land cover conversion table saved to: {output_file}")

    return df
