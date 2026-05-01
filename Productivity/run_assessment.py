"""
SDG 15.3.1 土地退化评估 — 完整运行脚本

用法:
    # 单独运行某个指标（方便调试）
    python run_assessment.py --metrics trend  --years 2015 2023
    python run_assessment.py --metrics state  --years 2015 2023
    python run_assessment.py --metrics performance --years 2015 2023

    # 三个指标全跑
    python run_assessment.py --metrics all --years 2015 2023

    # 组合三个指标（需要先跑完 trend/state/performance）
    python run_assessment.py --combine --years 2015 2023
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from land_productivity import (
    initalize_args,
    generate_report_period,
    combine_metrics_degradation,
    setup_logger,
    pixel_area_km2,
)


def run_single_metric(metrics: str, years: list[int], log: object) -> pd.DataFrame:
    """运行单个指标的评估，返回逐年结果 DataFrame。"""
    final_result = []
    args = None

    for year in years:
        print(f"\n{'='*60}")
        print(f"  报告期: {year}  |  指标: {metrics}")
        print(f"{'='*60}")

        args = initalize_args(metrics, year)

        # 指标特定的参数
        if metrics == "trend":
            args.trend_args.mask = True
            args.trend_args.slope_threshold = 0.15
        elif metrics == "state":
            args.state_args.mask = True
            args.state_args.slope_threshold = 1
            args.z_edges = [-np.inf, -2.56, 2.56, np.inf]
        elif metrics == "performance":
            args.z_edges = [-np.inf, 0.6, 1.5, np.inf]

        result = generate_report_period(metrics, args)
        final_result.append(result)

    df = pd.DataFrame(final_result)
    outdir = args.out_dir.parent / "overall"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{years[-1]}_{metrics}_by_year.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n逐年结果已保存: {csv_path}")
    print(df)
    return df


def run_combine(years: list[int]) -> pd.DataFrame:
    """组合三个指标的退化判断。"""
    final_result = []

    for year in years:
        trend_path = Path(f"../output/trend/{year}/overall.tif")
        state_path = Path(f"../output/state/{year}/overall.tif")
        perf_path = Path(f"../output/performance/{year}/overall.tif")

        missing = []
        for p, name in [(trend_path, "trend"), (state_path, "state"), (perf_path, "performance")]:
            if not p.exists():
                missing.append(f"{name}: {p}")

        if missing:
            print(f"\n⚠ 跳过 {year}，缺少文件:")
            for m in missing:
                print(f"   {m}")
            continue

        output_dir = Path("../output/combined")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"combined_degradation_{year}_table45.tif"
        combine_metrics_degradation(
            trend_degrading_path=trend_path,
            state_degrading_path=state_path,
            performance_degrading_path=perf_path,
            output_path=output_path,
            table_version="4-5",
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)
            px_area = pixel_area_km2(src)
            degraded = (data == 2).sum() * px_area
            improved = (data == 1).sum() * px_area

        print(f"  {year}: degraded={degraded:.2f} km², improved={improved:.2f} km²")
        final_result.append({"year": year, "degraded_km2": degraded, "improved_km2": improved})

    df = pd.DataFrame(final_result)
    outdir = Path("../output/combined/overall")
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"combined_{years[-1]}_table45.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n组合结果已保存: {csv_path}")
    print(df)
    return df


def main():
    parser = argparse.ArgumentParser(description="SDG 15.3.1 土地退化评估")
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        choices=["trend", "state", "performance", "all"],
        help="要运行的指标 (默认: all)",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs=2,
        default=[2015, 2024],
        metavar=("START", "END"),
        help="年份范围，包含起止年份 (默认: 2015 2024)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="组合三个指标的退化判断（需要先跑完 trend/state/performance）",
    )
    args = parser.parse_args()

    start_year, end_year = args.years
    years = list(range(start_year, end_year + 1))

    # 初始化日志
    log_path = Path("logs/assessment.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = setup_logger(name="sdg", log_file=log_path)

    if args.combine:
        run_combine(years)
        return

    metrics_list = ["trend", "state", "performance"] if args.metrics == "all" else [args.metrics]

    for metrics in metrics_list:
        print(f"\n{'#'*60}")
        print(f"#  开始运行: {metrics.upper()} 指标")
        print(f"{'#'*60}")
        run_single_metric(metrics, years, log)

    # 如果三个都跑完了，自动组合
    if args.metrics == "all":
        print(f"\n{'#'*60}")
        print(f"#  组合三个指标")
        print(f"{'#'*60}")
        run_combine(years)


if __name__ == "__main__":
    main()
