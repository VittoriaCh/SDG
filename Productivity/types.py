from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Union, Optional

import numpy as np
from rasterio.crs import CRS


@dataclass(frozen=False)
class MaskArgs:
    mask: bool = False
    slope_threshold: Union[float, list[float]] = 0.025


@dataclass(frozen=False)
class NppArgs:
    npp_path: Path
    height: int
    width: int
    nodata: int
    npp_bands: int
    base_year: int
    reporting_year: int
    npp_profile: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=False)
class BasicArgs:
    metrics: str
    name_field: str
    dst_crs: CRS
    dst_res: int
    reporting_year: int
    out_dir: Path
    shp_path: Path
    lecu_path: Path = Path("../data/LCEU/lecu.tif")

    z_edges: list[float] = field(default_factory=list)
    mask_low_productivity: bool = False

    npp_args: Optional[NppArgs] = None
    trend_args: MaskArgs = field(default_factory=MaskArgs)
    state_args: MaskArgs = field(default_factory=MaskArgs)
    performance_threshold: Union[float, list[float]] = field(
        default_factory=lambda: [0.5, 1.0]
    )


@dataclass
class PlotArgs:
    USE_BASEMAP: bool = True
    OUTDIR: Path = Path("../output/plots/emirate_local_figs")
    RADIUS_M: int = 30000
