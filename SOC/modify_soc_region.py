import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask


# ================= 1. 路径配置 =================

SHP_PATH = r"C:\deepOptica\deep\fromFeishu\abu_dhabi_all\abu_dhabi_all.shp"

SOC_DIR = r"C:\deepOptica\deep\UAE\Part2\SOC_qyx\SOC\soc_output"

OUTPUT_DIR = r"C:\deepOptica\deep\UAE\Part2\SOC_qyx\SOC\soc_output2"

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]


# ================= 2. shp NAME 到 Emirate 的归并规则 =================
# 你的 shp 里 Abu Dhabi 被拆成 3 个区域，Dubai 被拆成多个 Sector
# 这里统一归并成 7 个 Emirate

def map_to_emirate(name):
    name = str(name).strip()

    if name in ["Abu Dhabi Region", "Al - Ain Region", "Al - Dhafra Region"]:
        return "Abu Dhabi"

    if name.startswith("Dubai"):
        return "Dubai"

    if name == "Ajman":
        return "Ajman"

    if name == "Al - Fujairah":
        return "Fujairah"

    if name == "Ras al - Khaimah":
        return "Ras Al Khaimah"

    if name == "Sharjah":
        return "Sharjah"

    if name == "Umm al - Qiwain":
        return "Umm Al Quwain"

    return name


# ================= 3. 面积计算函数 =================

def area_km2_from_mask(mask_array, transform, crs):
    """
    根据 True/False 掩膜计算面积 km2。
    如果 raster 是投影坐标系，直接用像元面积。
    如果 raster 是经纬度坐标系，按纬度修正像元面积。
    """
    if np.sum(mask_array) == 0:
        return 0.0

    res_x = transform[0]
    res_y = -transform[4]
    height = mask_array.shape[0]

    is_geographic = False
    if crs is not None:
        is_geographic = crs.is_geographic

    if is_geographic:
        R = 6371.0  # km
        y_origin = transform[5]
        pixel_height = transform[4]

        rows = np.arange(height) + 0.5
        latitudes = y_origin + rows * pixel_height
        lat_rad = np.radians(latitudes)

        pixel_width_km = abs(res_x) * (np.pi / 180.0) * R * np.cos(lat_rad)
        pixel_height_km = abs(res_y) * (np.pi / 180.0) * R

        row_area_km2 = pixel_width_km * pixel_height_km
        count_per_row = np.sum(mask_array, axis=1)

        return float(np.sum(count_per_row * row_area_km2))

    else:
        pixel_area_km2 = abs(res_x * res_y) / 1_000_000.0
        return float(np.sum(mask_array) * pixel_area_km2)


def calculate_degraded_improved_area(tif_path, gdf_region):
    """
    对某个 tif，在给定矢量区域内统计 degraded / improved 面积。
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        # shp 重投影到 raster CRS
        if gdf_region.crs != crs:
            gdf_region = gdf_region.to_crs(crs)

        geoms = list(gdf_region.geometry)

        region_mask = geometry_mask(
            geoms,
            out_shape=data.shape,
            transform=transform,
            invert=True,
            all_touched=False
        )

        valid = region_mask & np.isfinite(data)

        if nodata is not None:
            valid = valid & (~np.isclose(data, nodata))

        mask_degraded = valid & (data == 2)
        mask_improved = valid & (data == 1)

        degraded_area = area_km2_from_mask(mask_degraded, transform, crs)
        improved_area = area_km2_from_mask(mask_improved, transform, crs)

        return degraded_area, improved_area


# ================= 4. 读取并整理 shp =================

gdf = gpd.read_file(SHP_PATH)

print("===== shp 字段 =====")
print(gdf.columns)

print("\n===== NAME 字段内容 =====")
print(gdf["NAME"].tolist())

gdf["Emirate"] = gdf["NAME"].apply(map_to_emirate)

# dissolve：把 Abu Dhabi 的三个区域、Dubai 的多个 sector 合并成一个 Emirate
gdf_emirate = gdf.dissolve(by="Emirate", as_index=False)

# 按报告表格顺序排列
emirate_order = [
    "Abu Dhabi",
    "Ajman",
    "Dubai",
    "Fujairah",
    "Ras Al Khaimah",
    "Sharjah",
    "Umm Al Quwain"
]

gdf_emirate["order"] = gdf_emirate["Emirate"].apply(
    lambda x: emirate_order.index(x) if x in emirate_order else 999
)
gdf_emirate = gdf_emirate.sort_values("order").drop(columns="order")

print("\n===== 归并后的 Emirate =====")
print(gdf_emirate["Emirate"].tolist())


# ================= 5. 输出 Table 4-5：2020 年按 Emirate 统计 =================

target_year = 2024
tif_2020 = os.path.join(SOC_DIR, f"soc_{target_year}.tif")

rows_2020 = []

for _, row in gdf_emirate.iterrows():
    emirate = row["Emirate"]
    region_gdf = gpd.GeoDataFrame([row], geometry="geometry", crs=gdf_emirate.crs)

    degraded_area, improved_area = calculate_degraded_improved_area(
        tif_path=tif_2020,
        gdf_region=region_gdf
    )

    rows_2020.append({
        "Emirate": emirate,
        "Area of Degraded Land (km2)": round(degraded_area, 3),
        "Area of Improved Land (km2)": round(improved_area, 3)
    })

df_2020 = pd.DataFrame(rows_2020)

out_csv_2020 = os.path.join(OUTPUT_DIR, "SOC_by_emirate_2024.csv")


df_2020.to_csv(out_csv_2020, index=False, encoding="utf-8-sig")


print("\n===== Table 4-5: SOC by Emirate for 2020 =====")
print(df_2020)
print(f"\n已保存: {out_csv_2020}")



