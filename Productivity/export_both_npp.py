"""Export both NPP methods for 2000-2025 (26 bands)."""
from __future__ import annotations
import ee
import time

ee.Initialize(project='robust-builder-418813')

ROI_ASSET = "projects/robust-builder-418813/assets/abu_dhabi_all"
ROI = ee.FeatureCollection(ROI_ASSET).geometry()
roi_region = ROI.bounds()

STEP_DAYS = 16
START_YEAR = 2000
END_YEAR = 2025

ref_proj = ee.ImageCollection("MODIS/061/MOD13Q1").first().select("NDVI").projection()


def prep_full(img):
    ndvi = img.select("NDVI").multiply(0.0001).rename("ndvi")
    doy = ee.Date(img.get("system:time_start")).getRelative("day", "year").add(1)
    doy_band = ee.Image.constant(doy).rename("doy").toInt16()
    return ndvi.addBands(doy_band).copyProperties(img, ["system:time_start"])


# ---------------------------------------------------------------------------
# Method 1: Fixed peak_doy=24
# ---------------------------------------------------------------------------
print("=== Method 1: Fixed peak_doy=24 ===")

FIXED_PEAK_DOY = 24
W_DAYS = 64
peak_doy_ee = ee.Number(FIXED_PEAK_DOY)
start_doy = peak_doy_ee.subtract(W_DAYS)
end_doy = peak_doy_ee.add(W_DAYS)


def in_window_fixed(img):
    doy = img.select("doy").toInt16()
    normal = ee.Number(start_doy).gte(1).And(ee.Number(end_doy).lte(366)).And(
        ee.Number(start_doy).lte(end_doy))

    def normal_mask():
        return doy.gte(start_doy).And(doy.lte(end_doy))

    def wrap_mask():
        return doy.gte(ee.Number(start_doy).add(366)).Or(doy.lte(end_doy))

    mask = ee.Image(ee.Algorithms.If(normal, normal_mask(), wrap_mask()))
    return img.updateMask(mask)


ic_full = (ee.ImageCollection("MODIS/061/MOD13Q1")
           .filterDate(f"{START_YEAR}-01-01", f"{END_YEAR+1}-01-01")
           .map(prep_full))

print(f"  Images: {ic_full.size().getInfo()}")


def make_auc_fixed(y):
    y = ee.Number(y)
    ys = ee.Date.fromYMD(y, 1, 1)
    ye = ys.advance(1, "year")
    year_ic = ic_full.filterDate(ys, ye).map(in_window_fixed)
    empty = ee.Image.constant(0).rename("auc").updateMask(ee.Image.constant(0))
    auc = ee.Image(
        ee.Algorithms.If(year_ic.size().gt(0),
                         year_ic.select("ndvi").sum().multiply(STEP_DAYS).rename("auc"),
                         empty)
    ).setDefaultProjection(ref_proj)
    return (auc.clip(ROI).set("year", y).set("system:time_start", ys.millis()))


years = ee.List.sequence(START_YEAR, END_YEAR)
ic_fixed = ee.ImageCollection(years.map(make_auc_fixed))
print(f"  AUC images: {ic_fixed.size().getInfo()}")

task_fixed = ee.batch.Export.image.toDrive(
    image=ic_fixed.toBands().clip(ROI),
    description="NPP_2000_2025_fixed24",
    folder="NDVI_PROXY",
    fileNamePrefix="annual_auc_toBands_2000_2025_fixed24",
    region=roi_region,
    crs="EPSG:4326",
    scale=250,
    maxPixels=1e13,
)
task_fixed.start()
print(f"  Task started: {task_fixed.id}")

# ---------------------------------------------------------------------------
# Method 2: Annual NDVI sum (no window)
# ---------------------------------------------------------------------------
print("\n=== Method 2: Annual NDVI sum (no window) ===")


def make_auc_sum(y):
    y = ee.Number(y)
    ys = ee.Date.fromYMD(y, 1, 1)
    ye = ys.advance(1, "year")
    year_ic = ic_full.filterDate(ys, ye)
    empty = ee.Image.constant(0).rename("auc").updateMask(ee.Image.constant(0))
    auc = ee.Image(
        ee.Algorithms.If(year_ic.size().gt(0),
                         year_ic.select("ndvi").sum().multiply(STEP_DAYS).rename("auc"),
                         empty)
    ).setDefaultProjection(ref_proj)
    return (auc.clip(ROI).set("year", y).set("system:time_start", ys.millis()))


ic_sum = ee.ImageCollection(years.map(make_auc_sum))
print(f"  AUC images: {ic_sum.size().getInfo()}")

task_sum = ee.batch.Export.image.toDrive(
    image=ic_sum.toBands().clip(ROI),
    description="NPP_2000_2025_annsum",
    folder="NDVI_PROXY",
    fileNamePrefix="annual_auc_toBands_2000_2025_annsum",
    region=roi_region,
    crs="EPSG:4326",
    scale=250,
    maxPixels=1e13,
)
task_sum.start()
print(f"  Task started: {task_sum.id}")

print("\nBoth tasks started. Polling until complete...")

while task_fixed.active() or task_sum.active():
    f = "DONE" if not task_fixed.active() else "running"
    s = "DONE" if not task_sum.active() else "running"
    print(f"  fixed: {f}  annsum: {s}")
    time.sleep(60)

print("\nBoth exports completed!")
