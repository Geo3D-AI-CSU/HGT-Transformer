import os
import glob
import xarray as xr

IO_DIR = r"D:/GCN-Transformer/data/CAMS-IO"
EGG4_DIR = r"D:/GCN-Transformer/data/CAMS-EGG4"
OUTPUT_DIR = r"D:/GCN-Transformer/processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_egg4_grid():
    files = sorted(glob.glob(os.path.join(EGG4_DIR, "*.nc")))
    if not files:
        raise FileNotFoundError("❌ EGG4 文件夹中未找到 .nc 文件")
    sample = xr.open_dataset(files[0])
    lats = sample["latitude"].values
    lons = sample["longitude"].values
    sample.close()
    return lats, lons

def interpolate_io_to_egg4(io_file, target_lats, target_lons):
    ds_io = xr.open_dataset(io_file)
    var = ds_io["XCO2"]

    # 使用最近邻插值
    interp = var.interp(
        latitude=target_lats,
        longitude=target_lons,
        method="nearest"
    )

    # 再进行双向补齐，确保 0 NaN
    interp = interp.ffill("latitude").bfill("latitude")
    interp = interp.ffill("longitude").bfill("longitude")

    ds_io.close()
    return interp

def main():
    target_lats, target_lons = load_egg4_grid()
    print(f"目标网格: {len(target_lats)} × {len(target_lons)}")

    files = sorted(glob.glob(os.path.join(IO_DIR, "*.nc")))
    print(f"找到 {len(files)} 个 CAMS-IO 文件")

    all_interp = []
    for fp in files:
        print(f"处理 {fp}")
        interp = interpolate_io_to_egg4(fp, target_lats, target_lons)
        all_interp.append(interp)

    ds_all = xr.concat(all_interp, dim="time")

    # 确认 0 NaN
    nan_total = int(ds_all.isnull().sum())
    print(f"插值后 NaN 数: {nan_total}")

    out_fp = os.path.join(OUTPUT_DIR, "CAMS-IO-interpolation.nc")
    ds_all.to_netcdf(out_fp)
    print(f"✔ 已保存: {out_fp}")

if __name__ == "__main__":
    main()
