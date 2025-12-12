import os
import glob
import xarray as xr

IO_DIR = r"D:/HGT-Transformer/data/CAMS-IO"
EGG4_DIR = r"D:/HGT-Transformer/data/CAMS-EGG4"
OUTPUT_DIR = r"D:/HGT-Transformer/processed_data"
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

    interp = var.interp(latitude=target_lats, longitude=target_lons)
    ds_io.close()
    return interp

def main():
    target_lats, target_lons = load_egg4_grid()
    print(f"✅ 目标网格: {len(target_lats)} × {len(target_lons)} (0.75°)")

    files = sorted(glob.glob(os.path.join(IO_DIR, "*.nc")))
    print(f"📂 找到 {len(files)} 个 IO 文件")

    all_interp = []
    for fp in files:
        print(f"👉 处理 {fp}")
        interp = interpolate_io_to_egg4(fp, target_lats, target_lons)
        all_interp.append(interp)

    ds_all = xr.concat(all_interp, dim="time")

    out_fp = os.path.join(OUTPUT_DIR, "CAMS-IO-interpolation.nc")
    ds_all.to_netcdf(out_fp)
    print(f"🎉 已保存插值后的数据: {out_fp}")

    print("\n🔍 数据检查:")
    print(ds_all)

    print(f"⏱ 时间范围: {str(ds_all['time'].values[0])} ~ {str(ds_all['time'].values[-1])}")

if __name__ == "__main__":
    main()
