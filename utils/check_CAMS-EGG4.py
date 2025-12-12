import os
import glob
import xarray as xr
import numpy as np

# === 配置 ===
DATA_DIR = r"D:/GCN-Transformer/data/CAMS-EGG4"


def check_cams_egg4(data_dir):
    # 找到一个文件
    files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    if not files:
        print(f"❌ 未找到任何文件: {data_dir}")
        return
    fp = files[0]
    print(f"正在读取文件: {fp}")

    ds = xr.open_dataset(fp)

    # 维度信息
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    time = ds["time"].values

    # 分辨率
    dlat = float(np.abs(lat[1] - lat[0]))
    dlon = float(np.abs(lon[1] - lon[0]))

    # 时间分辨率（小时）
    if len(time) > 1:
        dt = (time[1] - time[0]) / np.timedelta64(1, "h")
    else:
        dt = None

    print(f"🌍 空间分辨率: {dlat:.3f}° × {dlon:.3f}°")
    print(f"纬度范围: {lat.min()} ~ {lat.max()} (共 {len(lat)} 点)")
    print(f"经度范围: {lon.min()} ~ {lon.max()} (共 {len(lon)} 点)")

    if dt is not None:
        print(f"⏱ 时间分辨率: 平均 {dt:.1f} 小时")
    print(f"时间范围: {str(time[0])} ~ {str(time[-1])} (共 {len(time)} 个时间点)")

    print(f"📌 数据变量: {list(ds.data_vars.keys())}")

    ds.close()


if __name__ == "__main__":
    check_cams_egg4(DATA_DIR)
