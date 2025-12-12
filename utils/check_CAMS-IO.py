import os
import xarray as xr
import numpy as np

def check_cams_io_info(data_dir):
    # 找到文件夹下的第一个 nc 文件
    files = [f for f in os.listdir(data_dir) if f.endswith(".nc")]
    if not files:
        print("❌ 没有找到 nc 文件")
        return
    
    file_path = os.path.join(data_dir, files[0])
    print(f"正在读取文件: {file_path}")
    
    # 打开 NetCDF 文件
    ds = xr.open_dataset(file_path)
    
    # ========== 空间信息 ==========
    if "latitude" in ds and "longitude" in ds:
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        
        # 计算分辨率（取相邻差值的平均）
        lat_res = np.mean(np.diff(lats))
        lon_res = np.mean(np.diff(lons))
        
        print(f"🌍 空间分辨率: {abs(lat_res):.3f}° × {abs(lon_res):.3f}°")
        print(f"纬度范围: {lats.min()} ~ {lats.max()} (共 {len(lats)} 点)")
        print(f"经度范围: {lons.min()} ~ {lons.max()} (共 {len(lons)} 点)")
    else:
        print("❌ 没有找到 latitude / longitude 变量")
    
    # ========== 时间信息 ==========
    if "time" in ds:
        times = ds["time"].values
        if len(times) > 1:
            time_diffs = np.diff(times).astype("timedelta64[h]").astype(int)
            avg_time_res = np.mean(time_diffs)
            print(f"⏱ 时间分辨率: 平均 {avg_time_res:.1f} 小时")
            print(f"时间范围: {str(times[0])} ~ {str(times[-1])} (共 {len(times)} 个时间点)")
        else:
            print("⚠️ 时间维度只有 1 个点")
    else:
        print("❌ 没有找到 time 变量")
    
    # ========== 变量信息 ==========
    vars_list = [v for v in ds.data_vars]
    print(f"📌 数据变量: {vars_list}")
    
    ds.close()

if __name__ == "__main__":
    data_dir = r"D:\HGT-Transformer\data\CAMS-IO"
    check_cams_io_info(data_dir)
