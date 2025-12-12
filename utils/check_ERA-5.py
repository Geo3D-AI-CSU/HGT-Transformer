import os
import xarray as xr

def list_era5_variables(data_dir="D:/GCN-Transformer/data/ERA-5"):
    """
    遍历 ERA-5 文件夹下的所有 .nc 文件，输出变量名和维度信息
    """
    files = [f for f in os.listdir(data_dir) if f.endswith(".nc")]
    if not files:
        print("❌ 没有找到 .nc 文件，请检查路径。")
        return

    print(f"找到 {len(files)} 个 ERA5 文件。")

    # 打开第一个文件，通常所有文件的变量结构相同
    first_file = os.path.join(data_dir, files[0])
    print(f"正在读取: {first_file}")
    
    ds = xr.open_dataset(first_file)

    print("\n📌 ERA5 文件包含的变量有：\n")
    for var in ds.data_vars:
        print(f"- {var}: {ds[var].dims} {ds[var].attrs.get('long_name', '')} ({ds[var].attrs.get('units', '')})")

    print("\n📌 ERA5 文件的坐标维度有：\n")
    for coord in ds.coords:
        print(f"- {coord}: {ds[coord].shape}")

    ds.close()


if __name__ == "__main__":
    list_era5_variables()
