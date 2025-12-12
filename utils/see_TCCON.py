import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import scipy.stats
import pandas as pd

# === 全局绘图风格设置（中文 + Times New Roman） ===
plt.rcParams['font.family'] = ['SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 150


def plot_single_station_xco2(ds, station_code, output_dir):
    """绘制单站点 XCO₂ 时间序列（中文版本）"""
    os.makedirs(output_dir, exist_ok=True)

    if 'xco2' not in ds.variables:
        print(f"警告：{station_code} 文件缺少 XCO₂ 数据")
        return

    # ========= 获取站点名称与经纬度 =========
    location = ds.attrs.get('short_location', ds.attrs.get('location', '未知站点'))

    lat = ds['lat'].values if 'lat' in ds.variables else None
    lon = ds['long'].values if 'long' in ds.variables else None
    if lat is not None and lon is not None:
        latitude = float(scipy.stats.mode(lat, keepdims=True).mode[0])
        longitude = float(scipy.stats.mode(lon, keepdims=True).mode[0])
        location_info = f"{location}（{latitude:.2f}°N, {longitude:.2f}°E）"
    else:
        location_info = location

    # ========= 读取时间与 XCO₂ =========
    time = pd.to_datetime(ds.time.values)
    xco2 = ds.xco2.values

    # ========= 绘图 =========
    plt.figure(figsize=(15, 8))
    plt.plot(time, xco2, 'b.', alpha=0.5, markersize=2)

    # 线性拟合
    z = np.polyfit(range(len(time)), xco2, 1)
    p = np.poly1d(z)
    plt.plot(time, p(range(len(time))), "r--", linewidth=1.5)

    # ========= 图标题（中文）=========
    plt.title(f'TCCON 站点 {station_code} - {location_info}\nXCO₂ 时间序列')

    # ========= 坐标轴名称（中文）=========
    plt.xlabel('时间')
    plt.ylabel('XCO₂（ppm）')

    plt.grid(True, alpha=0.3)

    # ========= 平均值和标准差（中文）=========
    mean_xco2 = np.nanmean(xco2)
    std_xco2 = np.nanstd(xco2)

    plt.text(
        0.02, 0.98,
        f'平均值：{mean_xco2:.2f} ppm\n标准差：{std_xco2:.2f} ppm',
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # ========= 保存 =========
    plt.savefig(os.path.join(output_dir, f'{station_code}_xco2.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"已保存：{station_code}_xco2.png")


def main():
    input_dir = r"D:/GCN-Transformer/data/TCCON"
    output_dir = r"D:/GCN-Transformer/result"

    os.makedirs(output_dir, exist_ok=True)

    tccon_files = sorted(glob.glob(os.path.join(input_dir, "*.nc")))
    print(f"共发现 {len(tccon_files)} 个 TCCON 站点文件")

    for file in tccon_files:
        try:
            station_code = os.path.basename(file).split('_')[0]
            print(f"\n正在处理 {station_code} ...")

            ds = xr.open_dataset(file)
            plot_single_station_xco2(ds, station_code, output_dir)
            ds.close()

        except Exception as e:
            print(f"处理文件时出错：{file}，原因：{str(e)}")

    print("\n全部站点 XCO₂ 时间序列已绘制并保存到：", output_dir)


if __name__ == "__main__":
    main()
