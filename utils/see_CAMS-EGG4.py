import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

from matplotlib.colors import LinearSegmentedColormap

# =====================
# 保存路径设置
# =====================
output_dir = "D:/GCN-Transformer/results/"
os.makedirs(output_dir, exist_ok=True)

# =====================
# 绘图样式设置
# =====================
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.size'] = 10

colors = [
    (1.0, 1.0, 1.0),   # 白色
    (1.0, 0.6, 0.6),   # 浅红
    (0.8, 0.0, 0.0)    # 深红
]
cmap_red = LinearSegmentedColormap.from_list("white_to_red", colors)
vmin = 400
vmax = 420
# =====================
# 1. 读取 CAMS-IO 数据
# =====================
# 示例：替换为你的实际文件路径
ds_io = xr.open_dataset("D:\GCN-Transformer\data\CAMS-EGG4\predicted_xco2_2020-01-01.nc")

# 选择一个时间步（3 小时间隔）
xco2_io = ds_io['XCO2'].sel(time='2020-01-1T00:00:00')

# =====================
# 2. 绘制全球分布图
# =====================
fig = plt.figure(figsize=(10, 4))
ax = plt.axes(projection=ccrs.Robinson())

xco2_io.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap=cmap_red,
    vmin=vmin,
    vmax=vmax,
    cbar_kwargs={'label': 'XCO₂ (ppm)'}
)

ax.coastlines(linewidth=0.5)
ax.set_title('Global XCO₂ distribution from CAMS-EGG4\n(January 1st,2020,00:00 UTC)', fontsize=10, pad=8)

plt.tight_layout()

# =====================
# 3. 保存高分辨率图像
# =====================
plt.savefig(f"{output_dir}CAMS_EGG4.png", dpi=300, bbox_inches='tight')

plt.close()
