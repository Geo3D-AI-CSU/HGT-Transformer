import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

from matplotlib.colors import LinearSegmentedColormap

colors = [
    (1.0, 1.0, 1.0),   # 白色
    (1.0, 0.6, 0.6),   # 浅红
    (0.8, 0.0, 0.0)    # 深红
]
cmap_red = LinearSegmentedColormap.from_list("white_to_red", colors)
vmin = 0
vmax = 25
# =====================
# 文件与路径设置
# =====================
era5_file = "D:/GCN-Transformer/data/ERA-5/ERA5_2020.nc"  # 替换为你的ERA5文件路径
output_dir = "D:/GCN-Transformer/results/"
os.makedirs(output_dir, exist_ok=True)

# =====================
# 1️⃣ 读取ERA5数据
# =====================
ds = xr.open_dataset(era5_file)

# 常见变量名示例：
# ds['u10'], ds['v10']  -> 10米风
# 或 ds['u'], ds['v']   -> 高空风场
print(ds)

# 假设使用地表风 (10 m)
u = ds['u10'].sel(valid_time='2020-01-01T00:00:00')
v = ds['v10'].sel(valid_time='2020-01-01T00:00:00')

# 计算风速
wind_speed = np.sqrt(u**2 + v**2)

# 经纬度
lon = u['longitude']
lat = u['latitude']

# =====================
# 2️⃣ 绘制全球风场图
# =====================
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.size'] = 10

fig = plt.figure(figsize=(10, 4))
ax = plt.axes(projection=ccrs.Robinson())

# 绘制风速底色
im = plt.pcolormesh(
    lon, lat, wind_speed,
    transform=ccrs.PlateCarree(),
    cmap=cmap_red,
    vmin=vmin,
    vmax=vmax
)

# 绘制风向箭头（抽稀显示）
skip = (slice(None, None, 15), slice(None, None, 15))
ax.quiver(
    lon[skip[1]], lat[skip[0]],
    u.values[skip], v.values[skip],
    transform=ccrs.PlateCarree(),
    color='black', scale=700
)

ax.coastlines(linewidth=0.5)
ax.set_global()

# 颜色条
cbar = plt.colorbar(im, orientation='horizontal', pad=0.05, fraction=0.05)
cbar.set_label('wind speed (m/s)')

ax.set_title('Global 10 m wind field from ERA-5\n(January 1st,2020,00:00 UTC)', fontsize=10, pad=8)
plt.tight_layout()

# =====================
# 3️⃣ 保存高分辨率图片
# =====================
plt.savefig(f"{output_dir}ERA-5.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ ERA5 global wind field figure saved successfully.")
